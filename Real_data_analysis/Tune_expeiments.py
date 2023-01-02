


import numpy as np
import pandas as pd
import sys
from skbio.stats.composition import clr
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import MultivariateNormal

import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


sys.path.append('..')
sys.path.append('../jen_code')

import torch_wrapper_objects
import run_synthetic_simulations


def run_set(K, D, L, lr):

    met_embs=pd.read_csv('../Data/cdiff/met_embs_umap.csv', index_col=0)
    y_df=pd.read_csv('../Data/cdiff/met_data.csv', index_col=0)
    x_df=pd.read_csv('../Data/cdiff/mic_data.csv', index_col=0)
    mic_embs=pd.read_csv('../Data/cdiff/mic_embs.csv', index_col=0)

    met_embs=pd.read_csv('../Data/cdiff/med_embs_ginerprint_tanimoto.csv', index_col=0)

    met_embs=met_embs.loc[ ( ( y_df == y_df.min(axis=0) ).sum(axis=0) < 2 ), 
                           ( ( y_df == y_df.min(axis=0) ).sum(axis=0) < 2 )]
    y_df=y_df.loc[:, ( ( y_df == y_df.min(axis=0) ).sum(axis=0) < 2 ) ]


    y_df=pd.DataFrame( clr( abs( y_df.min() )+1e-4+y_df ),
                               columns=y_df.columns)


    x_df = x_df.div(x_df.sum(axis=1), axis=0)

    mic_embs=mic_embs.loc[ ((x_df>0).sum(axis=0)  > int( x_df.shape[0]/10 ) )]
    x_df = x_df.loc[:, ((x_df>0).sum(axis=0)  > int( x_df.shape[0]/10 ) ) ]

    y_df = y_df.loc[:, met_embs.sum(axis=1==0) > 0]
    met_embs = met_embs.loc[ met_embs.sum(axis=1==0) > 0 , met_embs.sum(axis=1==0) > 0 ]



    pc=KernelPCA(n_components=D, kernel='cosine')
    ss=StandardScaler()
    # pc=TSNE(n_components=2)
    mic_embs=pc.fit_transform(ss.fit_transform(mic_embs))


    pc=KernelPCA(n_components=D, kernel='cosine')
    ss=StandardScaler()
    # pc=TSNE(n_components=2)
    met_embs=pc.fit_transform(ss.fit_transform(met_embs))


    train_dataset = torch_wrapper_objects.m2mDataset(x_df.values[:140], y_df.values[:140], g=mic_embs)
    val_dataset = torch_wrapper_objects.m2mDataset(x_df.values[140:], y_df.values[140:], g=mic_embs)


    fitted_model = torch_wrapper_objects.run_training(train_dataset, 
                                                      val_dataset, 
                                                      met_embs,#.values,
                                                      mic_embs,#.values, 
                                                      learning_rate=lr,
                                                      logger_path='../Real_data_experiments/', 
                                                      batch_size=25, 
                                                      n_l=L, 
                                                      n_k=K, 
                                                      max_epochs=50, 
                                                      
                                                       )


    fitted=fitted_model.eval()
    
     ## calculate necessary components for preds
    eps = 1e-10

    temp = (1-2*eps)*torch.softmax(fitted.model.pi_met,1) + eps

    eye = torch.eye(fitted.model.met_embedding_dim).unsqueeze(0).expand(fitted.model.K, -1, -1)
    var = torch.exp(fitted.model.r_met).unsqueeze(-1).unsqueeze(-1).expand(-1,fitted.model.met_embedding_dim,
                                                                       fitted.model.met_embedding_dim)*eye
    mvn = MultivariateNormal(
        fitted.model.mu_met.unsqueeze(1).expand(-1,fitted.model.N_met,-1),var.unsqueeze(1).expand(
            -1,fitted.model.N_met,-1,-1)).log_prob(
        torch.Tensor(fitted.model.met_locs)).unsqueeze(1)

    ## run fitted models
    train_out = fitted.forward(train_dataset.x, train_dataset.y)[0]
    val_out = fitted.forward(val_dataset.x, val_dataset.y)[0]



    inds = torch.Tensor( 
            to_categorical(  
                        ( torch.log(temp.T) + mvn.squeeze(1) + \
                            Normal( train_out.T.unsqueeze(-1).expand(-1,-1,fitted.model.N_met),
                                                   torch.sqrt(torch.exp(fitted.model.sigma))  
                                          ).log_prob(train_dataset.y).sum(dim=1) ).argmax(dim=0), 
                            num_classes=mvn.shape[0] ) 
             ).bool()



    ## obtain train + validation predictions
    actual_train_preds = train_out.T.unsqueeze(-1).expand(-1,-1,fitted.model.N_met)\
                                       .transpose(0,2).transpose(0,1)[:, inds]


    actual_val_preds = val_out.T.unsqueeze(-1).expand(-1,-1,fitted.model.N_met)\
                                       .transpose(0,2).transpose(0,1)[:, inds]


    
    return( r2_score(train_dataset.y.flatten().detach().numpy(), 
                  actual_train_preds.flatten().detach().numpy()),
            r2_score(val_dataset.y.flatten().detach().numpy(), 
                  actual_val_preds.flatten().detach().numpy()),
           mean_squared_error(train_dataset.y.flatten().detach().numpy(), 
                  actual_train_preds.flatten().detach().numpy(), 
                  squared=False),
           mean_squared_error(val_dataset.y.flatten().detach().numpy(), 
                  actual_val_preds.flatten().detach().numpy(), 
                  squared=False)
          )


def main():
    k_set = [5, 10, 25, 50]
    l_set = [5,10, 25, 50]
    d_set = [2, 5, 10]
    lr_set = [1e-4, 1e-3, 1e-2]
    
    # runs the simulations for all cases outlined
    first_run=True
    for K in k_set:
        for L in l_set:
            for D in d_set:
                for lr in lr_set:
                    train_r2, val_r2, \
                        train_rmse, val_rmse = run_set(K, D, L, lr)
                    
                    run_summary={'K':[K], 
                                 'D':[D], 
                                 'L':[L], 
                                 'lr':[lr],
                                 'train_r2':[train_r2],
                                 'val_r2':[val_r2],
                                 'train_rmse':[train_rmse],
                                 'val_rmse':[val_rmse]
                                }
                    
                    if first_run:
                        pd.DataFrame({a:[b] for a,b in run_summary.items()})\
                                    .to_csv('simulation_results/result.csv')
                    
                    else:
                        pd.concat([pd.read_csv('simulation_results/result.csv', index_col=0), 
                                   pd.DataFrame({a:[b] for a,b in run_summary.items()})
                                  ]).to_csv('simulation_results/result.csv')
                        
                    first_run=False
    return(None)
    

if __name__=='__main__':
    main()