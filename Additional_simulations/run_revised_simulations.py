import numpy as np
import pandas as pd

import os
import sys
sys.path.append('..')
sys.path.append('../m2m')
sys.path.append('../jen_code')

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import seaborn as sns
import torch
from torch.distributions import MultivariateNormal
from sklearn.metrics import normalized_mutual_info_score

import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F
from sklearn.metrics import r2_score
import numpy as np

from sklearn.model_selection import KFold

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

import argparse
from torch_wrapper_objects import build_synthetic_datasets, run_training, LitM2M, m2mDataset
from evaluation import calculate_rsquared
import pandas as pd


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self



def run_kfold_simulation(case, seed=0, n_splits=5, max_epochs=100):
    
    np.random.seed(seed)
    idx=0

    # from the case (as described in the overleaf), format the args for jen's code
    args = AttrDict({'N_met':case['J'], 
                     'N_bug':case['M'], 
                     'xdim':case['D'],
                     'ydim':case['D'], 
                     'ydi':case['D'],
                     'N':case['N'],
                     'N_samples':case['N'],
                     'K':case['K'], ### this is fed into inits
                     'L':case['L'], 
                     'lr':1e-2,#0.01,
                     'noise_lvl':0.1
                     })
    
    print('1')
    generated_successfully=False
    while not generated_successfully:
        if True:
            ## build datasets
            train_dataset, val_dataset, gen_met_locs, gen_bug_locs, \
             x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
                        mu_met, r_bug, r_met = \
                build_synthetic_datasets(args, seed=seed+idx*9, return_all_info=True)
            generated_successfully=True
#         except:
#             idx+=1

    
    
    kf = KFold(n_splits=n_splits, shuffle=True)
    kf.get_n_splits(x)
#     first_run=True
    print('2')
    for train_index, test_index in kf.split(x[:args.N]):
        X_train, X_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]


        train_dataset=m2mDataset(X_train, y_train, g[train_index])
        val_dataset=m2mDataset(X_test, y_test, g[test_index])


        case_path = '_'.join( [a.replace('_', '-') + '-' + str(b) 
                               for a,b in case.items()] ) 
        
        print('3')

        ## builds model, runs training, logs results
        fitted = run_training(train_dataset, 
                              val_dataset, 
                              gen_met_locs, 
                              gen_bug_locs, 
                              learning_rate=args.lr,
                              logger_path=case_path,
                              seed=seed, 
                              max_epochs=max_epochs
                             )
        
        print('4')

        train_r2, val_r2, train_rmse, val_rmse, inds = calculate_rsquared(fitted,
                                                                          train_dataset,
                                                                          val_dataset, 
                                                                   return_inds = True)
        NMI = normalized_mutual_info_score(gen_z.argmax(axis=1), 
                                 inds.detach().numpy().argmax(axis=1) )
        run_summary={'K': case['K'], 
                     'D': case['D'], 
                     'L': case['L'], 
                     'M':case['M'],
                     'J':case['J'],
                     'N':case['N'],
                     'lr':args['lr'],
                     'train_r2':train_r2,
                     'val_r2':val_r2,
                     'train_rmse':train_rmse,
                     'val_rmse':val_rmse, 
                     'NMI':NMI
                    }


#         if first_run:
#             pd.DataFrame({a:[b] for a,b in run_summary.items()})\
#                         .to_csv('simulation_results/CV_result.csv')

#         else:
        pd.concat([pd.read_csv('simulation_results/CV_result.csv', index_col=0), 
                   pd.DataFrame({a:[b] for a,b in run_summary.items()})
                  ]).to_csv('simulation_results/CV_result.csv')

#         first_run=False



def main():
    # runs the simulations for all cases outlined in the overleaf scenarios
    
    # ordered lists of the different parametes for each case
    Ks = [2,2,2,3,10,10,20] # K: number of metabolite clusters
    Ls = [2,2,2,3,10,10,10] # L: number of microbial clusters
    Ms=[40,44,60,200,200,200,200] # M: number of microbial taxa
    Js=[40,40,40,200,200,200,200] # J: number of metabolites
    Ns=[100,100,100,200,200,200,200] # N: number of samples
    Ds=[2,2,2,2,2,2,2,5,10] # D: embedding space
    
    case_summaries=[{'K':k,
                     'L':l,
                     'M':m,
                     'J':j,
                     'N':n,
                     'D':d} for k,l,m,j,n,d in zip(Ks, Ls, Ms, Js, Ns, Ds)]
    
    
    for case in case_summaries[4:]:#[2:]
        for seed in range(3):
            run_kfold_simulation(case, seed=seed)
            

if __name__=='__main__':
    main()







        
        
