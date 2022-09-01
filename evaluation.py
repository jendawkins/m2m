import torch
from torch.distributions import MultivariateNormal, Normal
import torch.nn.functional as F
from sklearn.metrics import r2_score
import numpy as np





def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def calculate_rsquared(fitted, train_dataset, val_dataset):
    
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

    inds = torch.Tensor( to_categorical( mvn.squeeze(1).argmax(dim=0), 
                                         num_classes=mvn.shape[0]) 
                        ).bool()
    
    ## run fitted models
    train_out = fitted.forward(train_dataset.x, train_dataset.y)[0]
    val_out = fitted.forward(val_dataset.x, val_dataset.y)[0]
    
    
    ## obtain train + validation predictions
    actual_train_preds = train_out.T.unsqueeze(-1).expand(-1,-1,fitted.model.N_met)\
                                       .transpose(0,2).transpose(0,1)[:, inds]
                                   
    actual_preds = val_out.T.unsqueeze(-1).expand(-1,-1,fitted.model.N_met)\
                                       .transpose(0,2).transpose(0,1)[:, inds]
    ## calculated rsquared
        
    train_r2 = r2_score(train_dataset.y.detach().numpy(), actual_train_preds.detach().numpy() )
    val_r2 =  r2_score(train_dataset.y.detach().numpy(), actual_train_preds.detach().numpy() )
    
    return(train_r2, val_r2)
                                   
                                   

