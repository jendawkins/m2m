import scipy.stats as st
from collections import defaultdict
import pickle as pkl
from datetime import datetime
import random
import torch
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
from torch.distributions.negative_binomial import NegativeBinomial
from torch.distributions.beta import Beta
from torch.distributions.half_normal import HalfNormal
from torch.distributions.binomial import Binomial
import torch.nn as nn
import time
import numpy as np
from concrete import *
from helper import *
import torch.nn.functional as F

# This class calculates the loss and is called by the model from model.py
# inputs:
# - net: the model from model.py

class MAPloss():
    def __init__(self,net):
        self.net = net
        self.loss_dict = {}

    def compute_loss(self, outputs, true):
        # computes loss for each parameter specified in net.compute_loss_for
        # (default is all parameters, but if you want to fix a
        # parameter value or see if the model learns without a certain prior, you can specify that)
        # inputs:
        # - outputs: predicted metabolic cluster outputs
        # - true: true metabolite values
        self.loss_dict['y'] = self.marginalized_loss(outputs, true)

        total_loss = 0
#         for name, parameter in [p for p in self.net.named_parameters()]:# + [ ['alpha', ''] ]:
# #             print(name)
#             if 'NAM' in name:
# #                 print('am here')
#                 total_loss += self.NAM_loss(parameter)
#             elif 'batch' not in name:
#                 name=name.replace('.', '_')
#                 fun = getattr(self, name + '_loss')
#                 fun()
                
# #                     print(self.loss_dict[name])
#                 total_loss += self.loss_dict[name]
# #                 except:
#                     pass
        total_loss += self.loss_dict['y']
        return total_loss

    def marginalized_loss(self, outputs, true):
#         print(true.shape)
#         print(outputs.shape)
#         print(nn.functional.mse_loss(true, outputs ))
#         print('here')
#         return( nn.functional.mse_loss(true, outputs ) )
        
#         print('HEREREERERE')
#         print(outputs.shape)
#         print(true.shape)
    
#         Marginalized loss over z, the metabolic cluster indicator
        if self.net.met_locs is not None:
            eye = torch.eye(self.net.met_embedding_dim).unsqueeze(0).expand(self.net.K, -1, -1)
            var = torch.exp(self.net.r_met).unsqueeze(-1).unsqueeze(-1).expand(-1,self.net.met_embedding_dim,
                                                                               self.net.met_embedding_dim)*eye
            mvn = MultivariateNormal(
                self.net.mu_met.unsqueeze(1).expand(-1,self.net.N_met,-1),var.unsqueeze(1).expand(
                    -1,self.net.N_met,-1,-1)).log_prob(
                torch.Tensor(self.net.met_locs))#.unsqueeze(1)
        else:
            mvn = 0

            
        self.loss_dict['z'] = -mvn.sum()
        eps = 1e-10
        temp = (1-2*eps)*torch.softmax(self.net.pi_met,1) + eps
        
#         print('OUTPUT...')
#         print(outputs.T.unsqueeze(-1))
        
#         print( outputs[:5], self.net.all_loss_mean )
#         print(outputs.shape)
#         print(mvn.shape)
#         print( torch.log(temp.T).unsqueeze(1).shape )
#         print(true.shape)
#         print( Normal(outputs.T.unsqueeze(-1).expand(-1,-1,self.net.N_met), 
#                       torch.sqrt(torch.exp(self.net.sigma))
#                          ).log_prob(true).shape )
        
#         print( mvn + \
#                Normal(outputs.T.unsqueeze(-1).expand(-1,-1,self.net.N_met), 
#                       torch.sqrt(torch.exp(self.net.sigma))
#                          ).log_prob(true) )
        
#         print( F.softmax(mvn.T) )
        
#         print( (outputs.transpose(1,2) * F.softmax(mvn.T).unsqueeze(0) ).sum(dim=-1).shape )
    
#         z_log_probs = torch.log(temp.T).unsqueeze(1) + mvn.unsqueeze(1) + \
#                Normal(outputs.T.unsqueeze(-1).expand(-1,-1,self.net.N_met), 
#                       torch.sqrt(torch.exp(self.net.sigma))
#                          ).log_prob(true)
        
        
#         z_log_probs = Normal(outputs.T.unsqueeze(-1).expand(-1,-1,self.net.N_met), torch.sqrt(torch.exp(self.net.sigma))).log_prob(true)
        
#         print('LOSS')
#         print(mvn.shape)
#         print(temp.shape)
#         print(outputs.shape)
        
#         print(true.shape)
# #         print(  ( outputs.transpose(1,2)@temp.T ).squeeze(-1).shape )
        
        
#         print(outputs.shape)
#         print(F.softmax(mvn).T.unsqueeze(0).shape)
#         print( (outputs.transpose(1,2) * F.softmax(mvn).T.unsqueeze(0) ).sum(-1).shape)
#         raise(ValueError('stop'))
        loss = F.mse_loss(true, ( outputs.transpose(1,2) * F.softmax(mvn.T).unsqueeze(0) ).sum(-1) )
#         loss = F.mse_loss(true, outputs )#.T.unsqueeze(-1).expand(-1,-1,self.net.N_met) )
        
#         self.net.z_act = nn.functional.one_hot(torch.argmax(z_log_probs.sum(1),0),self.net.K)
#         loss = -torch.logsumexp(z_log_probs, 0).sum(1).sum()
        return loss

    def alpha_loss(self):
#         Computes loss for alpha, modeled as binary concrete with location self.net.alpha_loc and temperature
#         self.net.alpha_temp
#         Note that we compute the loss of alpha_act, which is the transformed version of alpha to be within 0 and 1
#         (rather than the unconstrained net.alpha, which is what the model learns through gradient descent)
#         print(self.net.alpha_act)

        self.loss_dict['alpha'] = -BinaryConcrete(self.net.alpha_loc, self.net.alpha_temp).log_prob(
                self.net.alpha_act.T).sum().sum()
    
    
#         self.loss_dict['alpha'] = -BinaryConcrete(self.net.alpha_loc, .99).log_prob(
#                 self.net.alpha_act.T).sum().sum()
        
        
#         print(self.loss_dict['alpha'])
# #         div=2
# #         while self.loss_dict['alpha'].isinf() and div < 112:
# #             self.loss_dict['alpha'] = -BinaryConcrete(self.net.alpha_loc, self.net.alpha_temp/div).log_prob(
# #                 self.net.alpha_act).sum().sum()
# #             print(self.loss_dict['alpha'])
# #             div=div*2
        
        
#         if self.loss_dict['alpha'].isinf():
#             raise(ValueError('Trouble calculating alpha prior'))

#         self.loss_dict['alpha']=0

#         pass


    def beta_loss(self):
        # Computes loss for beta, regression coefficients
        # Modeled as normally distributed
        temp_dist = self.net.distributions['beta']
        self.loss_dict['beta'] = -temp_dist.log_prob(self.net.beta).mean()

    def mu_bug_loss(self):
        # Mu_bug loss computed
        # Mu_bug is multivariate normally distributed
        temp_dist = self.net.distributions['mu_bug']
        self.loss_dict['mu_bug'] = -temp_dist.log_prob(self.net.mu_bug).mean()

    def r_bug_loss(self):
        # Computes loss for r_bug
        # Note that we have to exponentiate net.r_bug to compute the loss of the constrained parameter (constrained to
        # be greater than 0)
        val = torch.exp(self.net.r_bug)
        self.loss_dict['r_bug'] = -self.net.distributions['r_bug'].log_prob(val).mean()

    def mu_met_loss(self):
        # Loss for mu_met, same as loss for mu_bug
        temp_dist = self.net.distributions['mu_met']
        self.loss_dict['mu_met'] = -temp_dist.log_prob(self.net.mu_met).mean()

    def r_met_loss(self):
        # Computes loss for r_met, same as loss for r_bug
        val = torch.exp(self.net.r_met)
        self.loss_dict['r_met'] = -self.net.distributions['r_met'].log_prob(val).mean()

    def pi_met_loss(self):
        # Computes loss for pi_met
        # We have to transform pi_met using a softmax to it's constrained value
        epsilon = torch.exp(self.net.e_met)
        eps = 1e-10
        temp = (1-2*eps)*torch.softmax(self.net.pi_met,1) + eps
        self.loss_dict['pi_met'] = (torch.Tensor(1 - epsilon) * torch.log(temp)).mean()

    def e_met_loss(self):
        # Loss for e_met
        val = torch.exp(self.net.e_met)
        gamma = self.net.distributions['e_met']       
        self.loss_dict['e_met'] = -gamma.log_prob(val).mean()


    def NAM_loss(self, w):
        # Loss for NAM
        return -self.net.distributions['NAM'].log_prob(w).mean()
    
    def NAM_loss(self, w):
        # Loss for NAM
        return -self.net.distributions['NAM'].log_prob(w).mean()
    
    def f_weight_loss(self):
        self.loss_dict['f_weight'] = self.net.f.weight.data.pow(2).sum()
        
    def f_bias_loss(self):
        self.loss_dict['f_bias'] = self.net.f.bias.data.pow(2).sum()
    
        



