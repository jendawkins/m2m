

### This file contains functions used to create the pytorch-format datasets, dataloaders, training wrappers, and loggers
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger

import sys ## the key functions being used from the original codebase
sys.path.append('jen_code/') 
from model import Model
from synthetic_data_generation import generate_synthetic_data




### dataset functions

class m2mDataset(Dataset):
    """Dataset class.
    Args:
       df (dataframe): Pandas dataframe, must be from the deepmicro repo
 
       is_marker (bool): If it is a marker dataset, set to True.
    """
    def __init__(self, x, y, g):
        self.x=torch.Tensor(x)
        self.y=torch.Tensor(y)
        self.g=torch.Tensor(g)
        
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx], self.g[idx]]
    
def build_synthetic_datasets(args, val_size=500):
    ## inputs the args, outputs the torch dataset objects, 
    ## and the summary params necessary for the model construction
    
    
    # adjusting some args to set values for now                         
    x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
        mu_met, r_bug, r_met = \
                   generate_synthetic_data(p=0.5,
                                           N_met = args.N_met, 
                                           N_bug = args.N_bug, 
                                           N_met_clusters = args.K,
                                           N_bug_clusters = args.L,
                                           N_samples=args.N_samples + val_size, 
                                           linear = 1, #args.linear,
                                           nl_type = 'exp',# args.nltype, 
                                           xdim=args.xdim, 
                                           ydim = args.ydim
                                          )

    N=args.N_samples
    # build dataloader object
    train_dataset=m2mDataset(x[:N], y[:N], g[:N])
    val_dataset=m2mDataset(x[N:], y[N:], g[N:])
                   
    return(train_dataset, 
           val_dataset,
           gen_met_locs,
           gen_bug_locs)

    

### Model/training functions
    
class LitM2M(pl.LightningModule):
    def __init__(self, 
                 gen_met_locs, 
                 gen_bug_locs,
                 train_dataset,
                 val_dataset, 
                 batch_size=100, 
                 learning_rate=1e-2
                ):
        super().__init__()
         
        # build dataloader objects from inputted datasets
        self.train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        self.val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
        
        
        # build and initialize the model
        self.model=Model(gen_met_locs, 
                         gen_bug_locs, 
                         gen_met_locs.shape[0],
                         gen_bug_locs.shape[0]
                         )
        
        self.model.initialize(0, train_dataset.x, train_dataset.y)
        
        
        # set lr
        self.learning_rate = learning_rate
        
        
        # note to self -- at some point want to bring the loss function outside of the model object
        #      mainly because it would be good to separate the various components of the loss
        #     (also is just a personal preference, to match pl's standard framework)
        #
        # self.loss_func = lambda y, y_hat: ....
        
        
    def train_dataloader(self):
        return(self.train_loader)
    
    def val_dataloader(self):
        return(self.val_loader)
    
    def forward(self, x, y):
        yhat, loss = self.model(x, y)
        return yhat, loss

    def split_batch(self, batch):
        return batch[0], batch[1], batch[2]

    def training_step(self, batch, batch_idx):
        x, y, g = self.split_batch(batch)
        
        y_hat, loss = self(x, y)
        
        self.log('loss', loss)
        return {'loss':loss}
    

    def validation_step(self, batch, batch_idx):
        x, y, g = self.split_batch(batch)
        y_hat, loss = self(x, y)
        self.log('val_loss', loss)
        return {'val_loss':loss}
 
    def configure_optimizers(self):
        # doing rmsprop to match jen's code, 
        # for now I'm not adding the 'adjust lr by parameter size' approach
        return torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
    


    
def run_training(train_dataset,
                 val_dataset,
                 gen_met_locs, 
                 gen_bug_locs, 
                 learning_rate,
                 logger_path):
    

    
    # build pytorch-lightning wrapper
    litm2m=LitM2M(gen_met_locs, 
                  gen_bug_locs, 
                  train_dataset, 
                  val_dataset, 
                  learning_rate=learning_rate
                  )
    
    # callback for model saving, checkpoints
    checkpoint_callback=ModelCheckpoint(
                            dirpath = 'simulation_results',
                            save_top_k=1,
                            verbose=False,
                            monitor='val_loss',
                            mode='min'
                            )
    
    # object ot log model performance
    tube_logger = TestTubeLogger('simulation_results', 
                                  name=logger_path)#'test_tube_logger')


    # object ot train the model
    trainer = pl.Trainer(max_epochs = 500,
                         min_epochs=5,
                         logger=tube_logger,
                         gpus = int( torch.cuda.is_available() ),
    #                      progress_bar_refresh_rate=0,
                         weights_summary='full',
                         check_val_every_n_epoch=1,
                         checkpoint_callback=checkpoint_callback,
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)]
                        )
    
    
    # run training
    trainer.fit(litm2m)
    
    return(None)
    
    
    
    
    
    
    
    
    
    
    
    
    
    