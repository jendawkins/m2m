# M2M

This repository contains scripts for predicting metabolite levels from microbial relative abundances

## Conda environments
The conda environment .yml files needed are in the folder 'env_shareable'

Most scripts use M2M_CodeBase (python 3.8), but ete_tree/tree_plotter.py uses ete_env (python 3.6) and rdk_fingerprints.py uses my-rdkit-env (python 3.9)

These run for me on Mac and running on pycharm or on the command line, but if switching environments 
between scripts isn't working, you can create a new environment with python 3.6 (required for ete3) and you will just 
have to change some of the data loading in dataLoader.py (Also, I'm not sure if rdkit requires python > 3.6 as I haven't tried it)

## Runing scripts
To run the model, run main.py with the arguments specified. You don't have to activate an environment 
becaues the shebang at the top of each script should start the correct environment.
```bash
python3 ./main.py -yfile y.csv
```

To run in parallel, use local_dispatcher_MAP.py (if running locally) or use dispatcher.py on eristwo

To use these scripts, first specify in the script the models you want to run through the variable param_dict. A list of parameters will run a model 
with each listed parameter. 

For example, the following will run 18 models in parallel, with varying the number of metabolite clusters (L), the 
number of microbe clusters (K), the seed, and the learning rate. Any input arguments not specified in param_dict 
will default to the default argument value set in main.py

```python
param_dict = {('L', 'K'): [(30,30), (20,20)], 'seed': [0,1,2],
              ('learn','priors'): [('all', 'all')],
              'iter': 30000,
              'lr': [0.1,0.01, 0.001], 'meas_var': 0.10,
              'syn': 0,
              'load': 0, 'linear': 1,
              }
```

For the local dispatcher, make sure to specify max_load as an input argument so you don't run out of application memory. 
Max load is the maximum number of models running in parallel 

```bash
python3 ./local_dispatcher_MAP.py -max_load 6
```