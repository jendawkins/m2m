# M2M

This repository contains scripts for predicting metabolite levels from microbial relative abundances

See the overleaf pdf, "Joint_co_clustering_and_prediction.pdf" to see the mathematical model

## Conda environment
To make a conda environment with the necessary packages & versions, run:
```bash
chmod u+x create_env.sh
./create_env.sh
```

## Runing scripts
To run the model, run main.py with the arguments specified after activating the m2m environment

For example, to run the model with CDI data that is filtered to keep only metabolites with non-zero values in at least 75% of 
participants and with coefficients of variance in the top 95%, run:
```bash
conda activate m2m
python3 ./main.py -data 'cdi' -nzm 75 -cvm 5
```

To run multiple models in parallel, use local_dispatcher_MAP.py (if running locally) or use dispatcher.py on eristwo

To use these scripts, first specify in the script the models you want to run through the variable param_dict. A list of parameters will run a model 
with each listed parameter. 

For example, editing param_dict to the following will and then running local_dispatcher_MAP.py (or dispatcher.py on 
eristwo) will run 18 models in parallel, with varying the number of metabolite clusters (L), the 
number of microbe clusters (K), the seed, and the learning rate. Any input arguments not specified in param_dict 
will default to the default arguments set in main.py

```python
param_dict = {('L', 'K'): [(30,30), (20,20)], 'seed': [0,1,2],
              ('learn','priors'): [('all', 'all')],
              'iter': 30000,
              'lr': [0.1,0.01, 0.001], 'meas_var': 0.10,
              'data': 'cdi',
              'load': 0, 'linear': 1,
              }
```

For the local dispatcher, make sure to specify max_load as an input argument so you don't run out of application memory. 
Max load is the maximum number of models running in parallel.

```bash
python3 ./local_dispatcher_MAP.py -max_load 6
```

# Main To-Do's:
### Re-think priors for measurement variance and radii
### Run with additional datasets AND make model able to input various datasets more generally
### Run with and without metabolite location prior 
 - for now, stick to prior with pubchem fingerprints and tanimoto distance metric, where args.dtype = 'pubchem_tanimoto'
### Increase model ability to test different metabolic priors
### Determine best number of layers & nodes for neural network in non-linear model and assess how effectively the non-linear model is


Note: to-do's that include running the model are more complicated than just running the model, since there's not a super clear way to determine how well the model is performing. I've been doing a lot about thinking about what to plot and output to figure out exactly how the model is doing and how to improve it 

