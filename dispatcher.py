#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3

import subprocess
import argparse
import itertools
import time
import numpy as np
import os


my_str_orig = '''
#!/bin/bash
#BSUB -J m2m
#BSUB -o m2m.out
#BSUB -e m2m.err

rm *.err
rm *.out

# This is a sample script with specific resource requirements for the
# **bigmemory** queue with 64GB memory requirement and memory
# limit settings, which are both needed for reservations of
# more than 40GB.
# Copy this script and then submit job as follows:
# ---
# cd ~/lsf
# cp templates/bsub/example_8CPU_bigmulti_64GB.lsf .
# bsub < example_bigmulti_8CPU_64GB.lsf
# ---
# Then look in the ~/lsf/output folder for the script log
# that matches the job ID number

# Please make a copy of this script for your own modifications

#BSUB -q gpu
#BSUB -M 120000
#BSUB -R rusage[mem=120000]
#BSUB -n 8

# Some important variables to check (Can be removed later)
echo '---PROCESS RESOURCE LIMITS---'
ulimit -a
echo '---SHARED LIBRARY PATH---'
echo $LD_LIBRARY_PATH
echo '---APPLICATION SEARCH PATH:---'
echo $PATH
echo '---LSF Parameters:---'
printenv | grep '^LSF'
echo '---LSB Parameters:---'
printenv | grep '^LSB'
echo '---LOADED MODULES:---'
module list
echo '---SHELL:---'
echo $SHELL
echo '---HOSTNAME:---'
hostname
echo '---GROUP MEMBERSHIP (files are created in the first group listed):---'
groups
echo '---DEFAULT FILE PERMISSIONS (UMASK):---'
umask
echo '---CURRENT WORKING DIRECTORY:---'
pwd
echo '---DISK SPACE QUOTA---'
df .
echo '---TEMPORARY SCRATCH FOLDER ($TMPDIR):---'
echo $TMPDIR

# Add your job command here
# source activate dispatcher
# module load Anaconda3/5.2.0
cd /PHShome/jjd65/m2m
'''
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", help="case", type=str)
args = parser.parse_args()


param_dict = {('L', 'K'): [(10,10),(20,20),(30,30)], 'seed': [0,1,2,3,4],
              ('learn','priors'): [('all', 'all')],
              #'fix': ['sigma',''],
              'iter': 10000,
              'w_tau': [(-0.3, -3)],
              'a_tau': [(-0.3, -3)], 'gmm': [0],
              # 'N_met': 153, 'N_bug': 97,
              # 'N_samples': 48,
              'lr': [1, 0.1,0.01,0.001], 'meas_var': 0.108,
              'linear': [0],
              'syn': 0,
              'load': 0, 'hard': [0],
              'lm': [0], 'lb': [0], 'adjust_lr': [1], 'locs': ['none'],
              'yfile': ['y-95-5.csv']
              }




total_iters = np.prod([len(v) for v in param_dict.values() if hasattr(v, '__len__') and not isinstance(v, str)])
print(total_iters)

i = 0
list_keys = []
list_vals = []
for key, value in param_dict.items():
    list_keys.append(key)
    if hasattr(value, "__len__") and not isinstance(value, str):
        list_vals.append(value)
    else:
        list_vals.append([value])

zipped_params = list(itertools.product(*list_vals))

pid_list = []
outter_i = 0
for p in zipped_params:
    fin_list = []
    i = 0
    my_str = "python3 ./main.py"
    for l in list_keys:
        if isinstance(l, tuple) and isinstance(p[i], tuple):
            for ii in range(len(l)):
                if hasattr(p[i][ii], "__len__") and not isinstance(p[i][ii], str):
                    pout = [str(pp) for pp in p[i][ii]]
                    my_str = my_str + ' -' + l[ii] + ' ' + ' '.join(pout)
                else:
                    fin = p[i][ii]
                    my_str = my_str + ' -' + l[ii] + ' ' + str(fin)
        elif not isinstance(l, tuple) and isinstance(p[i], tuple):
            pout = [str(pp) for pp in p[i]]
            my_str = my_str + ' -' + l + ' ' + ' '.join(pout)
        else:
            my_str = my_str + ' -' + l + ' ' + str(p[i])
        i += 1
    if args.case == 'yfile':
        my_str = my_str + ' -case ' + 'yfile_' + p[-1].split('.')[0].replace('-','_')
    elif args.case is not None:
        my_str = my_str + ' -case ' + args.case
    # cmd = my_str
    f = open('m2m.lsf', 'w')
    f.write(my_str_orig + my_str)
    f.close()
    os.system('bsub < {}'.format('m2m.lsf'))
    # print(cmd)
    # args2 = cmd.split(' ')
    outter_i += 1
    # pid = subprocess.Popen(args2, cwd = base_path)
    # pid_list.append(pid)
    # time.sleep(0.5)
    # while sum([x.poll() is None for x in pid_list]) >= max_load:
    #     time.sleep(1)

