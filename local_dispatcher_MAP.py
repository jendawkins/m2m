#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3

import subprocess
import argparse
import itertools
import time
import numpy as np


max_load = 6
base_path = '/Users/jendawk/Dropbox (MIT)/M2M/'
parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", help="case", type=str)
parser.add_argument("-syn", "--syn", help="use synthetic data or not", type=int)
args = parser.parse_args()


param_dict = {('L', 'K'): [(10,10)], 'seed': [0,1,2,3,4],
              ('learn','priors'): [('all', 'all')],
              #'fix': ['sigma',''],
              'iter': 50000,
              'w_tau': [(-0.3, -3)],
              'a_tau': [(-0.3, -3)], 'gmm': [0],
              'N_met': 362, 'N_bug': 97,
              'N_samples': 48,
              'lr': [0.1,0.01], 'meas_var': 0.108,
              'syn': 0,
              'load': 0, 'linear': 0, 'hard': [0],
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
    if args.syn is not None and '-syn' not in my_str:
        my_str = my_str + ' -syn ' + str(args.syn)
    cmd = my_str
    print(cmd)
    args2 = cmd.split(' ')
    outter_i += 1
    pid = subprocess.Popen(args2, cwd = base_path)
    pid_list.append(pid)
    time.sleep(0.5)
    while sum([x.poll() is None for x in pid_list]) >= max_load:
        time.sleep(1)


# for seed in np.arange(5):
#     for learn, priors in list(zip(*[learn_list, learn_list])):
#         for lr in np.logspace(-1,-5,5):
#             # for local in [1, 5]:
#             # for priors in ['all', 'none']:
#             cmd = my_str.format(N_met, N_bug, local, L, K, meas_var, prior_meas_var, seed,
#                                 ' '.join(learn), ' '.join(priors), learn_num_clusters)
#             print(cmd)
#             args2 = cmd.split(' ')
#             pid = subprocess.Popen(args2)
#             pid_list.append(pid)
#             time.sleep(0.5)
#             while sum([x.poll() is None for x in pid_list]) >= max_load:
#                 time.sleep(30)
