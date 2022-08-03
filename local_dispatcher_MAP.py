#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3

import subprocess
import argparse
import itertools
import time
import numpy as np
import datetime
import os

# max_load = 4
base_path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("-case", "--case", help="case", type=str,
                    default=datetime.date.today().strftime('%m %d %Y').replace(' ', '-'))
parser.add_argument("-max_load", "--max_load", help="max_load", type=int, default = 4)
args = parser.parse_args()

max_load = args.max_load

param_dict = {('L', 'K'): [(2,2)], 'seed': [0,1,2],
              'iter': 30000,
              'lr': [0.1, 0.01, 0.001, 0.0001], 'meas_var': 0.010,
              'data': 'synthetic',
              'linear': 0, 'w_tau': [(-0.01, -5)], 'a_tau': [(-0.01, -5)],
              ('nltype', 'adjust_lr'): [('sigmoid', 0), ('linear', 1), ('linear', 0),
                                        ('poly', 0), ('exp', 0)],
              'lm': [0], 'lb': [0],
              'locs': ['true'], 'N_met': 40, 'N_bug': 40,
              # 'case': 'new_data_gen'
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
    cmd = my_str
    print(cmd)
    args2 = cmd.split(' ')
    outter_i += 1
    pid = subprocess.Popen(args2, cwd = base_path)
    pid_list.append(pid)
    time.sleep(0.5)
    while sum([x.poll() is None for x in pid_list]) >= max_load:
        time.sleep(1)
