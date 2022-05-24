import subprocess
import argparse
import itertools
import time
import numpy as np
max_load = 10
pid_list = []
my_str = "python3 ./rdk_fingerprints.py -fingerprint {0} -metric {1}"
base_path = '/Users/jendawk/Dropbox (MIT)/M2M/rdk/'
for metric in ['cosine', 'dice', 'tanimoto']:
    for fingerprint in ['pubchem']:
        cmd = my_str.format(fingerprint, metric)
        args2 = cmd.split(' ')
        pid = subprocess.Popen(args2, cwd=base_path)
        pid_list.append(pid)
        time.sleep(0.5)
        while sum([x.poll() is None for x in pid_list]) >= max_load:
            time.sleep(1)