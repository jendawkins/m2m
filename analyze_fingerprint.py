from helper import *
from plot_helper import *
from dataLoader import *
import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-dtype", "--dtype", help="case", type=str,default = '')
args = parser.parse_args()

dtype = args.dtype
name = 'test_'

yfile = 'y.csv'
ydist_file = 'y2_' + dtype + '_dist.csv'
met_newick_name = 'newick_tree_all_weeks.nhx'

# set data_path to point to directory with data
base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
data_path = base_path + "/inputs/"

# if dtype == 'clumps' or dtype == 'stratified' or dtype == 'ones':
all_mets = pd.read_csv(data_path + '/classy-fire/classy_fire_df.csv', header = 0, index_col = 0)
if not os.path.isfile('/ete_tree/' + met_newick_name):
    make_tree(all_mets.columns.values, base_path, '', 'metab_orig', newick_path='/ete_tree/' + met_newick_name, dist_type=dtype)

in_mets = pd.read_csv(data_path + yfile)
if ydist_file not in os.listdir(base_path + '/inputs/classy-fire/'):
    make_dist_mat(in_mets.columns.values, ydist_file, base_path, newick_path='/ete_tree/' + met_newick_name)
ydist = pd.read_csv(base_path + '/inputs/classy-fire/' + ydist_file, header=0, index_col=0)

# ydist = ydist / np.max(np.max(ydist))
# xdist = (xdist - xdist.mean().mean())/xdist.std().std()

# y_class = get_ytaxa(base_path + '/inputs/metab_classes.csv', ydist.columns.values, ydist, level='subclass')
# skbio_mds(ydist, y_class, path = base_path + '/figures/' + dtype + '-')

# ylocs_skbio= skbio_mds(ydist, path=base_path + '/figures/fast-' + dtype)
ylocs = plot_MDS(ydist, path=base_path + '/figures/classy-fire/metric-' + dtype, seed=0)
