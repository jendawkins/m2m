import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from copy import copy, deepcopy
from sklearn.model_selection import StratifiedKFold
import os
import scipy.stats as st
from collections import defaultdict
import pickle as pkl
from datetime import datetime
import random
import torch
from collections import Counter
import math
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.nn as nn
import time
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
import subprocess

def mds_choose_d(dat,dmax = 30, seed = 0):
    # Choose the lowest dimension at which the original distances and embedded distances are not significantly different

    # inputs:
    # - dat: NxN distance matrix
    # - dmax: maximum dimension to check
    # - seed: random seed

    # outputs:
    # - d: best dimension
    # - xlocs: embedded locations at dimension d (N x d matrix)
    # - embedding.stress_: stress at best embedded dimension
    true_dist = squareform(dat)
    for d in np.arange(2,dmax):
        embedding = MDS(n_components=d, dissimilarity='precomputed', random_state=seed)
        xlocs = embedding.fit_transform(dat)
        est_dist = pdist(xlocs)
        stat, pval = st.ks_2samp(est_dist, true_dist)
        if pval > 0.05:
            break
    return d, xlocs, embedding.stress_


def sample_gumbel(shape, eps=1e-20):
    # Sample from the gumbel distribution (for use in gumble_sigmoid() below)
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid(logits, temperature, epsilon):
    """
    Sample from the gumbel distribution to get binary value, but keep gradients for non-binary value
        (for use in forward() in model.py)
    input: logits, temperature, epsilon
    return: y (soft, unsampled value), y_hard (hard, sampled value)
    """
    temp = logits + sample_gumbel(logits.size())
    y = (1-2*epsilon)*torch.sigmoid(temp/temperature) + epsilon

    y_hard = torch.round(y)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y, y_hard

def smoothmax(x, tau = 1):
    # Calculates smooth maximum of x given temperature tau
    return torch.sum(x*torch.exp(x/tau))/torch.sum(torch.exp(x/tau))

def euc_dist(x,y):
    # Calculates euclidean distance between x and y
    return np.sqrt(np.sum([(x[i] - y[i])**2 for i in range(len(x))]))

def pairwise_eval(guess, true):
    """
    Evaluates clusters when the cluster indices may be scrambled by evaluating each pair
    Inputs:
    - guess: list of lists, where each list is the IDs (string, int, etc) of features predicted to be within that cluster
            ex: [[1,7,4], [2,3], [5,6,8]], where cluster 1 contains features 1, 7, and 4; cluster 2 contains features
            2 and 3, and cluster 3 contains features 5,6,and 8
    - true: list of lists, where each list is the IDs (string, int, etc) of features that are truely in the cluster
    Returns:
        - tp: # of true positive pairs (predicted to be in the same cluster and are)
        - fp: # of false positive pairs (predicted to be in the same cluster but are not)
        - tn: # of true negative pairs (predicted to be in different clusters adn are)
        - fn: # of false negative pairs (predicted to be in different clusters but are not)
        - ri: Rand index; i.e. percent of pairs that are correct
    """
    guess_dict = {i:guess[i] for i in range(len(guess))}
    true_dict = {i: true[i] for i in range(len(guess))}
    pairs = list(itertools.combinations(range(len(guess)),2))
    try:
        tp_fp = np.sum([math.comb(np.sum(guess == i), 2) for i in np.unique(guess)])
        tp = len([i for i in range(len(pairs)) if guess_dict[pairs[i][0]]==guess_dict[pairs[i][1]] and
                  true_dict[pairs[i][0]]==true_dict[pairs[i][1]]])
        fp = tp_fp - tp
        tn = len([i for i in range(len(pairs)) if guess_dict[pairs[i][0]]!=guess_dict[pairs[i][1]] and
                  true_dict[pairs[i][0]]!=true_dict[pairs[i][1]]])
        tn_fn = math.comb(guess.shape[0], 2) - tp_fp
    except:
        tp_fp = np.sum([math.comb(int(np.sum(guess[:,i])), 2) for i in np.arange(guess.shape[1])])
        tp = len([i for i in range(len(pairs)) if (guess_dict[pairs[i][0]]==guess_dict[pairs[i][1]]).all() and
                  (true_dict[pairs[i][0]]==true_dict[pairs[i][1]]).all()])
        fp = tp_fp - tp
        tn = len([i for i in range(len(pairs)) if (guess_dict[pairs[i][0]] != guess_dict[pairs[i][1]]).all() and
                  (true_dict[pairs[i][0]] != true_dict[pairs[i][1]]).all()])
        tn_fn = math.comb(guess.shape[0],2)*guess.shape[1] - tp_fp

    fn = tn_fn - tn
    ri = (tp + tn)/(tp + fp + tn + fn)
    return tp, fp, tn, fn, ri

def get_one_hot(x,l=None):
    # Get one-hot vector from index vector
    # x: single index or array of indices (i.e. x = 3 or x = [1, 3, 7])
    # l: length of one-hot vector
    # Returns: one-hot vector
    #   i.e. [0, 1, 0, 1, 0, 0, 0, 1]
    if l is None:
        l = max(x)
    if torch.is_tensor(x):
        vec = torch.zeros(l)
    else:
        vec = np.zeros(l)
    vec[x] = 1
    return vec

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def isclose(a, b, tol=1e-03):
    # Returns true if two matrices are the same within the given tolerance
    return (abs(a-b) <= tol).all()

# def concrete_sampler(a, T, dim, loc = 0, scale = 1):
#     # samples from concrete distribution
#     G = st.gumbel_r(loc, scale).rvs(size = a.shape)
#     return torch.softmax(T*(torch.log(torch.Tensor(a)) + torch.Tensor(G)), dim = dim)

def get_epsilon(data):
    # For log transforming data, get the epsilon to add to the data (so as to not log transform zeros) where epsilon is
    # 0.1 times the minimum absolute non-zero value of the data
    vals = np.array(data).flatten()
    vals[np.where(vals == 0)[0]] = np.max(vals)
    epsilon = 0.1*np.min(np.abs(vals))
    return epsilon

def filter_by_train_set(x_train, x_test, meas_key, key = 'metabs', log_transform = True, standardize_data = True):
    """
    Filters and standardizes test data according to the filtering and standardization of training data (not yet used
    in model since we are just using training data right now)

    Inputs:
    - x_train: training data
    - x_test: testing data
    - meas_key: nested dictionary of filtering criteria for metabolites and microbes
        first level: 'pt_perc': fraction of participants that the features must be non-zero in to pass the filter
                    'pt_tmpts': number of timepoints to filter on (keep to 1)
                    'meas_thresh': measurement threshold below which data is treated as 0
                    'var_perc': keep features with coefficients of variation in the top 'var_perc' percentile
            second level: key = 'metabs' or '16s'; provides the criteria for either metabolites or 16s data
    - key: 'metabs' or '16s' - whichever the data in x_train and x_test is
    - log_transform: whether or not to log transform data
    - standardize: whether or not to standardize data
    """
    if x_train.shape[1]>10:
        filt1 = filter_by_pt(x_train, perc=meas_key['pt_perc'][key], pt_thresh=meas_key['pt_tmpts'][key],
                             meas_thresh=meas_key['meas_thresh'][key], weeks=None)
        filt1_test = x_test[filt1.columns.values]
    else:
        filt1 = x_train
        filt1_test = x_test

    epsilon = get_epsilon(filt1)
    if '16s' in key:
        filt1_test = np.divide(filt1_test.T, np.sum(filt1_test,1)).T
        epsilon = get_epsilon(filt1_test)
        geom_means = np.exp(np.mean(np.log(filt1 + epsilon), 1))
        temp = np.divide(filt1.T, geom_means).T
        epsilon = get_epsilon(temp)
    xout = []
    for x in [filt1, filt1_test]:
        if '16s' not in key:
            if log_transform:
                transformed = np.log(x+ epsilon)
            else:
                transformed = x
        else:
            if log_transform:
                x = np.divide(x.T, np.sum(x,1)).T
                geom_means = np.exp(np.mean(np.log(x + epsilon), 1))
                transformed = np.divide(x.T, geom_means).T
                transformed = np.log(transformed + epsilon)
            else:
                transformed = x
        xout.append(transformed)
    xtr, xtst = xout[0], xout[1]

    if x_train.shape[1]>10:
        filt2 = filter_vars(xtr, perc=meas_key['var_perc'][key], weeks = None)
    else:
        filt2 = xtr
    filt2_test = xtst[filt2.columns.values]
    if standardize_data:
        dem = np.std(filt2,0)
        if (dem == 0).any():
            dem = np.where(np.std(filt2, 0) == 0, 1, np.std(filt2, 0))
        x_train_out = (filt2 - np.mean(filt2, 0))/dem
        x_test_out = (filt2_test - np.mean(filt2,0))/dem
    else:
        x_train_out = filt2
        x_test_out = filt2_test
    return x_train_out, x_test_out

def filter_vars(data, perc=5, weeks = [0,1,2]):
    # Filter data by coefficient of variance
    # - if weeks is not None, keep features with coefficient of variance in top 'perc' percentile in any of the
    #   weeks specified
    if weeks:
        tmpt = [float(x.split('-')[1]) for x in data.index.values]
        rm2 = []
        for week in weeks:
            t_ix = np.where(np.array(tmpt)==week)[0]
            dat_in = data.iloc[t_ix,:]
            variances = np.std(dat_in, 0) / np.abs(np.mean(dat_in, 0))
            rm = np.where(variances > np.percentile(variances, perc))[0].tolist()
            rm2.extend(rm)
        rm2 = np.unique(rm2)
    else:
        variances = np.std(data, 0) / np.abs(np.mean(data, 0))
        rm2 = np.where(variances > np.percentile(variances, perc))[0].tolist()

    return data.iloc[:,list(rm2)]

def standardize(x,override = True):
    """
    Standardize data,
        inputs: x, where columns of x are features and rows are samples
        override: put True if you know there are more samples than features; otherwise function will ensure that
        there are more columns than rows in your data (because most data we work with has more features than samples)
    """

    if not override:
        assert(x.shape[0]<x.shape[1])

    dem = np.std(x,0)
    if (dem == 0).any():
        dem = np.where(np.std(x, 0) == 0, 1, np.std(x, 0))
    stand = (x - np.mean(x, 0))/dem
    ix = np.where((stand == 1).all())[0]
    stand.iloc[:, ix] = 0
    return stand, np.mean(x,0), dem


def filter_by_pt(dataset, perc = .15, pt_thresh = 1, meas_thresh = 10, weeks = [0,1,2]):
    """
    Filter data. Remove features that are in too few participants.
    Inputs:
        - dataset: data frame where indexes are sample IDs and columns are features
        - perc: fraction of participants the feature must be non-zero in to keep
        - pt_thresh: filter based on presence of feature pt_thresh timepoints
        - meas_thresh: threshold below which we set the data value to 0
        - weeks: list containing which weeks to filter by; if len(weeks) > 1, keep features that pass filters in
                ANY of the weeks. Set to none if you want to use all possible weeks
    """
    # keep track of values greater than measurement threshold
    mets = np.zeros(dataset.shape)
    mets[np.abs(dataset) > meas_thresh] = 1

    if weeks is not None:
        # This is mostly for the cdi data, drop half weeks
        df_drop = [x for x in dataset.index.values if not x.split('-')[1].replace('.', '').isnumeric()]
        dataset = dataset.drop(df_drop)
        mets = np.zeros(dataset.shape)
        mets[np.abs(dataset) > meas_thresh] = 1
        if len(weeks)>1:
            pts = [x.split('-')[0] for x in dataset.index.values]
            ixs = dataset.index.values
            ix_add = [i for i in range(len(ixs)) if float(ixs[i].split('-')[1]) in weeks]
            oh = np.zeros(len(pts))
            oh[np.array(ix_add)] = 1
            index = pd.MultiIndex.from_tuples(list(zip(*[pts, oh.tolist()])), names = ['pts', 'add'])
            df = pd.DataFrame(mets, index = index)
            df2 = df.xs(1, level = 'add')
            df2 = df2.groupby(level=0).sum()
            mets = np.zeros(df2.shape)
            mets[np.abs(df2)>0] = 1

    # if measurement of a microbe/metabolite only exists in less than pt_thresh timepoints, set that measurement to zero
    if pt_thresh > 1:
        pts = [x.split('-')[0] for x in dataset.index.values]
        for pt in pts:
            ixs = np.where(np.array(pts) == pt)[0]
            mets_pt = mets[ixs,:]
            # tmpts_pt = np.array(tmpts)[ixs]
            mets_counts = np.sum(mets_pt, 0).astype('int')
            met_rm_ixs = np.where(mets_counts < pt_thresh)[0]
            for ix in ixs:
                mets[ix, met_rm_ixs] = 0

    # For each class, count how many measurements exist within that class and keep only measurements in X perc in each class
    met_counts = np.sum(mets, 0)
    mets_all_keep = np.where(met_counts >= np.round(perc * mets.shape[0]))[0]
    return dataset.iloc[:,np.unique(mets_all_keep)]

def get_meas_var(raw_data, repeat_data):
    """
    Get measurement variance of a dataset given the un-transformed (but filtered) data, and repeat data
    Inputs:
    - raw_data: un-transformed, but filtered, numpy array of data
    - repeat_data: list of arrays, where each array is a set of repeats
        for instance, if participant 101 had 3 repeats and participant 102 had 2 repeats,
        repeat_data = [3 x Nf array, 2 x Nf array] where Nf is the number of features
    """
    # Get epsilon, mean, and standard deviation of untransformed data
    epsilon = get_epsilon(raw_data)
    raw_log = np.log(raw_data + epsilon)
    mn = np.mean(raw_log,0)
    stdev = np.std(raw_log,0)

    # Apply transformations to each instance of repeat data and pool together to calculate approximate measurment variance
    pooled_numerator = 0
    pooled_denomenator = 0
    for rep_dat in repeat_data:
        rep_stand = np.log(rep_dat + epsilon)
        rep_stand = (rep_stand - mn) / stdev
        dof = rep_stand.shape[0]
        pooled_numerator = pooled_numerator + np.sum((dof-1)*np.var(rep_stand))
        pooled_denomenator = pooled_denomenator + (dof*rep_stand.shape[1]) - rep_stand.shape[1]

    pooled_var = pooled_numerator / pooled_denomenator
    return pooled_var


def load_data(base_path, xfile, yfile, dataLoader):
    """
    Load data based on filtering critera specefied in yfile filename, called in main.py
        Inputs:
            - base_path: current working path
            - xfile: file name to save xfile to
            - yfile: file name to save yfile to; also contains filtering criteria where the first number is the percent
                of participants the metabolite has to be non-zero in, and the second is the coef of var percentile
                ex: yfile = 'y_95_5.csv' means that we will filter the metabolite data such that all metabolites in the
                    file are non-zero in 95% of participants, and are in the top 95th percentile of coef of variation
            - dataLoader: dataLoader model

        Saves xfile and yfile to <base_path>/inputs/processed/ under the filenames xfile and yfile
    """
    data_path = base_path + "/inputs"
    pt_perc = float('0.' + yfile.split('-')[1])
    var_perc = float(yfile.split('-')[-1].split('.')[0])
    dl = dataLoader(path=data_path, pt_perc={'metabs': pt_perc, '16s': .1, 'scfa': 0, 'toxin': 0}, meas_thresh=
    {'metabs': 0, '16s': 10, 'scfa': 0, 'toxin': 0},
                    var_perc={'metabs': var_perc, '16s': 5, 'scfa': 0, 'toxin': 0}, pt_tmpts=1)

    x = dl.week_sm_filt['16s'][1]['x']
    x = np.divide(x.T, np.sum(x, 1)).T

    taxa_labels = pd.read_csv(data_path + '/taxa_labels.csv', index_col=[0])
    x.columns = taxa_labels['labels'].loc[x.columns.values]
    y = dl.week_sm['metabs'][1]['x']

    raw_dat = dl.cdiff_data_dict['data']
    replicate_ixs = [d for d in raw_dat.index.values if not d.split('-')[1].split('.')[-1].isnumeric()]
    repeat_dat = raw_dat[y.columns.values].loc[replicate_ixs]
    y_raw = dl.week_sm_filt['metabs'][1]['x']
    rep_pts = [ix.split('-')[0] for ix in replicate_ixs]
    unique_ixs = np.unique(rep_pts)
    rep_list = [repeat_dat.loc[[ix for ix in replicate_ixs if ix.split('-')[0] == unique_ix]] for unique_ix in unique_ixs]
    pooled_var = get_meas_var(y_raw, rep_list)
    with open(data_path + '/' + yfile.split('.')[0] + '-mvar.pkl', 'wb') as f:
        pkl.dump(pooled_var, f)

    met_classes = pd.read_csv(data_path + '/classy-fire/classy_fire_df.csv', header = 0, index_col = 0)
    inter = set(met_classes.columns.values).intersection(y.columns.values)
    y = y[inter]
    y = y.loc[x.index.values]
    # y.drop('linolenoyl-linolenoyl-glycerol (18:3/18:3) [2]*', axis = 1)
    if not os.path.isdir(data_path + '/' + 'processed/'):
        os.mkdir(data_path + '/' + 'processed/')
    x.to_csv(data_path + '/' + 'processed/' + xfile)
    y.to_csv(data_path + '/' + 'processed/' + yfile)

def make_tree(feats, base_path, case, func='asv',
              newick_path='/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
              dist_type = ''):
    """
    Runs one of the tree plotting scripts in tree_plotter.py given input information
    Inputs:
    - feats: features to make the tree with
    - base_path: current working dir
    - case: run case, so that tree is saved in correct directory
    - func: which tree function in 'tree_plotter.py' to run; options:
        'asv': checks for existing asv tree or alerts user to run phylogenetic tree script to get phylo tree
        'metab_orig': make metabolomic newick tree and save to newick_path; also plot tree pdf
        'metab': plot metabolomic tree
    - newick_path: path of existing tree or path to save tree
    - dist_type: for func='metab_orig', how to set the branch lengths (options are '', 'clumps', 'stratified')
    """
    if func == 'asv':
        if newick_path.split('/')[-1] not in os.listdir(base_path + '/ete_tree/phylo_placement/output/'):
            # taxa_labels = pd.read_csv(data_path + '/taxa_labels.csv', index_col=[0])
            for i, seq in enumerate(feats):
                if i == 0:
                    wa = 'w'
                else:
                    wa = 'a'
                f = open(base_path + "/ete3/phylo_placement/data/asvs_to_place.fa", wa)
                # f.writelines('>' + taxa_labels.loc[seq]['labels'] + '\n' + seq + '\n')
                f.writelines('>' + seq + '\n' + seq + '\n')
                f.close()
            raise Exception(
                "Run phylogenetic tree script in erisone to get ASV tree; documentation in ete_tree/phylo_placement/documentation.doc")

    if func == 'metab_orig':
        input_ls = ["python3", "tree_plotter.py", "-fun", func, "-name", case + '/' + func + '_in.pdf', "-newick",
             base_path + newick_path, "-dtype", dist_type, "-feat"]
        input_ls.extend(feats)
        subprocess.run(input_ls,
            cwd=base_path + "/ete_tree")
    else:
        input_ls = ["python3", "tree_plotter.py", "-fun", func, "-name", case + '/' + func + '_in.pdf', "-newick",
             base_path + newick_path, "-feat"]
        input_ls.extend(feats)
        subprocess.run(input_ls,
            cwd=base_path + "/ete_tree")

def make_dist_mat(dat, dist_file, base_path, newick_path, dist_type = '', yfile = '', outfile = ''):
    # Runs get_dist() function in tree_plotter.py
    # Inputs:
    # - dat: features to get distance matrix for
    # - dist_type: If getting metabolic distance matrix from fingerprings and not classification tree, dist_type should be
    #       specified as <fingerprintType>_<distanceMetric>;
    #       for example: dist_type = 'pubchem_tanimoto'
    # - dist_file: name of distance file that will be saved if dist_type is not of the form <fingerprintType>_<distanceMetric>;
    #       otherwise, distance matrix saved is saved as <fingerprintType>_<distanceMetric>.csv
    # - base_path: current working dir
    # - newick_path: path of newick tree if using phylogenetic or metabolomic tree to get distances
    # - yfile: if using metabolic fingerprints for distances, metabolic processed data
    # - outfile: output path for saved distance matrix csv
    if '_' in dist_type:
        in_list = ["python3", "rdk_fingerprints.py", "-fingerprint", dist_type.split('_')[0],
                   "-metric" + dist_type.split('_')[1], "-yfile", yfile, '-o', outfile]
        subprocess.run(in_list, cwd = base_path + '/rdk')
    else:
        in_list = ["python3", "tree_plotter.py", "-fun", 'dist', "-name", dist_file,
                   "-newick", base_path + newick_path, '-o', outfile, "-feat"]
        in_list.extend(dat)
        subprocess.run(in_list, cwd=base_path + "/ete_tree")

def get_rand_locs(dat, dim, seed):
    # Generate random embedded locations given metabolic/microbial data, embedding dimension, and random seed
    a_met = np.random.uniform(0, 10, size=(dat.shape[1], dat.shape[1]))
    a_met = (a_met + a_met.T) / 2
    np.fill_diagonal(a_met, 0)
    embedding = MDS(n_components=dim, dissimilarity='precomputed', random_state=seed)
    locs = embedding.fit_transform(a_met)
    return locs

def get_xtaxa(path, x):
    # Get families for each asv in x (where x a dataframe of samples x features);
    # path is the path to the taxanomic information
    x_taxa = pd.read_csv(path, header=0, index_col=0)
    x_fams = pd.Series([taxon.split('; ')[-3] for taxon in x_taxa['taxa_rdp']], index=x_taxa['labels'])
    x_fams_silva = pd.Series([taxon.split('; ')[-3] for taxon in x_taxa['taxa_silva']], index=x_taxa['labels'])
    x_fams[x_fams == 'NA'] = x_fams_silva[x_fams == 'NA']
    x_fams = x_fams.loc[x.columns.values]
    return x_fams

def edit_string(string):
    # For making the metabolomic classification trees, edit metabolite names so that trees can be stored in newick format
    # Input: metabolite name; output: edited metabolite name
    return string.replace('(', '_').replace(')', '_').replace(':',
                                                 '_').replace(','
                                                              , '_').replace('[', '_').replace(']', '_').replace(';',
                                                                                                                 '_')

def get_ytaxa(path, feats, ydist, level='subclass'):
    """
    Get metabolic classes from a given level of classy fire taxonomy
    Inputs:
    - path: path to classy-fire taxonomy csv (inputs/processed/classy-fire/classy_fire_df.csv)
    - feats: list of metabolites to get classification for
    - ydist: metabolite distance matrix
    - level: classy-fire level; options are 'superclass', 'class', 'subclass', 'level 5', 'level 6', ... 'level 10'
    Outputs:
    - list of classifications at specified level for all metabolites in feats
    """
    y_taxa = pd.read_csv(path, header=0, index_col=0)
    y_class = y_taxa[feats].loc[level]
    null_vals = y_class[y_class.isnull()].index.values
    for null_sub in y_class[y_class.isnull()].index.values:
        sorted_ls = ydist.loc[null_sub].sort_values().drop(null_vals)
        rep = y_class.loc[sorted_ls.index.values[0]]
        y_class.loc[null_sub] = rep
    return y_class

# def numpy_forward(x,w,beta,alpha,sigma,omega_temp, alpha_temp, K, mu_bug =None, r_bug = None, microbe_locs = None, gmm = True):
#
#     if microbe_locs is not None:
#         kappa = torch.stack(
#             [torch.sqrt(((mu_bug - torch.tensor(microbe_locs[m, :])).pow(2)).sum(-1)) for m in
#              np.arange(microbe_locs.shape[0])])
#         w_act = torch.sigmoid((torch.exp(r_bug) - kappa)/omega_temp)
#     else:
#         w_act = w
#     g = x@w_act
#     if gmm:
#         out_clusters = beta[0, :]+ st.norm(0,np.sqrt(np.exp(sigma))).rvs((g.shape[0], K))
#     else:
#         out_clusters = beta[0,:] + np.matmul(g, beta[1:,:]*alpha) + st.norm(0,np.sqrt(np.exp(sigma))).rvs((g.shape[0], K))
#
#     return out_clusters

