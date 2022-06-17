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
    true_dist = squareform(dat)
    for d in np.arange(2,dmax):
        embedding = MDS(n_components=d, dissimilarity='precomputed', random_state=seed)
        xlocs = embedding.fit_transform(dat)
        est_dist = pdist(xlocs)
        stat, pval = st.ks_2samp(est_dist, true_dist)
        if pval > 0.05:
            break
    return d, xlocs, embedding.stress_


def plot_predictions(pred_clusters, targets, z=None):
    fig, ax = plt.subplots(pred_clusters.shape[1],1,figsize = (pred_clusters.shape[1],4*pred_clusters.shape[1]))
    ymax = np.max([np.max(pred_clusters.detach().numpy()), np.max(targets.detach().numpy())])
    ymax = ymax + 0.1*ymax
    ymin = np.min([np.min(pred_clusters.detach().numpy()), np.min(targets.detach().numpy())])
    ymin = ymin - 0.1*ymin
    for s in np.arange(pred_clusters.shape[1]):
        if z is not None:
            ixs = np.where(z[:,s] == 1)[0]
        else:
            ixs = np.arange(targets.shape[1])
        ax[s].bar(np.arange(pred_clusters.shape[0])-0.2, pred_clusters[:,s].detach().numpy(), width = 0.4, label = 'Predicted')
        ax[s].bar(np.arange(targets.shape[0]) + 0.2, np.mean(targets[:, ixs].detach().numpy(),1), width=0.4, label = 'True Avg Over Metabolites')
        # ax[s].errorbar(np.arange(targets.shape[0]) + 0.2, np.mean(targets.detach().numpy(),1),
        #                yerr = np.std(targets.detach().numpy(), 1), fmt = 'o')
        ax[s].set_ylim(ymin, ymax)
        ax[s].axhline(y=0)
        ax[s].set_title('Cluster ' + str(s))
        ax[s].set_xlabel('Subjects')
        ax[s].legend(loc= 'upper right')
    fig.tight_layout()
    return fig

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sigmoid(logits, temperature, epsilon):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    temp = logits + sample_gumbel(logits.size())
    y = (1-2*epsilon)*torch.sigmoid(temp/temperature) + epsilon

    y_hard = torch.round(y)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y, y_hard

def get_lr(T, lr_max, lr_min, Tcurr):
    return lr_min + 0.5*(lr_max - lr_min)*(1 + np.cos(Tcurr*np.pi/T))

def unmix_clusters(mu,pred_mu,pred_r,locs):
    n_clusters = mu.shape[0]
    mapping = {}
    for cluster in range(n_clusters):
        closest_loc = np.argmin(np.sum((locs - mu[cluster,:])**2,1))
        dists = np.sum(np.sqrt((locs[closest_loc,:] - pred_mu)**2),1)
        new = np.argmin(dists)
        all_dists = np.mean((locs - pred_mu[new,:])**2,1)
        if (all_dists > pred_r[new]).all():
            continue
        if new not in mapping.values():
            mapping[cluster] = new
    unassigned = (set(range(n_clusters)) - set(mapping.keys())), set(range(n_clusters)) - set(mapping.values())
    for k, v in zip(*unassigned):
        mapping[k] = v
    return mapping


def smoothmax(x, tau = 1):
    return torch.sum(x*torch.exp(x/tau))/torch.sum(torch.exp(x/tau))

def euc_dist(x,y):
    return np.sqrt(np.sum([(x[i] - y[i])**2 for i in range(len(x))]))

def pairwise_eval(guess, true):
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
    if l is None:
        l = len(np.unique(x))
    if torch.is_tensor(x):
        vec = torch.zeros(l)
    else:
        vec = np.zeros(l)
    vec[x] = 1
    return vec

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def isclose(a, b, tol=1e-03):
    return (abs(a-b) <= tol).all()

def concrete_sampler(a, T, dim, loc = 0, scale = 1):
    G = st.gumbel_r(loc, scale).rvs(size = a.shape)
    return torch.softmax(T*(torch.log(torch.Tensor(a)) + torch.Tensor(G)), dim = dim)


def get_epsilon(data):
    vals = np.array(data).flatten()
    vals[np.where(vals == 0)[0]] = np.max(vals)
    epsilon = 0.1*np.min(np.abs(vals))
    return epsilon

def filter_by_train_set(x_train, x_test, meas_key, key = 'metabs', log_transform = True, standardize_data = True):
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

    temp = data.iloc[:,rm2]
    # import pdb; pdb.set_trace()
    # if len(np.where(np.sum(temp,0)==0)[0]) > 0:
    #     import pdb; pdb.set_trace()
    return data.iloc[:,list(rm2)]

def standardize(x,override = True):
    if not override:
        assert(x.shape[0]<x.shape[1])

    dem = np.std(x,0)
    if (dem == 0).any():
        dem = np.where(np.std(x, 0) == 0, 1, np.std(x, 0))
    stand = (x - np.mean(x, 0))/dem
    ix = np.where((stand == 1).all())[0]
    stand.iloc[:, ix] = 0
    return stand, np.mean(x,0), dem


def filter_by_pt(dataset, targets=None, perc = .15, pt_thresh = 1, meas_thresh = 10, weeks = [0,1,2]):
    # tmpts = [float(x.split('-')[1]) for x in dataset.index.values if x.replace('.','').isnumeric()]
    # mets is dataset with ones where data is present, zeros where it is not
    mets = np.zeros(dataset.shape)
    mets[np.abs(dataset) > meas_thresh] = 1


    if weeks is not None:
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

    mets_all_keep = []
    # For each class, count how many measurements exist within that class and keep only measurements in X perc in each class
    if targets is not None:
        if sum(['-' in x for x in targets.index.values]) > 1:
            labels = targets[dataset.index.values]
        else:
            labels = targets[np.array(pts)]
        for lab_cat in np.unique(labels):
            mets_1 = mets[np.where(labels == lab_cat)[0], :]
            met_counts = np.sum(mets_1, 0)
            met_keep_ixs = np.where(met_counts >= np.round(perc * mets_1.shape[0]))[0]
            mets_all_keep.extend(met_keep_ixs)
    else:
        met_counts = np.sum(mets, 0)
        mets_all_keep = np.where(met_counts >= np.round(perc * mets.shape[0]))[0]
    return dataset.iloc[:,np.unique(mets_all_keep)]

def get_meas_var(raw_data, repeat_data):
    epsilon = get_epsilon(raw_data)
    raw_log = np.log(raw_data + epsilon)
    mn = np.mean(raw_log,0)
    stdev = np.std(raw_log,0)
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
    x.to_csv(data_path + '/' + xfile)
    y.to_csv(data_path + '/' + yfile)

def make_tree(feats, base_path, case, func='asv',
              newick_path='/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
              dist_type = ''):
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
            raise Exception("Run phylogenetic tree script to get ASV tree")

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

def make_dist_mat(dat, dist_file, base_path, newick_path, dist_type = '', yfile = ''):
    if '_' in dist_type:
        in_list = ["python3", "rdk_fingerprints.py", "-fingerprint", dist_type.split('_')[0],
                   "-metric" + dist_type.split('_')[1], "-yfile", yfile]
        subprocess.run(in_list, cwd = base_path + '/rdk')
    else:
        in_list = ["python3", "tree_plotter.py", "-fun", 'dist', "-name", dist_file,
                   "-newick", base_path + newick_path, "-dtype", dist_type, "-feat"]
        in_list.extend(dat)
        subprocess.run(in_list, cwd=base_path + "/ete_tree")

def get_rand_locs(dat, dim, seed):
    a_met = np.random.uniform(0, 10, size=(dat.shape[1], dat.shape[1]))
    a_met = (a_met + a_met.T) / 2
    np.fill_diagonal(a_met, 0)
    embedding = MDS(n_components=dim, dissimilarity='precomputed', random_state=seed)
    locs = embedding.fit_transform(a_met)
    return locs

def get_xtaxa(path, x):
    x_taxa = pd.read_csv(path, header=0, index_col=0)
    x_fams = pd.Series([taxon.split('; ')[-3] for taxon in x_taxa['taxa_rdp']], index=x_taxa['labels'])
    x_fams_silva = pd.Series([taxon.split('; ')[-3] for taxon in x_taxa['taxa_silva']], index=x_taxa['labels'])
    x_fams[x_fams == 'NA'] = x_fams_silva[x_fams == 'NA']
    x_fams = x_fams.loc[x.columns.values]
    return x_fams

def edit_string(string):
    return string.replace('(', '_').replace(')', '_').replace(':',
                                                 '_').replace(','
                                                              , '_').replace('[', '_').replace(']', '_').replace(';',
                                                                                                                 '_')

def get_ytaxa(path, feats, ydist, level='subclass'):
    y_taxa = pd.read_csv(path, header=0, index_col=0)
    y_class = y_taxa[feats].loc[level]
    null_vals = y_class[y_class.isnull()].index.values
    for null_sub in y_class[y_class.isnull()].index.values:
        sorted_ls = ydist.loc[null_sub].sort_values().drop(null_vals)
        rep = y_class.loc[sorted_ls.index.values[0]]
        y_class.loc[null_sub] = rep
    return y_class

def numpy_forward(x,w,beta,alpha,sigma,omega_temp, alpha_temp, K, mu_bug =None, r_bug = None, microbe_locs = None, gmm = True):
    # This is just to keep alpha from getting to close to 0 or 1 and causing numerical issues
    omega_epsilon = omega_temp / 4
    alpha_epsilon = alpha_temp / 4
    if microbe_locs is not None:
        kappa = torch.stack(
            [torch.sqrt(((mu_bug - torch.tensor(microbe_locs[m, :])).pow(2)).sum(-1)) for m in
             np.arange(microbe_locs.shape[0])])
        w_act = torch.sigmoid((torch.exp(r_bug) - kappa)/omega_temp)
    else:
        w_act = w
    g = x@w_act
    if gmm:
        out_clusters = beta[0, :]+ st.norm(0,np.sqrt(np.exp(sigma))).rvs((g.shape[0], K))
    else:
        out_clusters = beta[0,:] + np.matmul(g, beta[1:,:]*alpha) + st.norm(0,np.sqrt(np.exp(sigma))).rvs((g.shape[0], K))

    return out_clusters

    # fig_path = base_path + '/figures/subclass/'
    # if not os.path.isdir(fig_path):
    #     os.mkdir(fig_path)
    # radii = []
    # means = []
    # locss = []
    # gps = []
    # # y_class = y_class.replace(np.nan, 'NAN')
    # for bc in np.unique(y_class.values):
    #     ixs = np.where(y_class.values == bc)[0]
    #     locs = ylocs[ixs, :]
    #     radii.append(np.sqrt(np.sum((np.max(locs, 0) - np.min(locs, 0)) ** 2)) / 2)
    #     means.append(np.mean(locs, 0))
    #     locss.append(locs)
    #     gps.append(bc)
    # scale = 10 * np.var(radii)
    # loc = np.mean(radii)
    # colors = cm.rainbow(np.linspace(0, 1, len(gps)))
    # # handles = []
    # for i, gp_rad in enumerate(radii):
    #     fig, ax = plt.subplots(figsize=(5, 5))
    #     ax.scatter(ylocs[:, 0], ylocs[:, 1], color='k')
    #     gp_mean = means[i]
    #     circle = plt.Circle((gp_mean[0], gp_mean[1]), gp_rad, color=
    #     colors[i], alpha=0.2)
    #
    #     ax.scatter(locss[i][:, 0], locss[i][:, 1], color=colors[i], marker='s')
    #     ax.add_patch(circle)
    #     ax.scatter([gp_mean[0]], [gp_mean[1]], marker='*', color=colors[i])
    #     ax.text(gp_mean[0], gp_mean[1], gps[i], fontsize=6)
    #     ax.add_patch(circle)
    #     ax.set_aspect('equal')
    #     fig.tight_layout()
    #     fig.savefig(fig_path + '/' + gps[i] + '-init-locs.pdf')
    #     plt.close(fig)
    # plot_locations(radii, means, locss, gps, name='met')


# points = [(0,1),(2,4),(3,2),(5,8),(0,2),(4,0),(3,7),(8,4)]
# plt.scatter(np.array(points)[:,0], np.array(points)[:,1])
# plt.show()
# hull = jarvis(points)