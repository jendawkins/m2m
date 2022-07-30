import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import torch.optim as optim
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from helper import *
from sklearn.model_selection import KFold
from scipy.special import logit
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score
from matplotlib.pyplot import cm
from collections import Counter
# from skbio.stats.ordination import pcoa
# from skbio.stats.distance import DistanceMatrix
import scipy

def plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs,
                  mu_bug, r_bug, mu_met, r_met, gen_u, gen_alpha, gen_beta):
    """
    Given outputs of data_gen.py,
    plot:
        - distribution of microbial relative abundances
        - distribution of microbial relative abundances summed into generated clusters
        - distribution of metabolite levels
        - 2D microbe locations and clusters (if embedding dim = 2)
        - 2D metabolite locations and clusters (if embedding dim = 2)
        - Microbe clusters vs metabolite cluster
    write:
        - file with list of lists where each list is a cluster of microbes
        - file with list of lists where each list is a cluster of metabolites
    """
    # distribution of microbial relative abundances
    plt.figure();
    plt.hist(x.flatten(), bins=20);
    plt.title('Microbe data distribution')
    plt.savefig(path + 'bug_hist.png')
    plt.close()

    # distribution of microbial relative abundances summed into generated clusters
    plt.figure();
    plt.hist(g.flatten(), bins=20);
    plt.title('Microbe cluster data distribution')
    plt.savefig(path + 'bug_hist.png')
    plt.close()

    # distribution of metabolite levels
    plt.figure();
    plt.hist(y.flatten(), bins=20);
    plt.title('Metabolite data distribution')
    plt.savefig(path + 'met_hist.png')
    plt.close()

    bug_clusters = [np.where(gen_u[:,i])[0] for i in np.arange(gen_u.shape[1])]

    # file with list of lists where each list is a cluster of microbes
    for ii,clust in enumerate(bug_clusters):
        if not os.path.isfile(path + 'microbe_clusters.txt'):
            with open(path + 'microbe_clusters.txt', 'w') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')
        else:
            with open(path + 'microbe_clusters.txt', 'a') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')


    # 2D microbe locations and clusters (if embedding dim = 2)
    fig2, ax2 = plt.subplots(3, 1, figsize = (8, 6*3))
    if mu_bug.shape[1]==2:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        p1 = ax[0].scatter(gen_bug_locs[:, 0], gen_bug_locs[:, 1], color = 'k', alpha = 0.5)
    if len(mu_bug.shape)>2:
        mu_bug = mu_bug[0,:,:]
        r_bug = r_bug[0,:]
    for i in range(mu_bug.shape[0]):
        if mu_bug.shape[1] == 2:
            p1 = ax[0].scatter(mu_bug[i,0], mu_bug[i,1], marker='*')
            circle1 = plt.Circle((mu_bug[i,0], mu_bug[i,1]), r_bug[i],
                                 alpha=0.2, label='Cluster ' + str(i),color=p1.get_facecolor().squeeze())
            # for ii in ix:
            #     ax[0].text(gen_bug_locs[ii,0], gen_bug_locs[ii,1], 'Bug ' + str(ii))
            ax[0].add_patch(circle1)
            ax[0].set_title('Microbes')
            ax[0].text(mu_bug[i,0], mu_bug[i,1], 'Cluster ' + str(i))
        # for ii in ix:
        bins = int((x.max() - x.min()) / 5)
        if bins<=10:
            bins = 10
        ix = np.where(gen_u[:, i]==1)[0]
        ax2[0].hist(x[:, ix].flatten(), range=(x.min(), x.max()), label='Cluster ' + str(i), alpha=0.5, bins = bins)
    ax2[0].set_xlabel('Microbial relative abundances')
    ax2[0].set_title('Microbes')
    if mu_bug.shape[1] == 2:
        ax[0].set_aspect('equal')

    b = x@gen_u
    bins = int((b.max() - b.min()) / 5)
    if bins <= 10:
        bins = 10
    for k in range(b.shape[1]):
        ax2[1].hist(b[:,k].flatten(), range = (b.min(), b.max()), label = 'Cluster ' + str(k), alpha = 0.5, bins = bins)
    ax2[1].set_title('Histogram of microbe cluster sums')
    ax2[1].legend(loc = 'upper right')

    # 2D metabolite locations and clusters (if embedding dim = 2)
    ax2[0].legend(loc = 'upper right')
    ax2[1].legend(loc = 'upper right')
    for i in range(gen_z.shape[1]):
        ix = np.where(gen_z[:, i] == 1)[0]
        if mu_bug.shape[1] == 2:
            p2 = ax[1].scatter(gen_met_locs[ix, 0], gen_met_locs[ix, 1])
            ax[1].scatter(mu_met[i, 0], mu_met[i, 1], marker='*', color=p2.get_facecolor().squeeze())
            ax[1].set_title('Metabolites')
            ax[1].text(mu_met[i, 0], mu_met[i, 1], 'Cluster ' + str(i))
            circle2 = plt.Circle((mu_met[i,0], mu_met[i,1]), r_met[i],
                                 alpha=0.2, color=p2.get_facecolor().squeeze(), label = 'Cluster ' + str(i))
            ax[1].add_patch(circle2)
        bins = int((y.max() - y.min())/5)
        if bins<=10:
            bins = 10
        ax2[2].hist(y[:, ix].flatten(), range=(y.min(), y.max()),
                    label='Cluster ' + str(i), alpha=0.5, bins = bins)
    ax2[2].set_xlabel('Standardized metabolite levels')
    ax2[2].set_title('Metabolites')

    if mu_bug.shape[1] == 2:
        ax[1].set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path + 'embedded_locations.png')
        plt.close(fig)
    fig2.savefig(path + 'cluster_histogram.png')
    plt.close(fig2)

    try:
        bug_active,met_active = np.where((gen_alpha * gen_beta[1:,:])>= 1e-6)
    except:
        bug_active, met_active = np.where((gen_alpha) >= 1e-6)

    K = gen_z.shape[1]
    L = gen_u.shape[1]
    fig, ax = plt.subplots(len(bug_active), 1, figsize=(8, 8 * len(bug_active)))
    ii = 0
    # Microbe clusters vs metabolite cluster
    for bug_clust, met_clust in zip(bug_active, met_active):
        ixs = np.where(gen_z[:, met_clust] == 1)[0]
        for ix in ixs:
            ax[ii].scatter(g[:,bug_clust], y[:, ix])
        ax[ii].set_xlabel('Microbe sum of cluster ' + str(bug_clust))
        ax[ii].set_ylabel('Metabolite values in cluster ' + str(met_clust))
        ax[ii].set_title('Bug cluster ' + str(bug_clust) + ' vs met cluster ' + str(met_clust))
        slope = np.round((np.max(y[:, ixs[0]]) - np.min(y[:, ixs[0]])) / ((np.max(g[:, bug_clust]) - np.min(g[:, bug_clust]))), 3)
        ax[ii].text(0.6, 0.8, 'slope = ' + str(slope), horizontalalignment='center',
                      verticalalignment='center', transform=ax[ii].transAxes)
        ii += 1
        try:
            ax[ii].text(0.6, 0.6, 'beta = ' + str(gen_beta[bug_clust + 1, met_clust]), horizontalalignment='center',
                          verticalalignment='center', transform=ax[ii].transAxes)
        except:
            continue
    fig.tight_layout()
    fig.savefig(path + '-sum_x_v_y.png')
    plt.close(fig)

    # Metabolite levels for first 8 subjects
    fig, ax = plt.subplots(8,1, figsize = (8,4*8))
    for i in range(gen_z.shape[1]):
        ixs = np.where(gen_z[:, i] == 1)[0]
        for s in range(8):
            ax[s].hist(y[s, ixs].flatten(), range=(y[s,:].min(), y[s,:].max()),
                        label='Cluster ' + str(i), alpha=0.5, bins=bins)
    fig.tight_layout()
    fig.savefig(path + '-per_part_metabolites.png')
    plt.close(fig)


def plot_distribution(dist, param, true_val = None, ptype = 'init', path = '', **kwargs):
    """
    Plot initial distributions
    Input:
        - dist: pytorch distribution
        - param: string name of parameter
        - true_val: whether the parameter has a true value or not (i.e. if data is generated, can set true_val =True)
        - ptype: options 'init' and 'priors' - whether we are plotting the distribution of the priors or initialized values
        - path: path to save files
    Output:
        - plot of distribution in path + '/' + ptype + '/' param.pdf
    """
    if true_val is not None:
        true_val = true_val[param]
    if ptype == 'init':
        label = 'Initialized values'
    vals = dist.sample([500])
    if 'r' in param:
        vals = 1/vals
    elif 'z' in param or 'w' in param and ptype != 'init':
        return
        # vals = dist.sample([500])
        # vals = torch.softmax(vals, 1)
    elif 'z' in param or 'w' in param and ptype == 'init':
        if true_val is not None:
            vals = dist.sample([500, true_val.shape[0], true_val.shape[1]])
            vals = torch.softmax(vals, 1)
            true_val = torch.softmax(true_val, 1)
    mean, std = np.round(vals.mean().item(),2), np.round(vals.std().item(),2)
    if len(vals.shape)>1:
        vals = vals.flatten()
    fig, ax = plt.subplots()
    bins = 10
    ax.hist(np.array(vals), bins = bins)
    ax.set_title(param + ', mean=' + str(mean) + ', std=' + str(std))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if true_val is not None:
        if isinstance(true_val, float):
            tv = [true_val]
        elif isinstance(true_val, list) or len(true_val.shape)<=1:
            tv = true_val
        else:
            tv = true_val.flatten()
        for k in np.arange(len(tv)):
            if k == 0:
                ax.axvline(tv[k], c = 'r', label = 'True value')
            else:
                ax.axvline(tv[k], c='r')
            ax.legend(loc = 'upper right')
    r = [k + '_' + str(np.round(item,2)).replace('.','d') for k, item in kwargs.items()]
    if not os.path.isdir(path + '/' + ptype + 's/'):
        os.mkdir(path + '/' + ptype + 's/')
    plt.tight_layout()
    plt.savefig(path + '/' + ptype + 's/' + param + '-' + '-'.join(r))
    plt.close(fig)

def plot_posterior(param_dict, seed, out_path):
    """
    Plot posterior distributions of each parameter saved in param_dict; plot histogram of the last fourth of iterations
    (i.e. if iterations = 12,000, plot histogram of parameter values from epoch = 9,000 to epoch = 12,000)
    Plots saved in out_path + '/posteriors/'
    """
    for key in param_dict.keys():
        start = len(param_dict[key])
        fig, ax = plt.subplots()

        all_dat = param_dict[key]
        if 'r_' in key or 'e_met' in key:
            all_dat = [np.exp(xx) for xx in all_dat]
        elif 'pi_met' in key:
            all_dat = [scipy.special.softmax(xx, 1) for xx in all_dat]

        try:
            all_dat1 = np.concatenate(all_dat)
            all_dat1 = all_dat1
        except:
            all_dat1 = all_dat

        range = (np.array(all_dat1).flatten().min(), np.array(all_dat1).flatten().max())

        iterset = np.int(len(param_dict[key])/4)
        posterior_list = all_dat[start - iterset: start]

        try:
            dat = np.concatenate(posterior_list)
            dat = dat.flatten()
        except:
            dat = posterior_list

        ax.hist(dat, range = range, bins=20, label='Iterations ' + str(start - iterset) + ' to ' + str(start), alpha=0.5)
        ax.legend()
        ax.set_title(key + ', ' + str(len(param_dict[key])) + ' iterations')
        if not os.path.isdir(out_path + '/posteriors'):
            os.mkdir(out_path + '/posteriors')
        fig.savefig(out_path + '/posteriors/' + str(seed) + '-' + key + '-posterior_dist.pdf')

def plot_param_traces(path, param_dict, true_vals, fold):
    """
    Plot parameter traces over the course of learning
    Inputs: path = path to save; param_dict = parameter dictionary; params2learn = whichever parameters the model is learning;
        true_vals = true val dictionary if data is synthetic, otherwise set to None
        net = model
        fold = seed
    """
    fig_dict, ax_dict = {},{}
    for name, plist in param_dict.items():
        if len(plist[0].shape) == 0:
            n = 1
        else:
            n = plist[0].squeeze().shape[0]
            if n > 5:
                n = 5
        if len(plist[0].squeeze().shape)<=1:
            fig_dict[name], ax_dict[name] = plt.subplots(n,
                                                         figsize=(5, 4 * n))
        else:
            nn = plist[0].squeeze().shape[1]
            if nn > 5:
                nn = 5
            fig_dict[name], ax_dict[name] = plt.subplots(n, nn,
                                                         figsize = (5*nn, 4*n))
        try:
            dat = np.concatenate([p.flatten() for p in plist])
        except:
            dat = plist

        for k in range(n):
            if len(plist[0].squeeze().shape) <= 1:
                if name == 'pi_met':
                    trace = [scipy.special.softmax(p,1).squeeze()[k] for p in plist]
                    mindat = -0.05
                    maxdat = 1.05
                elif name == 'e_met':
                    trace = [np.exp(p).squeeze()[k] for p in plist]
                    mindat = -0.05
                    maxdat = 1.05
                else:
                    trace = [p.squeeze()[k] for p in plist]
                    mindat = np.min(trace) - 0.1*np.mean(np.abs(trace))
                    maxdat = np.max(trace) + 0.1*np.mean(np.abs(trace))
                ax_dict[name][k].plot(trace, label='Trace')

                if true_vals is not None:
                    if name in true_vals.keys():
                        if not isinstance(true_vals[name], list):
                            true_vals[name] = true_vals[name].squeeze()
                        if np.array(true_vals[name]).shape[0] > k:
                            ax_dict[name][k].plot([true_vals[name][k]] * len(trace), c='r', label='True')
                    ax_dict[name][k].set_title(name + ', ' + str(k))
                # if name in net.range_dict.keys():
                ax_dict[name][k].set_ylim([mindat, maxdat])
                ax_dict[name][k].legend(loc = 'upper right')
                ax_dict[name][k].set_xlabel('Iterations')
                ax_dict[name][k].set_ylabel('Parameter Values')
            else:
                for j in range(nn):
                    new_k, new_j = k, j
                    if 'r_' in name:
                        trace = [np.exp(p).squeeze()[new_k, new_j] for p in plist]
                    else:
                        trace = [np.exp(p).squeeze()[new_k, new_j] for p in plist]
                    if 'alpha' in name or 'w_act' in name:
                        mindat = -0.05
                        maxdat = 1.05
                    else:
                        mindat = np.min(trace) - 0.1 * np.mean(np.abs(trace))
                        maxdat = np.max(trace) + 0.1 * np.mean(np.abs(trace))
                    ax_dict[name][k, j].plot(trace, label='Trace')
                    if true_vals is not None:
                        if name in true_vals.keys():
                            if np.array(true_vals[name]).shape[1]>j and np.array(true_vals[name]).shape[0] > k:
                                ax_dict[name][k, j].plot([true_vals[name][k, j]] * len(trace), c='r', label='True')
                    ax_dict[name][k, j].set_title(name + ', ' + str(k) + ', ' + str(j))
                    # if name in net.range_dict.keys():
                    ax_dict[name][k, j].set_ylim([mindat, maxdat])
                    ax_dict[name][k, j].legend(loc = 'upper right')
                    ax_dict[name][k, j].set_xlabel('Iterations')
                    ax_dict[name][k, j].set_ylabel('Parameter Values')

        if not os.path.isdir(path + 'traces/'):
            os.mkdir(path + 'traces/')
        fig_dict[name].tight_layout()
        fig_dict[name].savefig(path + 'traces/seed' + str(fold) + '_' + name + '_parameter_trace.png')
        plt.close(fig_dict[name])

def plot_locations(path, radii, means, locs, gps, name = 'bug'):
    """
    Plot 2D embedded locations and circles indicating cluster, only plot if embedded dimension = 2
    """
    colors = cm.rainbow(np.linspace(0, 1, len(gps)))
    fig, ax = plt.subplots(figsize = (10,10))
    # handles = []
    for i,gp_rad in enumerate(radii):
        gp_mean = means[i]
        gp_locs = locs[i]
        circle = plt.Circle((gp_mean[0], gp_mean[1]), gp_rad, color=
                            colors[i], alpha = 0.2)
        ax.add_patch(circle)
        ax.scatter(gp_locs[:,0], gp_locs[:,1], color = colors[i])
        ax.scatter([gp_mean[0]], [gp_mean[1]], marker = '*', color = colors[i])
        ax.text(gp_mean[0],gp_mean[1], gps[i], fontsize = 6)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        # handles.append(p)
    # ax.legend(handles, gps, prop = {'size': 3})
    fig.tight_layout()
    fig.savefig(path + '/' + name + '-init-locs.pdf')
    plt.close(fig)


def plot_output_locations(path, net, best_mod, param_dict, fold, type = 'best', plot_zeros = False):
    """
    Plot 2D embedded locations and circles indicating cluster, only plot if embedded dimension = 2
    """
    best_w = param_dict['w'][best_mod]
    best_mu = param_dict['mu_bug'][best_mod]
    best_r = np.exp(param_dict['r_bug'][best_mod])
    best_alpha = param_dict['alpha'][best_mod]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(net.microbe_locs[:,0], net.microbe_locs[:,1], facecolors='none',
               edgecolors='k')
    for i in range(best_w.shape[1]):
        ix = np.where(best_w[:,i] > 0.5)[0]
        if (len(ix) == 0 or np.sum(best_alpha[i,:])<0.5) and not plot_zeros:
            ax.scatter([], [])
            continue
        p2 = ax.scatter(net.microbe_locs[ix, 0], net.microbe_locs[ix, 1])
        ax.scatter(best_mu[i, 0], best_mu[i, 1], marker='*', color=p2.get_facecolor().squeeze())
        ax.text(best_mu[i, 0], best_mu[i, 1], 'predicted\ncluster ' + str(i) + ' mean')
        ax.set_title('Microbes')
        circle2 = plt.Circle((best_mu[i, 0], best_mu[i, 1]), best_r[i],
                             alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
        ax.add_patch(circle2)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(path + 'seed' + str(fold) + '-' + type + '-plot_zeros'*plot_zeros + '-bug_clusters.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    best_z = param_dict['z'][best_mod]
    best_mu = param_dict['mu_met'][best_mod]
    best_r = np.exp(param_dict['r_met'][best_mod])
    ax.scatter(net.met_locs[:, 0], net.met_locs[:, 1], facecolors='none',
               edgecolors='k')
    for i in range(param_dict['z'][0].shape[1]):
        ix = np.where(best_z[:,i] > 0.5)[0]
        if len(ix) == 0 and not plot_zeros:
            ax.scatter([], [])
            continue

        p2 = ax.scatter(net.met_locs[ix, 0], net.met_locs[ix, 1])
        ax.scatter(best_mu[i, 0], best_mu[i, 1], marker='*', color=p2.get_facecolor().squeeze())
        ax.text(best_mu[i, 0], best_mu[i, 1], 'predicted\ncluster ' + str(i) + ' mean')
        ax.set_title('Metabolites')
        circle2 = plt.Circle((best_mu[i, 0], best_mu[i, 1]), best_r[i],
                             alpha=0.2, color=p2.get_facecolor().squeeze(), label='Cluster ' + str(i))
        ax.add_patch(circle2)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(path + 'seed' + str(fold) + '-' + type +'-plot_zeros'*plot_zeros + '-predicted_metab_clusters.png')
    plt.close(fig)

def get_interactions_csv(path, best_mod, param_dict, seed):
    """
    Save csv of iteractions between each active cluster
    """
    best_w = param_dict['w'][best_mod]
    best_z = param_dict['z'][best_mod]
    best_alpha = param_dict['alpha'][best_mod]
    best_alpha_beta = param_dict['beta*alpha'][best_mod]
    active_microbes = list(set(np.where(np.sum(best_w,0)>1)[0]).intersection(set(np.where(best_alpha.sum(0)!= 0)[0])))
    active_mets = np.where(np.sum(best_z, 0) > 1)[0]
    df = {}
    for microbe, met in itertools.product(active_microbes, active_mets):
        if (microbe, met) not in df.keys() and (met, microbe) not in df.keys():
            df[(microbe, met)] = best_alpha_beta[microbe, met]

    pd.Series(df).to_csv(path + '/' + str(seed) + '-interactions.csv')


def plot_xvy(path, x, out_vec, best_mod, param_dict, seed):
    """
    Plot interactions between each active cluster
    """
    out = out_vec[best_mod].detach().numpy()
    best_w = param_dict['w'][best_mod]
    best_z = param_dict['z'][best_mod]
    best_alpha = param_dict['alpha'][best_mod]
    best_beta = param_dict['beta'][best_mod]

    # out = out[:, mapping['met']]
    microbe_sum = x.detach().numpy() @ best_w
    # true_sum = x.detach().numpy() @ gen_w
    # microbe_sum = microbe_sum[:, mapping['bug']]
    # target_sum = targets @ gen_z
    num_active_microbe = len(np.where(np.sum(best_w,0)>(0.1))[0])
    num_active_met = len(np.where(np.sum(best_z, 0) > 0)[0])
    if num_active_met==0:
        num_active_met = 1
    if num_active_microbe==0:
        num_active_microbe = 1
    fit_df = {}
    active_microbes = np.where(np.sum(best_w,0)>1)[0]
    active_mets = np.where(np.sum(best_z, 0) > 1)[0]
    ii=0
    x_dict = {}
    y_dict = {}
    for i in active_mets:
        ixs = np.where(best_z[:,i]==1)[0]
        if len(ixs) == 0:
            continue
        jj = 0
        if i not in y_dict.keys():
            y_dict[i] = out[:, i]
        for j in active_microbes:

            fig, ax = plt.subplots()
            ixs = np.where(best_w[:,j]>0.1)[0]
            if len(ixs) == 0 or best_alpha[j,i]<0.1:
                continue
            ax.scatter(microbe_sum[:, j], out[:, i], c = 'b', label = 'Predicted')
            if j not in x_dict.keys():
                x_dict[j] = microbe_sum[:, j]

            fit_df[(j, i)] = {}
            ax.set_xlabel('Microbe sum')
            ax.set_ylabel(r'$y_{i}$ when $i=$' + str(i))
            # ax.set_title('Met Clust ' + str(i) + ' vs Microbe Clust ' + str(j) + '\n ')
            ax.legend(loc = 'upper right')
            lreg = st.linregress(microbe_sum[:, j], out[:, i])
            ax.set_title('Met Clust ' + str(i) + ' vs Microbe Clust ' + str(j) +
                         '\n r2= ' + str(np.round(lreg.rvalue**2,3)))
            fit_df[(j,i)]['r-squared value'] = lreg.rvalue**2
            fit_df[(j,i)]['pvalue'] = lreg.pvalue
            fit_df[(j,i)]['slope'] = lreg.slope
            fit_df[(j,i)]['intercept'] = lreg.intercept

            fit_df[(j,i)]['alpha'] = np.round(best_alpha[j, i],3)
            if not os.path.isdir(path + '/seed' + str(seed)):
                os.mkdir(path + '/seed' + str(seed))
            # if not os.path.isdir(path + '/seed' + str(seed) + '/' + 'metclust' + str(i) + '_vs_' + 'microbeclust' + str(j)):
            #     os.mkdir(path + '/seed' + str(seed) + '/' + 'metclust' + str(i) + '_vs_' + 'microbeclust' + str(j))
            fig.savefig(path + '/seed' + str(seed)+ '/' + 'metclust' + str(i) + '_vs_' + 'microbeclust' + str(j) + '-sum_x_v_y.png')
            plt.close(fig)
            jj += 1
            try:
                fit_df[(j,i)]['beta'] = np.round(best_beta[j+1, i],3)
                fit_df[(j,i)]['alpha*beta'] = np.round(best_beta[j + 1, i] * best_alpha[j, i], 3)
            except:
                continue

        ii+=1

    try:
        pd.DataFrame(fit_df).T.to_csv(path + 'seed' + str(seed) + '-fit.csv')
        df = pd.DataFrame(fit_df).T
        if not os.path.isfile(path + 'rfit.txt'):
            with open(path + 'rfit.txt', 'w') as f:
                f.write('Seed ' + str(seed) + ': ' + str(np.round(np.mean(df['r-squared value']), 3)) +  ' +- ' +
                        str(np.round(np.std(df['r-squared value']), 3)) + '\n')
        else:
            with open(path + 'rfit.txt', 'a') as f:
                f.write('Seed ' + str(seed) + ': ' + str(np.round(np.mean(df['r-squared value']), 3)) +  ' +- ' +
                        str(np.round(np.std(df['r-squared value']), 3)) + '\n')
    except:
        print('')

    return x_dict, y_dict

def save_cluster_results(path, best_mod, true_vals,seed,
                param_dict, metabs = None, seqs = None):
    """
    Function:
        Saves dataframe of pairwise confusion matrices for metabolite cluster identities and microbe cluster identities
        Saves dataframe of cluster IDs
        Writes files with # of active metabolite and microbial clusters
    """
    pred_z = param_dict['z'][best_mod]
    pred_w = param_dict['w'][best_mod]
    if true_vals is not None:
        gen_z = true_vals['z']
        true = np.argmax(gen_z,1)
        z_guess = np.argmax(pred_z, 1)
        nmi = np.round(normalized_mutual_info_score(true, z_guess), 3)
        try:
            tp, fp, tn, fn, ri = pairwise_eval(z_guess, true)
            pairwise_cf = {'Same cluster': {'Predicted Same cluster': tp, 'Predicted Different cluster': fn},
                           'Different cluster':{'Predicted Same cluster': fp, 'Predicted Different cluster': tn}}
            pd.DataFrame(pairwise_cf).T.to_csv(
                path + 'seed' + str(seed) + '_PairwiseConfusionMetabs_' +  str(np.round(ri,3)).replace('.', 'd') + '.csv')
            ri = str(np.round(ri, 3))
        except:
            ri = 'NA'
        if not os.path.isfile(path + 'nmi_ri.txt'):
            with open(path + 'nmi_ri.txt', 'w') as f:
                f.write('Seed ' + str(seed) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')
        else:
            with open(path + 'nmi_ri.txt', 'a') as f:
                f.write('Seed ' + str(seed) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')

    cluster_df = {}
    for cluster in range(pred_z.shape[1]):
        met_ids = np.where(pred_z[:, cluster] == 1)[0]
        if len(met_ids) == 0:
            continue
        if metabs is not None:
            cluster_df[cluster] = str(metabs[met_ids])
        else:
            cluster_df[cluster] = str(met_ids)
    pd.Series(cluster_df).to_csv(path + 'seed' + str(seed) + '_metabolite_cluster_ids.csv')

    if os.path.isfile(path + 'number_of_met_clusters.txt'):
        with open(path + 'number_of_met_clusters.txt', 'a') as f:
            f.write('Seed ' + str(seed) + ': ' + str(np.sum(np.sum(pred_z,0)>0)) + '\n')
    else:
        with open(path + 'number_of_met_clusters.txt', 'w') as f:
            f.write('Seed ' + str(seed) + ': ' + str(np.sum(np.sum(pred_z,0)>0)) + '\n')

    if os.path.isfile(path + 'number_of_bug_clusters.txt'):
        with open(path + 'number_of_bug_clusters.txt', 'a') as f:
            f.write('Seed ' + str(seed) + ': ' + str(np.sum(np.sum(np.round(pred_w), 0) > 0)) + '\n')
    else:
        with open(path + 'number_of_bug_clusters.txt', 'w') as f:
            f.write('Seed ' + str(seed) + ': ' + str(np.sum(np.sum(np.round(pred_w), 0) > 0)) + '\n')


    cluster_df = {}
    for cluster in range(pred_w.shape[1]):
        met_ids = np.where(pred_w[:, cluster] > 0.5)[0]
        if len(met_ids) == 0:
            continue
        if seqs is not None:
            cluster_df[cluster] = str(seqs[met_ids])
        else:
            cluster_df[cluster] = str(met_ids)
    pd.Series(cluster_df).to_csv(path +  'seed' + str(seed) + 'microbe_cluster_ids.csv')

def plot_output(path, best_mod, out_vec, targets,
                param_dict, fold, meas_var = 0.1):
    """
    Function:
        Plots output metabolic predictions of first 10 subjects
        Writes files with # of active metabolite and microbial clusters
        Plots error per metabolite
        Writes file with normalized root-mean squared error for metabolite predictions

    """

    pred_clusters = out_vec[best_mod]
    pred_z = param_dict['z'][best_mod]
    preds = torch.matmul(pred_clusters + meas_var*torch.randn(pred_clusters.shape), torch.Tensor(pred_z).T)
    pred_w = param_dict['w'][best_mod]

    # err = np.sum((preds.detach().numpy() - targets)**2,0)/preds.shape[0]
    # std_err = np.std(np.sqrt((preds.detach().numpy() - targets)**2),0)
    # fig, ax = plt.subplots(figsize = (30,6))
    # start = 0
    # minor_ticks = []
    # minor_labels = []
    # major_ticks = []
    # major_labels = []
    # for cluster in range(pred_z.shape[1]):
    #     met_ids = np.where(pred_z[:, cluster]==1)[0]
    #     if len(met_ids) > 0:
    #         clust_err = err[met_ids]
    #         clust_std = std_err[met_ids]
    #         if metabs is not None:
    #             clust_mets = metabs[met_ids]
    #         sort_ix = np.argsort(clust_err)
    #         ax.scatter(np.arange(start, len(clust_err)+start), clust_err[sort_ix])
    #         ax.errorbar(np.arange(start, len(clust_err)+start), clust_err[sort_ix], yerr = clust_std[sort_ix], fmt = 'o')
    #         if metabs is not None:
    #             major_ticks.extend(np.arange(start,start + len(clust_err)))
    #             major_labels.extend(clust_mets[sort_ix])
    #         # print(len(major_ticks))
    #         md = start + len(clust_err)/2
    #         if int(md) == md:
    #             md = md + 0.5
    #         # print(md)
    #         minor_ticks.append(md)
    #         minor_labels.append('Cluster ' + str(cluster))
    #         start = start + len(clust_err) + 2
    #         # print(start)
    #
    # ax.set_xticks(major_ticks, minor = False)
    # ax.set_xticklabels(major_labels, rotation=45, ha='right', fontsize=6, minor = False)
    # ax.minorticks_on()
    # ax.set_xticks(minor_ticks, minor = True)
    # ax.set_xticklabels(minor_labels, minor = True, fontsize = 30)
    # ax.tick_params(axis='x', which='minor', pad=-30, direction='in')
    # ax.set_ylabel('Error')
    # fig.tight_layout()
    # fig.savefig(path + 'seed' + str(fold) + '-per_met_err.pdf')
    # plt.close(fig)

    num_mets = preds.shape[1]
    if np.int(num_mets/37) <= 1:
        val = 3
    else:
        val = np.int(num_mets/37) + 3
    fig, ax = plt.subplots(8,1,figsize = (val,8))
    for s in range(8):
        ax[s].bar(np.arange(preds.shape[1]), preds[s,:].flatten().detach().numpy(), width = 0.85, alpha= 0.5, label = 'Predicted')
        ax[s].bar(np.arange(targets.shape[1]), targets[s, :].flatten(), width=0.85, alpha = 0.5, label = 'True')
        ax[s].set_title('Subject ' + str(s))
        ax[s].legend(loc= 'upper right')
    fig.savefig(path + 'seed' + str(fold) + '-predictions.png')
    plt.close(fig)

    RMSE_est = np.sqrt(np.sum(((preds.detach().numpy() - targets)**2)) / len(preds.flatten()))
    N_RMSE_est = np.round(RMSE_est / st.iqr(targets.flatten()),3)


    # RMSE_avg = np.round(np.mean(RMSE),4)
    # RMSE_std = np.round(np.std(RMSE),4)
    if not os.path.isfile(path + 'NRMSE.txt'):
        with open(path + 'NRMSE.txt', 'w') as f:
            f.write('SEED ' + str(fold) + ' NRMSE: ' + str(N_RMSE_est) + '\n')
    else:
        with open(path + 'NRMSE.txt', 'a') as f:
            f.write('SEED ' + str(fold) + ' NRMSE: ' + str(N_RMSE_est) + '\n')

    # RMSE_df = pd.Series(RMSE, index = ['Metabolite ' + str(i) for i in range(len(RMSE))])
    # RMSE_df.to_csv(path + 'seed' + str(fold) + 'RMSE.csv')



def plot_loss(fold, loss_vec, test_loss=None, lowest_loss = None):
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    # Plot loss given loss vector
    ax3.set_title('Seed ' + str(fold))
    ax3.plot(np.arange(len(loss_vec)), loss_vec, label='training loss')
    if test_loss is not None:
        ax3.plot(np.arange(len(loss_vec)), test_loss, label='test loss')
        ax3.legend(loc = 'upper right')
    if lowest_loss is not None:
        ax3.plot(np.arange(len(loss_vec)), lowest_loss, label='lowest loss')
        ax3.legend(loc = 'upper right')
    ax3.set_yscale('log')
    return fig3, ax3

def plot_loss_dict(path, fold, loss_dict):
    # Plot loss per each learned parameter
    params = loss_dict.keys()
    fig, ax = plt.subplots(len(params),1, figsize = (8, 5*len(params)))
    for i,param in enumerate(params):
        loss = loss_dict[param]
        ax[i].plot(np.arange(len(loss)), loss)
        ax[i].set_xlabel('Iterations')
        ax[i].set_ylabel('Loss')
        ax[i].set_title(param)
        if param == 'y':
            ax[i].set_yscale('log')
    fig.savefig(path + 'seed' + str(fold) + '_loss_dict.png')


# def plot_cluster_outputs_vs_met_value(best_z, y, cluster_outputs, path, seed = 0):
#     cats = np.argmax(best_z, 1)
#     # ixs = [np.where(np.argmax(best_z, 1) == c)[0] for c in np.unique(cats)]
#     ixs = [np.where(np.argmax(best_z, 1) == c)[0][0] for c in np.unique(cats)]
#     unique_cats = np.unique(cats)
#     if best_z.shape[1] > 20:
#         n_clust = 20
#     else:
#         n_clust = best_z.shape[1]
#     clusters = np.append(unique_cats, np.arange(n_clust - len(unique_cats)))
#     for met_clust, jj in enumerate(ixs):
#         fig, ax = plt.subplots(1, n_clust, figsize=(8 * n_clust, 8))
#         dat = np.array(y)[:, jj]
#         for ct, ii in enumerate(clusters):
#             ax[ct].hist(dat, bins=20, label='True Met ' + str(jj));
#             if cats[ii] == ii:
#                 ax[ct].set_title('ACTIVE CLUSTER')
#             try:
#                 # ax[ct].hist(np.repeat(cluster_outputs.detach().numpy()[:, ii], len(ixs)), alpha=0.5,
#                 #             label='Cluster ' + str(ii));
#                 ax[ct].hist(cluster_outputs.detach().numpy()[:, ii], alpha=0.5,
#                              label='Cluster ' + str(ii));
#             except:
#                 ax[ct].hist(cluster_outputs[:, ii], alpha=0.5,
#                             label='Cluster ' + str(ii));
#             ax[ct].legend();
#         fig.tight_layout()
#         # fig.savefig(path + 'seed-' + str(seed) + '-met' + str(unique_cats[met_clust]) + '-vs-clusters.pdf')
#         fig.savefig(path + 'seed-' + str(seed) + '-met' + str(jj) + '-vs-clusters.pdf')
#         plt.close(fig)


def plot_classes(y_class, ylocs, path):
    # Plot 2D locations and class identity given classes (i.e. families) and embedded locations
    radii = []
    means = []
    locss = []
    gps = []
    for bc in np.unique(y_class.values):
        ixs = np.where(y_class.values == bc)[0]
        locs = ylocs[ixs, :]
        radii.append(np.sqrt(np.sum((np.max(locs, 0) - np.min(locs, 0)) ** 2)) / 2)
        means.append(np.mean(locs, 0))
        locss.append(locs)
        gps.append(bc)
    plot_locations(radii, means, locss, gps, name=path)

def plot_MDS(dat, dmax=30, seed=0, path = '/Users/jendawk/Dropbox (MIT)/M2M/figures/'):
    true_dist = squareform(dat)
    pvals = []
    locs = []
    times= []
    for d in np.arange(2,dmax):
        start = time.time()
        embedding = MDS(n_components=d, dissimilarity='precomputed', random_state=seed,metric = True)
        xlocs = embedding.fit_transform(dat)
        ed = time.time() - start
        times.append(ed)
        est_dist = pdist(xlocs)
        # est_dist = est_dist / np.max(est_dist)
        # true_dist = true_dist / np.max(true_dist)
        stat, pval = st.ks_2samp(est_dist, true_dist)
        pvals.append(pval)
        print(ed)
        locs.append(xlocs)

    best = np.argmax(pvals)
    plt.figure()
    plt.plot(np.arange(2, dmax), pvals)
    plt.yscale('log')
    plt.xlabel('dimensions')
    plt.ylabel('p-values')
    plt.title('Max p-value= ' + str(pvals[best]))
    plt.savefig(path + 'mds.pdf')
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[1].hist(pdist(locs[best]), bins = 20)
    ax[1].set_xlabel('Embedded distances')
    ax[1].set_title('Distribution of Embedded Distances, d = ' + str(best))
    ax[0].hist(true_dist,bins = 20)
    ax[0].set_xlabel('True distances')
    ax[0].set_title('Distribution of Taxonomic Distances')
    fig.savefig(path + 'hist.pdf')
    plt.close(fig)

    return locs[best]
    # plot_classes(y_class, locs[best][:, :2], path + 'mets_')


def plot_dist(dist, path = '/Users/jendawk/Dropbox (MIT)/M2M/figures/'):
    plt.figure()
    plt.hist(np.array(dist).flatten(), bins = 20)
    plt.xlabel('pairwise distances')
    plt.savefig(path + 'dist-hist.pdf')
    plt.close()



# if __name__ == "__main__":
#     dtype = 'clumps'
#     # args.case = args.case + '_100Bvar'
#     xfile = 'x.csv'
#     yfile = 'y.csv'
#     xdist_file = 'x_dist.csv'
#     ydist_file = 'y' + dtype + '_dist.csv'
#     met_newick_name = 'w1_newick_tree.nhx'
#
#     # set data_path to point to directory with data
#     base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
#     data_path = base_path + "/inputs"
#     # Option to change filtering criteria
#     x = pd.read_csv(data_path + '/' + xfile, index_col = [0])
#     y = pd.read_csv(data_path + '/' + yfile, index_col = [0])
#
#     xdist = pd.read_csv(base_path + '/inputs/' + xdist_file, header=0, index_col=0)
#     y = y.loc[x.index.values]
#     if ydist_file not in os.listdir(base_path + '/inputs/'):
#         make_dist_mat(y, ydist_file, base_path, newick_path = '/ete_tree/' + met_newick_name)
#     ydist = pd.read_csv(base_path + '/inputs/' + ydist_file, header = 0, index_col = 0)
#
#     x_fams = get_xtaxa(base_path + '/inputs/taxa_labels.csv', x)
#     y_class = get_ytaxa(base_path + '/inputs/metab_classes.csv', y, ydist, level='subclass')
#     # skbio_mds(ydist, y_class, path = base_path + '/figures/' + dtype + '-')
#     # skbio_mds(xdist, x_fams, path=base_path + '/figures/' + 'asvs-')
