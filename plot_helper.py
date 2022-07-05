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

def plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs,
                  mu_bug, r_bug, mu_met, r_met, gen_u, gen_alpha, gen_beta):
    plt.figure();
    plt.hist(x.flatten(), bins=20);
    plt.title('Microbe data distribution')
    plt.savefig(path + 'bug_hist.png')
    plt.close()

    plt.figure();
    plt.hist(g.flatten(), bins=20);
    plt.title('Microbe cluster data distribution')
    plt.savefig(path + 'bug_hist.png')
    plt.close()

    plt.figure();
    plt.hist(y.flatten(), bins=20);
    plt.title('Metabolite data distribution')
    plt.savefig(path + 'met_hist.png')
    plt.close()

    bug_clusters = [np.where(gen_u[:,i])[0] for i in np.arange(gen_u.shape[1])]
    for ii,clust in enumerate(bug_clusters):
        if not os.path.isfile(path + 'microbe_clusters.txt'):
            with open(path + 'microbe_clusters.txt', 'w') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')
        else:
            with open(path + 'microbe_clusters.txt', 'a') as f:
                f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')

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
        # dixs = np.where(gen_w[i, :] == 1)[0]
        # for dix in dixs:
        ix = np.where(gen_u[:, i]==1)[0]
        ax2[0].hist(x[:, ix].flatten(), range=(x.min(), x.max()), label='Cluster ' + str(i), alpha=0.5, bins = bins)
    ax2[0].set_xlabel('Microbial relative abundances')
    # ax2[0].set_ylabel('# Microbes in Cluster x\n# Samples Per Microbe', fontsize = 10)
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
    # ax3[0].set_aspect('equal')

    ax2[0].legend(loc = 'upper right')
    ax2[1].legend(loc = 'upper right')
    for i in range(gen_z.shape[1]):
        ix = np.where(gen_z[:, i] == 1)[0]
        if mu_bug.shape[1] == 2:
            p2 = ax[1].scatter(gen_met_locs[ix, 0], gen_met_locs[ix, 1])
            ax[1].scatter(mu_met[i, 0], mu_met[i, 1], marker='*', color=p2.get_facecolor().squeeze())
            ax[1].set_title('Metabolites')
            ax[1].text(mu_met[i, 0], mu_met[i, 1], 'Cluster ' + str(i))
            # p2 = None
            # for ii in ix:
                # p2 = ax[1].scatter(gen_met_locs[ii, 0], gen_met_locs[ii, 1], alpha = )
                # ax[1].text(gen_met_locs[ii,0], gen_met_locs[ii,1], 'Metabolite ' + str(ii))
            circle2 = plt.Circle((mu_met[i,0], mu_met[i,1]), r_met[i],
                                 alpha=0.2, color=p2.get_facecolor().squeeze(), label = 'Cluster ' + str(i))
            ax[1].add_patch(circle2)
        bins = int((y.max() - y.min())/5)
        if bins<=10:
            bins = 10
        ax2[2].hist(y[:, ix].flatten(), range=(y.min(), y.max()),
                    label='Cluster ' + str(i), alpha=0.5, bins = bins)
    ax2[2].set_xlabel('Standardized metabolite levels')
    # ax2[2].set_ylabel('# Metabolites in Cluster x\n# Samples Per Metabolite', fontsize = 10)
    ax2[2].set_title('Metabolites')

    if mu_bug.shape[1] == 2:
        ax[1].set_aspect('equal')
        fig.tight_layout()
        fig.savefig(path + 'embedded_locations.png')
        plt.close(fig)
    # fig2.tight_layout()
    fig2.savefig(path + 'cluster_histogram.png')
    plt.close(fig2)

    bug_active,met_active = np.where((gen_alpha * gen_beta[1:,:])>= 1e-6)

    K = gen_z.shape[1]
    L = gen_u.shape[1]
    fig, ax = plt.subplots(len(bug_active), 1, figsize=(8, 8 * len(bug_active)))
    ii = 0
    # g = x @ gen_u
    ax_ylim = (np.min(y.flatten()) - 0.01 * np.max(y.flatten()), np.max(y.flatten()) + 0.01 * np.max(y.flatten()))
    ax_xlim = (np.min(g.flatten()) - 0.01 * np.max(g.flatten()), np.max(g.flatten()) + 0.01 * np.max(g.flatten()))
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
        ax[ii].text(0.6, 0.6, 'beta = ' + str(gen_beta[bug_clust + 1, met_clust]), horizontalalignment='center',
                      verticalalignment='center', transform=ax[ii].transAxes)
        ax[ii].set_xlim(ax_xlim)
        # ax[ii].set_ylim(ax_ylim)
        ii += 1
    fig.tight_layout()
    fig.savefig(path + '-sum_x_v_y.png')
    plt.close(fig)

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
    if true_val is not None:
        true_val = true_val[param]
    if ptype == 'init':
        label = 'Initialized values'
    elif ptype == 'prior':
        label = 'True values'
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
                ax.axvline(tv[k], c = 'r', label = label)
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
    for key in param_dict[seed].keys():
        print(key)
        start = len(param_dict[seed][key])
        fig, ax = plt.subplots()
        all_dat = param_dict[seed][key]
        try:
            all_dat1 = np.concatenate(all_dat)
            all_dat1 = all_dat1
        except:
            all_dat1 = all_dat

        range = (np.array(all_dat1).flatten().min(), np.array(all_dat1).flatten().max())

        iterset = np.int(len(param_dict[seed][key])/4)
        posterior_list = param_dict[seed][key][start - iterset: start]

        try:
            dat = np.concatenate(posterior_list)
            dat = dat.flatten()
        except:
            dat = posterior_list
        # dat = dat.detach().numpy().flatten()

        ax.hist(dat, range = range, bins=20, label='Iterations ' + str(start - iterset) + ' to ' + str(start), alpha=0.5)
        ax.legend()
        ax.set_title(key + ', ' + str(len(param_dict[seed][key])) + ' iterations')
        if not os.path.isdir(out_path + '/posteriors'):
            os.mkdir(out_path + '/posteriors')
        fig.savefig(out_path + '/posteriors/' + str(seed) + '-' + key + '-posterior_dist.pdf')

def plot_param_traces(path, param_dict, params2learn, true_vals, net, fold):
    fig_dict, ax_dict = {},{}
    for name, plist in param_dict.items():
        if name in params2learn or 'all' in params2learn or name == 'z' or 'w_' in name:
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
            mindat = np.min(dat)
            maxdat = np.max(dat)

            for k in range(n):
                if len(plist[0].squeeze().shape) <= 1:
                    if name == 'sigma' or name == 'p':
                        trace = plist
                        ax_dict[name].plot(trace, label='Trace')
                        if true_vals is not None:
                            if name in true_vals.keys():
                                if not hasattr(true_vals[name], '__len__'):
                                    tv= [true_vals[name]]
                                else:
                                    tv = true_vals[name]
                                ax_dict[name].plot([tv[k]] * len(trace), c='r', label='True')
                        ax_dict[name].set_title(name + ', ' + str(k))
                        ax_dict[name].set_ylim([0, maxdat])
                        ax_dict[name].legend(loc='upper right')
                        ax_dict[name].set_xlabel('Iterations')
                        ax_dict[name].set_ylabel('Parameter Values')
                        ax_dict[name].set_yscale('log')
                    else:
                        trace = [p.squeeze()[k] for p in plist]
                        ax_dict[name][k].plot(trace, label='Trace')
                        if true_vals is not None:
                            if name in true_vals.keys():
                                if not isinstance(true_vals[name], list):
                                    true_vals[name] = true_vals[name].squeeze()
                                if np.array(true_vals[name]).shape[0] > k:
                                    ax_dict[name][k].plot([true_vals[name][k]] * len(trace), c='r', label='True')
                            ax_dict[name][k].set_title(name + ', ' + str(k))
                        if name in net.range_dict.keys():
                            ax_dict[name][k].set_ylim([mindat, maxdat])
                        ax_dict[name][k].legend(loc = 'upper right')
                        ax_dict[name][k].set_xlabel('Iterations')
                        ax_dict[name][k].set_ylabel('Parameter Values')
                else:
                    for j in range(nn):
                        new_k, new_j = k, j
                        trace = [p.squeeze()[new_k, new_j] for p in plist]
                        if name == 'z' or name == 'w' or name == 'alpha':
                            trace_ma = [np.sum(trace[i:i + 5:]) / 5 for i in np.arange(len(trace) - 5)]
                            trace_ma[:0] = [trace_ma[0]]*(len(trace)-len(trace_ma))
                            trace = trace_ma
                        ax_dict[name][k, j].plot(trace, label='Trace')
                        if true_vals is not None:
                            if name in true_vals.keys():
                                if np.array(true_vals[name]).shape[1]>j and np.array(true_vals[name]).shape[0] > k:
                                    ax_dict[name][k, j].plot([true_vals[name][k, j]] * len(trace), c='r', label='True')
                        ax_dict[name][k, j].set_title(name + ', ' + str(k) + ', ' + str(j))
                        if name in net.range_dict.keys():
                            ax_dict[name][k, j].set_ylim([mindat, maxdat])
                        ax_dict[name][k, j].legend(loc = 'upper right')
                        ax_dict[name][k, j].set_xlabel('Iterations')
                        ax_dict[name][k, j].set_ylabel('Parameter Values')
            fig_dict[name].tight_layout()
            fig_dict[name].savefig(path + 'seed' + str(fold) + '_' + name + '_parameter_trace.png')
            plt.close(fig_dict[name])

def plot_locations(radii, means, locs, gps, name = 'bug'):
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
    fig.savefig(name + '-init-locs.pdf')
    plt.close(fig)


def plot_output_locations(path, net, best_mod, param_dict, fold, type = 'best', plot_zeros = False):
    best_w = param_dict['w'][best_mod]
    # best_w = best_w[:, mapping['bug']]
    best_mu = param_dict['mu_bug'][best_mod]
    best_r = param_dict['r_bug'][best_mod]
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
    # [:,mapping['met']]
    best_mu = param_dict['mu_met'][best_mod]
    best_r = param_dict['r_met'][best_mod]
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
    best_w = param_dict[seed]['w'][best_mod]
    best_z = param_dict[seed]['z'][best_mod]
    best_alpha = param_dict[seed]['alpha'][best_mod]
    best_alpha_beta = param_dict[seed]['beta[1:,:]*alpha'][best_mod]
    active_microbes = list(set(np.where(np.sum(best_w,0)>1)[0]).intersection(set(np.where(best_alpha.sum(0)!= 0)[0])))
    active_mets = np.where(np.sum(best_z, 0) > 1)[0]
    df = {}
    for microbe, met in itertools.product(active_microbes, active_mets):
        if (microbe, met) not in df.keys() and (met, microbe) not in df.keys():
            df[(microbe, met)] = best_alpha_beta[microbe, met]

    pd.Series(df).to_csv(path + '/' + str(seed) + '-interactions.csv')


def plot_xvy(path, x, out_vec, best_mod, param_dict, seed):
    out = out_vec[best_mod].detach().numpy()
    best_w = param_dict[seed]['w'][best_mod]
    best_z = param_dict[seed]['z'][best_mod]
    best_alpha = param_dict[seed]['alpha'][best_mod]

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
    # fig, ax = plt.subplots(num_active_microbe, num_active_met, figsize = (8*num_active_met,8*num_active_microbe))
    # if num_active_microbe==1:
    #     ax = np.expand_dims(ax,0)
    # if num_active_met==1:
    #     ax = np.expand_dims(ax,1)
    # ax_xlim = (np.min(true_sum.flatten())-10, np.max(true_sum.flatten())+10)
    # ax_ylim = (np.min(targets.flatten()), np.max(targets.flatten()))

    # if np.min(microbe_sum.flatten())>ax_xlim[1] or np.max(microbe_sum.flatten()) < ax_xlim[0]:
    #     ax_xlim = (np.min(microbe_sum.flatten()), np.max(microbe_sum.flatten()))
    # if np.min(out.flatten())>ax_ylim[1] or np.max(out.flatten())<ax_ylim[0]:
    #     ax_ylim = (np.min(out.flatten()), np.max(out.flatten()))
    # if np.max(out.flatten())>ax_ylim[1]:
    #     ax_ylim = (ax_ylim[0], np.max(out.flatten()))
    # if np.min(out.flatten())< ax_ylim[0]:
    #     ax_ylim = (np.min(out.flatten()), ax_ylim[1])
    # if np.max(microbe_sum.flatten())>ax_xlim[1]:
    #     ax_xlim = (ax_xlim[0], np.max(microbe_sum.flatten()))
    # if np.min(microbe_sum.flatten())< ax_xlim[0]:
    #     ax_xlim = (np.min(microbe_sum.flatten()), ax_xlim[1])
    # ranges = [[np.max(microbe_sum[:,i]/out[:,j]) - np.min(microbe_sum[:,i]/out[:,j]) for i in range(out.shape[1])] for j in range(out.shape[1])]
    # ixs = [np.argmin(r) for r in ranges]
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
                         '\n r2= ' + str(np.round(lreg.rvalue,3)))
            fit_df[(j,i)]['rvalue'] = lreg.rvalue
            fit_df[(j,i)]['pvalue'] = lreg.pvalue
            fit_df[(j,i)]['slope'] = lreg.slope
            fit_df[(j,i)]['intercept'] = lreg.intercept
            if not os.path.isdir(path + '/seed' + str(seed)):
                os.mkdir(path + '/seed' + str(seed))
            # if not os.path.isdir(path + '/seed' + str(seed) + '/' + 'metclust' + str(i) + '_vs_' + 'microbeclust' + str(j)):
            #     os.mkdir(path + '/seed' + str(seed) + '/' + 'metclust' + str(i) + '_vs_' + 'microbeclust' + str(j))
            fig.savefig(path + '/seed' + str(seed)+ '/' + 'metclust' + str(i) + '_vs_' + 'microbeclust' + str(j) + '-sum_x_v_y.png')
            plt.close(fig)
            # slope = np.round((np.max(out[:,i]) - np.min(out[:, i]))/((np.max(microbe_sum[:,j]) - np.min(microbe_sum[:,j]))),3)
            # try:
            #     ax[jj,ii].text(0.6, 0.8,'slope = ' + str(slope), horizontalalignment='center',
            #          verticalalignment='center', transform=ax[ii,jj].transAxes)
            # except:
            #     return
            # ax[jj,ii].set_xlim(ax_xlim[0], ax_xlim[1])
            # ax[jj,ii].set_ylim(ax_ylim[0], ax_ylim[1])
            jj+=1
        ii+=1

    pd.DataFrame(fit_df).T.to_csv(path + 'seed' + str(seed) + '-fit.csv')
    df = pd.DataFrame(fit_df).T
    if not os.path.isfile(path + 'rfit.txt'):
        with open(path + 'perc-corr.txt', 'w') as f:
            f.write('Seed ' + str(seed) + ': ' + str(np.round(np.mean(df['rvalue']), 3)) +  ' +- ' +
                    str(np.round(np.std(df['rvalue']), 3)) + '\n')
    else:
        with open(path + 'perc-corr.txt', 'a') as f:
            f.write('Seed ' + str(seed) + ': ' + str(np.round(np.mean(df['rvalue']), 3)) +  ' +- ' +
                    str(np.round(np.std(df['rvalue']), 3)) + '\n')

    fig.savefig(path + 'seed' + str(seed) + '-sum_x_v_y.png')
    plt.close(fig)
    return x_dict, y_dict

def plot_output(path, best_mod, out_vec, targets, true_vals,
                param_dict, fold, type = 'unknown', meas_var = 0.1, metabs = None):
    if true_vals is not None:
        gen_z = true_vals['z']
        gen_w = true_vals['w']
        best_w = param_dict['w'][best_mod]
        bug_clusters = [np.where(best_w[:,i]>0.5)[0] for i in np.arange(best_w.shape[1])]
        for ii,clust in enumerate(bug_clusters):
            if not os.path.isfile(path + 'seed' + str(fold) + 'microbe_clusters.txt'):
                with open(path + 'seed' + str(fold) + 'microbe_clusters.txt', 'w') as f:
                    f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')
            else:
                with open(path + 'seed' + str(fold) + 'microbe_clusters.txt', 'a') as f:
                    f.writelines('Cluster ' + str(ii) + ': ' + str(clust) + '\n')

        tp, fp, tn, fn, ri = pairwise_eval(np.round(best_w), gen_w)
        pairwise_cf = {'Same cluster': {'Predicted Same cluster': tp, 'Predicted Different cluster': fn},
                       'Different cluster': {'Predicted Same cluster': fp, 'Predicted Different cluster': tn}}
        pd.DataFrame(pairwise_cf).T.to_csv(
            path + 'seed' + str(fold) + '_PairwiseConfusionBugs_' + type + '_' + str(np.round(ri, 3)).replace('.',
                                                                                                                'd') + '.csv')
        ri = str(np.round(ri, 3))
        if not os.path.isfile(path + type +'-ri_bug.txt'):
            with open(path + type +'-ri_bug.txt', 'w') as f:
                f.write('Seed ' + str(fold) + ':RI ' + ri + '\n')
        else:
            with open(path + type +'-ri_bug.txt', 'a') as f:
                f.write('Seed ' + str(fold) + ':RI ' + ri + '\n')

    # fig_dict2, ax_dict2 = plt.subplots(targets.shape[1], 1,
    #                                                figsize=(8, 4 * targets.shape[1]))
    # fig_dict3, ax_dict3 = plt.subplots(gen_z.shape[1], 1, figsize=(8, 4 * gen_z.shape[1]))
    pred_clusters = out_vec[best_mod]
    pred_z = param_dict['z'][best_mod]
    preds = torch.matmul(pred_clusters + meas_var*torch.randn(pred_clusters.shape), torch.Tensor(pred_z).T)
    pred_w = param_dict['w'][best_mod]
    # pred_z = pred_z[:, mapping['met']]
    if true_vals is not None:
        true = np.argmax(gen_z,1)
        z_guess = np.argmax(pred_z, 1)
        nmi = np.round(normalized_mutual_info_score(true, z_guess), 3)
        try:
            tp, fp, tn, fn, ri = pairwise_eval(z_guess, true)
            pairwise_cf = {'Same cluster': {'Predicted Same cluster': tp, 'Predicted Different cluster': fn},
                           'Different cluster':{'Predicted Same cluster': fp, 'Predicted Different cluster': tn}}
            pd.DataFrame(pairwise_cf).T.to_csv(
                path + 'seed' + str(fold) + '_PairwiseConfusionMetabs_' + type + '_' + str(np.round(ri,3)).replace('.', 'd') + '.csv')
            ri = str(np.round(ri, 3))
        except:
            ri = 'NA'
        if not os.path.isfile(path + type +'-nmi_ri.txt'):
            with open(path + type +'-nmi_ri.txt', 'w') as f:
                f.write('Seed ' + str(fold) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')
        else:
            with open(path + type +'-nmi_ri.txt', 'a') as f:
                f.write('Seed ' + str(fold) + ': NMI ' + str(nmi) + ', RI ' + ri + '\n')

    else:
        with open(path + type + '-metab_clust.txt', 'a') as f:
            f.write('Seed ' + str(fold) + ': ' + str(np.sum(pred_z, 0)) + '\n')
        with open(path + type + '-asv_clust.txt', 'a') as f:
            f.write('Seed ' + str(fold) + ': ' + str(np.sum(pred_w, 0)) + '\n')

    err = np.sum((preds.detach().numpy() - targets)**2,0)/preds.shape[0]
    std_err = np.std(np.sqrt((preds.detach().numpy() - targets)**2),0)
    fig, ax = plt.subplots(figsize = (30,6))
    start = 0
    minor_ticks = []
    minor_labels = []
    major_ticks = []
    major_labels = []
    for cluster in range(pred_z.shape[1]):
        met_ids = np.where(pred_z[:, cluster]==1)[0]
        if len(met_ids) > 0:
            clust_err = err[met_ids]
            clust_std = std_err[met_ids]
            if metabs is not None:
                clust_mets = metabs[met_ids]
                # print(clust_mets)
            sort_ix = np.argsort(clust_err)
            ax.scatter(np.arange(start, len(clust_err)+start), clust_err[sort_ix])
            ax.errorbar(np.arange(start, len(clust_err)+start), clust_err[sort_ix], yerr = clust_std[sort_ix], fmt = 'o')
            if metabs is not None:
                major_ticks.extend(np.arange(start,start + len(clust_err)))
                major_labels.extend(clust_mets[sort_ix])
            # print(len(major_ticks))
            md = start + len(clust_err)/2
            if int(md) == md:
                md = md + 0.5
            # print(md)
            minor_ticks.append(md)
            minor_labels.append('Cluster ' + str(cluster))
            start = start + len(clust_err) + 2
            # print(start)

    ax.set_xticks(major_ticks, minor = False)
    ax.set_xticklabels(major_labels, rotation=45, ha='right', fontsize=6, minor = False)
    ax.minorticks_on()
    ax.set_xticks(minor_ticks, minor = True)
    ax.set_xticklabels(minor_labels, minor = True, fontsize = 30)
    ax.tick_params(axis='x', which='minor', pad=-30, direction='in')
    ax.set_ylabel('Error')
    fig.tight_layout()
    fig.savefig(path + 'seed' + str(fold) + '-per_met_err.pdf')
    plt.close(fig)

    abs_err = np.abs(preds.detach().numpy() - targets)
    datstd = np.std(np.array(targets).flatten())
    perc_corr = len(np.where(abs_err.flatten() < (0.25*datstd))[0])/len(abs_err.flatten())
    if not os.path.isfile(path + 'perc-corr.txt'):
        with open(path + 'perc-corr.txt', 'w') as f:
            f.write('Seed ' + str(fold) + ': ' + str(np.round(perc_corr*100, 3)) + '% \n')
    else:
        with open(path + 'perc-corr.txt', 'a') as f:
            f.write('Seed ' + str(fold) + ': ' + str(np.round(perc_corr*100, 3)) + '% \n')

    num_mets = preds.shape[1]
    fig, ax = plt.subplots(8,1,figsize = (np.int(num_mets/37)*2,8))
    for s in range(8):
        ax[s].bar(np.arange(preds.shape[1]), preds[s,:].flatten().detach().numpy(), width = 0.85, alpha= 0.5, label = 'Predicted')
        ax[s].bar(np.arange(targets.shape[1]), targets[s, :].flatten(), width=0.85, alpha = 0.5, label = 'True')
        ax[s].set_title('Subject ' + str(s))
        ax[s].legend(loc= 'upper right')
    fig.savefig(path + 'seed' + str(fold) + '-predictions.png')
    plt.close(fig)

    RMSE_est = np.sqrt(np.sum(((preds.detach().numpy() - targets)**2)) / len(preds.flatten()))
    N_RMSE_est = np.round(RMSE_est / st.iqr(targets.flatten()),3)

    figx, axx = plt.subplots(figsize=(8, 8))
    color = cm.rainbow(np.linspace(0, 1, pred_z.shape[1]))
    RMSE = np.zeros(pred_z.shape[0])
    residuals = np.zeros(targets.shape)
    preds_t = np.zeros(targets.shape)
    for cluster in range(pred_z.shape[1]):
        met_ids = np.where(pred_z[:, cluster]==1)[0]
        if len(met_ids)==0:
            continue
    #     bins = int((total.max() - total.min()) / 5)
        residuals[:, met_ids] = targets[:, met_ids] - preds[:, met_ids].detach().numpy()
        preds_t[:, met_ids] = preds[:, met_ids].detach().numpy()
        for i in range(len(met_ids)):
            temp = np.sqrt(np.sum(((preds[:, met_ids[i]].detach().numpy() - targets[:, met_ids[i]])**2))/preds.shape[0])
            iqr = st.iqr(targets[:, met_ids[i]])
            RMSE[met_ids[i]] = temp / iqr

        axx.scatter(preds_t[:, met_ids], residuals[:, met_ids], c=[color[cluster]],
                    label='Cluster ' + str(cluster))

    RMSE_avg = np.round(np.mean(RMSE),4)
    RMSE_std = np.round(np.std(RMSE),4)
    if not os.path.isfile(path + 'NRMSE.txt'):
        with open(path + 'NRMSE.txt', 'w') as f:
            f.write('SEED ' + str(fold) + ' NRMSE: ' + str(N_RMSE_est) + '\n')
    else:
        with open(path + 'NRMSE.txt', 'a') as f:
            f.write('SEED ' + str(fold) + ' NRMSE: ' + str(N_RMSE_est) + '\n')

    RMSE_df = pd.Series(RMSE, index = ['Metabolite ' + str(i) for i in range(len(RMSE))])
    RMSE_df.to_csv(path + 'seed' + str(fold) + 'RMSE.csv')

    axx.set_title('Residuals plot for metabolite level predictions')
    axx.set_xlabel('Predicted Levels')
    axx.set_ylabel('Residuals')
    axx.legend()
    figx.savefig(path + 'seed' + str(fold) + '_residuals_' + type + '.png')
    plt.close(figx)


def plot_loss(fig3, ax3, fold, iterations, loss_vec, test_loss=None, lowest_loss = None):
    ax3.set_title('Seed ' + str(fold))
    ax3.plot(iterations, loss_vec, label='training loss')
    if test_loss is not None:
        ax3.plot(iterations, test_loss, label='test loss')
        ax3.legend(loc = 'upper right')
    if lowest_loss is not None:
        ax3.plot(iterations, lowest_loss, label='lowest loss')
        ax3.legend(loc = 'upper right')
    ax3.set_yscale('log')
    return fig3, ax3

def plot_loss_dict(path, fold, loss_dict):
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


def plot_cluster_outputs_vs_met_value(best_z, y, cluster_outputs, path, seed = 0):
    cats = np.argmax(best_z, 1)
    # ixs = [np.where(np.argmax(best_z, 1) == c)[0] for c in np.unique(cats)]
    ixs = [np.where(np.argmax(best_z, 1) == c)[0][0] for c in np.unique(cats)]
    unique_cats = np.unique(cats)
    if best_z.shape[1] > 20:
        n_clust = 20
    else:
        n_clust = best_z.shape[1]
    clusters = np.append(unique_cats, np.arange(n_clust - len(unique_cats)))
    for met_clust, jj in enumerate(ixs):
        fig, ax = plt.subplots(1, n_clust, figsize=(8 * n_clust, 8))
        dat = np.array(y)[:, jj]
        for ct, ii in enumerate(clusters):
            ax[ct].hist(dat, bins=20, label='True Met ' + str(jj));
            if cats[ii] == ii:
                ax[ct].set_title('ACTIVE CLUSTER')
            try:
                # ax[ct].hist(np.repeat(cluster_outputs.detach().numpy()[:, ii], len(ixs)), alpha=0.5,
                #             label='Cluster ' + str(ii));
                ax[ct].hist(cluster_outputs.detach().numpy()[:, ii], alpha=0.5,
                             label='Cluster ' + str(ii));
            except:
                ax[ct].hist(cluster_outputs[:, ii], alpha=0.5,
                            label='Cluster ' + str(ii));
            ax[ct].legend();
        fig.tight_layout()
        # fig.savefig(path + 'seed-' + str(seed) + '-met' + str(unique_cats[met_clust]) + '-vs-clusters.pdf')
        fig.savefig(path + 'seed-' + str(seed) + '-met' + str(jj) + '-vs-clusters.pdf')
        plt.close(fig)


def plot_classes(y_class, ylocs, path):
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


# def skbio_mds(dat, dmax=30, path='/Users/jendawk/Dropbox (MIT)/M2M/figures/'):
#     distmat = DistanceMatrix(dat)
#     true_dist = squareform(dat)
#     pvals = []
#     locs_ls = []
#     embeds = []
#     for d in np.arange(2, dmax):
#         embedding = pcoa(distmat, number_of_dimensions=d)
#         locs = embedding.samples
#         est_dist = pdist(locs)
#         # est_dist = est_dist / np.max(est_dist)
#         # true_dist = true_dist / np.max(true_dist)
#         stat, pval = st.ks_2samp(est_dist, true_dist)
#         pvals.append(pval)
#         locs_ls.append(locs)
#         embeds.append(embedding)
#     best = np.argmax(pvals)
#     plt.figure()
#     plt.plot(np.arange(2, dmax), pvals)
#     plt.yscale('log')
#     plt.xlabel('dimensions')
#     plt.ylabel('p-vals')
#     plt.savefig(path + 'skbio-pcoa.pdf')
#     plt.close()
#
#     fig, ax = plt.subplots(1, 2, figsize=(8, 5))
#     ax[1].hist(pdist(locs_ls[best]), bins = 20)
#     ax[1].set_xlabel('Embedded distances')
#     ax[1].set_title('Distribution of Embedded Distances, d = ' + str(best))
#     ax[0].hist(true_dist, bins = 20)
#     ax[0].set_xlabel('True distances')
#     ax[0].set_title('Distribution of Taxonomic Distances')
#     fig.savefig(path + 'hist-skbio.pdf')
#
#     # plot_classes(classes, np.array(locs_ls[best]), path + '-skbio-')
#
#     embeds[best].proportion_explained.to_csv(path + 'prop-exp.csv')
#     return locs_ls[best]


if __name__ == "__main__":
    dtype = 'clumps'
    # args.case = args.case + '_100Bvar'
    xfile = 'x.csv'
    yfile = 'y.csv'
    xdist_file = 'x_dist.csv'
    ydist_file = 'y' + dtype + '_dist.csv'
    met_newick_name = 'w1_newick_tree.nhx'

    # set data_path to point to directory with data
    base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
    data_path = base_path + "/inputs"
    # Option to change filtering criteria
    x = pd.read_csv(data_path + '/' + xfile, index_col = [0])
    y = pd.read_csv(data_path + '/' + yfile, index_col = [0])

    xdist = pd.read_csv(base_path + '/inputs/' + xdist_file, header=0, index_col=0)
    y = y.loc[x.index.values]
    if ydist_file not in os.listdir(base_path + '/inputs/'):
        make_dist_mat(y, ydist_file, base_path, newick_path = '/ete_tree/' + met_newick_name)
    ydist = pd.read_csv(base_path + '/inputs/' + ydist_file, header = 0, index_col = 0)

    x_fams = get_xtaxa(base_path + '/inputs/taxa_labels.csv', x)
    y_class = get_ytaxa(base_path + '/inputs/metab_classes.csv', y, ydist, level='subclass')
    # skbio_mds(ydist, y_class, path = base_path + '/figures/' + dtype + '-')
    # skbio_mds(xdist, x_fams, path=base_path + '/figures/' + 'asvs-')
