import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
import ete3
from skbio import TreeNode
import scipy.stats as st
from io import StringIO
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import seaborn as sns
from matplotlib import cm
import itertools
# from M2M import helper
# from M2M.dataLoader import *
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
import json
from scipy.cluster.hierarchy import dendrogram

def edit_string(string):
    return string.replace('(', '_').replace(')', '_').replace(':',
                                                 '_').replace(','
                                                              , '_').replace('[', '_').replace(']', '_').replace(';',
                                                                                                                 '_')

base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
t = ete3.TreeNode(base_path + '/ete_tree/newick_tree_all_weeks.nhx')
# dmat = pd.read_csv(base_path + '/inputs/pubchem/dice-dist_sm.csv', index_col=0)
dmat = pd.read_csv(base_path + '/inputs/zeros_full.csv', index_col = 0).T
coef_mat = pd.read_csv(base_path + '/inputs/coef_of_var.csv', index_col = 0)
# dmat = dmat['Num Zeros']
coefs = np.std(dmat, 1) / np.abs(np.mean(dmat, 1))
metabs = list(map(edit_string, dmat.index.values))

mets_keep = list(map(edit_string, dmat.index.values))
tree_leaves = [n.name for n in t.traverse() if n.is_leaf()]
rm_list = list(set(tree_leaves) - set(mets_keep))

while len(rm_list)>0:
    node_name = rm_list.pop()
    n = t.search_nodes(name = node_name)[0]
    parent = n.up
    n.detach()
    if len(parent.children)==0:
        rm_list.append(parent.name)

for n in t.traverse():
    if not n.is_leaf():
        n.add_face(ete3.TextFace(n.name+ '   ', fgcolor = 'red'), column = 0)
t.render(base_path + '/figures/test_tree.pdf')
node_order = [n.name for n in t.iter_descendants("postorder") if n.is_leaf()]

dmat.index = list(map(edit_string, dmat.index.values))
coef_mat.index = list(map(edit_string, coef_mat.index.values))
coef_mat = coef_mat.loc[node_order]
# dmat.columns = list(map(edit_string, dmat.columns.values))
# dmat = dmat[node_order].loc[node_order]
# sns.heatmap()
# mets_keep = [n for n in dmat.index.values if n in node_order]
# dmat_keep = dmat.loc[mets_keep]
dmat = dmat.loc[node_order]
darr = np.expand_dims(np.array(dmat),1)

fig, ax = plt.subplots(figsize = (1,25))
# sns.heatmap(darr, cmap='Blues', yticklabels=dmat.index.values, ax = ax)
sns.heatmap(coef_mat, cmap='Blues', ax = ax)
fig.tight_layout()
fig.savefig(base_path + '/figures/cv.pdf')

def num_parents(node, count=0):
    if node.up == None:
        return count
    else:
        return num_parents(node.up, count + 1)

t_array = dmat.to_csv()
t_array = '#Names' + t_array
t_array = t_array.replace(',','\t ')

ct = ete3.ClusterTree(t.write())
ct.link_to_arraytable(t_array)
for n_orig, n in zip(list(t.traverse()), list(ct.traverse())):
    if not n.is_leaf():
        count = num_parents(n_orig, 0)
        if count <= 5:
            n.add_face(ete3.TextFace(n_orig.name+ '   ', fgcolor = 'red'), column = 0)
# for n in ct.traverse():
#     if n.is_leaf():
#         n.add_face(ete3.ProfileFace(style = "heatmap", max_v = 1, min_v = 0, center_v=0.5, colorscheme=0), column = 0)
ts = ete3.TreeStyle()
ts.show_leaf_name = True
# ct.show("heatmap", tree_style=ts)
ct.render(base_path + '/figures/test.pdf', layout = 'heatmap', tree_style = ts)

link_mat = pd.DataFrame(np.zeros((len(metabs), len(metabs))), index = metabs, columns=metabs)
for mets in itertools.combinations(metabs, 2):
    met1, met2 = mets
    node1 = t.search_nodes(name = met1)[0]
    node2 = t.search_nodes(name = met2)[0]
    if node1.up == node2.up:
        link_mat[met1][met2] = 1
        link_mat[met2][met1] = 1
    else:
        link_mat[met1][met2] = 0
        link_mat[met2][met1] = 0

cluster = AgglomerativeClustering(connectivity=np.array(pd.DataFrame(link_mat)), linkage='average', n_clusters=None, compute_full_tree=True, compute_distances=True, distance_threshold=0.65).fit(dmat)

met_classes = pd.read_csv(base_path + '/inputs/metab_classes.csv', index_col = 0, header = 0)
cluster_dict = {}
for clust in np.unique(cluster.labels_):
    mets_in_cluster = dmat.index.values[np.where(cluster.labels_==clust)[0]]
    for i in np.arange(len(mets_in_cluster)):
        cluster_dict[(clust, mets_in_cluster[i])] = met_classes.loc[mets_in_cluster[i]]

pd.DataFrame(cluster_dict).T.to_csv('cluster_w_connectivity.csv')
# plot_dendrogram(cluster, truncate_mode = "level", p=7)
# dat = pd.read_csv(base_path + '/inputs/y.csv', index_col = 0, header = 0)
# df_corr, pval = st.spearmanr(dat, axis = 0)
# df_corr = np.triu(df_corr) + np.triu(df_corr).T
# np.fill_diagonal(df_corr, 1)
# dmat = pd.DataFrame(df_corr, index = dat.columns.values, columns = dat.columns.values)

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def mylayout(node):
    # If node is a leaf
    if node.is_leaf():
        # And a line profile
        node.add_face(ete3.TextFace(n.name), column = 0, position = 'branch-right')

linkage = hierarchy.linkage(squareform(dmat))


# dist_mat = pd.read_csv(base_path + '/inputs/y_dist.csv', index_col = 0)
# linkage = hierarchy.linkage(squareform(dmat))

# dat.columns = [met_to_newick[d] for d in dat.columns]

dat = pd.read_csv(base_path + '/inputs/y.csv', index_col = 0, header = 0)
# dat = dl.week['metabs'][1]['x']
df_corr, pval = st.spearmanr(dat, axis = 0)
df_corr = np.triu(df_corr) + np.triu(df_corr).T
np.fill_diagonal(df_corr, 1)
df_corr = pd.DataFrame(df_corr, index = dat.columns.values, columns = dat.columns.values)

dist_mat = pd.read_csv(base_path + '/inputs/y_dist_50.csv', index_col = 0, header = 0)
ydist = 10 * (dist_mat / dist_mat.max().max())
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
ylocs = embedding.fit_transform(ydist)
ylocs = pd.DataFrame(ylocs, index = dist_mat.index.values)

with open(base_path + '/ete_tree/met_to_inchikey.pkl', 'rb') as f:
    met_to_inchikey = pkl.load(f)
inchikey_to_met = {met_to_inchikey[i]: i for i in met_to_inchikey.keys()}
df_dict = {}
it = 0
for file in os.listdir(base_path + '/ete_tree/classy_fire_results/'):
    classification = pd.read_csv(base_path + '/ete_tree/classy_fire_results/' + file, index_col=0)
    met = inchikey_to_met[file.split('.csv')[0].split('=')[1]]
    # if met not in ylocs.index.values:
    #     continue
    if met not in df_dict.keys():
        df_dict[met] = {}
    for cl in classification.index.values:
        df_dict[met][cl] = classification['Classification'][cl]

fig, ax = plt.subplots(3,1, figsize = (12, 5*3))
fig2, ax2 = plt.subplots(3,1, figsize = (12, 5*3))
class_labels = pd.DataFrame(df_dict).T
class_labels.to_csv(base_path + '/inputs/metab_classes.csv')
h = []
for ii, level in enumerate(class_labels.columns.values[1:4]):
    level_dat = class_labels[level]
    ax[ii].scatter(ylocs.iloc[:,0], ylocs.iloc[:,1], color = 'k')
    avg_radii = []
    med_radii = []
    for val in level_dat.unique():
        mets_in_gp = level_dat.index.values[level_dat==val]
        if len(mets_in_gp)< 2:
            continue
        gp_locs = ylocs.loc[mets_in_gp,:]
        mu = gp_locs.sum(0)/gp_locs.shape[0]
        r = np.max(
            [np.sqrt(np.sum((mu - gp_locs.loc[l]) ** 2)) for l in gp_locs.index.values])
        avg_radii.append(r)
        med_radii.append(r)
        p1 = ax[ii].scatter([mu[0]], [mu[1]], marker = '*')
        circle = plt.Circle((mu[0], mu[1]), r, alpha = 0.1,
                            label = val, color = p1.get_facecolor().squeeze())
        ax[ii].add_patch(circle)
        h.append(circle)
    ax2[ii].hist(avg_radii)
    ax2[ii].set_title(level)
    ax2[ii].set_xlabel('Categories')
    ax2[ii].set_ylabel('Levels')
    print(level)
    print('Avg radii: ' + str(np.mean(avg_radii)))
    print('Median radii: ' + str(np.median(med_radii)))
    print('')
    ax[ii].set_xlabel('dim 1')
    ax[ii].set_ylabel('dim 2')
    ax[ii].set_title(level)
    ax[ii].legend(h, level_dat.unique())

# fig.tight_layout()
fig2.savefig(base_path + '/figures/radii_lessfilt.pdf')
fig.savefig(base_path + '/figures/locs2d_lessfilt.pdf')


fig2, ax2 = plt.subplots(4,1, figsize = (5, 5*4))
fig, ax = plt.subplots(4,1, figsize = (12, 5*4))
class_labels = pd.DataFrame(df_dict).T
for ii, level in enumerate(class_labels.columns.values[1:5]):
    level_dat = class_labels[level]
    print(level)
    print(len(level_dat.dropna()))
    print('')
    corr = (1 + df_corr)/2
    means = []
    stds = []
    cats = []
    lens = []
    medians = []
    for val in level_dat.unique():
        mets_in_gp = level_dat.index.values[level_dat==val]
        if len(mets_in_gp)< 2:
            continue
        gp_corr = []
        for met1, met2 in itertools.combinations(mets_in_gp, 2):
            gp_corr.append(corr[met1][met2])
        medians.append(np.median(gp_corr))
        means.append(np.mean(gp_corr))
        stds.append(np.std(gp_corr))
        cats.append(val)
        lens.append(len(mets_in_gp))
    ax2[ii].scatter(lens, means, label = 'means')
    ax2[ii].scatter(lens, medians, marker = 's', facecolors = "None", label = 'medians')
    ax2[ii].set_xlabel('# of metabolites in group')
    ax2[ii].set_ylabel('Avg correlation')
    ax2[ii].set_title(level)
    ax2[ii].set_ylim([0,1])
    ax2[ii].legend()
    # fig, ax = plt.subplots(figsize = (10,5))
    ax[ii].scatter(np.arange(len(cats)), means)
    ax[ii].errorbar(np.arange(len(cats)), means, yerr = stds)
    ax[ii].set_xticks(np.arange(len(cats)) + 0.1, minor = True)
    ax[ii].set_xticklabels(lens, minor = True, fontsize = 8)
    ax[ii].tick_params(axis = 'x', which = 'minor', pad = -30, direction = 'in')
    ax[ii].set_xticks(np.arange(len(cats)))
    ax[ii].axhline(y=0.5, color = 'r', alpha = 0.5)
    ax[ii].set_xticklabels(cats, rotation = 45, ha = 'right', fontsize = 8)
    ax[ii].set_title(level)
    ax[ii].set_ylabel('Avg Correlation')
    ax[ii].set_ylim([0,1])

fig.tight_layout()
fig.savefig(base_path + '/figures/' + 'morefilt_all.pdf')

fig2.tight_layout()
fig2.savefig(base_path + '/figures/more_filt_num_vs_corr.pdf')
    #
    # n = len(level_dat.unique())
    # lut = dict(zip(level_dat.unique(), cm.rainbow(np.linspace(0,1,n))))
    # row_colors = level_dat.map(lut)
    # # fig, ax = plt.subplots(figsize=(20,20))
    # g = sns.clustermap((1 + df_corr)/2, col_linkage = linkage, row_linkage = linkage, row_colors=row_colors,
    #                    col_colors=row_colors)
    # g.ax_heatmap.set_xticks(np.arange(df_corr.shape[0]))
    # g.ax_heatmap.set_xticklabels(df_corr.index.values, rotation = 90, ha = 'right', fontsize = 3)
    # g.ax_heatmap.set_yticks(np.arange(df_corr.shape[0]))
    # g.ax_heatmap.set_yticklabels(df_corr.index.values, fontsize = 3)
    # markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in lut.values()]
    # plt.legend(markers, lut.keys(), numpoints=1, prop={'size': 6})
    # g.savefig(base_path + '/cluster_' + level + '.pdf')

df_list = df_corr.to_csv(header=1, index=True, sep='\t').split('\n')
df_list[0] = '#Names' + df_list[0]
t_clust = ete3.ClusterTree(newick = base_path +'/ete_tree/w1_newick_tree.nhx', text_array = '\n'.join(df_list[:-1]))
t_clust.show()
# for n in t_clust.traverse():
#     if n.is_leaf():
#         n.name = newick_to_met[n.name]

ts = ete3.TreeStyle()
ts.show_leaf_name = True
t_clust.render(base_path + 'cluster.pdf', tree_style = ts)