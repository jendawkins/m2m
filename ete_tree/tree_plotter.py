#!/Users/jendawk/miniconda3/envs/ete_env/bin python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle as pkl
import sys
import ete3
import subprocess
import argparse
base_path = '/Users/jendawk/Dropbox (MIT)/M2M'

# This script plots phylogenetic trees for microbes and classy-fire classifcation trees for metabolites, saves
# the newick trees, and calculates distance matrices from trees
# NOTE: this script needs ete_env, not M2M_CodeBase env like the other scripts

def get_dist(seqs, newick_path=base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
             out_path = base_path + '/inputs/', name = 'x_dist.csv'):
    # This function saves a distance matrix to out_path based on an input phylogenetic/classification tree file
    # Inputs:
    #  - seqs: the sequences or metabolites to get a distance matrix for
    # - newick_path: the path of the newick tree (seqs needs to match the leaves on the newick tree)
    # - out_path: the path to save the distance matrix to
    out_path = out_path + '/classy-fire/'
    t = ete3.TreeNode(newick_path)
    nodes = [n.name for n in t.traverse() if n.is_leaf()]
    dist = {}
    for seq1 in seqs:
        if seq1.replace('(', '_').replace(')', '_').replace(':',
                                                            '_').replace(','
            , '_').replace('[', '_').replace(']','_').replace(';', '_') not in nodes:
            continue
        if seq1 not in dist.keys():
            dist[seq1] = {}
        else:
            if seq2 in dist[seq1].keys():
                continue
        for seq2 in seqs:
            if seq2.replace('(', '_').replace(')', '_').replace(':',
                                                                '_').replace(','
                , '_').replace('[', '_').replace(']', '_').replace(';', '_') not in nodes:
                continue
            if seq2 not in dist.keys():
                dist[seq2] = {}
            else:
                if seq1 in dist[seq2].keys():
                    continue
            if seq1 != seq2:
                try:
                    d = t.get_distance(seq1, seq2)
                except:
                    d = t.get_distance(seq1.replace('(', '_').replace(')', '_').replace(':','_').replace(',','_').replace('[','_').replace(']','_').replace(';','_'),
                                       seq2.replace('(', '_').replace(')', '_').replace(':','_').replace(',','_').replace('[','_').replace(']','_').replace(';','_'))
                dist[seq1][seq2] = d
                dist[seq2][seq1] = d
            else:
                dist[seq1][seq2] = 0
    pd.DataFrame(dist).to_csv(out_path + name)


def plot_asv_tree(newick_path=base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
                  out_path= base_path + '/outputs',
                  data_path = base_path + '/inputs', taxa_keep = None, name = None):
    # Plots the asv tree with taxonomic information given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - data_path: path of the taxonomic labels
    # - taxa_keep: which taxa to plot labels of on the tree
    # - name: name of the tree plot file
    t = ete3.TreeNode(newick_path)
    taxa_labels = pd.read_csv(data_path + '/taxa_labels.csv', index_col=[0])

    for n in t.traverse():
        if taxa_keep is not None:
            if n.is_leaf():
                if n.name not in taxa_keep:
                    n.detach()
                    continue
        if n.is_leaf():
            taxa = taxa_labels.loc[taxa_labels['labels'] == n.name]
            tax_lst = taxa['taxa_rdp'].values[0].split('; ')
            if tax_lst[-1] != 'NA':
                new_name = ' '.join(tax_lst[-2:]) + ', ' + n.name
                n.name = new_name
            else:
                temp = [tl for tl in tax_lst if tl != 'NA']
                num = len(tax_lst) - len(temp)
                new_name = ''.join(['*'] * num) + temp[-1] + ', ' + n.name
                n.name = new_name
    if name is not None:
        ts = ete3.TreeStyle()
        ts.show_leaf_name = True
        t.render(out_path + '/' + name, tree_style=ts)
    plt.close()

def plot_metab_tree(mets_keep, newick_path=base_path + '/ete_tree/w1_newick_tree.nhx',
                    out_path=base_path + '/outputs/', name='met_tree.pdf'):

    # Plots the metabolomic tree given an input newick tree
    # inputs:
    # - newick_path: path of the newick tree
    # - out_path: path to save the tree plot
    # - data_path: path of the taxonomic labels
    # - mets_keep: which metabolites to plot labels of on the tree
    # - name: name of the tree plot file

    t = ete3.TreeNode(newick_path)
    if mets_keep is not None:
        mets_keep = [m.replace('(', '_').replace(')', '_').replace(':','_').replace(',','_').replace('[','_').replace(']','_').replace(';','_') for m in mets_keep]

    for n in t.traverse():
        if n.is_leaf():
            if mets_keep is not None and n.name not in mets_keep:
                n.name = ''

    ts = ete3.TreeStyle()
    ts.show_leaf_name = True
    t.render(out_path + '/' + name, tree_style = ts)
    plt.close()

def plot_orig_metab_tree(out_path=base_path + '/outputs', name='mets_in.pdf', in_mets = None,
                         in_path = base_path + '/inputs/processed/',
                         newick_path = base_path + '/ete_tree/w1_newick_tree.nhx', dist_type = ''):

    # Constructs the metabolomic tree given the input metabolites and the classy-fire classifications
    # inputs:
    # - newick_path: path to save the newick tree to
    # - out_path: path to save the tree plot
    # - in_path: path for the input processed data
    # - in_mets: which metabolites to use in making the tree
    # - name: name of the tree plot file
    # - dist_type: How to construct the distance between branches of the tree; options:
    #           '' (empty string): set all branch distances = 1
    #           'clumps': set branch distance = 1 if within the same level 5 classification, or distance = 100 if not
    #           'stratified': increase branch distance exponentially with increasing classification levels

    met_classes = pd.read_csv(in_path + '/classy-fire/classy_fire_df.csv', index_col = 0, header = 0).T

    if dist_type == 'stratified':
        vals = [i**2 for i in range(1, met_classes.shape[1])]
        col_dicts = list(zip(met_classes.index.values, [np.min(vals)]*met_classes.shape[0]))
        for it, i in enumerate(np.arange(met_classes.shape[1]-1, 0, -1)):
            add_ls = list(zip(met_classes.iloc[:,i].str.upper(), [vals[it]]*met_classes.shape[0]))
            col_dicts.extend(add_ls)
        weight_dict = dict(col_dicts)
    elif dist_type == 'clumps':
        col_dicts = list(zip(met_classes.index.values, [1] * met_classes.shape[0]))
        for it, i in enumerate(np.arange(met_classes.shape[1]-1, 0, -1)):
            if i >= 6:
                add_ls = list(zip(met_classes.iloc[:,i].str.upper(), [1]*met_classes.shape[0]))
            else:
                add_ls = list(zip(met_classes.iloc[:, i].str.upper(), [100] * met_classes.shape[0]))
            col_dicts.extend(add_ls)
        weight_dict = dict(col_dicts)
    else:
        weight_dict = {}
    # query_child_dict = {}
    query_parent_dict = {}
    weights_dict = {}
    it = 0
    for met in met_classes.index.values:
        classification = met_classes.loc[met].dropna()
        if in_mets is not None:
            if met not in in_mets:
                continue
        it += 1
        for l in np.arange(1, len(classification)):
            if classification.iloc[l - 1].upper() not in query_parent_dict.keys():
                query_parent_dict[classification.iloc[l - 1].upper()] = [
                    classification.iloc[l].upper()]
            else:
                if classification.iloc[l].upper() not in query_parent_dict[
                    classification.iloc[l - 1].upper()]:
                    query_parent_dict[classification.iloc[l - 1].upper()].append(
                        classification.iloc[l].upper())
        if None not in query_parent_dict.keys():
            query_parent_dict[None] = [classification.iloc[0].upper()]
        else:
            if classification.iloc[0].upper() not in query_parent_dict[None]:
                query_parent_dict[None].append(classification.iloc[0].upper())
        if classification.iloc[-1].upper() not in query_parent_dict.keys():
            query_parent_dict[classification.iloc[-1].upper()] = [met]
        else:
            query_parent_dict[classification.iloc[-1].upper()].append(met)

    root = query_parent_dict[None][0]
    query_root = ete3.TreeNode(name=root)
    parents, added = [query_root], set([root])
    while parents:
        nxt = parents.pop()
        child_nodes = {child: ete3.TreeNode(name=child) for child in query_parent_dict[nxt.name]}
        for child in query_parent_dict[nxt.name]:
            if dist_type == '' or dist_type is None:
                nxt.add_child(child_nodes[child], name=child, dist=1)
            else:
                nxt.add_child(child_nodes[child], name=child, dist=weight_dict[child]/2)
            if child not in added:
                if child in query_parent_dict.keys():
                    parents.append(child_nodes[child])
                added.add(child)

    for n in query_root.traverse():
        if not n.is_leaf():
            n.add_face(ete3.TextFace(n.name + '   '), column = 0, position = 'branch-top')
    if name is not None:
        query_root.render(out_path + '/' + name)

    query_root.write(features=['name'], outfile=newick_path, format=0)
    plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-fun", "--fun", type = str)
    parser.add_argument("-name","--name", type = str)
    parser.add_argument("-out", "--out", type = str)
    parser.add_argument("-newick", "--newick", type = str)
    parser.add_argument("-feat", "--feat", nargs = '+', type = str)
    parser.add_argument("-dtype", "--dtype", type = str)
    args = parser.parse_args()

    if args.out is not None:
        if not os.path.isdir(args.out):
            os.mkdir(args.out)

        if args.fun == 'dist':
            get_dist(args.feat, out_path=args.out, name=args.name, newick_path = args.newick)
        if args.fun == 'metab_orig':
            plot_orig_metab_tree(name = args.name, in_mets = args.feat, out_path=args.out, newick_path = args.newick, dist_type = args.dtype)
        if args.fun == 'metab':
            plot_metab_tree(mets_keep = args.feat, name=args.name, out_path=args.out, newick_path = args.newick)
        if args.fun == 'asv':
            plot_asv_tree(taxa_keep=args.feat,name = args.name, out_path=args.out, newick_path = args.newick)
    else:
        if args.fun == 'dist':
            get_dist(args.feat, name=args.name, newick_path = args.newick)
        if args.fun == 'metab_orig':
            plot_orig_metab_tree(name = args.name, in_mets = args.feat, newick_path = args.newick, dist_type = args.dtype)
        if args.fun == 'metab':
            plot_metab_tree(mets_keep = args.feat, name=args.name, newick_path = args.newick)
        if args.fun == 'asv':
            plot_asv_tree(taxa_keep=args.feat,name = args.name, newick_path = args.newick)
