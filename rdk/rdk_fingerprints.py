# #!/Users/jendawk/miniconda3/envs/my-rdkit-env/bin python3

import numpy as np
import pandas as pd
import os
# import pubchempy as pcp
import pickle as pkl
import re
# import seaborn as sns
from scipy.spatial.distance import squareform
import scipy.stats as st
from itertools import permutations, combinations
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem import AllChem
import argparse
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
import matplotlib.pyplot as plt
import time
# this script is used to return a distance matrix if you don't want to use classy-fire to calculate metabolomic distances
# by calling the script and specifying the necessary arguments (see below where script is called also), script will
# return a distance matrix given the metabolites in yfile that was calculated using the fingerprint and distance metric
# specfied


def tanimoto(a,b):
    # Calculates tanimoto distance b/w fingerprints a and b
    both = np.sum(a*b)
    return both / ((np.sum(a) + np.sum(b)) - both)

def dice(a, b):
    # calculates dice distance b/w fingerprints a and b
    return 2*(np.sum(a * b)) / (np.sum(a) + np.sum(b))

def cosine(a, b):
    # calculates cosine distance b/w fingerprings a and b
    return np.sum(a* b) / (np.sqrt(np.sum(a)) * np.sqrt(np.sum(b)))

def get_non_zero(vec):
    # returns vector with zeros replaced with minimum non-zero value of vector
    vec_nzero = vec
    vec_nzero[vec == 0] = np.min(vec[vec != 0])
    return vec_nzero

def get_distances(met_to_fingerprint, fingerprint_type, fingerprint_metric, fingerprint_metric_name):
    # get distances given a dictionary that maps the metabolite to its fingerpring,
    # the fingerprint type, distance metric, and distance metric name

    # Save a dataframe of distances between each two metabolites
    # (faster way to do this would be to use itertools.prod but whatever for now)
    df = {}
    for met1, f1 in met_to_fingerprint.items():
        if met1 not in df.keys():
            df[met1] = {}
        for met2, f2 in met_to_fingerprint.items():
            if met2 in df[met1].keys():
                continue
            else:
                if met2 not in df[met1].keys():
                    if fingerprint_type != 'pubchem':
                        # if 'morgan' in fingerprint_type or 'atom_pairs' in fingerprint_type:
                        if fingerprint_metric_name == 'dice':
                            df[met1][met2] = DataStructs.DiceSimilarity(f1, f2)
                        elif fingerprint_metric_name == 'tanimoto':
                            df[met1][met2] = DataStructs.TanimotoSimilarity(f1, f2)
                        elif fingerprint_metric_name == 'cosine':
                            df[met1][met2] = DataStructs.CosineSimilarity(f1, f2)
                            # Sokal, Russel, Kulczynski, McConnaughey, and Tversky
                        elif fingerprint_metric_name == 'sokal':
                            df[met1][met2] = DataStructs.SokalSimilarity(f1, f2)
                        elif fingerprint_metric_name == 'russel':
                            df[met1][met2] = DataStructs.RusselSimilarity(f1, f2)
                        elif fingerprint_metric_name == 'kulczynski':
                            df[met1][met2] = DataStructs.KulczynskiSimilarity(f1, f2)
                        elif fingerprint_metric_name == 'mcconnaughey':
                            df[met1][met2] = DataStructs.McConnaugheySimilarity(f1, f2)
                        elif fingerprint_metric_name == 'tversky':
                            df[met1][met2] = DataStructs.TverskySimilarity(f1, f2)
                        # else:
                        #     df[met1][met2] = DataStructs.FingerprintSimilarity(f1, f2, fingerprint_metric)

                        if met2 not in df.keys():
                            df[met2] = {}
                        df[met2][met1] = df[met1][met2]
                    else:
                        df[met1][met2] = fingerprint_metric(np.fromstring(f1, 'u1') - ord('0'), np.fromstring(f2, 'u1') - ord('0'))
                        if met2 not in df.keys():
                            df[met2] = {}
                        df[met2][met1] = df[met1][met2]
    return df

def plot_MDS(dat, dmax=30, seed=0, path = '/Users/jendawk/Dropbox (MIT)/M2M/figures/'):
    # Find the lowest dimension (up to dmax) at which the distribution of the embedded distances are not significantly
    # different than the distribution of the original distances (i.e. p > 0.05 in a k2_samp test)
    # inputs:
    #   dat: distance matrix
    #   dmax: max embedded dimension to look at
    #   seed: random seed
    #   path: path to put plots

    # Returns the pvalues over each dimension and the locations at the best dimeanison

    # Calculate the pvalue of the embedded distances vs the true distances using ks_2samp
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

    # Plot the pvalues vs dimension
    best = np.argmax(pvals)
    plt.figure()
    plt.plot(np.arange(2, dmax), pvals)
    plt.yscale('log')
    plt.xlabel('dimensions')
    plt.ylabel('p-values')
    plt.title('Max p-value= ' + str(pvals[best]))
    plt.savefig(path + 'mds.pdf')
    plt.close()

    # Plot the embedded distances and original distances distributions at the dimension where the p-value is greater than
    # 0.05 or (if the p-value is never greater) at the highest allowed dimension d-max
    d = np.where(np.array(pvals) > 0.05)[0]
    if len(d) == 0:
        d = best
    else:
        d = d[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[1].hist(pdist(locs[d]), bins = 20)
    ax[1].set_xlabel('Embedded distances')
    ax[1].set_title('Distribution of Embedded Distances, d = ' + str(d))
    ax[0].hist(true_dist,bins = 20)
    ax[0].set_xlabel('True distances')
    ax[0].set_title('Distribution of Taxonomic Distances')
    fig.savefig(path + 'hist.pdf')
    plt.close(fig)

    return locs[best], pvals


if __name__ == "__main__":
    # Returns a distance matrix saved to outpath given calculated on the metabolites in yfile using the specified
    # fingerprint method and distance metric
    # TO DO:
    # - atom_paris and the morgan fingerprints don't seem to work yet; can figure out how to get them to work
    parser = argparse.ArgumentParser()
    parser.add_argument("-fingerprint", "--fingerprint",
                        help = "fingerprint method; choices are: 'pubchem', 'RDK' or 'MACCS' ",
                        type = str, default = 'pubchem')
    parser.add_argument("-metric", "--metric",
                        help="distance metric used to calculate distances between metabolic fingerprints"
                             "Choices are: 'tanimoto', 'dice', or 'cosine' ",
                        type=str, default = 'dice')
    parser.add_argument("-smiles_type", "--smiles_type", help='smiles type; keep as isomeric', type=str, default='isomeric')
    parser.add_argument("-yfile", "--yfile", type = str,
                        help = 'filename for processed metabolomic data, distance matrix calculated for metabolites in yfile')
    parser.add_argument("-ydist_file", "--ydist_file", type = str,
                        help = 'filename to save dist file to ')
    parser.add_argument("-o", "--outpath", type=str, help = 'path to save distance matrix to')
    parser.add_argument("-b", "--base_path", type= str, help = "base path")
    args = parser.parse_args()

    sm = pd.read_csv(args.base_path + '/inputs/processed/' + args.yfile)
    mets_keep = sm.columns.values

    if not os.path.isdir(args.base_path + '/inputs/processed/' + args.fingerprint):
        os.mkdir(args.base_path + '/inputs/processed/' + args.fingerprint)

    if os.path.isfile(args.base_path + '/inputs/processed/' + args.fingerprint + '/met_to_fingerprint_sm.pkl'):
        with open(args.base_path + '/inputs/processed/' + args.fingerprint + '/met_to_fingerprint_sm.pkl', 'rb') as f:
            met_to_fingerprint = pkl.load(f)
        with open(args.base_path + '/inputs/processed/' + 'met_to_' + args.smiles_type + '_smiles.pkl', 'rb') as f:
            met_to_smiles = pkl.load(f)


    else:
        with open(args.base_path + '/inputs/processed/' + 'met_to_' + args.smiles_type + '_smiles.pkl', 'rb') as f:
            met_to_smiles = pkl.load(f)
        if args.fingerprint == 'pubchem':
            with open(args.base_path + '/inputs/processed/' + args.fingerprint + '/met_to_fingerprint.pkl', 'rb') as f:
                init_met_to_fingerprint = pkl.load(f)
            met_to_fingerprint = {met: init_met_to_fingerprint[met] for met in mets_keep if met in init_met_to_fingerprint.keys()}
        else:
            met_to_fingerprint = {}
            for met, smile in met_to_smiles.items():
                if met not in mets_keep:
                    continue
                m = Chem.MolFromSmiles(smile)
                if args.fingerprint == 'RDK':
                    met_to_fingerprint[met] = Chem.RDKFingerprint(m)
                if args.fingerprint == 'MACCS':
                    met_to_fingerprint[met] = MACCSkeys.GenMACCSKeys(m)
                if args.fingerprint == 'atom_pairs':
                    met_to_fingerprint[met] = Pairs.GetAtomPairFingerprint(m)
                if args.fingerprint == 'morgan_ECFP4':
                    met_to_fingerprint[met] = AllChem.GetMorganFingerprint(m,2)
                if args.fingerprint == 'morgan_FCFP4':
                    met_to_fingerprint[met] = AllChem.GetMorganFingerprint(m,2, useFeatures = True)

        with open(args.base_path + '/inputs/processed/' + args.fingerprint + '/met_to_fingerprint_sm.pkl', 'wb') as f:
            pkl.dump(met_to_fingerprint, f)

    if args.fingerprint != 'pubchem':
        if args.metric == 'tanimoto':
            fingerprint_metric = DataStructs.TanimotoSimilarity
        elif args.metric == 'dice':
            fingerprint_metric = DataStructs.DiceSimilarity
        elif args.metric == 'cosine':
            fingerprint_metric = DataStructs.CosineSimilarity
        else:
            fingerprint_metric = ''
    else:
        if args.metric == 'tanimoto':
            fingerprint_metric = tanimoto
        if args.metric == 'dice':
            fingerprint_metric = dice
        if args.metric == 'cosine':
            fingerprint_metric = cosine

    if not os.path.isfile(args.base_path + '/inputs/processed/' + args.ydist_file):
        dist_dict = get_distances(met_to_fingerprint, args.fingerprint, fingerprint_metric, args.metric)
        df = pd.DataFrame(dist_dict)
        df = df[mets_keep].loc[mets_keep]
        df.to_csv(args.outpath + args.ydist_file)
    else:
        df = pd.read_csv(args.outpath  + args.ydist_file, header=0, index_col=0)




