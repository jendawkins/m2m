import pandas as pd
import numpy as np
import os
from helper import *

data_path = '/Users/jendawk/Dropbox (MIT)/Microbes to Metabolomes/Datasets/Safari_data/'
out_path = '/Users/jendawk/Dropbox (MIT)/M2M/inputs/'

def load_safari_data(data_path='/Users/jendawk/Dropbox (MIT)/Microbes to Metabolomes/Datasets/Safari_data/',
                     met_frac=0.9, bug_frac = 0.15):

    metadata = pd.read_csv(data_path + 'metadata.txt', sep = "\t", header = 0, index_col = 0)
    samples_keep = metadata.index.values[:101]
    print(samples_keep[-1])
    dat_dict = {}
    for file in os.listdir(data_path):
        if 'quant' not in file:
            continue

        fname = file.split('_quant')[0].split('18-')[-1].split('-04-')[-1]
        if 'polar' in fname:
            fname = 'polar'
        dat_dict[fname] = {}
        dat = pd.read_csv(data_path + file, sep = " ", header = 0, index_col = [0,1,2])
        dat_dict[fname]['all'] = dat

        col_name = 'File_' + fname.replace('-', '_')
        #     df = metadata.set_index(col_name)
        cols = metadata[col_name].loc[samples_keep]
        temp = pd.read_csv(data_path + file, sep="\t", header=0, index_col=[0, 1, 2])
        temp_ri = temp.T

        dd = cols.to_dict()
        dd = {v + ' Peak area': k for k, v in dd.items()}
        dat_filt = temp_ri.rename(index=dd)

        epsilon = get_epsilon(dat_filt)
        transformed = np.log(dat_filt + epsilon)
        dat_std = (transformed.values - np.mean(transformed.values, 0))/np.std(transformed.values,0)

        dat_dict[fname]['samples'] = dat_filt
        dat_dict[fname]['log'] = transformed
        dat_dict[fname]['log_std'] = pd.DataFrame(dat_std, index = transformed.index.values,
                                                  columns=transformed.columns.values)

        ct = np.ones(dat_filt.shape)
        ct[dat_filt < 2] = 0
        sm = np.sum(ct, 0)
        ixs = sm >= met_frac*dat_filt.shape[0]
        dat_fin = dat_std[:, ixs]

        dat_dict[fname]['log_std_filt'] = pd.DataFrame(dat_fin, index = transformed.index.values,
                                                       columns=transformed.columns.values[ixs])

    asv_dat = data_path + '16S-ASV-table.csv'
    asvs = pd.read_csv(asv_dat, index_col=0, header=0)
    ra = (asvs.values.T / np.sum(asvs.values, 1)).T

    ct = np.ones(ra.shape)
    ct[asvs.values==0] = 0
    sm = np.sum(ct,0)
    dat_fin_asvs = pd.DataFrame(ra[:, sm>= asvs.shape[0]*bug_frac], index = asvs.index.values,
                                columns = asvs.columns.values[sm>= asvs.shape[0]*bug_frac])

    asv_dat_dict = {'raw': asvs, 'ra': pd.DataFrame(ra, index = asvs.index.values, columns = asvs.columns.values),
                    'ra_filt': dat_fin_asvs}

    if 'safari_corr_filt.pkl' not in os.listdir(out_path):
        res_dict = {}
        for fname in dat_dict.keys():
            met_dat = dat_dict[fname]['log_std_filt']
            asv_dat = dat_fin_asvs
            res_dict[fname] = {}
            print(fname)
            for i, metab in enumerate(met_dat.columns.values):
                for microbe in asv_dat.columns.values:
                    r, p = st.spearmanr(met_dat[metab].loc[asv_dat.index.values].values, asv_dat[microbe].values)

                    res_dict[fname][(metab, microbe)] = {}
                    res_dict[fname][(metab, microbe)]['rho'] = r
                    res_dict[fname][(metab, microbe)]['p'] = p
                if i % 100 == 0:
                    print(str(i) + '/' + str(len(met_dat.columns.values)) + ' done')

        df = pd.DataFrame(res_dict[fname]).T
        pairs = df.index.values[df['rho'] > 0.5]
        metabs, microbes = list(zip(*pairs))
        with open(out_path + 'safari_corr_filt.pkl', 'wb') as f:
            pkl.dump(res_dict, f)

    else:
        with open(out_path + 'safari_corr_filt.pkl', 'rb') as f:
            res_dict = pkl.load(f)

    return dat_dict, asv_dat_dict, res_dict


