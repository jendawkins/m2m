import os
import pickle as pkl

param_dict = {}
x_sums = {}
y_sums = {}
path = '/Users/jendawk/M2M/outputs/synthetic_true_09-01-2022'
for folder in os.listdir(path):
    if '.' not in folder:
        files = os.listdir(path + '/' + folder)
        param_files = [f for f in files if 'param_dict.pkl' in f]
        x_files = [f for f in files if 'microbe_sum.pkl' in f]
        y_files = [f for f in files if 'met_clusters.pkl' in f]
        for (pfile, xfile, yfile) in zip(param_files, x_files, y_files):
            key = pfile.split('_')[0]
            with open(path + '/' + folder + '/' + pfile, 'rb') as f:
                param_dict[key] = pkl.load(f)
            with open(path + '/' + folder + '/' + xfile, 'rb') as f:
                x_sums[key] = pkl.load(f)
            with open(path + '/' + folder + '/' + yfile, 'rb') as f:
                y_sums[key] = pkl.load(f)

            preds = torch.matmul(pred_clusters + meas_var * torch.randn(pred_clusters.shape), torch.Tensor(pred_z).T)
