# #!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from torch.distributions.dirichlet import Dirichlet
from helper import *
from plot_helper import *
from concrete import *
import argparse
import re
from synthetic_data_generation import *
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

import sys
from dataLoader import *
# from rdk.rdk_fingerprints import *
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import subprocess
from sklearn.cluster import KMeans
import scipy
from torch.distributions.half_normal import HalfNormal
from safariLoader import *
from model import *
import datetime
from ete_tree.tree_plotter import *
from sklearn.model_selection import train_test_split
import json
from joblib import Parallel, delayed


class LitM2M(pl.LightningModule):
    def __init__(self, args, a_met, a_bug, met_class, bug_class, x, y, x_df, y_df):
        super().__init__()
        self.args = args
        self.net = Model(a_met, a_bug, K=args.K, L=args.L,
                    N_met=y.shape[1], N_bug=x.shape[1], alpha_temp=10 ** args.a_tau[0], omega_temp=10 ** args.w_tau[0],
                    learn_num_bug_clusters=args.lb, learn_num_met_clusters=args.lm, linear=args.linear == 1,
                    p_nn=args.p_num, data_meas_var=args.meas_var, met_class=met_class, bug_class=bug_class)

        self.alpha_tau_logspace = np.logspace(args.a_tau[0], args.a_tau[1], 0.70*(args.epochs + 1))
        self.omega_tau_logspace = np.logspace(args.w_tau[0], args.w_tau[1], 0.70*(args.epochs + 1))
        x = torch.Tensor(np.array(x)).to(device)
        y = torch.Tensor(np.array(y))
        self.x_df = x_df
        self.y_df = y_df
        self.net.initialize(args.seed, x, y)

        self.param_dict = {}
        start = 0
        for name, parameter in self.net.named_parameters():
            if 'NAM' in name:
                continue
            if name == 'alpha':
                param = getattr(self.net, name + '_act').detach().numpy()
            else:
                param = parameter.detach().numpy()
            self.param_dict[name] = [param.copy()]

        self.param_dict['z'] = [self.net.z_act.detach().numpy().copy()]
        self.param_dict['w'] = [self.net.w_act.detach().numpy().copy()]

        self.loss_dict = {}
        self.loss_vec = {'train':[], 'val':[]}

        self.scores_dict = {'train':{'R2':[], 'NRMSE':[], 'NMI':[],'RI':[]}, 'val':{'R2':[], 'NRMSE':[], 'NMI':[],'RI':[]}}

        self.out_vec = {'train': [], 'val': []}

    def get_metrics(self, clusters, y):
        preds = torch.matmul(clusters +
                             self.args.meas_var * torch.randn(clusters.shape), self.net.z_act.float().T)

        r2 = r2_score(y.detach().numpy(), preds.detach().numpy())

        RMSE_est = np.sqrt(np.sum(((preds.detach().numpy() - y.detach().numpy()) ** 2)) / len(preds.detach().flatten()))
        N_RMSE_est = np.round(RMSE_est / st.iqr(y.detach().numpy().flatten()), 3)

        # z_guess = np.argmax(self.net.z_act.detach().numpy(), 1)
        # nmi = np.round(normalized_mutual_info_score(y.detach().numpy(), z_guess), 3)
        # try:
        #     tp, fp, tn, fn, ri = pairwise_eval(z_guess, y.detach().numpy())
        #     ri = str(np.round(ri, 3))
        # except:
        #     ri = 'NA'
        return {'R2':r2, 'NRMSE':N_RMSE_est}


    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, self.train_y = batch
        if self.current_epoch > int(self.args.epochs*0.15) and self.current_epoch < int(self.args.epochs*(1-0.15)):
            self.net.alpha_temp = self.alpha_tau_logspace[self.current_epoch - int(self.args.epochs*0.15)]
            self.net.omega_temp = self.omega_tau_logspace[self.current_epoch - int(self.args.epochs*0.15)]
        self.train_clusters = self(x)
        self.out_vec['train'].append(self.train_clusters.detach().numpy().copy())
        loss = self.net.MAPloss.compute_loss(self.train_clusters, self.train_y)
        self.loss_vec['train'].append(loss)
        for param in self.net.MAPloss.loss_dict:
            if param not in self.loss_dict.keys():
                self.loss_dict[param] = [self.net.MAPloss.loss_dict[param].detach().item()]
            else:
                self.loss_dict[param].append(self.net.MAPloss.loss_dict[param].detach().item())

        scores = self.get_metrics(self.train_clusters, self.train_y)
        for k, v in scores.items():
            self.scores_dict['train'][k].append(v)

        self.log('train_loss', loss)
        return loss

    def on_train_epoch_end(self):
        for name, parameter in self.net.named_parameters():
            if 'NAM' in name:
                continue
            if name == 'alpha':
                param = getattr(self.net, name + '_act').detach().numpy()
            else:
                param = parameter.detach().numpy()
            self.param_dict[name].append(param.copy())
        self.param_dict['w'].append(self.net.w_act.detach().numpy().copy())
        self.param_dict['z'].append(self.net.z_act.detach().numpy().copy())


    def validation_step(self, batch, batch_idx):
        x, self.val_y = batch
        self.val_clusters = self(x)
        self.out_vec['val'].append(self.val_clusters.detach().numpy().copy())
        val_loss = self.net.MAPloss.compute_loss(self.val_clusters, self.val_y )
        self.loss_vec['val'].append(val_loss)

        self.val_scores = self.get_metrics(self.val_clusters, self.val_y)
        for k, v in self.val_scores.items():
            self.scores_dict['val'][k].append(v)

        self.log('val_loss', val_loss)
        return val_loss

    def on_validation_epoch_end(self):
        if self.current_epoch > 2:
            if not os.path.isdir(self.logger.log_dir + f'/epoch{self.current_epoch}/'):
                os.mkdir(self.logger.log_dir + f'/epoch{self.current_epoch}/')
            plot_tracked_metrics(self.scores_dict, self.logger.log_dir + f'/epoch{self.current_epoch}/')
            plot_loss_dict(self.logger.log_dir + f'/epoch{self.current_epoch}/', self.loss_dict)
            fig, ax = plot_loss(self.args.seed, self.loss_vec['train'], self.loss_vec['val'])
            fig.savefig(self.logger.log_dir + f'/epoch{self.current_epoch}/' + 'loss.pdf')
            plt.close(fig)

            plot_param_traces(self.logger.log_dir + f'/epoch{self.current_epoch}/', self.param_dict)
            best_mod = np.argmin(self.loss_vec[args.monitor.split('_')[0]])
            plot_output(self.logger.log_dir + f'/epoch{self.current_epoch}/' + '/train_', best_mod, self.out_vec['train'],
                        self.train_y.detach().numpy(),
                        self.param_dict, meas_var=args.meas_var)
            save_results(self.logger.log_dir + f'/epoch{self.current_epoch}/' + '/train_', best_mod, self.param_dict,
                         self.x_df.columns.values, self.y_df.columns.values)
            for name, param in self.net.named_parameters():
                if torch.isnan(param).any():
                    print(name)
            test_scores = pd.DataFrame(self.val_scores, index=[0])
            test_scores.to_csv(self.logger.log_dir + f'/epoch{self.current_epoch}/' + 'test_scores.csv')

    def test_step(self, batch, batch_idx):
        x, self.test_y = batch
        self.test_clusters = self(x)
        test_loss = self.net.MAPloss.compute_loss(self.test_clusters, self.test_y)
        self.test_scores = self.get_metrics(self.test_clusters, self.test_y)
        return test_loss

    def configure_optimizers(self):
        # Adjust each parameter's learning rate based on parameter size
        lr_dict = {'mu_bug': 0.01, 'r_bug': 0.005, 'r_met': 0.005, 'mu_met': 0.005, 'pi_met': 0.001,
                   'e_met': 0.001, 'beta': 0.005, 'alpha': 0.002}
        lr_list = []
        size_beta = torch.mean(torch.abs(self.net.beta.detach().flatten()))
        for name, parameter in self.net.named_parameters():
            size = torch.mean(torch.abs(parameter.detach().flatten()))
            if args.adjust_lr:
                lr_list.append({'params': parameter, 'lr': (args.lr / size_beta) * size})
                lr_dict[name] = [(args.lr / size_beta) * size]
            else:
                if name in lr_dict.keys():
                    lr_list.append({'params': parameter, 'lr': lr_dict[name]})
                else:
                    lr_list.append({'params': parameter})

                lr_dict = {k: [v] for k, v in lr_dict.items()}

        # initialize optimizer with learning rates
        optimizer = optim.RMSprop(lr_list, lr=args.lr)
        return optimizer

class m2mDataset(Dataset):
    def __init__(self, x, y, idxs):
        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        self.x = torch.Tensor(x[idxs,:])
        self.y=torch.Tensor(y[idxs,:])

    def __len__(self): return self.y.shape[0]

    def __getitem__(self, idx):
        return self.x[idx,:], self.y[idx,:]

class CVTrainer():
    def __init__(self, args, outpath, dataset):
        self.args = args
        self.outpath = outpath
        self.x = dataset['x']
        self.y = dataset['y']
        self.a_met = dataset['a_met']
        self.a_bug = dataset['a_bug']
        self.met_class = dataset['met_class']
        self.bug_class = dataset['bug_class']

    def train_loop(self, train_ixs, test_ixs=None, fold=None):
        if self.args.validate==1:
            train_ixs, val_ixs = train_test_split(train_ixs, test_size=0.1, random_state=args.seed)
            val_dataset = m2mDataset(self.x, self.y, val_ixs)
            val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)

        train_dataset = m2mDataset(self.x, self.y, train_ixs)
        train_loader = DataLoader(train_dataset, batch_size = 100, shuffle=True)
        if test_ixs is not None:
            test_loader=DataLoader(m2mDataset(self.x, self.y, test_ixs), batch_size=100, shuffle=False)

        else:
            test_loader = None
        os.makedirs(self.outpath + f'seed_{self.args.seed}', exist_ok=True)
        self.dirpath = self.outpath + f'seed_{self.args.seed}'
        with open(self.outpath + f'seed_{args.seed}' + '/commandline_args_eval.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
        model = LitM2M(self.args, self.a_met, self.a_bug, self.met_class, self.bug_class,
                       train_dataset.x, train_dataset.y, self.x, self.y)

        callbacks=[ModelCheckpoint(save_last=True, dirpath = self.dirpath, save_top_k=1, monitor=args.monitor,
                                   mode='min' if 'loss' in args.monitor else 'max',
                                   filename='{epoch}-{' + args.monitor + ':.2f}'), LearningRateMonitor()]

        if args.early_stopping == 1:
            callbacks.extend([EarlyStopping(monitor=args.monitor, patience=200)])

        tb_logger = TensorBoardLogger(save_dir=self.outpath, name=f'seed_{self.args.seed}', version=f'fold_{fold}')
        trainer = pl.Trainer(logger=tb_logger, max_epochs = args.epochs, min_epochs = args.min_epochs,
                             check_val_every_n_epoch=500,callbacks=callbacks, log_every_n_steps=1,
                             deterministic=True)
        for param, dist in model.net.distributions.items():
            if param != 'NAM':
                try:
                    plot_distribution(dist, param, true_val=None, ptype='prior', path=tb_logger.log_dir)
                except:
                    print(param + ' plot distribution error!!')

        if args.validate==1:
            trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
            best_mod = np.argmin(model.loss_vec['val'])
            plot_output(tb_logger.log_dir + '/val_', best_mod, model.out_vec['val'], val_dataset.y.detach().numpy(),
                        model.param_dict, meas_var=args.meas_var)
        elif args.test==1:
            trainer.fit(model, train_dataloader=train_loader, val_dataloaders=test_loader)
        else:
            trainer.fit(model, train_dataloaders=train_loader)

        plot_tracked_metrics(model.scores_dict, tb_logger.log_dir)
        plot_loss_dict(tb_logger.log_dir, model.loss_dict)
        fig, ax = plot_loss(args.seed, model.loss_vec['train'], model.loss_vec['val'])
        fig.savefig(tb_logger.log_dir + 'loss.pdf')
        plt.close(fig)

        plot_param_traces(tb_logger.log_dir, model.param_dict)
        best_mod = np.argmin(model.loss_vec[args.monitor.split('_')[0]])
        plot_output(tb_logger.log_dir + '/train_', best_mod, model.out_vec['train'], train_dataset.y.detach().numpy(),
                    model.param_dict, meas_var=args.meas_var)
        save_results(tb_logger.log_dir + '/train_', best_mod, model.param_dict, self.x.columns.values, self.y.columns.values)
        if test_ixs is not None:
            for name, param in model.net.named_parameters():
                if torch.isnan(param).any():
                    print(name)
            out=trainer.test(model, dataloaders=test_loader)
            test_scores = pd.DataFrame(model.test_scores, index=[0])
            test_scores.to_csv(tb_logger.log_dir + 'test_scores.csv')


def run_training_with_folds(args, dataset, OUTPUT_PATH = ''):
    seed_everything(args.seed, workers=True)

    cv_trainer = CVTrainer(args, OUTPUT_PATH, dataset)

    if args.test==1:
        if args.cv_type == 'kfold':
            train_ixs, test_ixs = cv_kfold_splits(np.zeros(y.shape[0]), y,num_splits=args.kfolds, seed=args.seed)
        elif args.cv_type=='loo':
            train_ixs, test_ixs = cv_loo_splits(np.zeros(y.shape[0]), y)
        elif args.cv_type=='one':
            # train_ixs, test_ixs = cv_kfold_splits(np.zeros(y.shape[0]), y, num_splits=args.kfolds, seed=args.seed)
            train_ixs, test_ixs = [np.arange(5,y.shape[0])], [np.arange(5)]
            # train_ixs, test_ixs = train_ixs[0], test_ixs[0]
        else:
            print("Please enter valid option for cv_type. Options are: 'kfold','loo','one'")
            return

        folds = list(range(len(train_ixs))) + ['EVAL']
        train_ixs = train_ixs + [np.arange(y.shape[0])]
        test_ixs = test_ixs + [np.arange(y.shape[0])]

        if args.parallel > 1 and args.cv_type !='one':
            Parallel(n_jobs=args.parallel)(delayed(cv_trainer.train_loop)(train_idx, test_idx, fold)
                                     for fold, train_idx, test_idx in zip(folds, train_ixs, test_ixs))
        else:
            for fold, train_idx, test_idx in zip(folds, train_ixs, test_ixs):
                print('FOLD {0}'.format(fold))
                cv_trainer.train_loop(train_idx, test_idx, fold)

    else:
        print('EVAL')
        cv_trainer.train_loop(np.arange(y.shape[0]), test_ixs=None, fold='EVAL')





if __name__ == "__main__":
    # Calls run_learner with input arguments
    # TO DO:
    # - make more generalizable to more input data; right now can use synthetic data (with or without location priors),
    # cdiff data (with or without location priors) and safari data (without location priors)
    # - could do the config method like Suhas does in MDITRE
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default = 0.001)
    parser.add_argument("-case", "--case", help="case", type=str,
                        default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 10)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 10)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 20)
    parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 20)
    parser.add_argument("-L_true", "--L_true", help="true number of microbe clusters "
                                                    "(for synthetic data generation)", type=int, default = 0)
    parser.add_argument("-K_true", "--K_true", help="true number of metab clusters "
                                                    "(for synthetic data generation)", type=int, default = 0)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.1)
    parser.add_argument("-epochs", "--epochs", help="number of iterations", type=int,default = 10000)
    parser.add_argument("-min_epochs","--min_epochs", type=int, default=100)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 99)
    parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 1)
    parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 1)
    parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int, default=240)
    parser.add_argument("-linear", "--linear", type = int, default = 0, help = 'whether to run linear model or not')
    parser.add_argument("-nltype", "--nltype", type = str, default = "poly",
                        help = 'if using synthetic data and linear == 0, how to non-linearly generate data'
                               'choices are: exp, poly, sine, linear, sigmoid')
    parser.add_argument("-adjust_lr", "--adjust_lr", type=int, default=1,
                        help = "whether to adjust the learning rate based on the size of the parameter")
    parser.add_argument("-p_num", "--p_num", type=int, default=1,
                        help = "if non-linear, how many neural networks per microbe cluster - metabolite cluster interaction"
                               "(i.e. p=1 means 1 NN per each interaction)")
    parser.add_argument("-xdim", "--xdim", type=int, default=10, help = 'embedding dimension for microbes')
    parser.add_argument("-ydim", "--ydim", type=int, default=10, help = 'embedding dimension for metabolites')
    parser.add_argument("-a_tau", "--a_tau", type=float, nargs = '+', default=[-0.1, -2.5], help = 'annealing for alpha temparature')
    parser.add_argument("-w_tau", "--w_tau", type=float, nargs='+', default=[-0.1, -1.5], help = 'anealing for omega temperature')
    parser.add_argument("-locs","--locs", type = str, default = 'true',
                        help = 'true= use true microbe and metabolite embedded locations; '
                                                                               'none= dont use locations;'
                                                                               'rand= use random locations;')
    parser.add_argument("-dtype", "--dtype", type=str, default='pubchem_tanimoto',
                        help = "which type of distance embedding to use, choices are:" 
                               " 'stratified', 'clumps', '', 'pubchem_tanimoto', 'RDK_tanimoto','MACCS_tanimoto' ")
    parser.add_argument("-data", "--data", type = str, default = 'cdi', help = "which input data to use; choices are: "
                                                                               "'cdi', 'safari', 'synthetic' ")

    # Filtering criteria if args.data == 'cdi' or args.data == 'safari'
    parser.add_argument("-nzm", "--non_zero_perc_met", type=float, default=98,
                        help='percent of participants with non-zero metabolites in filtered data')
    parser.add_argument("-nzb", "--non_zero_perc_bug", type=float, default=15,
                        help='percent of participants with non-zero microbes in filtered data')
    parser.add_argument("-cvm", "--coef_var_perc_met", type=float, default=50,
                        help='coefficient of variation percentile for metabolites')
    parser.add_argument("-cvb", "--coef_var_perc_bug", type=float, default=5,
                        help='coefficient of variation percentile for microbes')

    parser.add_argument("-most_corr", "--most_corr", type=int, default = 0,
                        help = 'whether to use the data with high correlation bw microbes and metabolites or not')

    parser.add_argument("-early_stopping", "--early_stopping", default = None,
                        help = 'whether to stop training early')

    parser.add_argument("-validate", "--validate", type=int,default =0)
    parser.add_argument('--test', default=1, type = int)
    parser.add_argument('--parallel', type=int, default=2,
                        help='run in parallel')
    parser.add_argument('--cv_type', type=str, default='one',
                        choices=['loo', 'kfold', 'one','None'],
                        help='Choose cross val type')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='Number of folds for k-fold cross val')
    parser.add_argument('--monitor', type=str, default='train_loss')

    args = parser.parse_args()
    print(sys.executable)

    if args.L_true == 0:
        args.L_true = args.L
    if args.K_true == 0:
        args.K_true = args.K

    args.case = args.data + '_' + args.locs + '_' + args.case
    dtype = args.dtype
    base_path = os.getcwd()
    if '/M2M' not in base_path:
        base_path = '/Users/jendawk/M2M/'
    if not os.path.isdir(base_path + '/outputs/'):
        os.mkdir(base_path + '/outputs/')
    if not os.path.isdir(base_path + '/outputs/' + args.case):
        os.mkdir(base_path + '/outputs/' + args.case)

    outpath = base_path + '/outputs/' + args.case
    args.raw_data_path = '/inputs/' + args.data + '/'
    dataset={}
    if args.data == 'cdi':
        if args.most_corr == 1:
            xfile = 'x_high_corr.csv'
            yfile = 'y_high_corr.csv'
        else:
            yfile = 'y_' + str(int(args.non_zero_perc_met)) + '_' + str(int(args.coef_var_perc_met)) + '.csv'
            xfile = 'x_' + str(int(args.non_zero_perc_bug)) + '_' + str(int(args.coef_var_perc_bug)) + '.csv'

        args.met_newick_name = 'newick_' + yfile.split('.csv')[0] + '.nhx'

        # Option to change filtering criteria
        # if xfile not in os.listdir(base_path + "/inputs/processed/") or yfile not in os.listdir(base_path + "/inputs/processed/"):
        load_data(xfile, yfile, dataLoader,data_path=base_path +args.raw_data_path, out_path = base_path + "/inputs/processed/", )

        # set data_path to point to directory with processed data
        data_path = base_path + "/inputs/processed/"
        x = pd.read_csv(data_path + '/' + xfile, index_col = [0])
        y = pd.read_csv(data_path + '/' + yfile, index_col=[0])

        y = y.loc[x.index.values]

        with open(data_path + '/' + yfile.split('.')[0] + '-mvar.pkl', 'rb') as f:
            args.meas_var = pkl.load(f)

        args.N_met = y.shape[1]
        args.N_bug = x.shape[1]
        args.N_samples = y.shape[0]

        if args.locs == 'true':
            xdist_file = xfile.split('.')[0] + '-dist.csv'
            if xdist_file not in os.listdir(base_path + '/inputs/processed/'):
                get_dist(x.columns.values, newick_path=base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
                         out_path=base_path+'/inputs/processed/', name = xdist_file)
            xdist = pd.read_csv(base_path + '/inputs/processed/' + xdist_file, header=0, index_col=0)
            xdist = xdist / np.max(np.max(xdist))
            x = x[xdist.columns.values]

            if '_' in dtype:
                ydist_file = dtype.split('_')[0] + '/' + dtype.split('_')[1] + '-dist.csv'
            else:
                ydist_file = 'y' + '-' + yfile.split('.')[0] + '-' + dtype + '_dist.csv'

            if ydist_file.split('/')[-1] not in os.listdir(base_path + '/inputs/processed/' + ydist_file.split('/')[0]):
                if '_' in args.dtype:
                    in_list = ["python3", "rdk_fingerprints.py", "-fingerprint", dtype.split('_')[0],
                               "-metric", dtype.split('_')[1], "-yfile", yfile, '-o', base_path+'/inputs/processed/',
                               "-ydist_file", ydist_file, "-b", base_path]
                    subprocess.run(in_list, cwd=base_path + '/rdk')
                else:
                    get_dist(y.columns.values, newick_path=base_path +'/ete_tree/' + args.met_newick_name,
                             out_path=base_path+'/inputs/processed/', name = ydist_file)
            ydist = pd.read_csv(base_path + '/inputs/processed/' + ydist_file, header = 0, index_col = 0)
            ydist = 1- (ydist / np.max(np.max(ydist)))

            if ydist.shape[0] != y.shape[1]:
                try:
                    ixs = list(set(y.columns.values).intersection(ydist.columns.values))
                    y = y[ixs]
                    ydist = ydist[ixs].loc[ixs]
                except:
                    y.columns = [edit_string(ii) for ii in y.columns.values]
                    ixs = list(set(y.columns.values).intersection(ydist.columns.values))
                    y = y[ixs]
                    ydist = ydist[ixs].loc[ixs]

            if args.xdim is None:
                args.xdim, xlocs, xstress = mds_choose_d(xdist,seed = args.seed)
            else:
                embedding = MDS(n_components=args.xdim, dissimilarity='precomputed', random_state=args.seed)
                xlocs = embedding.fit_transform(xdist)

            if args.ydim is None:
                args.ydim, ylocs, ystress = mds_choose_d(ydist, seed=args.seed)
            else:
                embedding = MDS(n_components=args.ydim, dissimilarity='precomputed', random_state=args.seed)
                ylocs = embedding.fit_transform(ydist)

            dataset['x'], dataset['y'] = x, y
            dataset['bug_class'] = get_xtaxa(base_path + '/' + args.raw_data_path + '/taxa_labels.csv', x)
            dataset['met_class'] = get_ytaxa(base_path + '/' + args.raw_data_path + '/classy_fire_df.csv', y.columns.values,
                                ydist, level='class')

            dataset['a_bug'] = (xlocs-np.mean(np.mean(xlocs)))/np.max(np.max(xlocs))
            dataset['a_met'] = (ylocs-np.mean(np.mean(ylocs)))/np.max(np.max(ylocs))
        elif args.locs == 'random':
            dataset['a_met'] = get_rand_locs(y, args.ydim, args.seed)
            dataset['a_bug'] = get_rand_locs(x, args.xdim, args.seed)
            dataset['bug_class'], dataset['met_class'] = None, None
        else:
            dataset['a_bug'], dataset['a_met'] = None, None
            dataset['bug_class'], dataset['met_class'] = None, None

    elif args.data=='synthetic':
        x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
        mu_met, r_bug, r_met = generate_synthetic_data(p=0.5,
            N_met = args.N_met, N_bug = args.N_bug, N_met_clusters = args.K_true,
            N_bug_clusters = args.L_true ,N_samples=args.N_samples, linear = args.linear,
            nl_type = args.nltype, xdim=args.xdim, ydim = args.ydim)
        if args.lm:
            gen_z = np.hstack((gen_z, np.zeros((args.N_met, args.N_met - 1 - args.K))))
            mu_met = np.vstack((mu_met, np.zeros((args.N_met - args.K - 1, mu_met.shape[1]))))
            r_met = np.append(r_met, np.zeros(args.N_met - 1 - len(r_met)))
            if args.linear:
                gen_beta = np.hstack((gen_beta, np.zeros((gen_beta.shape[0], args.N_met - args.K - 1))))
            gen_alpha = np.hstack((gen_alpha, np.zeros((gen_alpha.shape[0], args.N_met - args.K - 1))))
        if args.lb:
            r_bug = np.append(r_bug, np.zeros(args.N_bug - 1 - len(r_bug)))
            mu_bug = np.vstack((mu_bug, np.zeros((args.N_bug - args.L - 1, mu_bug.shape[1]))))
            gen_w = np.hstack((gen_w, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
            if args.linear:
                gen_beta = np.vstack((gen_beta, np.zeros((args.N_bug - args.L - 1, gen_beta.shape[1]))))
            gen_alpha = np.vstack((gen_alpha, np.zeros((args.N_bug - args.L - 1, gen_alpha.shape[1]))))
        # Dictionary of true values to compare to learned values
        true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': mu_bug,
                     'mu_met': mu_met, 'r_bug':1.2*r_bug, 'r_met': 1.2*r_met, 'z': gen_z,
                     'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                     'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0), 'bug_locs': gen_bug_locs,
                     'met_locs':gen_met_locs, 'e_met': np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0)}

        tr_ids, val_ids = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=args.seed)
        x_val, y_val, g_val = x[val_ids,:], y[val_ids,:], g[val_ids,:]
        x_val = torch.Tensor(np.array(x_val)).to(device)
        y_val = torch.Tensor(np.array(y_val))

        x, y, g = x[tr_ids, :], y[tr_ids, :], g[tr_ids, :]
        if not args.linear:
            gen_beta = gen_beta[0,:]
        plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                      r_bug, mu_met, r_met, gen_w, gen_alpha, gen_beta)

        dataset['a_met'] = gen_met_locs
        dataset['a_bug'] = gen_bug_locs
        if args.locs == 'none':
            dataset['a_met'], dataset['a_bug']=None, None

        dataset['bug_class'], dataset['met_class'] = None, None


    print(f'OTUS shape:{dataset["x"].shape}')
    print(f'Metabs shape:{dataset["y"].shape}')
    print(f'OTUS locs:{dataset["a_bug"].shape}')
    print(f'Metabs locs:{dataset["a_met"].shape}')
    plot_data(dataset['x'], dataset['y'], outpath)
    run_training_with_folds(args, dataset, OUTPUT_PATH=outpath)

    # run_learner(args, device, x=x, y=y, base_path = base_path, a_met = ylocs, a_bug = xlocs, met_class = y_class, bug_class = x_fams)
