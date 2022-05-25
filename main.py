#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from torch.distributions.dirichlet import Dirichlet
from helper import *
from plot_helper import *
from MAP_loss import *
from concrete import *
import argparse
import re
from data_gen import *
import sys
from dataLoader import *
import datetime
import subprocess
from sklearn.cluster import KMeans
import scipy
# from tree_plotter import plot_asv_tree, plot_orig_metab_tree, plot_metab_tree

# this model initializes model parameters and contains all the model equations in the overleaf
class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, N_met, N_bug, K = 2, L = 3,
                 seed = 0, alpha_temp = 1, omega_temp = 1, data_meas_var = 1,
                 compute_loss_for = ['alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met'],
                 learn_num_met_clusters = False, learn_num_bug_clusters = False, linear = True,
                l1= 1, p_nn = 1, sample_a = True, sample_w = True, met_class = None, bug_class = None, gmm = 0):
        super(Model, self).__init__()
        # This function initializes the Model;
        # *note that the inputs above are just defaults; the actual inputs are set when the model is initialized in run_learner() below
        # *note: omega and w are used interchangeably, but always mean the same variable (the microbial cluster indicator matrix)
        # Inputs:
        # - met_locs: the metabolite embedded locations (Matrix with size [N_met X #embedding-dimensions])
        # - microbe_locs: microbe embedded locations (matrix w/ size [N_bug X #embedding-dimensions])
        # - N_met: # metabolites
        # - N_bug: # microbes
        # - K: # metabolite clusters (if not learning the number of clusters)
        # - L: # microbe clusters (if not learning the number of clusters)
        # - seed: random seed
        # - alpha_temp: temperature parameter for alpha (decreases during learning)
        # - omega_temp: tempurature parameter for omega (only used if no microbial locations are given and omega is
        #           learned; also decreases during learning)
        # - data_meas_var: data measurement variance
        # - compute_loss_for: parameters that have a prior (useful if we want to take away the priors on some or all params)
        # - learn_num_met_clusters: Whether or not to learn the number of metabolite clusters
        # - learn_num_bug_clusters: Whether or not to learn the number of microbe clusters
        # - linear: Whether or not to learn a linear model or a non-linear model
        # - l1: Whether to have L1 regularziation (only used if linear=0/False; in trials with generated data, having l1 = True seemed to help)
        # - p_nn: The number of additive neural networks per each metabolite cluster interaction; if p_nn = 1,
        #           there is one network per each interaction between a metabolite cluster and a microbial cluster
        # - sample_a: Whether or not to sample a binary value for alpha (ensures the model is binary, but more difficult to learn)
        # - sample_w: Whether or not to sample a binary value for omega (only used if omega is learned)
        # - met_class: List of metabolite taxonomic categories if available (used for initialization and setting priors); when
        #           calling run_learner() in main, the user can specify which level of microbial taxonomy to use here (i.e. you can
        #           use super-classes, subclasses, etc)
        # - bug_class: List of microbial taxonomic categories if available (used for initialization and setting priors); when
        #           calling run_learner() in main, the user can specify which level of microbial taxonomy to use here (i.e. you can
        #           use families, genus, species, etc)
        # - gmm: 0 or 1, and indicates whether or not to make the model independent of microbial values (i.e. a gaussian mixture model) for debugging purposes


        # gmm is 0 or 1, and indicates whether or not to make the model independent of microbial values (i.e. a gaussian mixture model) for debugging purposes
        self.gmm = gmm
        # parameters to compute the loss for (default is all parameters, but can set to ignore priors of some parameters)
        self.compute_loss_for = compute_loss_for
        # class for loss function
        self.MAPloss = MAPloss(self)
        # metabolite locations (a^{metab})
        self.met_locs = met_locs
        # microbe locations (a^{taxa})
        self.microbe_locs = microbe_locs
        # metabolite levels or bug levels to inform priors
        self.met_class = met_class
        self.bug_class = bug_class
        # embedding dimension (d)
        if self.microbe_locs is None:
            if 'r_bug' in self.compute_loss_for:
                self.compute_loss_for.remove('r_bug')
            if 'mu_bug' in self.compute_loss_for:
                self.compute_loss_for.remove('mu_bug')
        if self.met_locs is None:
            if 'r_met' in self.compute_loss_for:
                self.compute_loss_for.remove('r_met')
            if 'mu_met' in self.compute_loss_for:
                self.compute_loss_for.remove('mu_met')
        if met_locs is not None:
            self.met_embedding_dim = met_locs.shape[1]
        else:
            self.met_embedding_dim = None
        if microbe_locs is not None:
            self.bug_embedding_dim = microbe_locs.shape[1]
        else:
            self.bug_embedding_dim = None
        # tau_{\alpha}
        self.alpha_temp = alpha_temp
        # tau_{\omega}
        self.omega_temp = omega_temp
        # whether or not to learn the NUMBER of metabolite or bug clusters
        self.learn_num_met_clusters = learn_num_met_clusters
        self.learn_num_bug_clusters = learn_num_bug_clusters
        # whether or not to run the linear model vs the generalized NAM
        self.linear = linear
        # number of metabolites
        self.N_met = N_met
        # number of microbes
        self.N_bug = N_bug
        # whether or not to perform l1 regularization on the NAM model parameters (only if non-linear)
        self.l1 = l1
        # whether or not to sample alpha in the forward pass
        self.sample_a = sample_a
        self.sample_w = sample_w
        # number of neural networks per metabolite cluster (default = 1)
        self.p_nn = p_nn
        # number of microbe and metabolite clusters
        self.L, self.K = L, K

        # If we learn the number of microbe/metabolite clusters and met_class is not None and bug_class is not None,
        # set the number of metabolomic clusters to the number of unique metabolite classes and the number of microbes
        # to the number of unique microbial classes
        # Otherwise, if met_class is none or bug_class is None, set to 1/3 times the number of features
        if self.learn_num_met_clusters:
            self.K = len(np.unique(self.met_class.values))
        if self.learn_num_bug_clusters:
            self.L = len(np.unique(self.bug_class.values))
        # the location parameter for the binaryConcrete distribution on alpha
        self.alpha_loc = 1 / (self.L * self.K)

        # define the NAM model if not linear
        # This NAM is a set of neural networks for each metab-microbe cluster interaction. Each neural network has
        # 2 hidden layers with 32 nodes in the first layer and 16 nodes in the second layer
        if not self.linear:
            self.NAM = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Sequential(
                nn.Linear(1, 16, bias = True),
                nn.ELU(),
                nn.Linear(16, 16, bias = True),
                nn.ELU(),
                nn.Linear(16, 8, bias = True),
                nn.ELU(),
                nn.Linear(8,1, bias = True)
            ) for p in np.arange(self.p_nn)]) for l in np.arange(self.L)]) for k in np.arange(self.K)])


        # Set the prior distributions parameters for each model parameter (to be called in MAP_loss.py when we find the
        # loss of each parameter according to its prior)
        # self.params is a dictionary of hyperparameters for each model parameter
        self.params = {}
        # self.distributions is a dictionary of each parameter's prior distribution
        self.distributions = {}
        # If we have metabolite locations, we set a prior distribution for r_met and mu_met
        if self.met_locs is not None:
            # If we have metabolite categories (from taxonomy or elsewhere), we set the prior for r_met to be centered
            # around the average embedded size of the given metabolite categories
            if self.met_class is not None:
                radii = []
                means = []
                locss = []
                gps = []
                for bc in np.unique(self.met_class.values):
                    ixs = np.where(self.met_class.values == bc)[0]
                    locs = self.met_locs[ixs,:]
                    radii.append(np.sqrt(np.sum((np.max(locs,0) - np.min(locs,0))**2)) / 2)
                    means.append(np.mean(locs,0))
                    locss.append(locs)
                    gps.append(bc)
                loc = np.mean(radii)
            # Otheriwse, we set r_met to be the size of the embedding space divided by K
            else:
                scale = np.sqrt(np.sum((np.max(self.met_locs,0) - np.min(self.met_locs,0))**2))
                loc = scale / self.K
            # r_met is parameterized by a scale-inv-chi-squared distribution with DOF = 0.1 and tau^{2} = loc
            # We re-parameterize to be able to use a Gamma dist since pytorch doesn't have an inv-gamma or inv-chi-squared
            v = 0.1
            tau2 = loc
            self.params['r_met'] = {'loc': loc, 'scale': (tau2*v)/2, 'dof': v/2}
            self.distributions['r_met'] = Gamma(rate=self.params['r_met']['scale'],
                                                      concentration=self.params['r_met']['dof'])
            # We define the prior for mu_met to be a multivariate normal with mean 0 and variance 100 (since the input
            # locations are already scaled to have variance 1)
            self.params['mu_met'] = {'mean': 0, 'var': 100}
            self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.met_embedding_dim),
                                                              (self.params['mu_met']['var']) * torch.eye(
                                                                  self.met_embedding_dim))

        # If we have input microbial locations, we do the same procedure as for input metabolic locations described above
        if self.microbe_locs is not None:
            if self.bug_class is not None:
                radii = []
                means =[]
                locss = []
                gps = []
                for bc in np.unique(self.bug_class.values):
                    ixs = np.where(self.bug_class.values == bc)[0]
                    locs = self.microbe_locs[ixs,:]
                    radii.append(np.sqrt(np.sum((np.max(locs,0) - np.min(locs,0))**2)) / 2)
                    means.append(np.mean(locs,0))
                    locss.append(locs)
                    gps.append(bc)
                loc = np.mean(radii)
            else:
                scale = np.sqrt(np.sum((np.max(self.microbe_locs,0) - np.min(self.microbe_locs,0))**2))
                loc = scale / self.L
            v = 0.1
            tau2 = loc
            self.params['r_bug'] = {'scale': (tau2*v)/2, 'dof': v/2}
            self.distributions['r_bug'] = Gamma(rate=self.params['r_bug']['scale'],
                                                      concentration=self.params['r_bug']['dof'])
            self.params['mu_bug'] = {'mean': 0, 'var': 100}
            self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.bug_embedding_dim), (self.params['mu_bug']['var'])*torch.eye(self.bug_embedding_dim))

        # If we don't have input microbial or metabolic locations, we need to learn omega/w, so we need to set a prior for w
        else:
            # We set a binary concrete prior with location = 0.1 and tau = self.omega_temp (as a reminder, this
            # parameter decreases throughout learning and is reset each epoch in run_learner() below)
            self.w_loc = (0.1*self.N_bug)/self.N_bug
            self.params['w'] = {'loc': self.w_loc, 'temp': self.omega_temp}
            self.distributions['w'] = BinaryConcrete(self.params['w']['loc'], self.omega_temp)
        # beta is parameterized by a normal dist with mean=0, var = 1000
        self.params['beta'] = {'mean': 0, 'scale': np.sqrt(1000)}
        self.distributions['beta'] = Normal(self.params['beta']['mean'], self.params['beta']['scale'])
        # alpha is parameterized by a binary concrete with loc=1/(K*L) (defined above) and self.alpha_temp, which
        # decreases throughout learning like self.omega_temp
        self.params['alpha'] = {'loc': self.alpha_loc, 'temp':self.alpha_temp}
        self.distributions['alpha'] = BinaryConcrete(self.params['alpha']['loc'], self.params['alpha']['temp'])
        # e_met is only learned (and thus the prior only used) if we are learning the number of metabolite clusters;
        # we set dof=10 and scale = 10*K according to the recommendations in Malsiner-Walli et al
        self.params['e_met'] = {'dof': 10, 'scale': 10*self.K} # based on Malsiner-Walli
        self.distributions['e_met'] = Gamma(self.params['e_met']['dof'], self.params['e_met']['scale'])
        # pi_met is set to a dirichlet prior with epsilon = 1/K; if we learn the number of metabolite clusters,
        # we use epsilon=e_met instead of epsilon=1/K
        self.params['pi_met'] = {'epsilon': [1/self.K]*self.K}
        self.distributions['pi_met'] = Dirichlet(torch.Tensor(self.params['pi_met']['epsilon']))
        tau2 = data_meas_var
        v = 0.1
        self.params['sigma'] = {'scale': (tau2 * v) / 2, 'dof': v/2}
        self.distributions['sigma'] = Gamma(rate=self.params['sigma']['scale'],
                                            concentration=self.params['sigma']['dof'])

        # For each model parameter, sample from the prior to get the parameter range, for adjusting the learning rate
        # based on parameter size
        # Also, get the expected range of the parameter for plotting purposes (the expected parameter range is use when
        # plotting parameter traces that are nicer and easier to interpret / compare between different model runs)
        self.range_dict = {} # stores ranges for plotting purposes
        self.lr_range = {} # stores ranges for adjusting the learning rate based on parameter size
        for param, dist in self.distributions.items():
            sampler = dist.sample([1000])
            # To find the expected range of each parameter, I sample from the prior. However, all of the parameters that
            # have priors with bounded support (i.e. sigma, r_met, r_bug, w, z, alpha, pi_met, and e_met)
            # have to be transformed to be learned in an unconstrained form.
            # The learning rates should be set based on the size of the unconstrained parameter (self.lr_range), while the
            # ranges for plotting are based on the constrained parameter (for increased interpretability of the parameter
            # trace plots)

            # Sigma, r_met, and r_bug are even more special because they are parameterized by an inverse Gamma, but
            # pytorch only has a Gamma distribution. To get the expected size of the un-constrained and learned parameter,
            # we sample from the Gamma dist but then take the log of the inverse
            if 'sigma' in param or 'r_met' in param or 'r_bug' in param:
                vals = np.log(1/self.distributions[param].sample([1000]))
                self.range_dict[param] = (-0.1, np.exp(vals.max()))
                self.lr_range[param] = torch.abs((torch.mean(vals) + torch.std(vals)) - (torch.mean(vals) - torch.std(vals)))
            elif 'w' in param or 'z' in param or 'alpha' in param:
                self.range_dict[param] = (-0.1,1.1)
                self.lr_range[param] = np.abs(2*(torch.log(torch.tensor(self.params[param]['loc']).float())/self.params[param]['temp']))
            elif param == 'pi_met' or param == 'e_met':
                vals = torch.log(sampler)
                self.lr_range[param] = torch.abs((torch.mean(vals) + torch.std(vals)) - (torch.mean(vals) - torch.std(vals)))
                range = sampler.max() - sampler.min()
                self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
            else:
                vals = dist.sample([1000])
                range = sampler.max() - sampler.min()
                self.range_dict[param] = (sampler.min() - range * 0.1, sampler.max() + range * 0.1)
                self.lr_range[param] = torch.abs((torch.mean(vals) + torch.std(vals)) - (torch.mean(vals) - torch.std(vals)))
        self.range_dict['beta'] = (-20,20)
        self.range_dict['beta[1:,:]*alpha'] = self.range_dict['beta']
        self.range_dict['z'] = (-0.1,1.1)


    # Initialize parameter values
    def initialize(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        # self.sigma is the measurement variance, initialized to be the log of the measurement variance (since we will
        # transform it to exp(self.sigma))
        self.sigma = nn.Parameter(torch.log(torch.tensor(self.params['sigma']['scale']).float()), requires_grad=True) # measurement variance
        if self.linear:
            # regression weights beta
            temp = Normal(0,1).sample([self.L+1, self.K])
            temp[0,:] = Normal(0,0.000001).sample([self.K])
            self.beta = nn.Parameter(temp, requires_grad=True)
            # self.beta = nn.Parameter(torch.ones(self.L+1, self.K), requires_grad=True)
        else:
            self.beta = nn.Parameter(Normal(0,1).sample([self.K]), requires_grad=True)
        # self.alpha is the un-transformed parameter that is learned by the model; self.alpha_act is the parameter constrained to be
        # between 0 and 1 that is seen in the model equations
        self.alpha = nn.Parameter(Normal(0,1).sample([self.L, self.K]), requires_grad=True)
        self.alpha_act = torch.sigmoid(self.alpha)
        # self.alpha = nn.Parameter(10*torch.ones(self.L, self.K), requires_grad=False)
        # self.alpha_act = torch.sigmoid(self.alpha)

        # Initialize the microbe cluster means and radii
        if self.microbe_locs is not None:
            kmeans = KMeans(n_clusters=self.L, random_state=self.seed).fit(self.microbe_locs)
            self.mu_bug = nn.Parameter(torch.Tensor(kmeans.cluster_centers_), requires_grad=True)
            r = []
            for clust in np.arange(self.L):
                cdist = scipy.spatial.distance.cdist(self.microbe_locs[kmeans.labels_==clust,:],
                                                     kmeans.cluster_centers_[clust:clust+1,:])
                r.append(np.max(cdist.squeeze()))
            eps = 0.01*np.max(r)
            self.r_bug = nn.Parameter(torch.Tensor(1.5*np.array(r + eps)), requires_grad=True)
            # self.mu_bug = nn.Parameter(MultivariateNormal(torch.zeros(self.bug_embedding_dim), 0.1*torch.eye(self.bug_embedding_dim)).sample([self.L]), requires_grad=True)
            # temp = np.sqrt(np.sum((np.max(self.microbe_locs,0) - np.min(self.microbe_locs,0))**2)) / 2
            # self.r_bug = nn.Parameter(torch.log(1.2*temp*torch.ones(self.L)), requires_grad=True)
            kappa = torch.stack([((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1) for m in
                                 range(self.microbe_locs.shape[0])])
            self.w_act = torch.sigmoid((self.r_bug - kappa))
        else:
            self.w = nn.Parameter(Normal(0,1).sample([self.N_bug, self.L]), requires_grad=True)
            self.w_act = torch.sigmoid(self.w / self.omega_temp)

        # Initialize the metabolite cluster means and radii
        if self.met_locs is not None:
            if self.met_class is not None:
                kmeans = KMeans(n_clusters=self.K, random_state=self.seed).fit(self.met_locs)
                self.mu_met = nn.Parameter(torch.Tensor(kmeans.cluster_centers_), requires_grad=True)
                r = []
                for clust in np.arange(self.K):
                    cdist = scipy.spatial.distance.cdist(self.met_locs[kmeans.labels_ == clust, :],
                                                         kmeans.cluster_centers_[clust:clust + 1, :])
                    r.append(np.max(cdist.squeeze()))
                eps = 0.01 * np.max(r)
                self.r_met = nn.Parameter(torch.Tensor(1.5 * np.array(r + eps)), requires_grad=True)
            else:
                ix = np.random.choice(range(self.met_locs.shape[0]), self.L, replace=False)
                self.mu_met = nn.Parameter(torch.Tensor(self.met_locs[ix, :]), requires_grad=True)
                r_temp = self.params['r_met']['loc'] * torch.ones((self.L)).squeeze()
                self.r_met = nn.Parameter(torch.log(r_temp), requires_grad=True)

        # initialize cluster weights
        pi_init = (1 / self.K) * torch.ones(self.K)
        if self.learn_num_met_clusters:
            self.e_met = nn.Parameter(torch.log(pi_init.unsqueeze(0)), requires_grad=True)
            self.pi_met = nn.Parameter(torch.log(Dirichlet(pi_init.unsqueeze(0)).sample()), requires_grad=True)
        else:
            self.e_met = torch.log(pi_init.unsqueeze(0))
            self.pi_met = nn.Parameter(torch.log(pi_init.unsqueeze(0)), requires_grad=True)

        # Initialize z and w (these initialitions arer not used in model, but the parameters are tracked along with the
        # others so this is the easiest way to initialize)
        self.z_act = Bernoulli(self.K/self.N_met).sample([self.N_met, self.K])

    # computing l1 loss
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    # Forward function, contains all the model equations except priors (those are in self.MAPlos.compute_loss())
    def forward(self, x, y):
        # This is just to keep alpha from getting to close to 0 or 1 and causing numerical issues
        omega_epsilon = self.omega_temp / 4
        alpha_epsilon = self.alpha_temp / 4
        if self.microbe_locs is not None:
            kappa = torch.stack(
                [torch.sqrt(((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1)) for m in
                 np.arange(self.microbe_locs.shape[0])])
            self.w_act = torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.omega_temp)
        else:
            if not self.sample_w:
                self.w_act = (1-2*omega_epsilon)*torch.sigmoid(self.w/self.omega_temp) + omega_epsilon
            else:
                self.w_soft, self.w_act = gumbel_sigmoid(self.w, self.omega_temp, omega_epsilon)
        # self.w_act = torch.ones(self.w_act.shape)
        g = x@self.w_act.float()
        # if sampling alpha in the forward pass
        if not self.sample_a:
            self.alpha_act = (1-2*alpha_epsilon)*torch.sigmoid(self.alpha/self.alpha_temp) + alpha_epsilon
        else:
            self.alpha_soft, self.alpha_act = gumbel_sigmoid(self.alpha, self.alpha_temp, alpha_epsilon)
        if self.linear:
            # out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act) + Normal(0,torch.sqrt(torch.exp(self.sigma))).sample([g.shape[0], self.K])
            if self.gmm:
                out_clusters = self.beta[0, :]+ Normal(0,torch.sqrt(torch.exp(self.sigma))).sample([g.shape[0], self.K])
            else:
                out_clusters = self.beta[0,:] + torch.matmul(g, self.beta[1:,:]*self.alpha_act) + Normal(0,torch.sqrt(torch.exp(self.sigma))).sample([g.shape[0], self.K])

        # out_clusters = torch.matmul(g, self.beta[1:, :] * self.alpha_act)
        else:
            out_clusters = self.beta + torch.cat([torch.cat([self.alpha_act[l,k]*torch.stack([
                self.NAM[k][l][p](g[:,l:l+1]) for p in np.arange(self.p_nn)],-1).sum(-1)
                                                  for l in np.arange(self.L)],1).sum(1).unsqueeze(1)
                                                  for k in np.arange(self.K)],1) + Normal(
                0,torch.sqrt(torch.exp(self.sigma))).sample([g.shape[0], self.K])
        # compute loss via the priors
        loss = self.MAPloss.compute_loss(out_clusters,y)

        # add l1 regularization to loss
        if not self.linear and self.l1:
            l1_parameters = []
            for parameter in self.NAM.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = self.compute_l1_loss(torch.cat(l1_parameters))
            loss += l1
        return out_clusters, loss


# runs model for given input arguments
def run_learner(args, device, x=None, y=None, a_met=None, a_bug = None, base_path = '', plot_params = True,
                met_class = None, bug_class = None):
    if args.linear == 1:
        args.l1 = 0
    if x is not None and y is not None:
        metabs = y.columns.values
        seqs = x.columns.values
        args.learn = 'all'
        # set path for saving results
        path = base_path + '/outputs/'
    else:
        path = base_path + '/outputs_gen/'
        metabs = None
        seqs = None
    if not os.path.isdir(path):
        os.mkdir(path)

    if 'none' in args.priors:
        args.priors = []

    params2learn = args.learn
    priors2set = args.priors
    # set path for saving results
    path = path + args.case.replace(' ','_')
    if not os.path.isdir(path):
        os.mkdir(path)

    if 'all' in priors2set:
        priors2set = ['alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_met','e_met','sigma']
        if a_bug is None:
            priors2set.append('w')
    if 'all' in params2learn:
        params2learn = ['alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_met','e_met','sigma']
    if a_bug is None and 'w' not in params2learn:
        params2learn.append('w')
    # fix parameters specified in args.fix, and don't learn these parameters
    if args.fix and x is None and y is None:
        for p in args.fix:
            if p in priors2set:
                priors2set.remove(p)
            if p in params2learn:
                params2learn.remove(p)

    # if we fix some parameters, add another folder to path
    if 'all' not in args.learn or 'all' not in args.priors or (args.fix and x is None and y is None):
        path = path + '/learn_' + '_'.join(params2learn) + '-priors_' + '_'.join(priors2set) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)
    if 'sigma' in args.fix:
        params2learn.remove('sigma')
        priors2set.remove('sigma')
        path = path + '/fix-sigma/'
        if not os.path.isdir(path):
            os.mkdir(path)


    # add all other specified inputs to path to prevent overwriting results
    info = '-lr' + str(args.lr) + '-linear'*(args.linear) + '-adj_lr'*args.adjust_lr + '-hard'*args.hard + \
           '-l1'*(args.l1) + '-'*(1-args.linear) +args.nltype*(1-args.linear) + '-lm'*args.lm + '-lb'*args.lb + \
            '-meas_var' + str(args.meas_var).replace('.', '_') +  '-Nmet' + str(args.N_met) + '-Nbug' + str(args.N_bug) + \
           '-L' + str(args.L) + '-K' + str(args.K) + '-gmm'*args.gmm + \
           '-atau' + str(args.a_tau).replace('.','_') + '-wtau' + str(args.w_tau).replace('.', '_')

    path = path + '/' + info + '/'
    if not os.path.isdir(path):
        os.mkdir(path)

    # Generate data by calling data_gen.py and then plot
    if x is None and y is None:
        x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
        mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
            N_met = args.N_met, N_bug = args.N_bug, N_met_clusters = 1,
            N_bug_clusters = args.L,meas_var = args.meas_var,
            repeat_clusters= args.rep_clust, N_samples=args.N_samples, linear = args.linear,
            nl_type = args.nltype, dist_var_frac=args.dist_var_perc, embedding_dim=args.dim)
        if not args.linear:
            gen_beta = gen_beta[0,:]
        try:
            plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                          r_bug, mu_met, r_met, gen_u, gen_alpha, gen_beta)
        except:
            print('no plots of gen data')

        if ylocs is None:
            a_met = None
        else:
            a_met = gen_met_locs

        if xlocs is None:
            a_bug = None
        else:
            a_bug = gen_bug_locs

        # get true values from data_gen.py to compare to learned parameter values
        if args.lm:
            gen_z = np.hstack((gen_z, np.zeros((args.N_met, args.N_met - 1 - args.K))))
            if ylocs is not None:
                mu_met = np.vstack((mu_met, np.zeros((args.N_met - args.K - 1, mu_met.shape[1]))))
                r_met = np.append(r_met, np.zeros(args.N_met - 1 - len(r_met)))
            if args.linear:
                gen_beta = np.hstack((gen_beta, np.zeros((gen_beta.shape[0], args.N_met - args.K - 1))))
            gen_alpha = np.hstack((gen_alpha, np.zeros((gen_alpha.shape[0], args.N_met - args.K - 1))))
        if args.lb:
            if xlocs is not None:
                r_bug = np.append(r_bug, np.zeros(args.N_bug - 1 - len(r_bug)))
                mu_bug = np.vstack((mu_bug, np.zeros((args.N_bug - args.L - 1, mu_bug.shape[1]))))
            gen_w = np.hstack((gen_w, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
            gen_u = np.hstack((gen_u, np.zeros((args.N_bug, args.N_bug - 1 - args.L))))
            if args.linear:
                gen_beta = np.vstack((gen_beta, np.zeros((args.N_bug - args.L - 1, gen_beta.shape[1]))))
            gen_alpha = np.vstack((gen_alpha, np.zeros((args.N_bug - args.L - 1, gen_alpha.shape[1]))))
        true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': mu_bug,
                     'mu_met': mu_met, 'u': gen_u,'w_soft': gen_w,'r_bug':1.2*r_bug, 'r_met': 1.2*r_met, 'z': gen_z,
                     'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                     'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0), 'bug_locs': gen_bug_locs,
                     'met_locs':gen_met_locs,
                     'e_met': np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),'b': mu_met, 'sigma': args.meas_var}
        # just for plotting
        if args.linear:
            true_vals['beta[1:,:]*alpha'] = gen_beta[1:,:]*sigmoid(gen_alpha)
    else:
        true_vals = None

    # Define model and initialize with input seed
    net = Model(a_met, a_bug, K=args.K, L=args.L,
                compute_loss_for=priors2set, N_met = y.shape[1], N_bug = x.shape[1],
                learn_num_bug_clusters=args.lb,learn_num_met_clusters=args.lm, linear = args.linear==1,
                p_nn = args.p_num, data_meas_var = args.meas_var, met_class = met_class, bug_class = bug_class,
                sample_w = args.hard, sample_a=args.hard, gmm = args.gmm)
    net.initialize(args.seed)
    net.to(device)

    # setattr(net, 'w', nn.Parameter(torch.zeros(net.w.shape), requires_grad=False))

    # plot prior distributions for all parameters
    for param, dist in net.distributions.items():
        parameter_dict = net.params[param]
        try:
            plot_distribution(dist, param, true_val = true_vals, ptype = 'prior', path = path, **parameter_dict)
        except:
            print(param + ' plot distribution error!!')

    # Set tau schedules for alpha and omega given inputs
    alpha_tau_logspace = np.logspace(args.a_tau[0], args.a_tau[1], args.iterations)
    omega_tau_logspace = np.logspace(args.w_tau[0], args.w_tau[1], args.iterations)
    net.alpha_temp = alpha_tau_logspace[0]
    net.omega_temp = omega_tau_logspace[0]

    # Record initial parameter values (we will also record per epoch for plotting purposes)
    param_dict = {}
    param_dict[args.seed] = {}
    start = 0
    for name, parameter in net.named_parameters():
        if 'NAM' in name or 'lambda_mu' in name or name=='b' or name == 'C':
            continue
        if name not in params2learn:
            if true_vals is not None:
                if name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p' or name == 'pi_met':
                    setattr(net, name, nn.Parameter(torch.tensor(np.log(true_vals[name])).float(), requires_grad=False))
                else:
                    setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
            elif name == 'sigma':
                setattr(net, name, nn.Parameter(torch.tensor(np.log(args.meas_var)).float(), requires_grad=False))
        if name == 'z' or name == 'alpha' or name == 'w':
            parameter = getattr(net, name + '_act')
        if name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p':
            parameter = np.exp(parameter.clone().detach().numpy())
        if name == 'pi_met':
            parameter = torch.softmax(parameter.clone().detach(),1).numpy()
        if torch.is_tensor(parameter):
            param_dict[args.seed][name] = [parameter.clone().detach().numpy()]
        else:
            param_dict[args.seed][name] = [parameter]
    param_dict[args.seed]['z'] = [net.z_act.clone().numpy()]
    if 'w' not in param_dict[args.seed].keys():
        param_dict[args.seed]['w'] = [net.w_act.clone().detach().numpy()]
    if net.linear:
        param_dict[args.seed]['beta[1:,:]*alpha'] = [net.beta[1:,:].clone().detach().numpy()*net.alpha_act.clone().detach().numpy()]

    if not os.path.isdir(path + '/init_clusters/'):
        os.mkdir(path + '/init_clusters/')
    best_z = param_dict[args.seed]['z'][0]
    best_w = np.round(param_dict[args.seed]['w'][0])
    best_alpha = np.round(param_dict[args.seed]['alpha'][0])
    if args.linear == 1:
        get_interactions_csv(path, 0, param_dict, args.seed)
    active_asv_clust = list(set(np.where(np.sum(best_w, 0) != 0)[0]).intersection(
        set(np.where(np.sum(best_alpha, 1) != 0)[0])))
    active_met_clust = np.where(np.sum(best_z, 0) != 0)[0]
    for asv_clust in active_asv_clust:
        asv_ix = np.where(best_w[:, asv_clust] != 0)[0]
        if seqs is not None:
            asv_ix = seqs[asv_ix]
        if not isinstance(asv_ix[0], str):
            asv_ix = [str(a) for a in asv_ix]
        inputs = ["python3", "tree_plotter.py", "-fun", 'asv', "-name", 'ASV_cluster_' + str(asv_clust) + '_tree_init.pdf',
                  "-out", path + '/init_clusters/', "-newick", base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
                  "-feat"]
        inputs.extend(asv_ix)
        print(inputs)
        subprocess.run(inputs, cwd=base_path + "/ete_tree")
    for met_clust in active_met_clust:
        met_ix = np.where(best_z[:, met_clust] != 0)[0]
        if metabs is not None:
            met_ix = metabs[met_ix]
        if not isinstance(met_ix[0], str):
            met_ix = [str(a) for a in met_ix]
        inputs = ["python3", "tree_plotter.py", "-fun", 'metab', "-name", 'Met_cluster_' + str(met_clust) + '_tree_init.pdf',
                  "-out", path + '/init_clusters/', "-newick", base_path + '/ete_tree/w1_newick_tree.nhx', "-feat"]
        inputs.extend(met_ix)
        subprocess.run(inputs, cwd=base_path + "/ete_tree")

    if a_met is not None and args.xdim == 2 and args.ydim == 2:
        plot_output_locations(path, net, 0, param_dict[args.seed], args.seed, plot_zeros=1)
    loss_vec = []
    train_out_vec = []
    lr_dict = {}
    matching_dict = {}

    # Adjust each parameter's learning rate based on parameter size
    lr_list = []
    ii = 0
    for name, parameter in net.named_parameters():
        if name in params2learn or 'all' in params2learn or 'NAM' in name:
            if name not in net.lr_range.keys():
                range = np.abs(np.max(parameter.detach().view(-1).numpy()) - np.min(parameter.detach().view(-1).numpy()))
            else:
                range = net.lr_range[name]
            matching_dict[name] = ii
            ii+= 1
            if args.adjust_lr:
                lr_list.append({'params': parameter, 'lr': (args.lr / net.lr_range['beta']) * range})
            else:
                lr_list.append({'params': parameter})
            lr_dict[name] = [(args.lr / net.lr_range['beta'].item()) * range.item()]
        # initialize optimizer
    optimizer = optim.RMSprop(lr_list, lr=args.lr)
    if args.adjust_lr:
        pd.Series(net.lr_range).to_csv(path + 'param_estimated_sizes.csv')
        pd.DataFrame(lr_dict).T.to_csv(path + 'per_param_lr.csv')

    # If args.load == 1, load previously trained model and re-start training at the last saved epoch
    epochs = re.findall('epoch\d+', ' '.join(os.listdir(path)))
    path_orig = path
    if len(epochs)>0:
        if os.path.isfile(path_orig + 'seed' + str(args.seed) + '.txt'):
            with open(path_orig + 'seed' + str(args.seed) + '.txt', 'r') as f:
                largest = int(f.readlines()[0])
        else:
            largest = max([int(num.split('epoch')[-1]) for num in epochs])
        foldername = path + 'epoch' + str(largest) + '/'
        if 'seed' + str(args.seed) + '_checkpoint.tar' in os.listdir(foldername) and args.load==1:
            checkpoint = torch.load(foldername + 'seed' + str(args.seed) + '_checkpoint.tar')
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = int(checkpoint['epoch'] - 1)
            ix = int(checkpoint['epoch'])-1000
            if ix >= len(alpha_tau_logspace):
                ix = -1
            if ix > -1:
                net.alpha_temp = alpha_tau_logspace[ix]
                net.omega_temp = omega_tau_logspace[ix]
            else:
                net.alpha_temp = alpha_tau_logspace[0]
                net.omega_temp = omega_tau_logspace[0]
            if args.iterations-1 <= start:
                print('training complete')
                sys.exit()
            print('model loaded')
        else:
            print('no model loaded')
    else:
        print('no model loaded')

    # plot initialized cluster locations & means
    if not os.path.isdir(path + '/epoch0'):
        os.mkdir(path + '/epoch0')
    # plot_syn_data(path + '/epoch0/seed' + str(args.seed), x, y, gen_z, gen_bug_locs, gen_met_locs, net.mu_bug.detach().numpy(),
    #               torch.exp(net.r_bug.detach()).numpy(), net.mu_met.detach().numpy(), torch.exp(net.r_met.detach()).numpy(),
    #               gen_w)

    # Train model over the number of specified input iterations in args.iterations
    x = torch.Tensor(np.array(x)).to(device)
    loss_dict_vec = {}
    ix = 0
    stime = time.time()
    last_epoch = 0
    for epoch in np.arange(start, args.iterations):
        net.alpha_temp = alpha_tau_logspace[ix]
        net.omega_temp = omega_tau_logspace[ix]
        optimizer.zero_grad()
        cluster_outputs, loss = net(x, torch.Tensor(np.array(y)))
        train_out_vec.append(cluster_outputs)
        try:
            loss.backward()
            loss_vec.append(loss.item())
            for param in net.MAPloss.loss_dict:
                if param not in loss_dict_vec.keys():
                    loss_dict_vec[param] = [net.MAPloss.loss_dict[param].detach().item()]
                else:
                    loss_dict_vec[param].append(net.MAPloss.loss_dict[param].detach().item())
            optimizer.step()
            last_epoch = args.iterations-1
        except:
            last_epoch = epoch - 1

        # keep track of updated parameter values
        for name, parameter in net.named_parameters():
            if 'NAM' in name or 'lambda_mu' in name or name=='b' or name == 'C':
                continue
            if name == 'z' or name == 'alpha' or name == 'w':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p':
                parameter = np.exp(parameter.clone().detach().numpy())
            elif name == 'pi_met':
                parameter = torch.softmax(parameter.clone().detach(), 1).numpy()
            if torch.is_tensor(parameter):
                param_dict[args.seed][name].append(parameter.clone().detach().numpy())
            else:
                param_dict[args.seed][name].append(parameter)
        if 'w' not in net.named_parameters():
            param_dict[args.seed]['w'].append(net.w_act.clone().detach().numpy())
        param_dict[args.seed]['z'].append(net.z_act.clone().numpy())
        if net.linear:
            param_dict[args.seed]['beta[1:,:]*alpha'].append(
                net.beta[1:, :].clone().detach().numpy() * net.alpha_act.clone().detach().numpy())

        # if epoch % 100 == 0:
        #     temp = torch.softmax(net.pi_met, 1)
        #     fig, ax = plt.subplots(net.K, 1, figsize = (8, 8*net.K))
        #     for ixx in np.arange(net.K):
        #         ax[ixx].hist(np.repeat(cluster_outputs[:,ixx].detach().numpy(), y.shape[1]), bins = 20, label = 'Predicted')
        #         ax[ixx].hist(np.array(y).flatten(), bins = 20, label = 'True')
        #         if torch.sum(net.z_act[:,ixx]) != 0:
        #             ax[ixx].set_title('Cluster ' + str(ixx) + ' ON, pi=' + str(temp[0][ixx]))
        #         else:
        #             ax[ixx].set_title('Cluster ' + str(ixx) + ', pi=' + str(temp[0][ixx]))
        #         ax[ixx].legend()
        #     fig.tight_layout()
        #     fig.savefig(path + str(epoch) + 'cluster_dist.pdf')
        #     plt.close(fig)

        # if epoch % 100 == 0 and epoch > 0:
        #     fig = plot_predictions(cluster_outputs, torch.Tensor(np.array(y)), param_dict[args.seed]['z'][-1])
        #     fig.savefig(path + str(args.seed) + '-per_clust_predictions.pdf')
        if epoch % 5000 == 0:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
            save_dict = {'model_state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'epoch': epoch}
            torch.save(save_dict,
                       path + 'seed' + str(args.seed) + '_checkpoint.tar')

        # at the last epoch, plot results
        if epoch == last_epoch or epoch % 5000 == 0:
                # or epoch%10000==0:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))
            if 'epoch' not in path:
                path = path + 'epoch' + str(epoch) + '/'
            else:
                path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            if net.met_embedding_dim is not None and net.met_embedding_dim == 2 and net.bug_embedding_dim==2:
                plot_output_locations(path, net, -1, param_dict[args.seed], args.seed,
                                      type='best_train', plot_zeros=False)
                plot_output_locations(path, net, -1, param_dict[args.seed], args.seed,
                                      type='best_train', plot_zeros=True)
            print('Epoch ' + str(epoch))

            if epoch >= 1:
                if 'epoch' not in path:
                    path = path + 'epoch' + str(epoch) + '/'
                else:
                    path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
                if not os.path.isdir(path):
                    os.mkdir(path)
                if not os.path.isdir(path + 'seed' + str(args.seed) + '-clusters/'):
                    os.mkdir(path + 'seed' + str(args.seed) + '-clusters/')
                best_mod = np.argmin(loss_vec)
                best_z = param_dict[args.seed]['z'][best_mod]
                best_w = np.round(param_dict[args.seed]['w'][best_mod])
                best_alpha = np.round(param_dict[args.seed]['alpha'][best_mod])

                # plot_cluster_outputs_vs_met_value(best_z, y, cluster_outputs, path, seed = args.seed)
                if args.linear == 1:
                    get_interactions_csv(path, best_mod, param_dict, args.seed)
                if args.hard != 1:
                    pd.DataFrame(param_dict[args.seed]['alpha'][best_mod]).to_csv(path + 'seed' + str(args.seed) + 'alpha.csv')
                    pd.DataFrame(param_dict[args.seed]['w'][best_mod]).to_csv(path + 'seed' + str(args.seed) + 'omega.csv')

                if not args.syn:
                    met_newick_name = 'newick_' + args.yfile.split('.csv')[0] + '.nhx'
                    active_asv_clust = list(set(np.where(np.sum(best_w,0) != 0)[0]).intersection(
                        set(np.where(np.sum(best_alpha,1)!= 0)[0])))
                    active_met_clust = np.where(np.sum(best_z,0) != 0)[0]
                    for asv_clust in active_asv_clust:
                        asv_ix = np.where(best_w[:,asv_clust]!= 0)[0]
                        if seqs is not None:
                            asv_ix = seqs[asv_ix]
                        if not isinstance(asv_ix[0], str):
                            asv_ix = [str(a) for a in asv_ix]
                        inputs = ["python3", "tree_plotter.py", "-fun", 'asv', "-name", 'ASV_cluster_' + str(asv_clust) + '_tree.pdf',
                             "-out", path + 'seed' + str(args.seed) + '-clusters/',
                                  "-newick",base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx', "-feat"]
                        inputs.extend(asv_ix)
                        subprocess.run(inputs,cwd=base_path + "/ete_tree")
                    if len(active_met_clust) > 20:
                        active_met_clust = active_met_clust[:20]
                    for met_clust in active_met_clust:
                        met_ix = np.where(best_z[:, met_clust]!=0)[0]
                        if metabs is not None:
                            met_ix = metabs[met_ix]
                        if not isinstance(met_ix[0], str):
                            met_ix = [str(a) for a in met_ix]
                        inputs = ["python3", "tree_plotter.py", "-fun", 'metab', "-name", 'Met_cluster_' + str(met_clust) + '_tree.pdf',
                             "-out", path+ 'seed' + str(args.seed) + '-clusters/',
                                  "-newick", base_path + '/ete_tree/' + met_newick_name,"-feat"]
                        inputs.extend(met_ix)
                        subprocess.run(inputs,cwd=base_path + "/ete_tree")


                if not os.path.isfile(path + 'Num_Clusters.txt'):
                    with open(path + 'Num_Clusters.txt', 'w') as f:
                        f.writelines('Seed ' + str(args.seed) + ', K: ' + str(len(active_met_clust)) + ', L: ' + str(len(active_asv_clust)) + '\n')
                else:
                    with open(path + 'Num_Clusters.txt', 'a') as f:
                        f.writelines('Seed ' + str(args.seed) + ', K: ' + str(len(active_met_clust)) + ', L: ' + str(len(active_asv_clust)) + '\n')

                if not os.path.isfile(path + 'Loss.txt'):
                    with open(path + 'Loss.txt', 'w') as f:
                        f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec)) + '\n')
                else:
                    with open(path + 'Loss.txt', 'a') as f:
                        f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec))+ '\n')

                # fig = plot_predictions(cluster_outputs, torch.Tensor(np.array(y)), param_dict[args.seed]['z'][best_mod])
                # fig.savefig(path + str(args.seed) + '-predictions.pdf')
                # plt.close(fig)
                try:
                    plot_loss_dict(path, args.seed, loss_dict_vec)
                except:
                    print('no loss dict')
                plot_xvy(path, x, train_out_vec, best_mod, param_dict, args.seed)
                if plot_params and args.load == 0:
                    plot_param_traces(path, param_dict[args.seed], params2learn, true_vals, net, args.seed)
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                fig3, ax3 = plot_loss(fig3, ax3, args.seed, np.arange(len(loss_vec)), loss_vec, lowest_loss=None)
                fig3.tight_layout()
                fig3.savefig(path_orig + 'loss_seed_' + str(args.seed) + '.pdf')
                plt.close(fig3)

                plot_output(path, best_mod, train_out_vec, np.array(y), true_vals, param_dict[args.seed],
                                     args.seed, type = 'best_train', metabs = metabs, meas_var=args.meas_var)

                save_dict = {'model_state_dict':net.state_dict(),
                           'optimizer_state_dict':optimizer.state_dict(),
                           'epoch': epoch}
                torch.save(save_dict,
                           path + 'seed' + str(args.seed) + '_checkpoint.tar')
                if 'beta' in param_dict[args.seed].keys():
                    with open(path + 'seed' + str(args.seed) + '_beta.txt', 'w') as f:
                        f.writelines(str(param_dict[args.seed]['beta'][best_mod]) + '\n')

                if 'beta[1:,:]*alpha' in param_dict[args.seed].keys():
                    with open(path + 'seed' + str(args.seed) + '_beta-alpha.txt', 'w') as f:
                        f.writelines(str(param_dict[args.seed]['beta[1:,:]*alpha'][best_mod]) + '\n')

                with open(path + str(args.seed) + '_param_dict.pkl', 'wb') as f:
                    pkl.dump(param_dict, f)
                with open(path + str(args.seed) + '_loss.pkl', 'wb') as f:
                    pkl.dump(loss_vec, f)

                with open(path_orig + 'seed' + str(args.seed) + '.txt', 'w') as f:
                    f.writelines(str(epoch))

                etime= time.time()
                with open(path_orig + 'seed' + str(args.seed) + '_min_per_epoch.txt', 'w') as f:
                    f.writelines(str(epoch) + ': ' + str(np.round((etime - stime)/60, 3)) + ' minutes')


    etime = time.time()
    print('total time:' + str(etime - stime))
    print('delta loss:' + str(loss_vec[-1] - loss_vec[0]))

if __name__ == "__main__":
    # input arguments
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", "--learn", help="params to learn", type=str, nargs='+', default = 'all')
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default = 0.1)
    parser.add_argument("-fix", "--fix", help="params to fix", type=str, nargs='+', default = 'all')
    parser.add_argument("-priors", "--priors", help="priors to set", type=str, nargs='+', default = '')
    parser.add_argument("-case", "--case", help="case", type=str, default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))
    # parser.add_argument("-case", "--case", help="case", type=str,
    #                     default='mixture_model')
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 30)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 30)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 10)
    parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 10)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.001)
    parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int,default = 1001)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 99)
    parser.add_argument("-load", "--load", help="0 to not load model, 1 to load model", type=int, default = 1)
    parser.add_argument("-rep_clust", "--rep_clust", help = "whether or not bugs are in more than one cluster", default = 0, type = int)
    parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 0)
    parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 0)
    parser.add_argument("-hard", "--hard", help="whether or not to sample alpha and omega in the forward pass", type=int, default=0)
    parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int, default=1000)
    parser.add_argument("-linear", "--linear", type = int, default = 1)
    parser.add_argument("-nltype", "--nltype", type = str, default = "exp")
    parser.add_argument("-adjust_lr", "--adjust_lr", type=int, default=1)
    parser.add_argument("-l1", "--l1", type=int, default=0)
    parser.add_argument("-dist_var_perc", "--dist_var_perc", type=float, default=0.5)
    parser.add_argument("-p_num", "--p_num", type=int, default=1)
    parser.add_argument("-xdim", "--xdim", type=int, default=2)
    parser.add_argument("-ydim", "--ydim", type=int, default=2)
    parser.add_argument("-a_tau", "--a_tau", type=float, nargs = '+', default=[-0.3, -3])
    parser.add_argument("-w_tau", "--w_tau", type=float, nargs='+', default=[-0.3, -3])
    parser.add_argument("-locs","--locs", type = str, default = 'none')
    parser.add_argument("-dtype", "--dtype", type=str, default='')
    parser.add_argument("-dim", "--dim", type=float, default=2)
    parser.add_argument("-syn", "--syn", type=int, default=0)
    parser.add_argument("-yfile", "--yfile", type=str, default='y_lt-one-stand.csv')
    parser.add_argument("-gmm", "--gmm", type=int, default=0)
    args = parser.parse_args()
    print(sys.executable)
    args.case = args.locs + '_' + args.case
    dtype = args.dtype
    base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
    if not os.path.isdir(base_path + '/outputs/' + args.case):
        os.mkdir(base_path + '/outputs/' + args.case)
    gen_data = args.syn==1
    if not gen_data:
        # args.case = args.case + '_100Bvar'
        calc_dim = False
        xfile = 'x.csv'
        yfile = args.yfile
        xdist_file = 'x_dist.csv'
        ydist_file = 'y' + dtype + '_dist.csv'
        met_newick_name = 'newick_' + args.yfile.split('.csv')[0] + '.nhx'

        # set data_path to point to directory with data
        data_path = base_path + "/inputs"
        # Option to change filtering criteria
        if xfile not in os.listdir(data_path) or yfile not in os.listdir(data_path):
            load_data(base_path, xfile, yfile, dataLoader)
        x = pd.read_csv(data_path + '/' + xfile, index_col = [0])
        y = pd.read_csv(data_path + '/' + yfile, index_col = [0])
        y = y.loc[x.index.values]

        args.N_met = y.shape[1]
        args.N_bug = x.shape[1]
        args.N_samples = y.shape[0]

        corr, p = st.spearmanr(y)
        corr_dist = (1 + corr)/2
        embedding = MDS(n_components=2, dissimilarity='precomputed', metric = True, random_state = args.seed)
        y_mds = embedding.fit_transform(corr_dist)

        # clusters = sklearn.cluster.AffinityPropagation(affinity = 'precomputed', random_state =  args.seed).fit(corr_dist)
        # # cluster_centers = clusters.predict(corr_dist)
        # plt.figure()
        # # plt.scatter(clusters.cluster_centers[:,0], clusters.cluster_centers[:,1], marker = '*', c = 'k')
        # for clust in np.arange(len(np.unique(clusters.labels_))):
        #   plt.scatter(y_mds[clusters.labels_ == clust, 0], y_mds[clusters.labels_ == clust, 1], label = 'Cluster ' + str(clust))
        # plt.legend()
        # plt.title('499 Metabolites')
        # plt.show()

        print(x.shape)
        print(y.shape)
        make_tree(x.columns.values, base_path, args.case, 'asv',
                  newick_path='/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx')
        make_tree(y.columns.values, base_path, args.case, 'metab_orig', newick_path='/ete_tree/' + met_newick_name,
                  dist_type=dtype)

        if args.locs == 'true':
            if xdist_file not in os.listdir(base_path + '/inputs/'):
                make_dist_mat(x, xdist_file, base_path, newick_path = '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx')
            xdist = pd.read_csv(base_path + '/inputs/' + xdist_file, header=0, index_col=0)
            xdist = xdist / np.max(np.max(xdist))

            if ydist_file not in os.listdir(base_path + '/inputs/'):
                make_dist_mat(y, ydist_file, base_path, newick_path = '/ete_tree/' + met_newick_name)
            ydist = pd.read_csv(base_path + '/inputs/' + ydist_file, header = 0, index_col = 0)
            ydist = ydist / np.max(np.max(ydist))
            # xdist = (xdist - xdist.mean().mean())/xdist.std().std()
            if calc_dim:
                args.xdim, xlocs, xstress = mds_choose_d(xdist,seed = args.seed)
                args.ydim, ylocs, ystress = mds_choose_d(ydist, seed=args.seed)
            else:
                args.xdim = 7
                embedding = MDS(n_components=args.xdim, dissimilarity='precomputed', random_state=args.seed)
                xlocs = embedding.fit_transform(xdist)
                args.ydim = 15
                embedding = MDS(n_components=args.ydim, dissimilarity='precomputed', random_state=args.seed)
                ylocs = embedding.fit_transform(ydist)

            x_fams = get_xtaxa(base_path + '/inputs/taxa_labels.csv', x)
            y_class = get_ytaxa(base_path + '/inputs/metab_classes.csv', y.columns.values, ydist, level='subclass')
            # skbio_mds(ydist, y_class, path = base_path + '/figures/' + dtype + '-')

            ylocs = plot_MDS(ydist, path=base_path + '/figures/' + dtype, seed=args.seed)
            # plot_MDS(xdist, path=base_path + '/figures/' + 'asvs', seed=args.seed)
            # plot_dist(ydist, path=base_path + '/figures/' + dtype + '-true-')
            # plot_dist(pdist(ylocs), path = base_path + '/figures/' + dtype + '-est-d' + str(args.ydim) + '-')

            xlocs = (xlocs - np.mean(xlocs, 0))/np.std(xlocs,0)
            ylocs = (ylocs - np.mean(ylocs, 0))/np.std(ylocs, 0)
            plot_classes(y_class, ylocs[:,:2], base_path + '/figures/mets_' + dtype)
            # plot_classes(x_fams, xlocs[:,:2], base_path + '/figures/bugs')
            # plot_MDS(xdist, path=base_path + '/figures/' + 'asvs_', seed=args.seed)
        elif args.locs == 'random':
            ylocs = get_rand_locs(y, args.ydim, args.seed)
            xlocs = get_rand_locs(x, args.xdim, args.seed)
            x_fams, y_class = None, None
        else:
            xlocs, ylocs = None, None
            x_fams, y_class = None, None
    else:
        x,y,ylocs,xlocs,y_class,x_fams = None, None, None, None, None, None
        # args.N_samples = 49


    run_learner(args, device, x=x, y=y, base_path = base_path, a_met = ylocs, a_bug = xlocs, met_class = y_class, bug_class = x_fams)
