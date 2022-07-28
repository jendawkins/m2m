#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from torch.distributions.dirichlet import Dirichlet
from helper import *
from torch.distributions.half_normal import HalfNormal
from MAP_loss import *
from sklearn.cluster import KMeans
import scipy

# this model initializes model parameters and contains all the model equations in the overleaf
# TO DO:
# - make sure all initializations / priors are the same as in the overleaf
# - r_bug and r_met prior distributions - maybe don't use gamma? dof shouldn't be fraction?
class Model(nn.Module):
    def __init__(self, met_locs, microbe_locs, N_met, N_bug, K = 2, L = 3,
                 alpha_temp = 1, omega_temp = 1, data_meas_var = 1,
                 compute_loss_for = ['alpha','beta','w','z','mu_bug','r_bug','pi_bug','mu_met','r_met','pi_met'],
                 learn_num_met_clusters = False, learn_num_bug_clusters = False, linear = True,
                p_nn = 1, sample_a = True, sample_w = True, met_class = None, bug_class = None, gmm = 0, l1 = None):
        super(Model, self).__init__()
        # This function initializes the Model and calls the loss class

        # *note that the inputs above are just defaults; the actual inputs are set when the model is initialized in run_learner() below
        # *note: omega and w are used interchangeably, but always mean the same variable (the microbial cluster indicator matrix)
        # Inputs:
        # - met_locs: the metabolite embedded locations (Matrix with size [N_met X #embedding-dimensions])
        # - microbe_locs: microbe embedded locations (matrix w/ size [N_bug X #embedding-dimensions])
        # - N_met: # metabolites
        # - N_bug: # microbes
        # - K: # metabolite clusters (if not learning the number of clusters)
        # - L: # microbe clusters (if not learning the number of clusters)
        # - alpha_temp: temperature parameter for alpha (decreases during learning)
        # - omega_temp: tempurature parameter for omega (only used if no microbial locations are given and omega is
        #           learned; also decreases during learning) (note: omega and w are used interchangeably; they refer
        #           to the same parameter)
        # - compute_loss_for: parameters that have a prior (useful if we want to take away the priors on some or all params)
        # - learn_num_met_clusters: Whether or not to learn the number of metabolite clusters
        # - learn_num_bug_clusters: Whether or not to learn the number of microbe clusters
        # - linear: Whether or not to learn a linear model or a non-linear model
        # - l1: Whether to have L1 regularziation (only used if linear=0/False; in trials with generated data, having l1 = True seemed to help)
        # - p_nn: The number of additive neural networks per each metabolite cluster interaction; if p_nn = 1,
        #           there is one network per each interaction between a metabolite cluster and a microbial cluster
        # - sample_a: Whether or not to sample a binary value for alpha (ensures the model is binary, but more difficult to learn)
        # - sample_w: Whether or not to sample a binary value for omega (only used if omega is learned and/or no input microbial
        #           embedded locations are given)
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
        # whether or not to sample alpha in the forward pass
        self.sample_a = sample_a
        self.sample_w = sample_w
        # number of neural networks per metabolite cluster (default = 1)
        self.p_nn = p_nn
        # number of microbe and metabolite clusters
        self.L, self.K = L, K
        # Measurment variance of data
        self.data_meas_var = data_meas_var

        self.l1 = l1

        # If we learn the number of microbe/metabolite clusters and met_class is not None and bug_class is not None,
        # set the number of metabolomic clusters to the number of unique metabolite classes and the number of microbes
        # to the number of unique microbial classes
        # Otherwise, if met_class is none or bug_class is None, set to 1/3 times the number of features
        if self.learn_num_met_clusters:
            if self.met_class is not None:
                self.K = len(np.unique(self.met_class.values))
            else:
                self.K = np.int(self.N_met/3)
        if self.learn_num_bug_clusters:
            if self.bug_class is not None:
                self.L = len(np.unique(self.bug_class.values))
            else:
                self.L = np.int(self.N_bug/3)
        # the location parameter for the binaryConcrete distribution on alpha
        self.alpha_loc = 1 / (self.L * self.K)

        # define the NAM model if not linear
        # This NAM is a set of neural networks for each metab-microbe cluster interaction. Each neural network has
        # 2 hidden layers with 32 nodes in the first layer and 16 nodes in the second layer
        if not self.linear:
            self.NAM = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Sequential(
                nn.Linear(1, 8, bias = True),
                nn.ReLU(),
                nn.Linear(8, 8, bias = True),
                nn.ReLU(),
                nn.Linear(8, 6, bias = True),
                nn.ReLU(),
                nn.Linear(6,1, bias = True)
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
            # self.params['r_met'] = {'loc': loc, 'scale': (tau2*v)/2, 'dof': v/2}
            # self.distributions['r_met'] = Gamma(rate=self.params['r_met']['scale'],
            #                                           concentration=self.params['r_met']['dof'])

            self.params['r_met'] = {'scale': 10, 'loc': loc}
            self.distributions['r_met'] = HalfNormal(10)

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
            # self.params['r_bug'] = {'scale': (tau2*v)/2, 'dof': v/2}
            self.params['r_bug'] = {'scale': 10, 'loc': loc}
            self.distributions['r_bug'] = HalfNormal(10)
            # self.distributions['r_bug'] = Gamma(rate=self.params['r_bug']['scale'],
            #                                           concentration=self.params['r_bug']['dof'])
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
        # tau2 = data_meas_var
        # v = 0.1
        # self.params['sigma'] = {'scale': (tau2 * v) / 2, 'dof': v/2}
        # self.distributions['sigma'] = Gamma(rate=self.params['sigma']['scale'],
        #                                     concentration=self.params['sigma']['dof'])
        self.params['sigma'] = {'scale': 10}
        self.distributions['sigma'] = HalfNormal(self.params['sigma']['scale'])


        # For each model parameter, sample from the prior to get the parameter range, for adjusting the learning rate
        # based on parameter size
        # Also, get the expected range of the parameter for plotting purposes (the expected parameter range is use when
        # plotting parameter traces that are nicer and easier to interpret / compare between different model runs)
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
            if 'r_met' in param or 'r_bug' in param:
                # vals = np.log(1/self.distributions[param].sample([1000]))
                vals = np.log(self.distributions[param].sample([1000]))
                self.lr_range[param] = torch.abs((torch.mean(vals) + torch.std(vals)) - (torch.mean(vals) - torch.std(vals)))
            elif 'w' in param or 'z' in param or 'alpha' in param:
                self.lr_range[param] = np.abs(2*(torch.log(torch.tensor(self.params[param]['loc']).float())/self.params[param]['temp']))
            elif param == 'pi_met' or param == 'e_met' or param == 'sigma':
                vals = torch.log(sampler)
                self.lr_range[param] = torch.abs((torch.mean(vals) + torch.std(vals)) - (torch.mean(vals) - torch.std(vals)))
                range = sampler.max() - sampler.min()
            else:
                vals = dist.sample([1000])
                range = sampler.max() - sampler.min()
                self.lr_range[param] = torch.abs((torch.mean(vals) + torch.std(vals)) - (torch.mean(vals) - torch.std(vals)))



    # Initialize parameter values with given random seed
    def initialize(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        # self.sigma is the measurement variance, initialized to be the log of the measurement variance (since we will
        # transform it to exp(self.sigma))
        self.sigma = nn.Parameter(torch.log(torch.tensor(self.data_meas_var).float()), requires_grad=True) # measurement variance
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

        # Initialize the microbe cluster means and radii using K-means
        if self.microbe_locs is not None:
            kmeans = KMeans(n_clusters=self.L, random_state=self.seed).fit(self.microbe_locs)
            self.mu_bug = nn.Parameter(torch.Tensor(kmeans.cluster_centers_), requires_grad=True)
            r = []
            for clust in np.arange(self.L):
                cdist = scipy.spatial.distance.cdist(self.microbe_locs[kmeans.labels_==clust,:],
                                                     kmeans.cluster_centers_[clust:clust+1,:])
                r.append(np.max(cdist.squeeze()))
            eps = 0.01*np.max(r)
            self.r_bug = nn.Parameter(torch.Tensor(1.1*np.array(r + eps)), requires_grad=True)
            # self.mu_bug = nn.Parameter(MultivariateNormal(torch.zeros(self.bug_embedding_dim), 0.1*torch.eye(self.bug_embedding_dim)).sample([self.L]), requires_grad=True)
            # temp = np.sqrt(np.sum((np.max(self.microbe_locs,0) - np.min(self.microbe_locs,0))**2)) / 2
            # self.r_bug = nn.Parameter(torch.log(1.2*temp*torch.ones(self.L)), requires_grad=True)
            kappa = torch.stack([((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1) for m in
                                 range(self.microbe_locs.shape[0])])
            self.w = (self.r_bug - kappa)
            if 'w' in self.compute_loss_for:
                self.compute_loss_for.remove('w')
        else:
            self.w = nn.Parameter(Normal(0,1).sample([self.N_bug, self.L]), requires_grad=True)

        if self.sample_w:
            omega_epsilon = self.omega_temp / 4
            self.w_soft, self.w_act = gumbel_sigmoid(self.w, self.omega_temp, omega_epsilon)
        else:
            self.w_act = torch.sigmoid(self.w / self.omega_temp)

        self.w_loc = 0.1
        # Initialize the metabolite cluster means and radii
        if self.met_locs is not None:
            kmeans = KMeans(n_clusters=self.K, random_state=self.seed).fit(self.met_locs)
            self.mu_met = nn.Parameter(torch.Tensor(kmeans.cluster_centers_), requires_grad=True)
            r = []
            for clust in np.arange(self.K):
                cdist = scipy.spatial.distance.cdist(self.met_locs[kmeans.labels_ == clust, :],
                                                     kmeans.cluster_centers_[clust:clust + 1, :])
                r.append(np.max(cdist.squeeze()))
            eps = 0.01 * np.max(r)
            self.r_met = nn.Parameter(torch.Tensor(1.1 * np.array(r + eps)), requires_grad=True)
        else:
            ix = np.random.choice(range(self.met_locs.shape[0]), self.L, replace=False)
            self.mu_met = nn.Parameter(torch.Tensor(self.met_locs[ix, :]), requires_grad=True)
            r_temp = 0.5 * torch.ones((self.L)).squeeze()
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

    def forward(self, x, y):
        # Forward function, contains all the model equations except priors (those are in self.MAPlos.compute_loss())

        # Omega and alpha epsilon are to keep w and alpha from getting to close to 0 or 1 and causing numerical issues
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
        if self.l1 and not self.linear:
            l1_parameters = []
            for parameter in self.NAM.parameters():
                l1_parameters.append(parameter.view(-1))
            l1 = self.compute_l1_loss(torch.cat(l1_parameters))
            loss += l1

        return out_clusters, loss
