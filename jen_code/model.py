#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from torch.distributions.dirichlet import Dirichlet
from helper import *
from torch.distributions.log_normal import LogNormal
from MAP_loss import *
from sklearn.cluster import KMeans
import scipy
from torch import optim

class NAM(nn.Module):
    def __init__(self, L, K, p_nn=1):
        super(NAM, self).__init__()
        self.L = L
        self.K = K
        self.p_nn = p_nn
        self.NAM = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Sequential(
            nn.Linear(1, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 6, bias=True),
            nn.ReLU(),
            nn.Linear(6, 1, bias=True)
        ) for p in np.arange(p_nn)]) for l in np.arange(L)]) for k in np.arange(K)])

    # def initialize(self, seed):
    #     self.seed = seed

    def forward(self, x, alpha, beta, sigma):
        out = torch.stack([torch.cat([torch.stack([self.NAM[k][l][p](x[:, l:l + 1]) for p in np.arange(self.p_nn)
                 ], -1).sum(-1) for l in np.arange(self.L)], 1) for k in np.arange(self.K)], 2)
        out = beta + (out * alpha).sum(1) 
#         + Normal(
#             0, torch.sqrt(torch.exp(sigma))).sample([x.shape[0], self.K])
        return out

class Model(nn.Module):
    def __init__(self, 
                 met_locs, 
                 microbe_locs, 
                 N_met, 
                 N_bug, 
                 K = 25,#2
                 L = 25,#3,
                 alpha_temp = 1, 
                 omega_temp = 1, 
                 data_meas_var = 1,
                 linear = False, #True, 
                 learn_num_met_clusters = True, 
                 learn_num_bug_clusters = True,
                 p_nn = 1,
                 met_class = None, 
                 bug_class = None
                ):
        super(Model, self).__init__()
        """
                This function initializes the Model and calls the loss class

        *note that the inputs above are just defaults; the actual inputs are set when the model is initialized in run_learner() below
        *note: omega and w are used interchangeably, but always mean the same variable (the microbial cluster indicator matrix)
        Inputs:
        - met_locs: the metabolite embedded locations (Matrix with size [N_met X #embedding-dimensions])
        - microbe_locs: microbe embedded locations (matrix w/ size [N_bug X #embedding-dimensions])
        - N_met: # metabolites
        - N_bug: # microbes
        - K: # metabolite clusters (if not learning the number of clusters)
        - L: # microbe clusters (if not learning the number of clusters)
        - alpha_temp: temperature parameter for alpha (decreases during learning)
        - omega_temp: tempurature parameter for omega (only used if no microbial locations are given and omega is
                  learned; also decreases during learning) (note: omega and w are used interchangeably; they refer
                  to the same parameter)
        - compute_loss_for: parameters that have a prior (useful if we want to take away the priors on some or all params)
        - learn_num_met_clusters: Whether or not to learn the number of metabolite clusters
        - learn_num_bug_clusters: Whether or not to learn the number of microbe clusters
        - linear: Whether or not to learn a linear model or a non-linear model
        - p_nn: The number of additive neural networks per each metabolite cluster interaction; if p_nn = 1,
                  there is one network per each interaction between a metabolite cluster and a microbial cluster
        - met_class: List of metabolite taxonomic categories if available (used for initialization and setting priors); when
                  calling run_learner() in main, the user can specify which level of microbial taxonomy to use here (i.e. you can
                  use super-classes, subclasses, etc)
        - bug_class: List of microbial taxonomic categories if available (used for initialization and setting priors); when
                  calling run_learner() in main, the user can specify which level of microbial taxonomy to use here (i.e. you can
                  use families, genus, species, etc)

        """
        linear=False
        

        # class for loss function
        self.MAPloss = MAPloss(self)

        # metabolite locations (a^{metab})
        self.met_locs = met_locs

        # microbe locations (a^{taxa})
        self.microbe_locs = microbe_locs

        # metabolite levels or bug levels to inform priors
        self.met_class = met_class
        self.bug_class = bug_class

        # Don't learn r and mu parameters if no locations are given
        if self.microbe_locs is None:
            if 'r_bug' in self.compute_loss_for:
                self.compute_loss_for.remove('r_bug')
            if 'mu_bug' in self.compute_loss_for:
                self.compute_loss_for.remove('mu_bug')
            self.bug_embedding_dim = None
        else:
            self.bug_embedding_dim = microbe_locs.shape[1]

        if self.met_locs is None:
            if 'r_met' in self.compute_loss_for:
                self.compute_loss_for.remove('r_met')
            if 'mu_met' in self.compute_loss_for:
                self.compute_loss_for.remove('mu_met')
            self.met_embedding_dim = None
        else:
            self.met_embedding_dim = met_locs.shape[1]

        # alpha parameter temperature, annealed over the course of learning
        self.alpha_temp = alpha_temp

        # omega parameter temperature, annealed over the course of learning
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

        # number of neural networks per metabolite cluster (default = 1)
        self.p_nn = p_nn

        # Measurment variance of data
        self.sigma = torch.tensor(data_meas_var).float()

        # If we learn the number of microbe/metabolite clusters and met_class is not None and bug_class is not None,
        # set the number of metabolomic clusters to the number of unique metabolite classes and the number of microbes
        # to the number of unique microbial classes
        # Otherwise, set to input L and K
        self.L, self.K = L, K
        if self.learn_num_met_clusters:
            if self.met_class is not None:
                self.K = len(np.unique(self.met_class.values))
        if self.learn_num_bug_clusters:
            if self.bug_class is not None:
                self.L = len(np.unique(self.bug_class.values))


        # Set the prior distributions parameters for each model parameter (to be called in MAP_loss.py when we find the
        # loss of each parameter according to its prior)
        # self.distributions is a dictionary of each parameter's prior distribution
        self.distributions = {}


        # define the NAM model if not linear
        # This NAM is a set of neural networks for each metab-microbe cluster interaction. Each neural network has
        # 2 hidden layers with 32 nodes in the first layer and 16 nodes in the second layer
        if not self.linear:
            self.nam = NAM(self.L, self.K, p_nn = self.p_nn)
            self.distributions['NAM'] = Normal(0, 1000)
        self.batch_norm = nn.BatchNorm1d(self.L)


        # If we have metabolite locations, we set a prior distribution for r_met and mu_met
        if self.met_locs is not None:
            # If we have metabolite categories (from taxonomy or elsewhere), we set the prior for r_met to be centered
            # around the average embedded size of the given metabolite categories
            if self.met_class is not None:
                radii = []
                for bc in np.unique(self.met_class.values):
                    ixs = np.where(self.met_class.values == bc)[0]
                    locs = self.met_locs[ixs,:]
                    radii.extend(pdist(locs))
                loc = np.median(radii)

            # Otheriwse, we set r_met to be the size of the embedding space divided by K
                scale = np.var(radii)
            else:
                scale = np.sum((np.max(self.met_locs,0) - np.min(self.met_locs,0))**2)
                loc = scale / self.K

            # r_met is parameterized by a lognormal
            # NEW: variance is set to the log of the variance of the radii (or the scale based on the locations) rather than set to 1
            mu = np.log(loc)
            var = np.log(scale)
            self.distributions['r_met'] = LogNormal(torch.tensor(mu), scale = torch.sqrt(torch.tensor(var)))

            # We define the prior for mu_met to be a multivariate normal with mean 0 and variance 100 (since the input
            # locations are already scaled to have variance 1)
            self.distributions['mu_met'] = MultivariateNormal(torch.zeros(self.met_embedding_dim),
                                                              (torch.tensor(1.0e4)) * torch.eye(
                                                                  self.met_embedding_dim))

        # If we have input microbial locations, we do the same procedure as for input metabolic locations described above
        if self.microbe_locs is not None:
            if self.bug_class is not None:
                radii = []
                for bc in np.unique(self.bug_class.values):
                    ixs = np.where(self.bug_class.values == bc)[0]
                    locs = self.microbe_locs[ixs,:]
                    radii.extend(pdist(locs))
                loc = np.mean(radii)
                scale = np.var(radii)
            else:
                scale = np.sum((np.max(self.microbe_locs,0) - np.min(self.microbe_locs,0))**2)
                loc = scale / self.L

            # NEW: variance is set to the log of the variance of the radii (or the scale based on the locations) rather than set to 1
            mu = np.log(loc)
            var = np.log(scale)
            self.distributions['r_bug'] = LogNormal(torch.tensor(mu), scale = torch.sqrt(torch.tensor(var)))

            self.distributions['mu_bug'] = MultivariateNormal(torch.zeros(self.bug_embedding_dim),
                                                              1e4*torch.eye(self.bug_embedding_dim))


        # beta is parameterized by a normal dist with mean=0, var = 1000
        self.distributions['beta'] = Normal(0, np.sqrt(1e4))

        # alpha is parameterized by a binary concrete with loc=1/(K*L) (defined above) and self.alpha_temp, which
        # decreases throughout learning like self.omega_temp
        self.alpha_loc = 0.5/L
        self.distributions['alpha'] = BinaryConcrete(self.alpha_loc, self.alpha_temp)

        # e_met is only learned (and thus the prior only used) if we are learning the number of metabolite clusters;
        # we set dof=10 and scale = 10*K according to the recommendations in Malsiner-Walli et al
        self.distributions['e_met'] = Gamma(10, 10*self.K)
#         self.distributions['e_met'] = Gamma(1, 1./self.K)

        # pi_met is set to a dirichlet prior with epsilon = 1/K; if we learn the number of metabolite clusters,
        # we use epsilon=e_met instead of epsilon=1/K
        self.distributions['pi_met'] = Dirichlet(torch.Tensor([1/self.K]*self.K))

    def train_NAM_for_init(self, x, y, iterations=1000, lr=0.1):
        """
        Trains NAM for initialization
        inputs: x (microbe sums), y (metabolite cluster targets), model (NAM model), iterations and lr
        """

        # get initialization NAM
        init_nam = NAM(self.L, self.K, self.p_nn)
        optimizer = optim.RMSprop(init_nam.parameters(), lr=lr)
        loss_func = nn.L1Loss()
        loss_vec = []
        model_save = init_nam.state_dict()
        for epoch in range(iterations):
            optimizer.zero_grad()
            out = init_nam(x, self.alpha_act.detach(), self.beta.detach(), self.sigma.detach())
            loss = loss_func(out, y)

            # For some reason, it doesn't run without retaining the graph. It's pretty quick regardless but this might be
            # something to look into, since I'm not sure why this is
            loss.backward(retain_graph = True)
            optimizer.step()
            loss_vec.append(loss.detach().item())
            if epoch > 100:
                if loss.detach().item() == np.min(loss_vec):
                    model_save = init_nam.state_dict()
                if (np.max(loss_vec[-10:]) - np.min(loss_vec[-10:])) <= 1:
                    break

        return model_save, loss_vec

    # Initialize parameter values with given random seed
    def initialize(self, seed, x, y):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        # Initialize the microbe cluster means and radii using K-means
        if self.microbe_locs is not None:
            kmeans = KMeans(n_clusters=self.L, random_state=self.seed).fit(self.microbe_locs)
            self.mu_bug = nn.Parameter(torch.Tensor(kmeans.cluster_centers_), requires_grad=True)
            r = []
            for clust in np.arange(self.L):
                cdist = scipy.spatial.distance.cdist(self.microbe_locs[kmeans.labels_==clust,:],
                                                     kmeans.cluster_centers_[clust:clust+1,:])
                if np.max(cdist.squeeze())==0:
                    r.append(0.01)
                else:
                    r.append(np.max(cdist.squeeze()))

            self.r_bug = nn.Parameter(torch.log(torch.Tensor(np.array(r))), requires_grad=True)
            kappa = torch.stack([((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1) for m in
                                 range(self.microbe_locs.shape[0])])
            # NEW!! Fixed error here (didn't have torch.exp)
            self.w = (torch.exp(self.r_bug) - kappa)
        else:
            self.w = Normal(0,1).sample([self.N_bug, self.L])

        self.w_act = torch.sigmoid(self.w / self.omega_temp)

        # Initialize the metabolite cluster means and radii
        if self.met_locs is not None:
            kmeans = KMeans(n_clusters=self.K, random_state=self.seed).fit(self.met_locs)
            self.mu_met = nn.Parameter(torch.Tensor(kmeans.cluster_centers_), requires_grad=True)
            r = []
            gp_size = []
            for clust in np.arange(self.K):
                cdist = scipy.spatial.distance.cdist(self.met_locs[kmeans.labels_ == clust, :],
                                                     kmeans.cluster_centers_[clust:clust + 1, :])
                if np.max(cdist.squeeze()) == 0:
                    r.append(0.01)
                else:
                    r.append(np.max(cdist.squeeze()))
                gp_size.append(np.sum(kmeans.labels_ == clust))
            r = np.array(r)
            r[r < 0.1] = 0.1
            self.r_met = nn.Parameter(torch.log(torch.Tensor(np.array(r))), requires_grad=True)
            self.z_act = torch.Tensor(get_one_hot(kmeans.labels_, l = self.K))
            if self.learn_num_met_clusters:
                self.e_met = nn.Parameter(torch.Tensor(np.array(gp_size)), requires_grad=True)
            else:
                self.e_met = torch.Tensor(np.array(gp_size))
            self.pi_met = nn.Parameter(torch.log(torch.Tensor(np.array(gp_size)/self.N_met).unsqueeze(0)), requires_grad=True)
            
        else:
            if self.learn_num_met_clusters:
                self.e_met = nn.Parameter(torch.log(((1 / self.K) * torch.ones(self.K)).unsqueeze(0)), requires_grad=True)
                self.e_met = nn.Parameter( torch.Tensor(np.ones(self.K)/self.K), requires_grad=True)
            else:
                self.e_met = torch.log(((1 / self.K) * torch.ones(self.K)).unsqueeze(0))
                
            self.pi_met = nn.Parameter(torch.log(Dirichlet(((1 / self.K) * torch.ones(self.K)).unsqueeze(0)).sample()), requires_grad=True)
            z = Categorical(self.pi_met).sample([self.N_met])
            self.z_act = torch.Tensor(get_one_hot(z, l = self.K))
            
        self.e_met = nn.Parameter( (1 / self.K) * torch.ones(self.K).unsqueeze(0), requires_grad=True )

        # Get microbial cluster inputs and metabolic cluster outputs for lasso regression to initialize biases and alphas
        temp = x.detach()@self.w_act.float()
        eps = get_epsilon(temp)
        temp = torch.log(temp + eps)
        X = (temp - torch.mean(temp, 0))/torch.std(temp, 0)
        Y = (y.detach()@self.z_act.float())/self.z_act.sum(0)
        lreg = sklearn.linear_model.Lasso(alpha = 1.0, random_state=self.seed, selection='random')
        lreg.fit(X.detach().numpy(), Y.detach().numpy())

        # Set alpha_act to 1 if coefficient is present, or 0 otherwise
        # Set alpha (unconstrained to be within 0 to 1) to soft version
        selection = (lreg.coef_ > 1e-5).astype(float)
        smooth_selection = selection.copy()
        smooth_selection[np.where(selection == 1)[0], np.where(selection==1)[1]] = 1 - 0.1*np.random.rand(np.sum(selection==1))
        smooth_selection[np.where(selection == 0)[0], np.where(selection==0)[1]] = 0 + 0.1*np.random.rand(np.sum(selection==0))
        self.alpha = nn.Parameter(torch.Tensor(np.log(smooth_selection/(1-smooth_selection)).T),requires_grad=True)
        self.alpha_act = torch.Tensor(selection.T)

        if self.linear:
            # If we are learning the linear model, set beta to the linear regression intercepts + coefficients
            self.beta = nn.Parameter(torch.Tensor(np.vstack((np.expand_dims(lreg.intercept_,0), lreg.coef_.T))), requires_grad=True)
        else:
            # If we are learning the non-linear model, set beta to the linear regression intercepts and train NAM
            self.beta = nn.Parameter(torch.Tensor(lreg.intercept_), requires_grad=True)
            init_state_dict, loss_vec = self.train_NAM_for_init(X, Y)
            self.nam.load_state_dict(deepcopy(init_state_dict))


    def forward(self, x, y):        
        # Forward function, contains all the model equations and calls MAP_loss.py to calculate loss
        # Omega and alpha epsilon are to keep w and alpha from getting to close to 0 or 1 and causing numerical issues
        omega_epsilon = 1e-13
        alpha_epsilon = self.alpha_temp / 4
        if self.microbe_locs is not None:
            kappa = torch.stack(
                [torch.sqrt(((self.mu_bug - torch.tensor(self.microbe_locs[m, :])).pow(2)).sum(-1)) for m in
                 np.arange(self.microbe_locs.shape[0])])
            self.w_act = (1-2*omega_epsilon)*torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.omega_temp)+ omega_epsilon
        else:
            self.w_act = (1-2*omega_epsilon)*torch.sigmoid(self.w/self.omega_temp) + omega_epsilon
        g = x@self.w_act.float()


        # log-transform g with set epsilon
        g = torch.log(g + 0.001)
        bn = (g - torch.mean(g, 0)) / (torch.std(g, 0) + 1e-5)
        self.alpha_act = (1-2*alpha_epsilon)*torch.sigmoid(self.alpha/self.alpha_temp) + alpha_epsilon


        if self.linear:
#             out_clusters = self.beta[0,:] + torch.matmul(bn, self.beta[1:,:])*self.alpha_act
            out_clusters = self.beta[0,:] + torch.matmul(bn, self.beta[1:,:]*self.alpha_act) # + \
#                            Normal(0,torch.sqrt(torch.exp(self.sigma))).sample([bn.shape[0], self.K])

        else:
            out_clusters = self.nam(bn, self.alpha_act, self.beta, self.sigma)

        if torch.isnan(out_clusters).any() or torch.isinf(out_clusters).any():
            print('debug')
        # compute loss via the priors
#         print(out_clusters.shape)
#         print(out_clusters[:4])
        loss = self.MAPloss.compute_loss(out_clusters,y)
#         print(loss)
        return out_clusters, loss, self.MAPloss.loss_dict
