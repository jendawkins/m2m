from torch.distributions.dirichlet import Dirichlet
import torch.nn.functional as F
from helper import *
from torch.distributions.log_normal import LogNormal
from MAP_loss_newer import *
from sklearn.cluster import KMeans
import scipy
from torch import optim
import hdbscan

from typing import Union, Iterable, Sized, Tuple
import torch
import torch.nn.functional as F


def truncated_normal_(tensor, mean: float = 0., std: float = 1.):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class ActivationLayer(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((in_features, out_features)))
        self.bias = torch.nn.Parameter(torch.empty(in_features))

    def forward(self, x):
        raise NotImplementedError("abstract method called")


class ExULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        truncated_normal_(self.weight, mean=4.0, std=0.5)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        exu = (x - self.bias) @ torch.exp(self.weight)
        return torch.clip(exu, 0, 1)


class ReLULayer(ActivationLayer):
    def __init__(self,
                 in_features: int,
                 out_features: int):
        super().__init__(in_features, out_features)
        torch.nn.init.xavier_uniform_(self.weight)
        truncated_normal_(self.bias, std=0.5)

    def forward(self, x):
        return F.relu((x - self.bias) @ self.weight)


class FeatureNN(torch.nn.Module):
    def __init__(self,
                 out_dim: int,
                 shallow_units: int,
                 hidden_units: Tuple = (),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 dropout: float = .5,
                 ):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            hidden_layer(shallow_units if i == 0 else hidden_units[i - 1], hidden_units[i])
            for i in range(len(hidden_units))
        ])
        self.layers.insert(0, shallow_layer(1, shallow_units))

        self.dropout = torch.nn.Dropout(p=dropout)
        self.linear = torch.nn.Linear(shallow_units if len(hidden_units) == 0 else hidden_units[-1], out_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.linear.weight)


    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        return self.linear(x)


class NAM(torch.nn.Module):
    def __init__(self,
                 L: int,
                 K: int,
                 J: int,
                 shallow_units: int=2,
                 hidden_units: Tuple = (8,6),
                 shallow_layer: ActivationLayer = ExULayer,
                 hidden_layer: ActivationLayer = ReLULayer,
                 feature_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 ):
        super().__init__()
        self.input_size = L
        self.output_size = K
        self.N_mets = J
#         hidden_units = [a for a in hidden_units] + K 

        if isinstance(shallow_units, list):
            assert len(shallow_units) == self.input_size
        elif isinstance(shallow_units, int):
            shallow_units = [shallow_units for _ in range(self.input_size)]

        self.feature_nns = torch.nn.ModuleList([
            FeatureNN(out_dim=K*self.N_mets,
                      shallow_units=shallow_units[i],
                      hidden_units=hidden_units,
                      shallow_layer=shallow_layer,
                      hidden_layer=hidden_layer,
                      dropout=hidden_dropout)
            for i in range(self.input_size)
        ])
        self.feature_dropout = torch.nn.Dropout(p=feature_dropout)
        self.bias = torch.nn.Parameter(torch.zeros(1,K,1))

    def forward(self, x):
        f_out = torch.cat([a.unsqueeze(-1) for a in self._feature_nns(x)], dim=-1)
        f_out = self.feature_dropout(f_out)
#         print(f_out.shape)
        return f_out.sum(dim=2).reshape(x.shape[0],self.output_size, self.N_mets) + self.bias #f_out.sum(axis=-1) + self.bias, f_out

    def _feature_nns(self, x):
#         print('forward....')
        return [self.feature_nns[i](x[:, i]) for i in range(self.input_size)]


# class NAM(nn.Module):
#     def __init__(self, L, K, p_nn=1):
#         super(NAM, self).__init__()
#         self.L = L
#         self.K = K
#         self.p_nn = p_nn
#         self.NAM = nn.ModuleList([nn.ModuleList([nn.ModuleList([nn.Sequential(
#             nn.Linear(1, 8, bias=True),
#             nn.ReLU(),
#             nn.Linear(8, 8, bias=True),
#             nn.ReLU(),
#             nn.Linear(8, 6, bias=True),
#             nn.ReLU(),
#             nn.Linear(6, 1, bias=True)
#         ) for p in np.arange(p_nn)]) for l in np.arange(L)]) for k in np.arange(K)])

#     # def initialize(self, seed):
#     #     self.seed = seed

#     def forward(self, x, alpha, beta, sigma):
#         out = torch.stack([torch.cat([torch.stack([self.NAM[k][l][p](x[:, l:l + 1]) for p in np.arange(self.p_nn)
#                  ], -1).sum(-1) for l in np.arange(self.L)], 1) for k in np.arange(self.K)], 2)
#         out = beta + (out * alpha).sum(1) + Normal(
#             0, torch.sqrt(torch.exp(sigma))).sample([x.shape[0], self.K])
#         return F.softmax( out )

class Model(nn.Module):
    def __init__(self, 
                 met_locs, 
                 microbe_locs, 
                 alpha_temp = 1,
                 omega_temp = 1, 
                 data_meas_var = 1,
                 linear = True,
                 p_nn = 1,
                 met_class = None, 
                 bug_class = None):
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
        self.bug_embedding_dim = microbe_locs.shape[1]
        self.met_embedding_dim = met_locs.shape[1]
        
        # alpha parameter temperature, annealed over the course of learning
        self.alpha_temp = alpha_temp

        # omega parameter temperature, annealed over the course of learning
        self.omega_temp = omega_temp

        # whether or not to run the linear model vs the generalized NAM
        self.linear = linear

        # number of metabolites
        self.N_met = met_locs.shape[0]
        self.J=met_locs.shape[0]
        print(met_locs.shape)
        print(microbe_locs.shape)
        
        # number of microbes
        self.N_bug = microbe_locs.shape[0]

        # number of neural networks per metabolite cluster (default = 1)
        self.p_nn = p_nn

        # Measurment variance of data
        self.sigma = torch.tensor(data_meas_var).float()
        
        self.omega_epsilon = 1e-13
        
        

#     def train_NAM_for_init(self, x, y, iterations=1000, lr=0.1):
#         """
#         Trains NAM for initialization
#         inputs: x (microbe sums), y (metabolite cluster targets), model (NAM model), iterations and lr
#         """

#         # get initialization NAM
#         init_nam = NAM(self.L, self.K)#, self.p_nn)
#         optimizer = optim.RMSprop(init_nam.parameters(), lr=lr)
#         loss_func = nn.L1Loss()
#         loss_vec = []
#         model_save = init_nam.state_dict()
#         for epoch in range(iterations):
#             optimizer.zero_grad()
#             out = init_nam(x, self.alpha_act.detach(), self.beta.detach(), self.sigma.detach())
#             loss = loss_func(out, y)

#             # For some reason, it doesn't run without retaining the graph. It's pretty quick regardless but this might be
#             # something to look into, since I'm not sure why this is
#             loss.backward(retain_graph = True)
#             optimizer.step()
#             loss_vec.append(loss.detach().item())
#             if epoch > 100:
#                 if loss.detach().item() == np.min(loss_vec):
#                     model_save = init_nam.state_dict()
#                 if (np.max(loss_vec[-10:]) - np.min(loss_vec[-10:])) <= 1:
#                     break

#         return model_save, loss_vec

    
    def initialize_distributions(self, K=2, L=2):
        # Set the prior distributions parameters for each model parameter (to be called in MAP_loss.py when we find the
        # loss of each parameter according to its prior)
        # self.distributions is a dictionary of each parameter's prior distribution
        self.distributions = {}

        # define the NAM model if not linear
        # This NAM is a set of neural networks for each metab-microbe cluster interaction. Each neural network has
        # 2 hidden layers with 32 nodes in the first layer and 16 nodes in the second layer
        if not self.linear:
            self.f = NAM(L, K, self.J)#, p_nn = self.p_nn)
#             self.distributions['NAM'] = Normal(0, 1000)
#         self.batch_norm = nn.BatchNorm1d(self.L)

        
        
        # If we have metabolite locations, we set a prior distribution for r_met and mu_met
        scale = np.sum((np.max(self.met_locs,0) - np.min(self.met_locs,0))**2)
        loc = scale / K

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
        self.distributions['e_met'] = Gamma(10, 10*K)
#         self.distributions['e_met'] = Gamma(1, 1./self.K)

        # pi_met is set to a dirichlet prior with epsilon = 1/K; if we learn the number of metabolite clusters,
        # we use epsilon=e_met instead of epsilon=1/K
        self.distributions['pi_met'] = Dirichlet(torch.Tensor([1/K]*K))
        return(None)

    def init_clusters(self):
        ## hdscan to determine and fit # of clusters

        ### Fitting for metabolites
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        clusterer.fit(self.met_locs)

        cluster_centers = np.vstack( [ np.array( clusterer.weighted_cluster_centroid(a) ) if a>=0 
                                    else np.zeros_like(clusterer.weighted_cluster_centroid(0))
                                     for a in np.unique( clusterer.labels_ ) ] )

        self.mu_met = nn.Parameter(torch.Tensor(
                        np.vstack( cluster_centers )
                                        ), 
                                   requires_grad=True)

        cdists = [ scipy.spatial.distance.cdist(self.met_locs[np.array(clusterer.labels_) == clust, :],
                                                       cluster_centers[i:i + 1, :])
                   for i,clust in enumerate(np.unique(clusterer.labels_) ) ]

        
        r = [ np.max(a) for a in cdists]
        gp_size= list( pd.Series(clusterer.labels_).value_counts().sort_index().values )
        self.K = len(gp_size)

        self.r_met = nn.Parameter(torch.log(torch.Tensor(np.array(r))), requires_grad=True)
        self.z_act = torch.Tensor(get_one_hot(clusterer.labels_, l = self.K) )
#         self.e_met = nn.Parameter( torch.Tensor(np.array(gp_size)), requires_grad=True)
        self.e_met = nn.Parameter( torch.Tensor(np.ones(self.K)/self.K), requires_grad=True)
        self.pi_met = nn.Parameter( torch.log( torch.Tensor(np.array(gp_size)/self.N_met).unsqueeze(0) ), 
                                    requires_grad=True)

        
        print(cluster_centers.shape)
#         self.met_locs= torch.tensor( cluster_centers )



        ### Fitting for microbes
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2)
        clusterer.fit(self.microbe_locs)

        cluster_centers = np.vstack( [ clusterer.weighted_cluster_centroid(a) if a>=0 
                                    else np.zeros_like(clusterer.weighted_cluster_centroid(0))
                                     for a in np.unique( clusterer.labels_ ) ] )
        self.mu_bug = nn.Parameter(torch.Tensor(
                        np.vstack( cluster_centers )
                                        ), 
                                   requires_grad=True)

        cdists = [ scipy.spatial.distance.cdist(self.microbe_locs[clusterer.labels_ == clust, :],
                                                       cluster_centers[i:i + 1, :])
                   for i,clust in enumerate(np.unique(clusterer.labels_)) ]
        
        self.L = len(cdists)

        r = [ np.max(a) for a in cdists ]
        self.r_bug = nn.Parameter(torch.log(torch.Tensor(np.array(r))), requires_grad=True)
        kappa = torch.stack( [ ( (self.mu_bug - torch.tensor(self.microbe_locs[m, :]) 
                                             ).pow(2)).sum(-1) 
                              for m in range(self.microbe_locs.shape[0])
                             ])
        self.w = ( self.r_bug - kappa )
        self.w_act = torch.sigmoid(self.w / self.omega_temp)
#         self.microbe_locs = torch.Tensor( cluster_centers )
        return(None)

    # Initialize parameter values with given random seed
    def initialize(self, seed, x, y):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed
        
        
        self.init_clusters()
        self.initialize_distributions(K=self.K, L=self.L)


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
#         self.alpha = nn.Parameter(torch.Tensor(np.log(smooth_selection/(1-smooth_selection)).T),requires_grad=True)
        self.alpha = torch.nn.Parameter( torch.ones((self.L, self.K))/self.K, requires_grad=True)#nn.Linear(self.L, self.K)
        self.alpha_act = torch.Tensor(selection.T)
        
        
        # set this as tensor so we don't need to adjust during the forward eqn
        self.met_locs=torch.Tensor( self.met_locs )
        if self.linear:
            # If we are learning the linear model, set beta to the linear regression intercepts + coefficients
            self.f = nn.Linear(self.L, self.K) #self.K) #self.met_locs.shape[1])
            self.beta = nn.Parameter( torch.zeros(self.K), requires_grad=True )
            
            
#             self.beta = nn.Parameter(torch.Tensor(np.vstack((np.expand_dims(lreg.intercept_,0), lreg.coef_.T))), requires_grad=True)
        else:
            # If we are learning the non-linear model, set beta to the linear regression intercepts and train NAM
            self.beta = nn.Parameter(torch.Tensor(lreg.intercept_), requires_grad=True)
#             init_state_dict, loss_vec = self.train_NAM_for_init(X, Y)
#             self.nam.load_state_dict(deepcopy(init_state_dict))



    def forward(self, x, y):
#         print(self.K)
#         print(self.L)
#         print('ITER')
#         print(y)
#         print([p for p in self.named_parameters()])
        # Forward function, contains all the model equations and calls MAP_loss.py to calculate loss
        # Omega and alpha epsilon are to keep w and alpha from getting to close to 0 or 1 and causing numerical issues
#         print('LINEAR')
#         print(self.linear)
        alpha_epsilon = self.alpha_temp / 4
        
        kappa = torch.stack(
                [ torch.sqrt(
                    ( ( self.mu_bug - torch.tensor( self.microbe_locs[m, :] ) ).pow(2)
                       ).sum(-1) ) 
                 for m in np.arange( self.microbe_locs.shape[0] ) ] )
        
#         print(kappa)
        
            # (1-2*self.omega_epsilon) * \             
        self.w_act = torch.sigmoid((torch.exp(self.r_bug) - kappa)/self.omega_temp) + \
                     self.omega_epsilon
        
#         print(self.w_act)
    
        g = x@self.w_act.float()
        
#         print(g)
        
        # log-transform g with set epsilon
#         g = torch.log(g + 0.001)
#         print(g)
#         bn = (g - torch.mean(g, 0)) / (torch.std(g, 0) + 1e-5)
#         print(bn)
#         print(self.alpha)
        self.alpha_act = F.softmax( (1-2*alpha_epsilon)*torch.sigmoid(self.alpha/self.alpha_temp) + alpha_epsilon )

#         print(self.alpha_act)
#         print('Beta')
#         print(self.beta.shape)
#         print('G')
# #         print(g.shape)
#         print(self.K)
#         print(self.L)
#         print(self.met_locs.shape)
    
#         if self.linear:
# #             print(self.f(g).shape)
# #             print(self.alpha_act.shape)
# #             print(self.mu_met.shape)
# #             print(self.met_locs.shape)
# #             print(self.beta.shape)
            
#             out_clusters = ( self.beta + self.f(g) )#* self.alpha_act ) #@ self.mu_met @ self.met_locs.T
# #             [0,:] + torch.matmul(bn, self.beta[1:,:]*self.alpha_act) 
# #             + \
# #                            Normal(0,torch.sqrt(torch.exp(self.sigma))).sample([bn.shape[0], self.K])

#         else:
#             out_clusters = self.nam(bn, self.alpha_act, self.beta, self.sigma)
            
#         print('NAM OUT')
#         print(self.L)
#         print(self.K)
#         print(g[:5])
#         print(self.f(g).shape)
        out_clusters = self.f(g)#( self.beta + self.f(g) )#
            
#         print(out_clusters)
#         print(out_clusters.sum())
        
#         print(out_clusters)
        if torch.isnan(out_clusters).any() or torch.isinf(out_clusters).any():
            print('debug')
            
#         print(out_clusters.shape)
        # compute loss via the priors
        loss = self.MAPloss.compute_loss(out_clusters,y)
#         print(loss)
#         print(self.MAPloss.loss_dict)
        return out_clusters, loss, self.MAPloss.loss_dict



