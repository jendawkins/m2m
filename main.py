#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from torch.distributions.dirichlet import Dirichlet
from helper import *
from plot_helper import *
from concrete import *
import argparse
import re
from data_gen import *
import sys
from dataLoader import *

import subprocess
from sklearn.cluster import KMeans
import scipy
from torch.distributions.half_normal import HalfNormal
from safariLoader import *
import datetime
from model import *

def run_learner(args, device, x=None, y=None, a_met=None, a_bug = None, base_path = '', plot_params = True,
                met_class = None, bug_class = None):
    # Calls model and trains over set epochs for given input arguments
    # inputs:
    # - args: input arguments from argument parser
    # - device: cuda or not
    # - x: microbial relative abundances
    # - y: metabolic standardized levels
    # - a_met: metabolite embedded locations
    # - a_bug: microbial embedded locations
    # - base_path: current working dir
    # - plot_params: Whether to plot parameter traces or not (this can take time so sometimes I don't)
    # - met_class: Which metabolite sub-class/level to use for initializing metabolite cluster centers and radii (only
    #           used if a_met is not none
    # - bug_class: Which microbe level (family, genus, species, etc) to use for initializing microbial cluster centers
    #           and radii (I use family level)
    # TO DO:
    # - automatic stopping after loss stops decreasing / model has converged
    # - make more effecient / faster
    # - decide which plotting to keep and which to get rid of since that's a major time burden
    # - torch distributed learning (like suhas does in MDITRE)

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
    # add case to path for saving results
    path = path + args.case.replace(' ','_')
    if not os.path.isdir(path):
        os.mkdir(path)

    # priors2set is for specfying specific parameters to put priors on, default is 'all' but you can specify a list if
    # you don't want to put priors on all parameters (i.e. see how the model does without those priors)
    if 'all' in priors2set:
        priors2set = ['alpha','beta','mu_bug','mu_met','r_bug','r_met','pi_met','e_met','sigma']
        if a_bug is None:
            priors2set.append('w')

    # params2learn is the parameters that the model will learn; all other parameters will be fixed to their true value
    # (args.fix does the same thing, but it's easier to use args.fix if you want to fix 1 parameter and easier to use
    # params2learn if you want to learn only 1 parameter)
    # This is pretty much for debugging; default is 'all'
    # Also this only works if you are generating data, since otherwise you don't know the true values
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
    if args.fix is not None:
        if 'sigma' in args.fix:
            params2learn.remove('sigma')
            priors2set.remove('sigma')
            path = path + '/fix-sigma/'
            if not os.path.isdir(path):
                os.mkdir(path)


    # add all other specified inputs to path to prevent overwriting results & keep a record of the parameters each model is run with
    info = 'lr' + str(args.lr) + '-linear'*(args.linear) + '-adj_lr'*args.adjust_lr + '-hard'*args.hard + \
           '-l1'*(args.l1) + '-'*(1-args.linear) +args.nltype*(1-args.linear)*args.syn + '-lm'*args.lm + '-lb'*args.lb + \
            '-meas_var' + str(np.round(args.meas_var,3)).replace('.', '_') +  '-Nmet' + str(args.N_met) + '-Nbug' + str(args.N_bug) + \
           '-L' + str(args.L) + '-K' + str(args.K) + '-gmm'*args.gmm + \
           '-atau' + str(args.a_tau).replace('.','_') + '-wtau' + str(args.w_tau).replace('.', '_')

    path = path + '/' + info + '/'
    if not os.path.isdir(path):
        os.mkdir(path)

    # If function is called without input x and y data, generate synthetic data by calling data_gen.py and then plot
    if x is None and y is None:
        x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
        mu_met, r_bug, r_met, gen_u = generate_synthetic_data(
            N_met = args.N_met, N_bug = args.N_bug, N_met_clusters = args.K,
            N_bug_clusters = args.L,meas_var = args.meas_var,
            repeat_clusters= args.rep_clust, N_samples=args.N_samples, linear = args.linear,
            nl_type = args.nltype, dist_var_frac=args.dist_var_perc, embedding_dim=args.dim)
        if not args.linear:
            gen_beta = gen_beta[0,:]

        plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                          r_bug, mu_met, r_met, gen_u, gen_alpha, gen_beta)

        if a_met is not None:
            a_met = gen_met_locs

        if a_bug is not None:
            a_bug = gen_bug_locs

        # get true values from data_gen.py to compare to learned parameter values

        # If we are learning the number of metabolite clusters or the number of microbial clusters,
        # need to expand the generated true parameter matrices to compare to learned parameters
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
        # Dictionary of true values to compare to learned values
        true_vals = {'y':y, 'beta':gen_beta, 'alpha':gen_alpha, 'mu_bug': mu_bug,
                     'mu_met': mu_met, 'u': gen_u,'w_soft': gen_w,'r_bug':1.2*r_bug, 'r_met': 1.2*r_met, 'z': gen_z,
                     'w': gen_w, 'pi_met':np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),
                     'pi_bug':np.expand_dims(np.sum(gen_w,0)/np.sum(np.sum(gen_w)),0), 'bug_locs': gen_bug_locs,
                     'met_locs':gen_met_locs,
                     'e_met': np.expand_dims(np.sum(gen_z,0)/np.sum(np.sum(gen_z)),0),'b': mu_met, 'sigma': args.meas_var}
        # just for plotting; see the interaction b/w clusters
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

    # Print the parameters that we have set a prior for and will compute a posterior loss for
    print(net.compute_loss_for)

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
            parameter = np.exp(parameter.detach().numpy())
        if name == 'pi_met':
            parameter = torch.softmax(parameter.detach(),1).numpy()
        if torch.is_tensor(parameter):
            param_dict[args.seed][name] = [parameter.detach().numpy()]
        else:
            param_dict[args.seed][name] = [parameter]
    param_dict[args.seed]['z'] = [net.z_act.detach().numpy()]
    if 'w' not in param_dict[args.seed].keys():
        param_dict[args.seed]['w'] = [net.w_act.detach().numpy()]
    if net.linear:
        param_dict[args.seed]['beta[1:,:]*alpha'] = [net.beta[1:,:].detach().numpy()*net.alpha_act.detach().numpy()]

    # Plot initial metabolite and microbial classification / phylogenetic trees, to compare to learned trees
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
        # print(inputs)
        if args.safari == 0 and args.syn == 0:
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
        if args.safari == 0 and args.syn == 0:
            subprocess.run(inputs, cwd=base_path + "/ete_tree")

    # If embedding dimension = 2 and we have embedded locations input, plot output locations
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

    # initialize optimizer with learning rates
    optimizer = optim.RMSprop(lr_list, lr=args.lr)
    # record parameter size estimations and per-parameter learning rates
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

    # Train model over the number of specified input iterations in args.iterations
    x = torch.Tensor(np.array(x)).to(device)
    loss_dict_vec = {}
    ix = 0

    for epoch in np.arange(start, args.iterations+1):
        if epoch ==1:
            stime = time.time()
        net.alpha_temp = alpha_tau_logspace[ix]
        net.omega_temp = omega_tau_logspace[ix]
        optimizer.zero_grad()
        cluster_outputs, loss = net(x, torch.Tensor(np.array(y)))
        train_out_vec.append(cluster_outputs)

        # If model can't learn for whatever reason, set last_epoch = epoch so that the last successful epoch is plotted
        try:
            loss.backward()
            loss_vec.append(loss.detach().item())
            for param in net.MAPloss.loss_dict:
                if param not in loss_dict_vec.keys():
                    loss_dict_vec[param] = [net.MAPloss.loss_dict[param].detach().item()]
                else:
                    loss_dict_vec[param].append(net.MAPloss.loss_dict[param].detach().item())
            optimizer.step()
            last_epoch = args.iterations
        except:
            last_epoch = epoch
            loss_vec.append(loss_vec[-1])
            for param in net.MAPloss.loss_dict:
                loss_dict_vec[param].append(loss_dict_vec[param][-1])

        # keep track of updated parameter values
        for name, parameter in net.named_parameters():
            if 'NAM' in name or 'lambda_mu' in name or name=='b' or name == 'C':
                continue
            if name == 'z' or name == 'alpha' or name == 'w':
                parameter = getattr(net, name + '_act')
            elif name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'sigma' or name == 'p':
                parameter = np.exp(parameter.detach().numpy())
            elif name == 'pi_met':
                parameter = torch.softmax(parameter.detach(), 1).numpy()
            if torch.is_tensor(parameter):
                param_dict[args.seed][name].append(parameter.detach().numpy())
            else:
                param_dict[args.seed][name].append(parameter)
        if 'w' not in net.named_parameters():
            param_dict[args.seed]['w'].append(net.w_act.detach().numpy())
        param_dict[args.seed]['z'].append(net.z_act.detach().numpy())
        if net.linear:
            param_dict[args.seed]['beta[1:,:]*alpha'].append(
                net.beta[1:, :].detach().numpy() * net.alpha_act.detach().numpy())


        # Print the epoch and loss along the way to track progress
        if epoch % np.int(last_epoch/10) == 0:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))

        # at the last epoch, or at each 5000 epochs, plot results
        if epoch == last_epoch or epoch % 5000 == 0:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))

            # Make new path for new results
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

            if not os.path.isdir(path + 'seed' + str(args.seed) + '-clusters/'):
                os.mkdir(path + 'seed' + str(args.seed) + '-clusters/')
            best_mod = np.argmin(loss_vec)
            best_z = param_dict[args.seed]['z'][best_mod]
            best_w = np.round(param_dict[args.seed]['w'][best_mod])
            best_alpha = np.round(param_dict[args.seed]['alpha'][best_mod])

            # Save interactions if linear
            if args.linear == 1:
                get_interactions_csv(path, best_mod, param_dict, args.seed)

            # save alpha and omega to see how close they are to one
            if args.hard != 1:
                pd.DataFrame(param_dict[args.seed]['alpha'][best_mod]).to_csv(path + 'seed' + str(args.seed) + 'alpha.csv')
                pd.DataFrame(param_dict[args.seed]['w'][best_mod]).to_csv(path + 'seed' + str(args.seed) + 'omega.csv')

            # plot output locations if synthetic data
            if args.syn:
                plot_output_locations(path, net, best_mod, param_dict[args.seed], args.seed, plot_zeros = False)

            if not args.syn:

                # Plot each learned cluster's phylogenetic tree and metabolic classification tree
                met_newick_name = 'newick_' + args.yfile.split('.csv')[0] + '.nhx'
                active_asv_clust = list(set(np.where(np.sum(best_w,0) != 0)[0]).intersection(
                    set(np.where(np.sum(best_alpha,1)!= 0)[0])))
                active_met_clust = np.where(np.sum(best_z,0) != 0)[0]
                asv_df = {}
                for asv_clust in active_asv_clust:
                    asv_ix = np.where(best_w[:,asv_clust]!= 0)[0]
                    if seqs is not None:
                        asv_ix = seqs[asv_ix]
                    asv_df['Cluster ' + str(asv_clust)] = asv_ix
                    if not isinstance(asv_ix[0], str):
                        asv_ix = [str(a) for a in asv_ix]
                    inputs = ["python3", "tree_plotter.py", "-fun", 'asv', "-name", 'ASV_cluster_' + str(asv_clust) + '_tree.pdf',
                         "-out", path + 'seed' + str(args.seed) + '-clusters/',
                              "-newick",base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx', "-feat"]
                    inputs.extend(asv_ix)
                    if args.safari == 0 and args.syn == 0:
                        subprocess.run(inputs,cwd=base_path + "/ete_tree")
                if len(active_met_clust) > 20:
                    active_met_clust = active_met_clust[:20]
                met_df = {}
                for met_clust in active_met_clust:
                    met_ix = np.where(best_z[:, met_clust]!=0)[0]
                    if metabs is not None:
                        met_ix = metabs[met_ix]
                    met_df['Cluster ' + str(met_clust)] = met_ix
                    if not isinstance(met_ix[0], str):
                        met_ix = [str(a) for a in met_ix]
                    inputs = ["python3", "tree_plotter.py", "-fun", 'metab', "-name", 'Met_cluster_' + str(met_clust) + '_tree.pdf',
                         "-out", path+ 'seed' + str(args.seed) + '-clusters/',
                              "-newick", base_path + '/ete_tree/' + met_newick_name,"-feat"]
                    inputs.extend(met_ix)
                    if args.safari == 0 and args.syn == 0:
                        subprocess.run(inputs,cwd=base_path + "/ete_tree")

                temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in met_df.items()]))
                temp.to_csv(path + str(args.seed) + 'mets_in_clusters.csv')
                temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in asv_df.items()]))
                temp.to_csv(path + str(args.seed) + 'asvs_in_clusters.csv')

            # Save the number of active metabolite and microbial clusters to a text file
            if not os.path.isfile(path + 'Num_Clusters.txt'):
                with open(path + 'Num_Clusters.txt', 'w') as f:
                    f.writelines('Seed ' + str(args.seed) + ', K: ' + str(len(active_met_clust)) + ', L: ' + str(len(active_asv_clust)) + '\n')
            else:
                with open(path + 'Num_Clusters.txt', 'a') as f:
                    f.writelines('Seed ' + str(args.seed) + ', K: ' + str(len(active_met_clust)) + ', L: ' + str(len(active_asv_clust)) + '\n')


            # Save the lowest loss over the epochs to a text file
            if not os.path.isfile(path + 'Loss.txt'):
                with open(path + 'Loss.txt', 'w') as f:
                    f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec)) + '\n')
            else:
                with open(path + 'Loss.txt', 'a') as f:
                    f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec))+ '\n')


            # Plot the loss per learned parameter
            try:
                plot_loss_dict(path_orig, args.seed, loss_dict_vec)
            except:
                print('no loss dict')

            # Plot scatterplots of the active microbial cluster outputs versus the active metabolic cluster outputs to
            # see the relationship
            xdict, ydict= plot_xvy(path, x, train_out_vec, best_mod, param_dict, args.seed)

            # Plot parameter traces
            if plot_params and args.load == 0:
                plot_param_traces(path, param_dict[args.seed], params2learn, true_vals, net, args.seed)
            fig3, ax3 = plt.subplots(figsize=(8, 8))
            fig3, ax3 = plot_loss(fig3, ax3, args.seed, np.arange(len(loss_vec)), loss_vec, lowest_loss=None)
            fig3.tight_layout()
            fig3.savefig(path_orig + 'loss_seed_' + str(args.seed) + '.pdf')
            plt.close(fig3)

            # Plot posterior distribution histograms of each parameter
            plot_posterior(param_dict, args.seed, path_orig)

            # Plot predicted metabolic values per the first 5 participants
            plot_output(path, best_mod, train_out_vec, np.array(y), true_vals, param_dict[args.seed],
                                 args.seed, type = 'best_train', metabs = metabs, meas_var=args.meas_var)

            # Save the model at this epoch
            save_dict = {'model_state_dict':net.state_dict(),
                       'optimizer_state_dict':optimizer.state_dict(),
                       'epoch': epoch}
            torch.save(save_dict,
                       path_orig + 'seed' + str(args.seed) + '_checkpoint.tar')

            # Save the parameter dict, loss vector, microbial cluster sum for the model with the lowest loss, and
            # metabolite cluster outputs for the model with the lowest loss
            with open(path_orig + str(args.seed) + '_param_dict.pkl', 'wb') as f:
                pkl.dump(param_dict, f)
            with open(path_orig + str(args.seed) + '_loss.pkl', 'wb') as f:
                pkl.dump(loss_vec, f)
            with open(path_orig + str(args.seed) + '_microbe_sum.pkl', 'wb') as f:
                pkl.dump(xdict, f)
            with open(path_orig + str(args.seed) + '_met_clusters.pkl', 'wb') as f:
                pkl.dump(ydict, f)

            # Save the epoch per seed (needed when loading model)
            with open(path_orig + 'seed' + str(args.seed) + '.txt', 'w') as f:
                f.writelines(str(epoch))

            # Save the time (in minutes) per epoch
            etime= time.time()
            with open(path_orig + 'seed' + str(args.seed) + '_min_per_epoch.txt', 'w') as f:
                f.writelines(str(epoch) + ': ' + str(np.round((etime - stime)/60, 3)) + ' minutes')


    etime = time.time()
    print('total time:' + str(etime - stime))
    print('delta loss:' + str(loss_vec[-1] - loss_vec[0]))

if __name__ == "__main__":
    # Calls run_learner with input arguments
    # TO DO:
    # - make more generalizable to more input data; right now can use synthetic data (with or without location priors),
    # cdiff data (with or without location priors) and safari data (without location priors)
    # - could do the config method like Suhas does in MDITRE
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", "--learn", help="params to learn", type=str, nargs='+', default = 'all')
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default = 0.1)
    parser.add_argument("-fix", "--fix", help="params to fix", type=str, nargs='+')
    parser.add_argument("-priors", "--priors", help="priors to set", type=str, nargs='+', default = 'all')
    parser.add_argument("-case", "--case", help="case", type=str,
                        default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 10)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 10)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 3)
    parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 3)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.1)
    parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int,default = 10000)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 99)
    parser.add_argument("-load", "--load", help="0 to not load model, 1 to load model", type=int, default = 0)
    parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 0)
    parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 0)
    parser.add_argument("-hard", "--hard", help="whether or not to sample alpha and omega in the forward pass", type=int, default=0)
    parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int, default=1000)
    parser.add_argument("-linear", "--linear", type = int, default = 0, help = 'whether to run linear model or not')
    parser.add_argument("-nltype", "--nltype", type = str, default = "exp",
                        help = 'if using synthetic data and linear == 0, how to non-linearly generate data'
                               'choices are: exp, poly, sine, linear, sigmoid')
    parser.add_argument("-adjust_lr", "--adjust_lr", type=int, default=1,
                        help = "whether to adjust the learning rate based on the size of the parameter")
    parser.add_argument("-dist_var_perc", "--dist_var_perc", type=float, default=0.5,
                        help = "if generating data, how far apart to generate embedding points within a cluster. "
                               "0.1 = very close, 0.9 = far")
    parser.add_argument("-p_num", "--p_num", type=int, default=1,
                        help = "if non-linear, how many neural networks per microbe cluster - metabolite cluster interaction"
                               "(i.e. p=1 means 1 NN per each interaction)")
    parser.add_argument("-xdim", "--xdim", type=int, default=2, help = 'embedding dimension for microbes')
    parser.add_argument("-ydim", "--ydim", type=int, default=2, help = 'embedding dimension for metabolites')
    parser.add_argument("-a_tau", "--a_tau", type=float, nargs = '+', default=[-0.3, -3], help = 'annealing for alpha temparature')
    parser.add_argument("-w_tau", "--w_tau", type=float, nargs='+', default=[-0.3, -3], help = 'anealing for omega temperature')
    parser.add_argument("-locs","--locs", type = str, default = 'true',
                        help = 'true= use true microbe and metabolite embedded locations; '
                                                                               'none= dont use locations;'
                                                                               'rand= use random locations;')
    parser.add_argument("-dtype", "--dtype", type=str, default='',
                        help = "which type of distance embedding to use, choices are:" 
                               " 'stratified', 'clumps', '', 'pubchem', 'RDK','MACCS' ")
    parser.add_argument("-dim", "--dim", type=float, default=2, help = 'embedding dimension if using synthetic data')
    parser.add_argument("-syn", "--syn", type=int, default=0, help = 'whether or not to use synthetic data')
    parser.add_argument("-yfile", "--yfile", type=str, default='y_high_corr.csv', help = 'which y data file to use')
    parser.add_argument("-gmm", "--gmm", type=int, default=0, help = 'whether to run as gaussian mixture model or not '
                                                                     '(i.e. if 1, zero out input and just cluster metabolites)')
    parser.add_argument("-safari", "--safari", type=int, default=1, help = 'whether to run with safari data or not')
    parser.add_argument("-saf_type", "--saf_type", type=str, default='polar', help= 'if safari == 1, which safari data type to run')
    parser.add_argument("-most_corr", "--most_corr", type=int, default = 1,
                        help = 'if safari == 1, whether to use the data with high correlation bw microbes and metabolites or not')
    args = parser.parse_args()
    print(sys.executable)
    args.case = args.locs + '_' + args.case
    dtype = args.dtype
    base_path = os.getcwd()
    if '/M2M' not in base_path:
        base_path = '/Users/jendawk/Dropbox (MIT)/M2M/'
    if not os.path.isdir(base_path + '/outputs/'):
        os.mkdir(base_path + '/outputs/')
    if not os.path.isdir(base_path + '/outputs/' + args.case):
        os.mkdir(base_path + '/outputs/' + args.case)
    gen_data = args.syn==1
    if args.safari==1 and args.syn==0:
        # TO DO: calculate and use actual measurement variance of safari data, not just 0.1 default
        met_dict, asv_dict, res_dict = load_safari_data(met_frac = 0.9, bug_frac = 0.15)
        y = met_dict[args.saf_type]['log_std_filt']
        x = asv_dict['ra_filt']
        if args.most_corr:
            df = pd.DataFrame(res_dict[args.saf_type]).T
            pairs = df.index.values[df['rho'] > 0.5]
            metabs, microbes = list(zip(*pairs))
            y = y[set(metabs)]
            x = x[set(microbes)]

        x = (x.T/np.sum(x,1)).T

        y = y.loc[x.index.values]
        args.meas_var = 0.1
        args.locs = 'none'
        args.N_met = y.shape[1]
        args.N_bug = x.shape[1]
        args.N_samples = y.shape[0]
        ylocs, xlocs, y_class, x_fams = None, None, None, None

    elif not gen_data:
        # args.case = args.case + '_100Bvar'
        calc_dim = True
        xfile = 'x_high_corr.csv'
        yfile = args.yfile
        xdist_file = 'x_dist.csv'
        if '_' in dtype:
            ydist_file = dtype.split('_')[0] + '/' + dtype.split('_')[1] + '-dist.csv'
        else:
            ydist_file = 'y' + dtype + '_dist.csv'
        # ydist_file =
        met_newick_name = 'newick_' + args.yfile.split('.csv')[0] + '.nhx'

        # set data_path to point to directory with data
        data_path = base_path + "/inputs/processed/"
        # Option to change filtering criteria
        if xfile not in os.listdir(data_path) or yfile not in os.listdir(data_path):
            load_data(base_path, xfile, yfile, dataLoader)
        x = pd.read_csv(data_path + '/' + xfile, index_col = [0])
        if yfile not in os.listdir(data_path):
            ml = metabLoader(non_zero_perc=int(yfile.split('-')[1]), meas_thresh=0,
                             var_perc=int(yfile.split('-')[-1].split('.')[0]), week=1)
            y = ml.data['x']
            y.to_csv(data_path + '/' + yfile, index_col = 0)
        else:
            y = pd.read_csv(data_path + '/' + yfile, index_col = [0])
        y = y.loc[x.index.values]

        if yfile.split('.')[0] + '-mvar.pkl' not in os.listdir(data_path):
            ml = metabLoader(non_zero_perc=0, meas_thresh=0, var_perc=0, week=1)
            raw_dat = ml.cdiff_dat
            replicate_ixs = [d for d in raw_dat.index.values if not d.split('-')[1].split('.')[-1].isnumeric()]
            repeat_dat = raw_dat[y.columns.values].loc[replicate_ixs]
            y_raw = ml.data_ntr_filt['x']
            rep_pts = [ix.split('-')[0] for ix in replicate_ixs]
            unique_ixs = np.unique(rep_pts)
            rep_list = [repeat_dat.loc[[ix for ix in replicate_ixs if ix.split('-')[0] == unique_ix]] for unique_ix in
                        unique_ixs]
            pooled_var = get_meas_var(y_raw, rep_list)
            with open(data_path + '/' + yfile.split('.')[0] + '-mvar.pkl', 'wb') as f:
                pkl.dump(pooled_var, f)

        with open(data_path + '/' + yfile.split('.')[0] + '-mvar.pkl', 'rb') as f:
            args.meas_var = pkl.load(f)

        args.N_met = y.shape[1]
        args.N_bug = x.shape[1]
        args.N_samples = y.shape[0]

        print(x.shape)
        print(y.shape)
        make_tree(x.columns.values, base_path, args.case, 'asv',
                  newick_path='/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx')
        make_tree(y.columns.values, base_path, args.case, 'metab_orig', newick_path='/ete_tree/' + met_newick_name,
                  dist_type=dtype)

        if args.locs == 'true':
            if xdist_file not in os.listdir(base_path + '/inputs/processed/'):
                make_dist_mat(x, xdist_file, base_path, outfile = base_path + '/inputs/processed/',
                              newick_path = '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx')
            xdist = pd.read_csv(base_path + '/inputs/processed/' + xdist_file, header=0, index_col=0)
            xdist = xdist / np.max(np.max(xdist))

            if ydist_file not in os.listdir(base_path + '/inputs/processed/'):
                make_dist_mat(y, ydist_file, base_path, newick_path = '/ete_tree/' + met_newick_name,
                              yfile = yfile, outfile = base_path + '/inputs/processed/')
            ydist = pd.read_csv(base_path + '/inputs/processed/' + ydist_file, header = 0, index_col = 0)
            ydist = 1- (ydist / np.max(np.max(ydist)))

            if calc_dim:
                args.xdim, xlocs, xstress = mds_choose_d(xdist,seed = args.seed)
                args.ydim, ylocs, ystress = mds_choose_d(ydist, seed=args.seed)
            else:
                embedding = MDS(n_components=args.xdim, dissimilarity='precomputed', random_state=args.seed)
                xlocs = embedding.fit_transform(xdist)
                embedding = MDS(n_components=args.ydim, dissimilarity='precomputed', random_state=args.seed)
                ylocs = embedding.fit_transform(ydist)

            x_fams = get_xtaxa(base_path + '/inputs/taxa_labels.csv', x)
            y_class = get_ytaxa(base_path + '/inputs/processed/classy-fire/classy_fire_df.csv', y.columns.values, ydist, level='subclass')

            xlocs = (xlocs - np.mean(xlocs, 0))/np.std(xlocs,0)
            ylocs = (ylocs - np.mean(ylocs, 0))/np.std(ylocs, 0)
        elif args.locs == 'random':
            ylocs = get_rand_locs(y, args.ydim, args.seed)
            xlocs = get_rand_locs(x, args.xdim, args.seed)
            x_fams, y_class = None, None
        else:
            xlocs, ylocs = None, None
            x_fams, y_class = None, None
    else:
        if args.locs == 'none':
            x,y,ylocs,xlocs,y_class,x_fams = None, None, None, None, None, None
        else:
            x, y, ylocs, xlocs, y_class, x_fams = None, None, True, True, None, None
        # args.N_samples = 49


    run_learner(args, device, x=x, y=y, base_path = base_path, a_met = ylocs, a_bug = xlocs, met_class = y_class, bug_class = x_fams)
