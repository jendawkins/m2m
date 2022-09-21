# #!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from torch.distributions.dirichlet import Dirichlet
from helper import *
from plot_helper import *
from concrete import *
import argparse
import re
from synthetic_data_generation import *
import sys
from dataLoader import *
# from rdk.rdk_fingerprints import *

import subprocess
from sklearn.cluster import KMeans
import scipy
from torch.distributions.half_normal import HalfNormal
from safariLoader import *
from model import *
import datetime
from ete_tree.tree_plotter import *
from sklearn.model_selection import train_test_split



def run_learner(args, device, x=None, y=None, a_met=None, a_bug = None, base_path = '', plot_params = True,
                met_class = None, bug_class = None):
    """
    Calls model and trains over set epochs for given input arguments
    inputs:
    - args: input arguments from argument parser
    - device: cuda or not
    - x: microbial relative abundances
    - y: metabolic standardized levels
    - a_met: metabolite embedded locations
    - a_bug: microbial embedded locations
    - base_path: current working dir
    - plot_params: Whether to plot parameter traces or not (this can take time so sometimes I don't)
    - met_class: Which metabolite sub-class/level to use for initializing metabolite cluster centers and radii (only
              used if a_met is not none
    - bug_class: Which microbe level (family, genus, species, etc) to use for initializing microbial cluster centers
              and radii (I use family level)
    TO DO:
    - automatic stopping after loss stops decreasing / model has converged
    - make more effecient / faster
    - decide which plotting to keep and which to get rid of since that's a major time burden
    - torch distributed learning (like suhas does in MDITRE)

    """

    if x is not None and y is not None:
        metabs = y.columns.values
        seqs = x.columns.values
        args.learn = 'all'
        # set path for saving results
    else:
        metabs = None
        seqs = None
    path = base_path + '/outputs/'
    if not os.path.isdir(path):
        os.mkdir(path)

    # add case to path for saving results
    path = path + args.case.replace(' ','_')
    if not os.path.isdir(path):
        os.mkdir(path)


    # args.fix fixes parameters to their generated value (if using synthetic data)
    # This is pretty much for debugging
    if args.fix and args.data != 'synthetic':
        args.fix = None


    # if we fix some parameters, add another folder to path
    if args.fix and args.data == 'synthetic':
        path = path + '/fix_' + '_'.join(args.fix) + '/'
        if not os.path.isdir(path):
            os.mkdir(path)


    # add all other specified inputs to path to prevent overwriting results & keep a record of the parameters each model is run with
    info = 'lr' + str(args.lr) + '-linear'*(args.linear) + '-adj_lr'*args.adjust_lr + \
            '-'*(1-args.linear) +args.nltype*(1-args.linear)*(args.data=='synthetic') + '-lm'*args.lm + '-lb'*args.lb + \
            '-Nmet' + str(args.N_met) + '-Nbug' + str(args.N_bug) + '-N_samples' + str(args.N_samples) +\
           '-L' + str(args.L) + '-K' + str(args.K) + '-dim' + str(args.xdim) + \
           '-atau' + str(args.a_tau).replace('.','_') + '-wtau' + str(args.w_tau).replace('.', '_')

    path = path + '/' + info + '/'
    if not os.path.isdir(path):
        try:
            os.mkdir(path)
        except:
            pass


    path = path + '/seed_{0}/'.format(args.seed)
    if not os.path.isdir(path):
        os.mkdir(path)

    # If function is called without input x and y data, generate synthetic data by calling data_gen.py and then plot
    if args.data == 'synthetic':
        x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
        mu_met, r_bug, r_met = generate_synthetic_data(p=0.5,
            N_met = args.N_met, N_bug = args.N_bug, N_met_clusters = args.K_true,
            N_bug_clusters = args.L_true ,N_samples=args.N_samples, linear = args.linear,
            nl_type = args.nltype, xdim=args.xdim, ydim = args.ydim)

        tr_ids, val_ids = train_test_split(np.arange(x.shape[0]), test_size=0.2, random_state=args.seed)
        x_val, y_val, g_val = x[val_ids,:], y[val_ids,:], g[val_ids,:]
        x_val = torch.Tensor(np.array(x_val)).to(device)
        y_val = torch.Tensor(np.array(y_val))

        x, y, g = x[tr_ids, :], y[tr_ids, :], g[tr_ids, :]
        if not args.linear:
            gen_beta = gen_beta[0,:]
        plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                      r_bug, mu_met, r_met, gen_w, gen_alpha, gen_beta)

        if a_met is not None:
            a_met = gen_met_locs
        if a_bug is not None:
            a_bug = gen_bug_locs

        # If we are learning the number of metabolite clusters or the number of microbial clusters,
        # need to expand the generated true parameter matrices to compare to learned parameters
        if args.lm:
            gen_z = np.hstack((gen_z, np.zeros((args.N_met, args.N_met - 1 - args.K))))
            if a_met is not None:
                mu_met = np.vstack((mu_met, np.zeros((args.N_met - args.K - 1, mu_met.shape[1]))))
                r_met = np.append(r_met, np.zeros(args.N_met - 1 - len(r_met)))
            if args.linear:
                gen_beta = np.hstack((gen_beta, np.zeros((gen_beta.shape[0], args.N_met - args.K - 1))))
            gen_alpha = np.hstack((gen_alpha, np.zeros((gen_alpha.shape[0], args.N_met - args.K - 1))))
        if args.lb:
            if a_bug is not None:
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
        # just for plotting; see the interaction b/w clusters
        if args.linear:
            true_vals['beta[1:,:]*alpha'] = gen_beta[1:,:]*sigmoid(gen_alpha)
    else:
        true_vals = None

    # Define model and initialize with input seed
    net = Model(a_met, a_bug, K=args.K, L=args.L,
                N_met = y.shape[1], N_bug = x.shape[1], alpha_temp=10**args.a_tau[0], omega_temp=10**args.w_tau[0],
                learn_num_bug_clusters=args.lb,learn_num_met_clusters=args.lm, linear = args.linear==1,
                p_nn = args.p_num, data_meas_var = args.meas_var, met_class = met_class, bug_class = bug_class)

    x = torch.Tensor(np.array(x)).to(device)
    y = torch.Tensor(np.array(y))
    net.initialize(args.seed, x, y)
    net.to(device)

    # plot prior distributions for all parameters
    for param, dist in net.distributions.items():
        if param != 'NAM':
            try:
                plot_distribution(dist, param, true_val = true_vals, ptype = 'prior', path = path)
            except:
                print(param + ' plot distribution error!!')

    # Set tau schedules for alpha and omega given inputs
    alpha_tau_logspace = np.logspace(args.a_tau[0], args.a_tau[1], args.iterations+1)
    omega_tau_logspace = np.logspace(args.w_tau[0], args.w_tau[1], args.iterations+1)

    # Record initial parameter values (we will also record per epoch for plotting purposes)
    param_dict = {}
    start = 0
    for name, parameter in net.named_parameters():
        if 'NAM' in name:
            continue
        if name in args.fix:
            if true_vals is not None:
                if name == 'r_bug' or name == 'r_met' or name == 'e_met' or name == 'pi_met':
                    setattr(net, name, nn.Parameter(torch.tensor(np.log(true_vals[name])).float(), requires_grad=False))
                else:
                    setattr(net, name, nn.Parameter(torch.Tensor(true_vals[name]), requires_grad=False))
        else:
            if name == 'alpha':
                param = getattr(net, name + '_act').detach().numpy()
            else:
                param = parameter.detach().numpy()
            param_dict[name] = [param.copy()]

    param_dict['z'] = [net.z_act.detach().numpy().copy()]
    param_dict['w'] = [net.w_act.detach().numpy().copy()]
    if net.linear:
        param_dict['beta*alpha'] = [net.beta[1:,:].detach().numpy()*net.alpha_act.detach().numpy()]

    # Plot initial metabolite and microbial classification / phylogenetic trees, to compare to learned trees
    if not os.path.isdir(path + '/init_clusters/'):
        os.mkdir(path + '/init_clusters/')
    best_z = param_dict['z'][0]
    best_w = np.round(param_dict['w'][0])
    best_alpha = np.round(param_dict['alpha'][0])
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
        if args.data == 'cdi':
            plot_asv_tree(newick_path=base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
                          out_path= path + '/init_clusters/', data_path=base_path + '/inputs/' + args.data + '/',
                          taxa_keep=asv_ix, name = 'ASV_cluster_' + str(asv_clust) + '_tree_init.pdf')
    for met_clust in active_met_clust:
        met_ix = np.where(best_z[:, met_clust] != 0)[0]
        if metabs is not None:
            met_ix = metabs[met_ix]
        if not isinstance(met_ix[0], str):
            met_ix = [str(a) for a in met_ix]
        if args.data == 'cdi':
            plot_metab_tree(mets_keep = met_ix, newick_path=base_path + '/ete_tree/' + args.met_newick_name,
                            out_path=path + '/init_clusters/', name = 'Met_cluster_' + str(met_clust) + '_tree_init.pdf')

    # If embedding dimension = 2 and we have embedded locations input, plot output locations
    if a_met is not None and args.xdim == 2 and args.ydim == 2:
        plot_output_locations(path + 'init_clusters_', net, 0, param_dict, args.seed, plot_zeros=1)


    # Adjust each parameter's learning rate based on parameter size
    lr_dict = {}
    lr_list = []
    size_beta = torch.mean(torch.abs(net.beta.detach().flatten()))
    for name, parameter in net.named_parameters():
        size = torch.mean(torch.abs(parameter.detach().flatten()))
        if args.adjust_lr:
            lr_list.append({'params': parameter, 'lr': (args.lr / size_beta) * size})
        else:
            lr_list.append({'params': parameter})
        lr_dict[name] = [(args.lr / size_beta) * size]

    # initialize optimizer with learning rates
    optimizer = optim.RMSprop(lr_list, lr=args.lr)

    # record per-parameter learning rates
    if args.adjust_lr:
        pd.DataFrame(lr_dict).T.to_csv(path + 'per_param_lr.csv')

    # Train model over the number of specified input iterations in args.iterations
    loss_dict_vec = {}
    ix = 0
    loss_vec = []
    train_out_vec = []
    val_loss_vec = []
    val_out_vec = []
    path_orig = path
    for epoch in np.arange(start, args.iterations+1):
        if epoch ==1:
            stime = time.time()
        net.alpha_temp = alpha_tau_logspace[ix]
        net.omega_temp = omega_tau_logspace[ix]
        ix += 1
        optimizer.zero_grad()
        cluster_outputs, loss = net(x, y)
        train_out_vec.append(cluster_outputs)

        # If model can't learn for whatever reason, set last_epoch = epoch so that the last successful epoch is plotted
        # try:
        loss.backward()
        loss_vec.append(loss.detach().item())
        for param in net.MAPloss.loss_dict:
            if param not in loss_dict_vec.keys():
                loss_dict_vec[param] = [net.MAPloss.loss_dict[param].detach().item()]
            else:
                loss_dict_vec[param].append(net.MAPloss.loss_dict[param].detach().item())
        optimizer.step()
        last_epoch = args.iterations
        # except:
        #     last_epoch = epoch
        #     loss_vec.append(loss_vec[-1])
        #     for param in net.MAPloss.loss_dict:
        #         loss_dict_vec[param].append(loss_dict_vec[param][-1])

        # keep track of updated parameter values
        for name, parameter in net.named_parameters():
            if 'NAM' in name:
                continue
            if name == 'alpha':
                param = getattr(net, name + '_act').detach().numpy()
            else:
                param = parameter.detach().numpy()
            param_dict[name].append(param.copy())
        param_dict['w'].append(net.w_act.detach().numpy().copy())
        param_dict['z'].append(net.z_act.detach().numpy().copy())
        if net.linear:
            param_dict['beta*alpha'].append(
                net.beta[1:, :].detach().numpy().copy() * net.alpha_act.detach().numpy().copy())

        if epoch > 200 and args.early_stopping == 1:
            if epoch != last_epoch and (np.max(loss_vec[-100:]) - np.min(loss_vec[-100:])) <= 1:
                last_epoch = epoch

        # if epoch % 10 == 0 or epoch == last_epoch and epoch > 1:
        # net.eval()
        # val_cluster_outputs, val_loss = net(x_val, y_val)
        # val_out_vec.append(val_cluster_outputs.detach().numpy())
        # val_loss_vec.append(val_loss.detach().numpy())
        # net.train()
        # at the last epoch, or at each 5000 epochs, plot results
        if epoch == last_epoch or (epoch % 5000 == 0 and epoch > 1):
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))

            # Make new path for new results
            if 'epoch' not in path:
                path = path + 'epoch' + str(epoch) + '/'
            else:
                path = path.split('epoch')[0] + 'epoch' + str(epoch) + '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            if net.met_embedding_dim == 2 and net.bug_embedding_dim==2 and args.locs != 'none':
                plot_output_locations(path, net, -1, param_dict, args.seed,
                                      type='best_train', plot_zeros=False)
                plot_output_locations(path, net, -1, param_dict, args.seed,
                                      type='best_train', plot_zeros=True)

            if not os.path.isdir(path + 'seed' + str(args.seed) + '-clusters/'):
                os.mkdir(path + 'seed' + str(args.seed) + '-clusters/')
            best_train_mod = np.argmin(loss_vec)
            # best_val_mod = np.argmin(loss_vec)

            train_path = path + '/train/'
            test_path = path + '/test/'
            if not os.path.isdir(train_path):
                os.mkdir(train_path)
            if not os.path.isdir(test_path):
                os.mkdir(test_path)

            best_mod = best_train_mod
            cur_path = path
            out_vec = train_out_vec
            # for best_mod, cur_path, out_vec, ids in zip([best_train_mod, best_train_mod], [train_path, test_path],
            #                                             [train_out_vec, val_out_vec], [tr_ids, val_ids]):
            best_z = param_dict['z'][best_mod]
            best_w = np.round(param_dict['w'][best_mod])
            best_alpha = np.round(param_dict['alpha'][best_mod])

            # Save interactions if linear
            if args.linear == 1:
                get_interactions_csv(cur_path, best_mod, param_dict, args.seed)

            # save alpha and omega to see how close they are to one
            pd.DataFrame(param_dict['alpha'][best_mod]).to_csv(cur_path + 'alpha.csv')
            pd.DataFrame(param_dict['w'][best_mod]).to_csv(cur_path + 'omega.csv')

            if args.data != 'synthetic':
                # Plot each learned cluster's phylogenetic tree and metabolic classification tree
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
                    if args.data == 'cdi':
                        plot_asv_tree(newick_path=base_path + '/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
                                      out_path=cur_path + '-clusters/', data_path=base_path + '/inputs/' + args.data + '/',
                                      taxa_keep=asv_ix, name = 'ASV_cluster_' + str(asv_clust) + '_tree.pdf')
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
                    if args.data == 'cdi':
                        plot_metab_tree(mets_keep=met_ix, newick_path=base_path + '/ete_tree/' + args.met_newick_name,
                                        out_path= cur_path+ '-clusters/', name = 'Met_cluster_' + str(met_clust) + '_tree.pdf')

                temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in met_df.items()]))
                temp.to_csv(cur_path +'mets_in_clusters.csv')
                temp = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in asv_df.items()]))
                temp.to_csv(cur_path +'asvs_in_clusters.csv')


            # Save the lowest loss over the epochs to a text file
            if not os.path.isfile(cur_path + 'Loss.txt'):
                with open(cur_path + 'Loss.txt', 'w') as f:
                    f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec)) + '\n')
            else:
                with open(cur_path + 'Loss.txt', 'a') as f:
                    f.writelines('Seed ' + str(args.seed) + ', Lowest Loss: ' + str(np.min(loss_vec))+ '\n')

            net.eval()
            val_cluster_outputs, val_loss = net(x_val, y_val)
            # val_out_vec.append(val_cluster_outputs.detach().numpy())
            # val_loss_vec.append(val_loss.detach().numpy())
            net.train()

            pred_clusters = val_cluster_outputs
            preds = torch.matmul(pred_clusters + args.meas_var * torch.randn(pred_clusters.shape), torch.Tensor(best_z).T)
            pd.DataFrame(preds.detach().numpy()).to_csv(path_orig + '/' + 'val_predictions.csv', header=False, index=True)

            if not os.path.isfile(cur_path + 'ValLoss.txt'):
                with open(cur_path + 'ValLoss.txt', 'w') as f:
                    f.writelines('Seed ' + str(args.seed) + ', Current Val Loss: ' + str(val_loss.detach().item()) + '\n')
            else:
                with open(cur_path + 'Loss.txt', 'a') as f:
                    f.writelines('Seed ' + str(args.seed) + ', Current Val Loss: ' + str(val_loss.detach().item())+ '\n')

            # Plot the loss per learned parameter
            plot_loss_dict(cur_path, args.seed, loss_dict_vec)

            # Plot scatterplots of the active microbial cluster outputs versus the active metabolic cluster outputs to
            # see the relationship
            xdict, ydict= plot_xvy(cur_path, x, out_vec, best_mod, param_dict, args.seed)

            # Plot parameter traces
            if plot_params:
                plot_param_traces(cur_path, param_dict, true_vals, args.seed)

            # plot loss
            plot_loss(args.seed, loss_vec)

            # Plot posterior distribution histograms of each parameter
            plot_posterior(param_dict, args.seed, cur_path)

            # Plot predicted metabolic values per the first 5 participants
            plot_output(cur_path, best_mod, out_vec, np.array(y.detach().numpy()), param_dict,
                                 args.seed, meas_var=args.meas_var)

            plot_annealing(cur_path, param_dict['w'], omega_tau_logspace, param_name='omega')
            plot_annealing(cur_path, param_dict['alpha'], alpha_tau_logspace, param_name='alpha')

            # save info about predicted metabolite clusters and microbe groups
            save_cluster_results(cur_path, best_mod, true_vals, args.seed,
                                 param_dict, metabs=metabs, seqs=seqs)

            # pd.DataFrame(train_out_vec[best_train_mod], index = ids).to_csv(cur_path + '/predictions.csv', index=True, header=False)

            # Save the model at this epoch
            save_dict = {'model_state_dict':net.state_dict(),
                       'optimizer_state_dict':optimizer.state_dict(),
                       'epoch': epoch}
            torch.save(save_dict,
                       path_orig + '_checkpoint.tar')

            # Save the parameter dict, loss vector, microbial cluster sum for the model with the lowest loss, and
            # metabolite cluster outputs for the model with the lowest loss
            # if not os.path.isdir(path_orig + '/seed_{0}'.format(args.seed)):
            #     os.mkdir(path_orig + '/seed_{0}'.format(args.seed))

            with open(path_orig + '/param_dict.pkl', 'wb') as f:
                pkl.dump(param_dict, f)
            with open(path_orig +'/loss.pkl', 'wb') as f:
                pkl.dump(loss_vec, f)
            with open(path_orig+ '/microbe_sum.pkl', 'wb') as f:
                pkl.dump(xdict, f)
            with open(path_orig + '/met_clusters.pkl', 'wb') as f:
                pkl.dump(ydict, f)

            for key in param_dict.keys():
                pd.DataFrame(param_dict[key][best_train_mod]).to_csv(path_orig + '/' + key + '.csv',
                                                     header=False, index=False)

            x_tr, y_tr = pd.DataFrame(x.detach().numpy(), index = tr_ids), pd.DataFrame(y.detach().numpy(), index = tr_ids)
            x_tr.to_csv(path_orig+ '/' + 'training_data.csv', header=False, index=True)
            y_tr.to_csv(path_orig + '/' + 'training_labels.csv', header=False, index=True)

            x_val, y_val = pd.DataFrame(x_val.detach().numpy(), index = val_ids), pd.DataFrame(y_val.detach().numpy(), index = val_ids)
            x_val.to_csv(path_orig +  '/' + 'training_data.csv', header=False, index=True)
            y_val.to_csv(path_orig  + '/' + 'training_labels.csv', header=False, index=True)

            best_z = param_dict['z'][best_train_mod]
            pred_clusters = train_out_vec[best_train_mod]
            preds = torch.matmul(pred_clusters + args.meas_var * torch.randn(pred_clusters.shape), torch.Tensor(best_z).T)
            pd.DataFrame(preds.detach().numpy()).to_csv(path_orig + '/' + 'train_predictions.csv', header=False, index=True)

            # Save the epoch per seed (needed when loading model)
            with open(path_orig +'.txt', 'w') as f:
                f.writelines(str(epoch))

            # Save the time (in minutes) per epoch
            etime= time.time()
            with open(path_orig + '_min_per_epoch.txt', 'w') as f:
                f.writelines(str(epoch) + ': ' + str(np.round((etime - stime)/60, 3)) + ' minutes')

            if epoch == last_epoch:
                break


        elif epoch % np.int(last_epoch/10) == 0:
            print('Epoch ' + str(epoch) + ' Loss: ' + str(loss_vec[-1]))


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
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default = 0.001)
    parser.add_argument("-fix", "--fix", help="params to fix", type=str, nargs='+', default = [])
    parser.add_argument("-case", "--case", help="case", type=str,
                        default = datetime.date.today().strftime('%m %d %Y').replace(' ','-'))
    parser.add_argument("-N_met", "--N_met", help="N_met", type=int, default = 50)
    parser.add_argument("-N_bug", "--N_bug", help="N_bug", type=int, default = 30)
    parser.add_argument("-L", "--L", help="number of microbe rules", type=int, default = 3)
    parser.add_argument("-K", "--K", help="metab clusters", type=int, default = 4)
    parser.add_argument("-L_true", "--L_true", help="true number of microbe clusters "
                                                    "(for synthetic data generation)", type=int, default = 0)
    parser.add_argument("-K_true", "--K_true", help="true number of metab clusters "
                                                    "(for synthetic data generation)", type=int, default = 0)
    parser.add_argument("-meas_var", "--meas_var", help="measurment variance", type=float, default = 0.1)
    parser.add_argument("-iterations", "--iterations", help="number of iterations", type=int,default = 100)
    parser.add_argument("-seed", "--seed", help = "seed for random start", type = int, default = 99)
    parser.add_argument("-lb", "--lb", help = "whether or not to learn bug clusters", type = int, default = 0)
    parser.add_argument("-lm", "--lm", help = "whether or not to learn metab clusters", type = int, default = 0)
    parser.add_argument("-N_samples", "--N_samples", help="num of samples", type=int, default=1000)
    parser.add_argument("-linear", "--linear", type = int, default = 0, help = 'whether to run linear model or not')
    parser.add_argument("-nltype", "--nltype", type = str, default = "poly",
                        help = 'if using synthetic data and linear == 0, how to non-linearly generate data'
                               'choices are: exp, poly, sine, linear, sigmoid')
    parser.add_argument("-adjust_lr", "--adjust_lr", type=int, default=1,
                        help = "whether to adjust the learning rate based on the size of the parameter")
    parser.add_argument("-p_num", "--p_num", type=int, default=1,
                        help = "if non-linear, how many neural networks per microbe cluster - metabolite cluster interaction"
                               "(i.e. p=1 means 1 NN per each interaction)")
    parser.add_argument("-xdim", "--xdim", type=int, default=2, help = 'embedding dimension for microbes')
    parser.add_argument("-ydim", "--ydim", type=int, default=2, help = 'embedding dimension for metabolites')
    parser.add_argument("-a_tau", "--a_tau", type=float, nargs = '+', default=[-0.1, -2.5], help = 'annealing for alpha temparature')
    parser.add_argument("-w_tau", "--w_tau", type=float, nargs='+', default=[-0.1, -1.5], help = 'anealing for omega temperature')
    parser.add_argument("-locs","--locs", type = str, default = 'true',
                        help = 'true= use true microbe and metabolite embedded locations; '
                                                                               'none= dont use locations;'
                                                                               'rand= use random locations;')
    parser.add_argument("-dtype", "--dtype", type=str, default='pubchem_tanimoto',
                        help = "which type of distance embedding to use, choices are:" 
                               " 'stratified', 'clumps', '', 'pubchem_tanimoto', 'RDK_tanimoto','MACCS_tanimoto' ")
    parser.add_argument("-data", "--data", type = str, default = 'synthetic', help = "which input data to use; choices are: "
                                                                               "'cdi', 'safari', 'synthetic' ")
    parser.add_argument("-saf_type", "--saf_type", type=str, default='polar', help= 'if args.data == safari, which safari data type to run'
                                                                                    'options are: polar, lipids-neg, or lipids-pos')

    # Filtering criteria if args.data == 'cdi' or args.data == 'safari'
    parser.add_argument("-nzm", "--non_zero_perc_met", type=float, default=80,
                        help='percent of participants with non-zero metabolites in filtered data')
    parser.add_argument("-nzb", "--non_zero_perc_bug", type=float, default=15,
                        help='percent of participants with non-zero microbes in filtered data')
    parser.add_argument("-cvm", "--coef_var_perc_met", type=float, default=5,
                        help='coefficient of variation percentile for metabolites')
    parser.add_argument("-cvb", "--coef_var_perc_bug", type=float, default=0,
                        help='coefficient of variation percentile for microbes')

    parser.add_argument("-most_corr", "--most_corr", type=int, default = 0,
                        help = 'whether to use the data with high correlation bw microbes and metabolites or not')

    parser.add_argument("-early_stopping", "--early_stopping", default = None,
                        help = 'whether to stop training early')


    args = parser.parse_args()
    print(sys.executable)

    if args.L_true == 0:
        args.L_true = args.L
    if args.K_true == 0:
        args.K_true = args.K

    if args.L_true < args.L:
        args.lb = 1
    else:
        args.lb = 0
    if args.K_true < args.K:
        args.lm = 1
    else:
        args.lb = 0

    args.case = args.data + '_' + args.locs + '_' + args.case
    dtype = args.dtype
    base_path = os.getcwd()
    if '/M2M' not in base_path:
        base_path = '/Users/jendawk/M2M/'
    if not os.path.isdir(base_path + '/outputs/'):
        os.mkdir(base_path + '/outputs/')
    if not os.path.isdir(base_path + '/outputs/' + args.case):
        os.mkdir(base_path + '/outputs/' + args.case)

    args.raw_data_path = '/inputs/' + args.data + '/'
    if args.data == 'safari':
        # TO DO: calculate and use actual measurement variance of safari data, not just 0.1 default
        met_dict, asv_dict, res_dict = load_safari_data(data_path= base_path + args.raw_data_path,
                                                        out_path = base_path + '/inputs/processed/',
            met_frac = args.non_zero_perc_met/100, bug_frac = args.non_zero_perc_bug/100)
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

    elif args.data == 'cdi':
        if args.most_corr == 1:
            xfile = 'x_high_corr.csv'
            yfile = 'y_high_corr.csv'
        else:
            yfile = 'y_' + str(int(args.non_zero_perc_met)) + '_' + str(int(args.coef_var_perc_met)) + '.csv'
            xfile = 'x_' + str(int(args.non_zero_perc_bug)) + '_' + str(int(args.coef_var_perc_bug)) + '.csv'

        args.met_newick_name = 'newick_' + yfile.split('.csv')[0] + '.nhx'

        # Option to change filtering criteria
        if xfile not in os.listdir(base_path + "/inputs/processed/") or yfile not in os.listdir(base_path + "/inputs/processed/"):
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

        print(x.shape)
        print(y.shape)
        plot_asv_tree(newick_path=base_path +'/ete_tree/phylo_placement/output/newick_tree_query_reads.nhx',
                      out_path=base_path + '/outputs/' + args.case, data_path = base_path + '/' + args.raw_data_path,
                      name = 'init_phylo_tree.pdf', taxa_keep = x.columns.values)

        try:
            plot_orig_metab_tree(out_path = base_path + '/outputs/' + args.case, name = 'init_metab_tree.pdf',
                                 data_path = base_path + '/' + args.raw_data_path, newick_path = base_path +'/ete_tree/' + args.met_newick_name,
                                 dist_type = dtype, in_mets = y.columns.values)
        except:
            print('cant plot metab tree')


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

            x_fams = get_xtaxa(base_path + '/' + args.raw_data_path + '/taxa_labels.csv', x)
            y_class = get_ytaxa(base_path + '/' + args.raw_data_path + '/classy_fire_df.csv', y.columns.values,
                                ydist, level='subclass')

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


    run_learner(args, device, x=x, y=y, base_path = base_path, a_met = ylocs, a_bug = xlocs, met_class = y_class, bug_class = x_fams)
