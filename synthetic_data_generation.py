from helper import *
from plot_helper import *

def generate_synthetic_data(N_met = 10, N_bug = 14, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2,
                            seed = 0, measurement_var=0.1, xdim = 2, ydim = 2, p = 0.5, linear = False,
                            nl_type = 'linear',r_bug_desired = 0.3, r_met_desired = 0.3,
                            nuisance = 0):
    np.random.seed(seed)
    pi_bug = st.dirichlet([10/N_bug_clusters]*N_bug_clusters).rvs().squeeze()
    mu_bug = st.multivariate_normal(np.zeros(xdim), np.eye(xdim, xdim)).rvs(N_bug_clusters)
    temp = st.norm(np.log(r_bug_desired),0.1).rvs(N_bug_clusters)

    r_bug = np.exp(temp)
    # r_bug = st.lognorm(0.01, 1).rvs(N_bug_clusters)
    w_id = st.multinomial(1, pi_bug).rvs(N_bug)
    bug_locs = np.zeros((N_bug, xdim))
    fig, ax = plt.subplots()
    for clusters in range(N_bug_clusters):
        cluster_id = np.where(w_id[:,clusters]==1)[0]
        bug_locs[cluster_id, :] = st.multivariate_normal(mu_bug[clusters,:], (1/9)*r_bug[clusters]*np.eye(xdim,
                                                                                     xdim)).rvs(len(cluster_id))
    w_one_hot = w_id
    for cluster in np.arange(N_bug_clusters):
        ixs = np.linalg.norm(bug_locs - mu_bug[cluster,:], axis = 1) < r_bug[cluster]
        w_one_hot[ixs, cluster] = 1

    if nuisance > 0:
        nuis_ix = np.random.choice(np.arange(N_bug_clusters), nuisance)
        w_one_hot[nuis_ix, :] = 1

    pi_met = st.dirichlet([10/N_met_clusters]*N_met_clusters).rvs().squeeze()
    mu_met = st.multivariate_normal(np.zeros(ydim), np.eye(ydim, ydim)).rvs(N_met_clusters)
    # r_met = st.lognorm(0.01, 1).rvs(N_met_clusters)
    temp = st.norm(np.log(r_met_desired),0.1).rvs(N_met_clusters)
    r_met = np.exp(temp)
    z_id = st.multinomial(1, pi_met).rvs(N_met)
    met_locs = np.zeros((N_met, ydim))
    for clusters in range(N_met_clusters):
        cluster_id = np.where(z_id[:, clusters] == 1)[0]
        met_locs[cluster_id, :] = st.multivariate_normal(mu_met[clusters,:], (1/9)*r_met[clusters]*np.eye(ydim,
                                                                                     ydim)).rvs(len(cluster_id))

    beta = st.norm(0,1).rvs((N_bug_clusters + 1, N_met_clusters))

    alpha = st.bernoulli(p).rvs((N_bug_clusters, N_met_clusters))


    a = st.uniform(10,100).rvs()
    g = np.zeros((N_samples, N_bug_clusters))
    # X_temp = np.zeros((N_samples, N_bug, N_bug_clusters))
    X = np.zeros((N_samples, N_bug))
    used_ixs = []
    for l in range(N_bug_clusters):
        outer_ixs = np.where(w_one_hot[:, l] == 1)[0]
        b = st.uniform(10,100).rvs()
        g[:, l] = st.uniform(a, b).rvs(N_samples)
        a = a + b + st.uniform(10,100).rvs()

        overlap_ixs = list(set(outer_ixs).intersection(used_ixs))
        outer_ixs = list(set(outer_ixs) - set(overlap_ixs))
        if len(outer_ixs)==0:
            g[:, l] = X[:, overlap_ixs].sum(1)
        else:
            g_func = g[:, l] - (X[:, overlap_ixs]).sum(1)
            conc = np.expand_dims(g_func/(len(outer_ixs)),1)
            p = [st.dirichlet(conc.squeeze()).rvs().squeeze() for n in range(N_samples)]
            X[:, outer_ixs] = np.stack([st.multinomial(int(np.round(g_func[n])), p[n]).rvs() for n in range(N_samples)]).squeeze()
            used_ixs.extend(outer_ixs)

    # for l in range(N_bug_clusters):
    #     outer_ixs = np.where(w_one_hot[:, l] == 1)[0]
    #     g_func = g[:, l:l+1]
    #     conc = np.expand_dims(g_func[outer_ixs, 0]/(len(outer_ixs)*w_one_hot[outer_ixs,:].sum(1)),1)
    #     p = [st.dirichlet(conc.squeeze()).rvs().squeeze() for n in range(N_samples)]
    #
    #     X_temp[:, outer_ixs, l] = np.stack([st.multinomial(int(np.round(g_func[n,:]*(
    #         len(outer_ixs)/np.sum(w_one_hot[outer_ixs,:])))), p[n]).rvs() for n in range(N_samples)]).squeeze()


    # X = X_temp.sum(-1)
    g_new = X@w_one_hot

    X = X / np.expand_dims(np.sum(X, 1), 1)
    g = g_new / np.expand_dims(np.sum(g_new, 1), 1)

    y = np.zeros((N_samples, N_met))
    for j in range(N_met):
        k = np.argmax(z_id[j,:])
        g_temp = (g - np.mean(g, 0)) / np.std(g - np.mean(g, 0))
        if not linear:
            if nl_type == 'exp':
                y[:, j] = np.random.normal(beta[0, k] + np.exp(g_temp) @ (beta[1:, k] * alpha[:, k]), np.sqrt(measurement_var))
            if nl_type == 'sigmoid':
                y[:, j] = np.random.normal(beta[0, k] + sigmoid(g_temp) @ (beta[1:, k] * alpha[:, k]), np.sqrt(measurement_var))
            if nl_type == 'sin':
                y[:, j] = np.random.normal(beta[0, k] + np.sin(g_temp) @ (beta[1:, k] * alpha[:, k]), np.sqrt(measurement_var))
            if nl_type == 'poly':
                y[:,j] = np.random.normal(beta[0, k] + (g_temp)**5 @ (beta[1:, k] * alpha[:, k]) - (g)**4 @ (
                        beta[1:, k] * alpha[:, k]), np.sqrt(measurement_var))
            if nl_type == 'linear':
                y[:, j] = np.random.normal(beta[0, k] + g_temp @ (beta[1:, k] * alpha[:, k]), np.sqrt(measurement_var))
        else:
            y[:, j] = np.random.normal(beta[0, k] + g_temp @ (beta[1:, k] * alpha[:, k]), np.sqrt(measurement_var))

    return X, y, g, beta, alpha, w_one_hot, z_id, bug_locs, met_locs, mu_bug, mu_met, r_bug, r_met

if __name__ == "__main__":
    N_bug = 50
    N_met = 80
    K=10
    L=10
    N = 200
    # path = datetime.date.today().strftime('%m %d %Y').replace(' ','-') + '/'
    orig_path = 'NEW_data_gen/'
    if not os.path.isdir(orig_path):
        os.mkdir(orig_path)

    r_default = 0.3
    nl_type = 'linear'
    meas_var = 0.01
    nuisance = 0

    # x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
    # mu_met, r_bug, r_met = generate_synthetic_data(N_met=N_met, N_bug=N_bug, N_met_clusters=K,
    #                                                N_bug_clusters=L, measurement_var=meas_var,
    #                                                N_samples=N, p=0.6, nuisance=nuisance,
    #                                                linear=False, nl_type=nl_type,
    #                                                r_bug_desired=r_default, r_met_desired=r_default)
    # path = orig_path + '/' + nl_type + '_r' + str(r_default).replace('.', '-') + \
    #        '_mvar' + str(meas_var).replace('.', '-') + 'nuisance' + str(nuisance) + '/'
    # if not os.path.isdir(path):
    #     os.mkdir(path)
    # plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
    #               r_bug, mu_met, r_met, gen_w, gen_alpha, gen_beta)

    # for nuisance in [4,6,8,10,15,20]:
    for meas_var in [0.01, 0.05, 0.1, 0.5, 1]:
        for nl_type in ['linear', 'poly', 'exp', 'sigmoid']:
            x, y, g, gen_beta, gen_alpha, gen_w, gen_z, gen_bug_locs, gen_met_locs, mu_bug, \
            mu_met, r_bug, r_met = generate_synthetic_data(N_met=N_met, N_bug=N_bug, N_met_clusters=K,
                                                                  N_bug_clusters=L, measurement_var=meas_var,
                                                                  N_samples=N, p = 0.6, nuisance= nuisance,
                                                                  linear=False, nl_type=nl_type,
                                                           r_bug_desired=r_default, r_met_desired=r_default)
            if np.isnan(y).any():
                print('NAN data, meas_var=' + str(meas_var) + ', nl_type=' + str(nl_type) + ', r=' + str(r_default))
                continue
            path = orig_path + '/' + nl_type + '_r' + str(r_default).replace('.','-') + \
                   '_mvar' + str(meas_var).replace('.','-') + 'nuisance' + str(nuisance) +  '/'
            if not os.path.isdir(path):
                os.mkdir(path)
            plot_syn_data(path, x, y, g, gen_z, gen_bug_locs, gen_met_locs, mu_bug,
                                            r_bug, mu_met, r_met, gen_w, gen_alpha, gen_beta)

            # fig, ax = plt.subplots(K, L, figsize=(8 * L, 8 * K))
            # # ranges = [[np.max(microbe_sum[:,i]/out[:,j]) - np.min(microbe_sum[:,i]/out[:,j]) for i in range(out.shape[1])] for j in range(out.shape[1])]
            # # ixs = [np.argmin(r) for r in ranges]
            # g = x @ gen_w
            # for i in range(K):
            #     ixs = np.where(gen_z[:, i] == 1)[0]
            #     for j in range(L):
            #         # ax[i].scatter(microbe_sum[:,ixs[i]], out[:,i])
            #         for ii in ixs:
            #             ax[i, j].scatter(g[:, j], y[:, ii])
            #         ax[i, j].set_xlabel('Microbe sum')
            #         ax[i, j].set_ylabel(r'$y_{i}$ when $i=$' + str(i))
            #         ax[i, j].set_title('Metabolite Cluster ' + str(i) + ' vs Microbe Cluster ' + str(j))
            # fig.tight_layout()
            # fig.savefig(orig_path + '-sum_x_v_y_v2.pdf')
            # plt.close(fig)



# from helper import *
# from sklearn.manifold import MDS
# from sklearn.cluster import KMeans
# from plot_helper import *
# import datetime
#
#
# def generate_synthetic_data(N_met = 10, N_bug = 14, N_samples = 200, N_met_clusters = 2, N_bug_clusters = 2,
#                             N_local_clusters=1, state = 3,
#                             beta_var = 2, cluster_disparity = 100, meas_var = 0.001,
#                             cluster_per_met_cluster = 0, repeat_clusters = 1,xdim = 2, ydim = 2,
#                             deterministic = True, linear = False, nl_type = "linear",dist_var_frac = 0.9,
#                             overlap_frac = 0.5, cluster_std = 1):
#
#     # Choose metabolite indices for each cluster
#     np.random.seed(state)
#     choose_from = np.arange(N_met)
#     met_gp_ids = []
#     for n in range(N_met_clusters-1):
#         # num_choose = np.random.choice(np.arange(2,len(choose_from)-(N_met_clusters-n)),1)
#         chosen = np.random.choice(choose_from, int(N_met/N_met_clusters),replace = False)
#         choose_from = list(set(choose_from) - set(chosen))
#         met_gp_ids.append(chosen)
#     met_gp_ids.append(np.array(choose_from))
#
#     # Generate distance matrix for metabolites
#     dist_met = np.zeros((N_met, N_met))
#     for i, gp in enumerate(met_gp_ids):
#         for met in gp:
#             dist_met[met, gp] = np.ones((1, len(gp)))
#             dist_met[gp, met] = dist_met[met,gp]
#     dist_met[dist_met == 0] = 1 + dist_var_frac
#     np.fill_diagonal(dist_met, 0)
#     dist_met = dist_met / np.max(np.max(dist_met))
#
#     # Choose microbial cluster indices for each microbe
#     if N_local_clusters <= 1:
#         N_clusters_by_dist = N_bug_clusters
#     else:
#         N_clusters_by_dist = N_local_clusters
#     choose_from = np.arange(N_bug)
#     bug_gp_ids = []
#     for n in range(N_clusters_by_dist-1):
#         chosen = np.random.choice(choose_from, int((N_bug/N_clusters_by_dist)),replace = False)
#         if repeat_clusters==1:
#             choose_from = np.array(list(set(choose_from) - set(chosen)) +
#                                    list(np.random.choice(chosen, np.int(len(chosen)*overlap_frac))))
#         else:
#             choose_from = np.array(list(set(choose_from) - set(chosen)))
#         bug_gp_ids.append(chosen)
#     if repeat_clusters==2:
#         bug_gp_ids.append(np.concatenate((choose_from, bug_gp_ids[-1])))
#     else:
#         bug_gp_ids.append(choose_from)
#
#     # Generate microbial cluster distance matrix
#     dist_bug = np.zeros((N_bug, N_bug))
#     for i, gp in enumerate(bug_gp_ids):
#         others = bug_gp_ids.copy()
#         others.pop(i)
#         others = np.concatenate(others)
#         for met in gp:
#             dist_bug[met, gp] = np.ones((1, len(gp)))
#             dist_bug[gp, met] = dist_bug[met,gp]
#     dist_bug[dist_bug == 0] = 1 + dist_var_frac
#     np.fill_diagonal(dist_bug, 0)
#     dist_bug = dist_bug / np.max(np.max(dist_bug))
#
#     # Generate cluster embedded locations
#     embedding = MDS(n_components=xdim, dissimilarity='precomputed', random_state=state)
#     met_locs = embedding.fit_transform(dist_met)
#     embedding = MDS(n_components=ydim, dissimilarity='precomputed', random_state=state)
#     bug_locs = embedding.fit_transform(dist_bug)
#
#     met_locs = (met_locs - np.mean(met_locs, 0)) / np.std(met_locs, 0)
#     bug_locs = (bug_locs - np.mean(bug_locs, 0)) / np.std(bug_locs, 0)
#
#     # Get cluster means as the center of each cluster, and cluster radii as the distance from the center
#     # to the outermost point
#     mu_met = np.array([[met_locs[bg, i].sum()/len(bg) for i in np.arange(met_locs.shape[1])] for bg in met_gp_ids])
#     z_gen = np.array([get_many_hot(kk, l=N_met) for kk in met_gp_ids]).T
#
#     cluster_centers = np.array([[bug_locs[bg, i].sum()/len(bg) for i in np.arange(bug_locs.shape[1])] for bg in bug_gp_ids])
#     w_gen = np.array([get_many_hot(kk, l=N_bug) for kk in bug_gp_ids]).T
#     temp = w_gen
#     mu_bug = cluster_centers
#     r_bug = np.array([np.max(
#         [np.sqrt(np.sum((cluster_centers[i, :] - l) ** 2)) for l in bug_locs[temp[:, i] == 1, :]]) for i
#                       in
#                       range(cluster_centers.shape[0])])
#     if cluster_per_met_cluster:
#         w_gen = np.stack([w_gen for i in range(N_met_clusters)], axis = -1)
#         temp = w_gen[:,:,0]
#         mu_bug = np.repeat(mu_bug[:,:,np.newaxis], N_met_clusters, axis = -1)
#         r_bug = np.repeat(r_bug[:, np.newaxis], N_met_clusters, axis = -1)
#     u = None
#     r_met = np.array([np.max([np.sqrt(np.sum((mu_met[i,:] - l)**2)) for l in met_locs[z_gen[:,i]==1,:]]) for i in
#              range(mu_met.shape[0])])
#
#     # Specify beta, alpha, and the range for each microbial cluster sum
#     if not deterministic:
#         alphas = st.bernoulli(0.5).rvs((N_bug_clusters, N_met_clusters))
#         cluster_starts = np.arange(1, np.int(N_bug_clusters * cluster_disparity) + 1, cluster_disparity)
#         cluster_ends = cluster_starts[1:] - cluster_disparity/10
#         cluster_ends = np.append(cluster_ends, cluster_starts[-1] + cluster_disparity - cluster_disparity/10)
#     else:
#         alphas = st.bernoulli(0.5).rvs((N_bug_clusters, N_met_clusters))
#         cluster_starts = [100,350,510,650,870,1000,1200,1400,1600,1800]
#         cluster_ends = [250,410,550,770,900,1100,1300,1500,1700,1900]
#         cluster_start_noise = st.norm(0,cluster_std/2).rvs(len(cluster_starts))
#         cluster_end_noise = st.norm(0, cluster_std / 2).rvs(len(cluster_starts))
#         cluster_starts = cluster_start_noise + cluster_starts
#         cluster_ends = cluster_end_noise + cluster_ends
#
#     # For each cluster, sample the cluster sum g from the cluster range, and then get the individual microbial
#     # counts by sampling X[:, cluster_ixs] from a multinomial
#     X = np.zeros((N_samples, N_bug))
#     temp2 = w_gen
#     g = np.zeros((N_samples, N_bug_clusters))
#     for i in range(N_bug_clusters):
#         g[:,i] = st.uniform(cluster_starts[i], cluster_ends[i]-cluster_starts[i]).rvs(size = N_samples)
#         outer_ixs = np.where(temp2[:,i]==1)[0]
#         conc = np.repeat(g[:, i:i+1], len(outer_ixs), axis = 1) / len(outer_ixs)
#         # conc = np.repeat(1, len(outer_ixs), axis=1) / len(outer_ixs)
#         p = [st.dirichlet(conc[n,:]).rvs() for n in range(conc.shape[0])]
#         X[:, outer_ixs] = np.stack([st.multinomial(int(np.round(g[n,i])), p[n].squeeze()).rvs() for n in range(len(p))]).squeeze()
#
#     X = X/np.sum(X)
#     g = g/np.sum(g)
#
#     y_est = st.norm(0,1).rvs((N_samples, N_met_clusters))
#     g_new = np.hstack((np.ones((g.shape[0], 1)), g))
#     betas = np.linalg.inv(g_new.T@g_new)@(g_new.T@(y_est))
#     # Get metabolic levels based on type of relationship (i.e. linear, exponential, etc)
#     y = np.zeros((N_samples, N_met))
#     for j in range(N_met):
#         k = np.argmax(z_gen[j,:])
#         if not linear:
#             g = (g - np.mean(g,0))/np.std(g-np.mean(g,0))
#             if nl_type == 'exp':
#                 y[:, j] = np.random.normal(betas[0, k] + np.exp(g) @ (betas[1:, k] * alphas[:, k]), np.sqrt(meas_var))
#             if nl_type == 'sigmoid':
#                 y[:, j] = np.random.normal(betas[0, k] + sigmoid(g) @ (betas[1:, k] * alphas[:, k]), np.sqrt(meas_var))
#             if nl_type == 'sin':
#                 y[:, j] = np.random.normal(betas[0, k] + np.sin(g) @ (betas[1:, k] * alphas[:, k]), np.sqrt(meas_var))
#             if nl_type == 'poly':
#                 y[:,j] = np.random.normal(betas[0, k] + (g)**5 @ (betas[1:, k] * alphas[:, k]) - (g)**4 @ (betas[1:, k] * alphas[:, k]), np.sqrt(meas_var))
#             if nl_type == 'linear':
#                 y[:, j] = np.random.normal(betas[0, k] + g @ (betas[1:, k] * alphas[:, k]), np.sqrt(meas_var))
#         else:
#             y[:, j] = np.random.normal(betas[0, k] + g @ (betas[1:, k] * alphas[:, k]), np.sqrt(meas_var))
#
#     return X, y, g, betas, alphas, w_gen, z_gen, bug_locs, met_locs, mu_bug, mu_met, r_bug, r_met, temp