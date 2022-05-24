#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3

from main import *

data_path = '/Users/jendawk/Dropbox (MIT)/M2M/inputs/'
dl = dataLoader(path=data_path, pt_perc={'metabs': .50, '16s': .1, 'scfa': 0, 'toxin': 0}, meas_thresh=
{'metabs': 0, '16s': 10, 'scfa': 0, 'toxin': 0},
                var_perc={'metabs': 75, '16s': 5, 'scfa': 0, 'toxin': 0}, pt_tmpts=1)

y_raw = dl.week_raw['metabs'][1]['x']
yfile = 'y.csv'
y_dat =pd.read_csv(data_path + '/' + yfile, index_col=[0])
# y_raw_filt = y_raw[y_dat.columns.values].loc[y_dat.index.values]
# y_ = (y_raw_filt - np.array([y_raw_filt.mean(1)]).T)/np.array([np.std(y_raw_filt,1)]).T
# y_.to_csv('y_re-norm.csv')

y_raw_filt = y_raw[y_dat.columns.values].loc[y_dat.index.values]
epsilon = get_epsilon(y_raw_filt)
y_raw_filt.to_csv(data_path + '/y_raw.csv')
y_log = np.log(y_raw_filt + epsilon)
y_log.to_csv(data_path +'y_log.csv')
y_ = (y_log - np.array([y_log.mean(1)]).T)/np.array([np.std(y_log,1)]).T
y_.to_csv(data_path +'y_log-re-norm.csv')

y_range = y_log / (np.max(y_log,0) - np.min(y_log,0))
y_range.to_csv(data_path +'y_log-range-norm')

y_range2 = y_raw_filt / (np.max(y_raw_filt,0) - np.min(y_raw_filt,0))
y_range2.to_csv(data_path +'y_range-norm')

transformed = np.log(y_raw_filt + epsilon)

# y_diff_stand = y_
# ix_z = np.where(y_raw_filt==0)
for met_z_ix in np.arange(y_raw_filt.shape[1]):
    samp_z_ix = np.where(y_raw_filt.iloc[:,met_z_ix]==0)[0]
    y_.iloc[samp_z_ix, met_z_ix] = st.norm(0,1).rvs(len(samp_z_ix))

y_.to_csv(data_path + 'y_re-norm_no-zeros.csv')

xfile = 'x.csv'
yfile = 'y.csv'
base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
data_path = base_path + '/inputs'
x = pd.read_csv(data_path + '/' + xfile, index_col=[0])
y = pd.read_csv(data_path + '/' + yfile, index_col=[0])


# plot omega prior vs tau
loc = 0.1
x = np.linspace(0,1,100)

for tau in [0.7,0.6,0.5,0.3,0.2]:
    pdf = BinaryConcrete(loc, tau).log_prob(x)
    plt.figure()
    plt.plot(np.linspace(0,1,100), pdf)
    plt.xlabel('x')
    plt.ylabel('log prob')
    plt.title('loc= ' + str(loc) + ', tau= ' + str(np.round(tau, 6)))
    plt.show()