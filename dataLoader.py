#!/Users/jendawk/miniconda3/envs/M2M_CodeBase/bin python3
from helper import *
import pandas as pd
import numpy as np

class dataLoader():
    def __init__(self, path = "/Users/jendawk/Dropbox (MIT)/Microbes to Metabolomes/Datasets/cdi/",
                 filename_cdiff = "CDiffMetabolomics.xlsx",
        filename_16s = 'seqtab-nochim-total.xlsx',
                 pt_perc = {'metabs': .25, '16s': .1, 'scfa': 0, 'toxin':0},
                 meas_thresh = {'metabs': 0, '16s': 10, 'scfa': 0, 'toxin':0},
                 var_perc={'metabs': 50, '16s': 5, 'scfa': 0, 'toxin':0}, pt_tmpts = 1):
        
        self.path = path
        self.filename_cdiff = filename_cdiff
        self.filename_16s = filename_16s
        # self.filename_ba = filename_ba

        self.pt_perc = pt_perc
        self.meas_thresh = meas_thresh
        self.var_perc = var_perc
        self.pt_tmpts = pt_tmpts

        self.load_cdiff_data()
        # self.load_ba_data()
        self.load_16s_data()
        self.keys = {'metabs':self.cdiff_data_dict,'16s':self.data16s_dict}

        self.combos = ['metabs_16s']
        # self.week_one = {}
        # for key, value in keys.items():
        #     temp = self.get_week_x(value['data'],value['targets_by_pt'], week = 1)
        #     self.week_one[key] = self.filter_transform(temp['x'], value['targets_by_pt'], key), temp['y']

        self.week = {}
        self.week_one = {}
        self.week_raw = {}
        self.week_filt = {}
        self.week_stand = {}
        self.week_sm = {}
        self.week_sm_filt = {}
        for key, value in self.keys.items():
            if isinstance(pt_perc, dict):
                if key not in pt_perc.keys():
                    continue
                self.pt_perc = pt_perc[key]
            if isinstance(var_perc, dict):
                if key not in pt_perc.keys():
                    continue
                self.var_perc = var_perc[key]
            if isinstance(pt_tmpts, dict):
                if key not in pt_perc.keys():
                    continue
                self.pt_tmpts = var_perc[key]
            if isinstance(meas_thresh, dict):
                if key not in pt_perc.keys():
                    continue
                self.meas_thresh = meas_thresh[key]
            self.week[key] = {}
            self.week_raw[key] = {}
            self.week_filt[key] = {}
            self.week_stand[key] = {}
            self.week_sm[key] = {}
            self.week_sm_filt[key] = {}
            value['data'] = value['data'].fillna(0)
            value['targets'] = value['targets'].replace('Recur', 'Recurrer').replace('Cleared','Non-recurrer')
            value['targets_by_pt'] = value['targets_by_pt'].replace('Recur', 'Recurrer').replace('Cleared', 'Non-recurrer')
            # if key == 'scfa':
            #     filter = False
            #     value['data'] = value['data'].drop('Heptanoate', axis = 1)
            # else:
            filter = True
            value['filtered_data'] = self.filter_transform(value['data'], targets_by_pt = None, key = key, filter = filter)
            # temp = self.get_week_x(value['filtered_data'], value['targets_by_pt'], week=1)
            #
            # self.week_one[key] = temp['x'], temp['y']
            temp_filt = filter_by_pt(value['data'], targets=None, perc=self.pt_perc, pt_thresh=self.pt_tmpts,
                                     meas_thresh=self.meas_thresh)

            for week in [0,1,2,3]:
                reps = None
                self.week[key][week] = self.get_week_x_step_ahead(value['filtered_data'], value['targets_by_pt'], week = week)
                self.week_raw[key][week] = self.get_week_x_step_ahead(value['data'], value['targets_by_pt'], week=week)
                self.week_filt[key][week] = self.get_week_x_step_ahead(temp_filt, value['targets_by_pt'], week=week)
                self.week_filt[key][week]['x'] = self.week_filt[key][week]['x'][self.week[key][week]['x'].columns.values]

                temp = self.get_week_x_step_ahead(value['data'], value['targets_by_pt'], week = week)
                x = self.filter_transform(temp['x'], targets_by_pt=None,key=key, filter=filter, weeks = [week])
                self.week_sm[key][week] = {'x': x, 'y': temp['y']}

                x = filter_by_pt(temp['x'], perc=self.pt_perc, targets = None,
                                                             pt_thresh = self.pt_tmpts, meas_thresh=self.meas_thresh,
                                                                  weeks = [week])
                x = x[self.week_sm[key][week]['x'].columns.values]
                self.week_sm_filt[key][week] = {'x': x,'y': temp['y']}

        for ck in self.combos:
            self.week[ck] = {}
            for week in [0,1,2,3]:
                self.week[ck][week] = {}
                ix_both = list(set(self.week[ck.split('_')[0]][week]['x'].index.values).intersection(
                    set(self.week[ck.split('_')[1]][week]['x'].index.values)))
                ix_pt = [ix.split('-')[0] for ix in ix_both]
                joint = np.hstack((self.week[ck.split('_')[0]][week]['x'].loc[ix_both,:],
                                   self.week[ck.split('_')[1]][week]['x'].loc[ix_both,:]))
                cols = list(self.week[ck.split('_')[0]][week]['x'].columns.values)
                cols.extend(self.week[ck.split('_')[1]][week]['x'].columns.values)
                self.week[ck][week]['x'] = pd.DataFrame(joint, index = ix_both, columns = cols)
                self.week[ck][week]['y'] = self.week[ck.split('_')[0]][week]['y'][ix_pt]
                self.week[ck][week]['event_times'] = self.week[ck.split('_')[0]][week]['event_times'][ix_pt]


    def load_cdiff_data(self):
        xl = pd.ExcelFile(self.path + '/' + self.filename_cdiff)
        self.cdiff_raw = xl.parse('OrigScale', header = None, index_col = None)
        ixs = np.where(self.cdiff_raw == 'MASS EXTRACTED')
        ix_row, ix_col = ixs[0].item(), ixs[1].item()
        act_data = self.cdiff_raw.iloc[ix_row + 2:, ix_col + 1:]
        feature_header = self.cdiff_raw.iloc[ix_row+2:, :ix_col+1]
        pt_header = self.cdiff_raw.iloc[:ix_row + 1, ix_col + 1:]
        pt_names = list(self.cdiff_raw.iloc[:ix_row+1, ix_col])
        feat_names = list(self.cdiff_raw.iloc[ix_row+1, :ix_col + 1])
        feat_names[-1] = 'HMDB'

        self.col_mat_mets = feature_header
        self.col_mat_mets.columns = feat_names
        self.col_mat_mets.index = np.arange(self.col_mat_mets.shape[0])
        #
        self.col_mat_pts = pt_header.T
        self.col_mat_pts.columns = pt_names
        self.col_mat_pts.index = np.arange(self.col_mat_pts.shape[0])

        self.targets_dict = pd.Series(self.col_mat_pts['PATIENT STATUS (BWH)'].values, index = self.col_mat_pts['CLIENT SAMPLE ID'].values).to_dict()

        self.cdiff_dat = pd.DataFrame(np.array(act_data), columns = self.col_mat_pts['CLIENT SAMPLE ID'].values,
                          index = self.col_mat_mets['BIOCHEMICAL'].values).fillna(0).T

        self.targets_by_pt = {key.split('-')[0]:value for key, value in self.targets_dict.items() if key.split('-')[1].isnumeric()}
        self.cdiff_data_dict = {'sampleMetadata':self.col_mat_pts, 'featureMetadata':self.col_mat_mets,
                                'data':self.cdiff_dat, 'targets':pd.Series(self.targets_dict),
                                'targets_by_pt': pd.Series(self.targets_by_pt)}
        self.col_mat_mets = self.col_mat_mets.set_index('BIOCHEMICAL')

    def load_16s_data(self):
        self.file16s = pd.ExcelFile(self.path + '/' + self.filename_16s)
        self.raw16s = self.file16s.parse(index_col = 0)
        dcol = []
        for x in self.raw16s.columns.values:
            if len(x.split('-')) == 3:
                dcol.append('.'.join([x[:5], x[-1]]))
            elif len(x.split('-')[1]) == 2:
                dcol.append('.'.join([x[:5], x[-1]]))
            else:
                dcol.append(x)
        self.data16s = pd.DataFrame(
            np.array(self.raw16s), columns=dcol, index=self.raw16s.index.values)
        self.targets_16s = {key: val for key, val in self.targets_dict.items() if key in self.data16s.columns.values}
        pt_both = [x for x in dcol if x in self.cdiff_dat.index.values]
        self.data16s_both = (self.data16s[pt_both]).T
        pts = np.unique([x.split('-')[0] for x in self.data16s_both.index.values])
        targets_by_pt = pd.Series(self.cdiff_data_dict['targets_by_pt'][pts], index = pts)
        self.data16s_dict = {'data': self.data16s_both, 'targets': pd.Series(self.targets_16s),
                             'targets_by_pt': pd.Series(targets_by_pt)}

    def filter_transform(self, data, targets_by_pt, key = 'metabs', filter = True, weeks = [0,1,2]):
        if filter:
            filt1 = filter_by_pt(data, targets_by_pt, perc=self.pt_perc, pt_thresh=self.pt_tmpts,
                                 meas_thresh=self.meas_thresh, weeks = weeks)
            # print(key + ', 1st filter: ' + str(filt1.shape))
        else:
            filt1 = data
        epsilon = get_epsilon(filt1)

        if '16s' not in key:
            transformed = np.log(filt1 + epsilon)
        else:
            data_prop = np.divide(filt1.T, np.sum(filt1, 1)).T
            epsilon = get_epsilon(data_prop)
            geom_means = np.exp(np.mean(np.log(data_prop + epsilon), 1))
            temp = np.divide(data_prop.T, geom_means).T
            epsilon = get_epsilon(temp)
            transformed = np.log(temp + epsilon)

            trans_rep = None
        if filter and 'toxin' not in key:
            filt2 = filter_vars(transformed, perc=self.var_perc)
        else:
            filt2 = transformed

        stand, mean, dem = standardize(filt2, override=True)
        # print(key + ', 2nd filter: ' + str(filt2.shape))
        # print(key)
        # print(filt1.shape)
        # print(filt2.shape)
        return stand


    def get_week_x(self, data, targets, week = 1):
        ixs = data.index.values
        pts = [x.split('-')[0] for x in ixs]
        tmpts = [x.split('-')[1] for x in ixs]
        rm_ix = []
        for pt in np.unique(pts):
            ix_pt = np.where(pt == np.array(pts))[0]
            tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.','').isnumeric()]
            if max(tm_floats) == week:
                rm_ix.append(pt)
        week_one = np.where(np.array(tmpts)==str(week))[0]
        pt_keys = np.array(pts)[week_one]
        pt_keys = np.array(list(set(pt_keys) - set(rm_ix)))
        pt_keys_1 = np.array([pt + '-' + str(week) for pt in pt_keys])
        data_w1 = data.loc[pt_keys_1]
        targs = targets[pt_keys]
        return {'x':data_w1,'y':targs}

    def get_week_x_step_ahead(self, data, targets, week = 1):

        ixs = data.index.values
        pts = [x.split('-')[0] for x in ixs]
        tmpts = [x.split('-')[1] for x in ixs]
        week_one = np.where(np.array(tmpts) == str(week))[0]
        pt_keys = np.array(pts)[week_one]

        rm_ix = []
        targets_out = {}
        event_time = {}
        for pt in np.unique(pts):
            targets_out[pt] = 'Non-recurrer'
            ix_pt = np.where(pt == np.array(pts))[0]
            tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.', '').isnumeric()]
            event_time[pt] = tm_floats[-1]
            if targets[pt] == 'Recurrer':
                ix_pt = np.where(pt == np.array(pts))[0]
                tm_floats = [float(tmpts[ix]) for ix in ix_pt if tmpts[ix].replace('.','').isnumeric()]
                if week not in tm_floats:
                    continue
                if max(tm_floats) == week:
                    rm_ix.append(pt)
                    continue
                tm_floats.sort()
                tmpt_step_before = tm_floats[-2]
                if tmpt_step_before == week:
                    targets_out[pt] = 'Recurrer'

        pt_keys = np.array(list(set(pt_keys) - set(rm_ix)))
        pt_keys_1 = np.array([pt + '-' + str(week) for pt in pt_keys])
        data_w1 = data.loc[pt_keys_1]
        targs = pd.Series(targets_out)[pt_keys]
        return {'x':data_w1,'y_step_ahead':targs, 'y':targets[pt_keys], 'event_times': pd.Series(event_time)[pt_keys]}

    def get_step_ahead(self, data, targets):
        ixs = targets.index.values
        data_all = data.iloc[ixs,:]
        targets_out = targets[data_all.index.values]
        return {'x':data_all,'y':targets_out}

    def is_float(self, element):
        try:
            float(element)
            return True
        except ValueError:
            return False

if __name__ == "__main__":
    base_path = '/Users/jendawk/Dropbox (MIT)/M2M'
    dl = dataLoader(pt_perc={'metabs': .25, '16s': .1, 'scfa': 0, 'toxin':0}, meas_thresh=
    {'metabs': 0, '16s': 10, 'scfa': 0, 'toxin':0}, var_perc={'metabs': 50, '16s': 5, 'scfa': 0, 'toxin':0})
    # dl = dataLoader(path=base_path + '/inputs/', pt_perc={'metabs': .25, '16s': .1, 'scfa': 0, 'toxin': 0}, meas_thresh=
    # {'metabs': 0, '16s': 10, 'scfa': 0, 'toxin': 0},
    #                 var_perc={'metabs': 50, '16s': 5, 'scfa': 0, 'toxin': 0}, pt_tmpts=1)
    y = dl.week['metabs'][1]['x']
    # y.to_csv(base_path + '/inputs/y_50.csv')

