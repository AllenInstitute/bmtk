import os
import numpy as np
import pandas as pd


metrics_dir = os.path.join(os.path.dirname(__file__), 'cell_metrics')
def get_data_metrics_for_each_subclass(ctype):
    # Load csv file into dataframe
    if ctype.find('_sus') >= 0:
        prs_fn = os.path.join(metrics_dir, '{}_cells_v3.csv'.format(ctype))
    else:
        prs_fn = os.path.join(metrics_dir, '{}_cell_data.csv'.format(ctype))

    prs_df = pd.read_csv(prs_fn)
    N_class, nmet = np.shape(prs_df)

    # Group data by subclasses based on max F0 vals
    exp_df = prs_df.iloc[:, [13, 14, 17, 18, 28, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                             54]].copy()  # Bl_lat,Wh_lat,Bl_si, wh_si, spont, 5 F0s, 5 F1s
    sub_df = exp_df.iloc[:, [5, 6, 7, 8, 9]]
    exp_df['max_tf'] = sub_df.idxmax(axis=1).values  # sub_df.idxmax(axis=1)

    exp_means = exp_df.groupby(['max_tf']).mean()
    exp_std = exp_df.groupby(['max_tf']).std()
    exp_nsub = exp_df.groupby(['max_tf']).size()

    max_ind_arr = np.where(exp_nsub == np.max(exp_nsub))
    max_nsub_ind = max_ind_arr[0][0]

    # Get means and std dev for subclasses
    exp_prs_dict = {}
    for scn in np.arange(len(exp_nsub)):
        f0_exp = exp_means.iloc[scn, 5:10].values
        f1_exp = exp_means.iloc[scn, 10:].values
        spont_exp = exp_means.iloc[scn, 4:5].values
        if ctype.find('OFF') >= 0:
            si_exp = exp_means.iloc[scn, 2:3].values
            ttp_exp = exp_means.iloc[scn, 0:1].values
        elif ctype.find('ON') >= 0:
            si_exp = exp_means.iloc[scn, 3:4].values
            ttp_exp = exp_means.iloc[scn, 1:2].values
        else:
            si_exp = np.NaN * np.ones((1, 5))
            ttp_exp = np.NaN * np.ones((1, 2))

        nsub = exp_nsub.iloc[scn]
        if nsub == 1:
            f0_std = np.mean(exp_std.iloc[max_nsub_ind, 5:10].values) * np.ones((1, 5))
            f1_std = np.mean(exp_std.iloc[max_nsub_ind, 10:].values) * np.ones((1, 5))
            spont_std = np.mean(exp_std.iloc[max_nsub_ind, 4:5].values) * np.ones((1, 5))
            if ctype.find('OFF') >= 0:
                si_std = np.mean(exp_std.iloc[max_nsub_ind, 2:3].values) * np.ones((1, 5))
            elif ctype.find('ON') >= 0:
                si_std = np.mean(exp_std.iloc[max_nsub_ind, 3:4].values) * np.ones((1, 5))
            else:
                si_std = np.NaN * np.ones((1, 5))

        else:
            f0_std = exp_std.iloc[scn, 5:10].values
            f1_std = exp_std.iloc[scn, 10:].values
            spont_std = exp_std.iloc[scn, 4:5].values
            if ctype.find('OFF') >= 0:
                si_std = exp_std.iloc[scn, 2:3].values
            elif ctype.find('ON') >= 0:
                si_std = exp_std.iloc[scn, 3:4].values
            else:
                si_std = np.NaN * np.ones((1, 5))

        if ctype.find('t') >= 0:
            tcross = 40.
            si_inf_exp = (si_exp - tcross / 200.) * (200. / (200. - tcross - 40.))
        elif ctype.find('s') >= 0:
            tcross = 60.
            si_inf_exp = (si_exp - tcross / 200.) * (200. / (200. - tcross - 40.))
        else:
            si_inf_exp = np.nan

        dict_key = exp_means.iloc[scn].name[3:]
        exp_prs_dict[dict_key] = {}
        exp_prs_dict[dict_key]['f0_exp'] = f0_exp
        exp_prs_dict[dict_key]['f1_exp'] = f1_exp
        exp_prs_dict[dict_key]['spont_exp'] = spont_exp
        exp_prs_dict[dict_key]['si_exp'] = si_exp
        exp_prs_dict[dict_key]['si_inf_exp'] = si_inf_exp
        exp_prs_dict[dict_key]['ttp_exp'] = ttp_exp
        exp_prs_dict[dict_key]['f0_std'] = f0_std
        exp_prs_dict[dict_key]['f1_std'] = f1_std
        exp_prs_dict[dict_key]['spont_std'] = spont_std
        exp_prs_dict[dict_key]['si_std'] = si_std
        exp_prs_dict[dict_key]['nsub'] = nsub
        exp_prs_dict[dict_key]['N_class'] = N_class

    return exp_prs_dict