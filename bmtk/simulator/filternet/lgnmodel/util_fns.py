import os
import re
import numpy as np
import scipy.io as sio
from scipy.fftpack import fft
import pandas as pd
from .movie import Movie, FullFieldFlashMovie


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 100)


#################################################
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


##################################################
def compute_FFT_OneCycle(FR, TF, downsample):
    one_cyc = np.int(((1000. / downsample) / TF))
    FR_cyc = list(chunks(FR, one_cyc))
    if (TF == 15. or TF == 8.):
        FR_cyc = FR_cyc[:-1]

    FR_cyc_avg = np.mean(FR_cyc, axis=0)
    y = FR_cyc_avg
    AMP = 2 * np.abs(fft(y) / len(y))
    F0 = 0.5 * AMP[0]
    assert (F0 - np.mean(y) < 1.e-4)
    F1 = AMP[1]

    return F0, F1


##################################################
def create_ff_mov(frame_rate, tst, tend, xrng, yrng):
    ff_mov_on = FullFieldFlashMovie(np.arange(xrng), np.arange(yrng), tst, tend, frame_rate=frame_rate,
                                    max_intensity=1).full(t_max=tend)  # +0.5)
    ff_mov_off = FullFieldFlashMovie(np.arange(xrng), np.arange(yrng), tst, tend, frame_rate=frame_rate,
                                     max_intensity=-1).full(t_max=tend)  # +0.5)

    return ff_mov_on, ff_mov_off


##################################################
def create_grating_movie_list(gr_dir_name):
    gr_fnames = os.listdir(gr_dir_name)
    gr_fnames_ord = sorted(gr_fnames, key=lambda x: (int(re.sub('\D', '', x)), x))

    gr_mov_list = []
    for fname in gr_fnames_ord[:5]:
        movie_file = os.path.join(gr_dir_name, fname)
        m_file = sio.loadmat(movie_file)
        m_data_raw = m_file['mov'].T
        swid = np.shape(m_data_raw)[1]
        res = int(np.sqrt(swid / (8 * 16)))
        m_data = np.reshape(m_data_raw, (3000, 8 * res, 16 * res))
        m1 = Movie(m_data[:500, :, :], row_range=np.linspace(0, 120, m_data.shape[1], endpoint=True), col_range=np.linspace(0, 120, m_data.shape[2], endpoint=True), frame_rate=1000.)
        gr_mov_list.append(m1)

    return gr_mov_list

"""
##################################################
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
"""


##################################################
def check_optim_results_against_bounds(bounds, opt_wts, opt_kpeaks):
    bds_wts0 = bounds[0]
    bds_wts1 = bounds[1]
    bds_kp0 = bounds[2]
    bds_kp1 = bounds[3]

    opt_wts0 = opt_wts[0]
    opt_wts1 = opt_wts[1]
    opt_kp0 = opt_kpeaks[0]
    opt_kp1 = opt_kpeaks[1]

    if (opt_wts0 == bds_wts0[0] or opt_wts0 == bds_wts0[1]):
        prm_on_bds = 'w0'
    elif (opt_wts1 == bds_wts1[0] or opt_wts1 == bds_wts1[1]):
        prm_on_bds = 'w1'
    elif (opt_kp0 == bds_kp0[0] or opt_kp0 == bds_kp0[1]):
        prm_on_bds = 'kp0'
    elif (opt_kp1 == bds_kp1[0] or opt_kp1 == bds_kp1[1]):
        prm_on_bds = 'kp1'
    else:
        prm_on_bds = 'None'

    return prm_on_bds


def cross_from_above(x, threshold):
    """Return the indices into *x* where *x* crosses some threshold from above."""
    x = np.asarray(x)
    ind = np.nonzero((x[:-1] >= threshold) & (x[1:] < threshold))[0]
    if len(ind):
        return ind+1
    else:
        return ind


#######################################################
def get_tcross_from_temporal_kernel(temporal_kernel):
    max_ind = np.argmax(temporal_kernel)
    min_ind = np.argmin(temporal_kernel)

    temp_tcross_ind = cross_from_above(temporal_kernel[max_ind:min_ind], 0.0)
    tcross_ind = max_ind + temp_tcross_ind[0]
    return tcross_ind
