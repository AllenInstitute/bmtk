import os
import pickle
import numpy as np

from bmtk.builder import NetworkBuilder


X_grids = 2  # 15
Y_grids = 2  # 10
X_len = 240.0  # In linear degrees
Y_len = 120.0  # In linear degrees


def generate_positions_grids(N, X_grids, Y_grids, X_len, Y_len):
    width_per_tile = X_len/X_grids
    height_per_tile = Y_len/Y_grids

    X = np.zeros(N * X_grids * Y_grids)
    Y = np.zeros(N * X_grids * Y_grids)

    counter = 0
    for i in range(X_grids):
        for j in range(Y_grids):
            X_tile = np.random.uniform(i*width_per_tile,  (i+1) * width_per_tile,  N)
            Y_tile = np.random.uniform(j*height_per_tile, (j+1) * height_per_tile, N)
            X[counter*N:(counter+1)*N] = X_tile
            Y[counter*N:(counter+1)*N] = Y_tile
            counter += 1
    return np.column_stack((X, Y))


def get_filter_spatial_size(N, X_grids, Y_grids, size_range):
    spatial_sizes = np.zeros(N * X_grids * Y_grids)
    counter = 0
    for i in range(X_grids):
        for j in range(Y_grids):
            if len(size_range) == 1:
                sizes = np.ones(N) * size_range[0]
            else:
                sizes = np.random.triangular(size_range[0], size_range[0] + 1, size_range[1], N)
            spatial_sizes[counter * N:(counter + 1) * N] = sizes
            counter += 1
    return spatial_sizes


def get_filter_temporal_params(N, X_grids, Y_grids, model):
    # Total number of cells
    N_total = N * X_grids * Y_grids

    # Jitter parameters
    jitter = 0.025
    lower_jitter = 1 - jitter
    upper_jitter = 1 + jitter

    # Directory of pickle files with saved parameter values
    basepath = 'optimized_params'

    # For two-subunit filter (sONsOFF and sONtOFF)
    sOFF_fn = os.path.join(basepath, 'sOFF_TF4_3.5_-2.0_10.0_60.0_15.0_ic.pkl')  # best chosen fit for sOFF 4 Hz
    tOFF_fn = os.path.join(basepath, 'tOFF_TF8_4.222_-2.404_8.545_23.019_0.0_ic.pkl')  # best chosen fit for tOFF 8 Hz
    sON_fn = os.path.join(basepath, 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.pkl')  # best chosen fit for sON 4 Hz

    sOFF_prs = pickle.load(open(sOFF_fn, 'rb')) #, encoding='latin1')
    tOFF_prs = pickle.load(open(tOFF_fn, 'rb'))
    sON_prs = pickle.load(open(sON_fn, 'rb'))

    model = model.replace('lgnmodel:', '') if model.startswith('lgnmodel:') else model

    # Choose cell type and temporal frequency
    if model == 'sONsOFF_001':
        kpeaks = sOFF_prs['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks = sON_prs['opt_kpeaks']
        kpeaks_non_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_non_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        wts = sOFF_prs['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        wts =  sON_prs['opt_wts']
        wts_non_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_non_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        delays = sOFF_prs['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        delays = sON_prs['opt_delays']
        delays_non_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_non_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        sf_sep = 6.
        sf_sep = np.random.uniform(lower_jitter * sf_sep, upper_jitter * sf_sep, N_total)
        tuning_angles = np.random.uniform(0, 360., N_total)

    elif model == 'sONtOFF_001':
        kpeaks = tOFF_prs['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        kpeaks = sON_prs['opt_kpeaks']
        kpeaks_non_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_non_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)

        wts = tOFF_prs['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        wts = sON_prs['opt_wts']
        wts_non_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_non_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)

        delays = tOFF_prs['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        delays = sON_prs['opt_delays']
        delays_non_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_non_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)

        sf_sep = 4.
        sf_sep = np.random.uniform(lower_jitter * sf_sep, upper_jitter * sf_sep, N_total)
        tuning_angles = np.random.uniform(0, 360., N_total)

    else:
        cell_type = model[0: model.find('_')]    #'sON'  # 'tOFF'
        tf_str = model[model.find('_') + 1:]

        # Load pickle file containing params for optimized temporal kernel, it it exists
        file_found = 0
        for fname in os.listdir(basepath):
            if os.path.isfile(os.path.join(basepath, fname)):
                pkl_savename = os.path.join(basepath, fname)
                if tf_str in pkl_savename.split('_') and pkl_savename.find(cell_type) >= 0 and pkl_savename.find('.pkl') >= 0:
                    file_found = 1
                    filt_file = pkl_savename

        if file_found != 1:
            print('File not found: Filter was not optimized for this sub-class')

        savedata_dict = pickle.load(open(filt_file, 'rb'))

        kpeaks = savedata_dict['opt_kpeaks']
        kpeaks_dom_0 = np.random.uniform(lower_jitter * kpeaks[0], upper_jitter * kpeaks[0], N_total)
        kpeaks_dom_1 = np.random.uniform(lower_jitter * kpeaks[1], upper_jitter * kpeaks[1], N_total)
        kpeaks_non_dom_0 = np.nan * np.zeros(N_total)
        kpeaks_non_dom_1 = np.nan * np.zeros(N_total)

        wts = savedata_dict['opt_wts']
        wts_dom_0 = np.random.uniform(lower_jitter * wts[0], upper_jitter * wts[0], N_total)
        wts_dom_1 = np.random.uniform(lower_jitter * wts[1], upper_jitter * wts[1], N_total)
        wts_non_dom_0 = np.nan * np.zeros(N_total)
        wts_non_dom_1 = np.nan * np.zeros(N_total)

        delays = savedata_dict['opt_delays']
        delays_dom_0 = np.random.uniform(lower_jitter * delays[0], upper_jitter * delays[0], N_total)
        delays_dom_1 = np.random.uniform(lower_jitter * delays[1], upper_jitter * delays[1], N_total)
        delays_non_dom_0 = np.nan * np.zeros(N_total)
        delays_non_dom_1 = np.nan * np.zeros(N_total)

        sf_sep = np.nan * np.zeros(N_total)
        tuning_angles =  np.nan * np.zeros(N_total)

    return np.column_stack(
        (kpeaks_dom_0, kpeaks_dom_1, wts_dom_0, wts_dom_1, delays_dom_0, delays_dom_1,
         kpeaks_non_dom_0, kpeaks_non_dom_1, wts_non_dom_0, wts_non_dom_1, delays_non_dom_0,
         delays_non_dom_1, tuning_angles, sf_sep)
    )


lgn_models = [
    # {
    #     'N': 7, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sON_TF1', 'size_range': [2, 10]
    # },
    # {
    #     'N': 5, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sON_TF2', 'size_range': [2, 10]
    # },
    # {
    #     'N': 7, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sON_TF4', 'size_range': [2, 10]
    # },
    # {
    #     'N': 15, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sON_TF8', 'size_range': [2, 10]
    # },
    # {
    #     'N': 8, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sOFF_TF1', 'size_range': [2, 10]
    # },
    # {
    #     'N': 8, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sOFF_TF2', 'size_range': [2, 10]
    # },
    # {
    #     'N': 15, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sOFF_TF4', 'size_range': [2, 10]
    # },
    # {
    #     'N': 8, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sOFF_TF8', 'size_range': [2, 10]
    # },
    # {
    #     'N': 7, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sOFF_TF15', 'size_range': [2, 10]
    # },
    # {
    #    'N': 10, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:tOFF_TF4', 'size_range': [2, 10]
    # },
    # {
    #    'N': 5, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:tOFF_TF8', 'size_range': [2, 10]
    # },
    {
        'N': 8, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:tOFF_TF15', 'size_range': [2, 10]
    },
    {
        'N': 8, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sONsOFF_001', 'size_range': [6]
    },
    {
        'N': 5, 'ei': 'e', 'model_type': 'virtual', 'model_template': 'lgnmodel:sONtOFF_001', 'size_range': [9]
    }
]

lgn = NetworkBuilder('lgn')
for params in lgn_models:
    # Get position of lgn cells and keep track of the averaged location
    # For now, use randomly generated values
    total_N = params['N'] * X_grids * Y_grids

    # Get positional coordinates of cells
    positions = generate_positions_grids(params['N'], X_grids, Y_grids, X_len, Y_len)

    # Get spatial filter size of cells
    filter_sizes = get_filter_spatial_size(params['N'], X_grids, Y_grids, params['size_range'])

    # Get filter temporal parameters
    filter_params = get_filter_temporal_params(params['N'], X_grids, Y_grids, params['model_template'])

    lgn.add_nodes(
        N=total_N,
        ei=params['ei'],
        model_type=params['model_type'],
        model_processing='preset_params',  # Used to indicate parameters are preset
        model_template=params['model_template'],
        x=positions[:, 0],
        y=positions[:, 1],
        spatial_size=filter_sizes,
        # kpeaks_dom_0=filter_params[:, 0],
        # kpeaks_dom_1=filter_params[:, 1],
        kpeaks=filter_params[:, 0:2],
        #weight_dom_0=filter_params[:, 2],
        #weight_dom_1=filter_params[:, 3],
        weights=filter_params[:, 2:4],
        #delay_dom_0=filter_params[:, 4],
        #delay_dom_1=filter_params[:, 5],
        delays=filter_params[:, 4:6],

        # kpeaks_non_dom_0=filter_params[:, 6],
        # kpeaks_non_dom_1=filter_params[:, 7],
        kpeaks_non_dom=filter_params[:, 6:8],
        weight_non_dom_0=filter_params[:, 8],
        weight_non_dom_1=filter_params[:, 9],
        delay_non_dom_0=filter_params[:, 10],
        delay_non_dom_1=filter_params[:, 11],
        tuning_angle=filter_params[:, 12],
        sf_sep=filter_params[:, 13],
    )

lgn.build()
lgn.save(output_dir='network')
