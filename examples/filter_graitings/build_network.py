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


lgn_models = [
    {
        'N': 8,
        'ei': 'e',
        'model_type': 'virtual',
        'model_template': 'lgnmodel:tOFF_TF15',
        'size_range': [2, 10],
        'dynamics_params': 'tOFF_TF15_3.44215357_-2.11509939_8.27421573_20.0_0.0_ic.json'
    },
    {
        'N': 8,
        'ei': 'e',
        'model_type': 'virtual',
        'model_template': 'lgnmodel:sONsOFF_001',
        'size_range': [6],
        'dynamics_params': 'sOFF_TF4_3.5_-2.0_10.0_60.0_15.0_ic.json',
        'non_dom_params': 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.json',
        'sf_sep': 6.0
    },
    {
        'N': 5,
        'ei': 'e',
        'model_type': 'virtual',
        'model_template': 'lgnmodel:sONtOFF_001',
        'size_range': [9],
        'dynamics_params': 'tOFF_TF8_4.222_-2.404_8.545_23.019_0.0_ic.json',
        'non_dom_params': 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.json',
        'sf_sep': 4.0
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

    lgn.add_nodes(
        N=total_N,
        ei=params['ei'],
        model_type=params['model_type'],
        model_template=params['model_template'],
        x=positions[:, 0],
        y=positions[:, 1],
        dynamics_params=params['dynamics_params'],

        # TODO: Come up with better name than non-dominate parameters (spatial-params?)
        non_dom_params=params.get('non_dom_params', None),

        # TODO: See if it's possible to calculate spatial sizes during simulation.
        spatial_size=filter_sizes,

        # NOTE: If tuning angle is not defined, then it will be randomly generated during the simulation. But
        #  when evaluating a large network many times it will be more efficent to store it in the nodes file.
        tuning_angle=np.random.uniform(0.0, 360.0, total_N),

        # TODO: Can sf-sperator be stored in the params json file.
        sf_sep=params.get('sf_sep', None)
    )

lgn.build()
lgn.save(output_dir='network')
