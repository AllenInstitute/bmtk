import os
import pickle
import numpy as np

from bmtk.builder import NetworkBuilder


x_grids = 2  # 15
y_grids = 2  # 10
x_len = 240.0  # In linear degrees
y_len = 120.0  # In linear degrees


# def generate_positions_grids(N, xs_grids, ys_grids, X_len, Y_len):
#     width_per_tile = X_len/xs_grids
#     height_per_tile = Y_len/ys_grids

#     X = np.zeros(N * xs_grids * ys_grids)
#     Y = np.zeros(N * xs_grids * ys_grids)

#     counter = 0
#     for i in range(xs_grids):
#         for j in range(ys_grids):
#             X_tile = np.random.uniform(i*width_per_tile,  (i+1) * width_per_tile,  N)
#             Y_tile = np.random.uniform(j*height_per_tile, (j+1) * height_per_tile, N)
#             X[counter*N:(counter+1)*N] = X_tile
#             Y[counter*N:(counter+1)*N] = Y_tile
#             counter += 1
#     return np.column_stack((X, Y))


def get_filter_spatial_size(N, X_grids, Y_grids, size_range):
    N=10
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

    print(spatial_sizes)
    print(X_grids, Y_grids, size_range)
    print('>>')
    exit()
    return spatial_sizes


# x_grids, y_grids = 15, 10
# field_size = (240.0, 120.0)
# lgn_fraction = 1.0

lgn_models = {
  "sON_TF1": {
    "N": 1,
    "subtype": "sON",
    "model_type": "virtual",
    "model_template": "lgnmodel:sON_TF1",
    "dynamics_params": "sON_TF1.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sON_TF2": {
    "N": 1,
    # "model_name": "sON_TF2",
    "subtype": "sON",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sON_TF2",
    "dynamics_params": "sON_TF2.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sON_TF4": {
    "N": 1,
    # "model_name": "sON_TF4",
    "subtype": "sON",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sON_TF4",
    "dynamics_params": "sON_TF4.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sON_TF8": {
    "N": 1,
    # "model_name": "sON_TF8",
    "subtype": "sON",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sON_TF8",
    "dynamics_params": "sON_TF8.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sOFF_TF1": {
    "N": 1,
    # "model_name": "sOFF_TF1",
    "subtype": "sOFF",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sOFF_TF1",
    "dynamics_params": "sOFF_TF1.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sOFF_TF2": {
    "N": 1,
    # "model_name": "sOFF_TF2",
    "subtype": "sOFF",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sOFF_TF2",
    "dynamics_params": "sOFF_TF2.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sOFF_TF4": {
    "N": 1,
    # "model_name": "sOFF_TF4",
    "subtype": "sOFF",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sOFF_TF4",
    "dynamics_params": "sOFF_TF4.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sOFF_TF8": {
    "N": 1,
    # "model_name": "sOFF_TF8",
    "subtype": "sOFF",
    # "ei": "e",
    "model_type": "virtual",
    "model_template": "lgnmodel:sOFF_TF8",
    "dynamics_params": "sOFF_TF8.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sOFF_TF15": {
    "N": 1,
    "subtype": "sOFF",
    "model_type": "virtual",
    "model_template": "lgnmodel:sOFF_TF15",
    "dynamics_params": "sOFF_TF15.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "tOFF_TF4": {
    "N": 1,
    "subtype": "tOFF",
    "model_type": "virtual",
    "model_template": "lgnmodel:tOFF_TF4",
    "dynamics_params": "tOFF_TF4.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "tOFF_TF8": {
    "N": 1,
    "subtype": "tOFF",
    "model_type": "virtual",
    "model_template": "lgnmodel:tOFF_TF8",
    "dynamics_params": "tOFF_TF8.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "tOFF_TF15": {
    "N": 1,
    "subtype": "tOFF",
    "model_type": "virtual",
    "model_template": "lgnmodel:tOFF_TF15",
    "dynamics_params": "tOFF_TF15.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "sONsOFF_001": {
    "N": 1,
    "subtype": "sONsOFF",
    "model_type": "virtual",
    "model_template": "lgnmodel:sONsOFF_001",
    "dynamics_params": "sOFF_TF4.json",
    "non_dom_params": "sON_TF4.json",
    "size_range": [6, 6],
    "sf_sep": 6.0,
    "tuning_angle": True
  },
  "sONtOFF_001": {
    "N": 1,
    "subtype": "sONtOFF",
    "model_type": "virtual",
    "model_template": "lgnmodel:sONtOFF_001",
    "dynamics_params": "tOFF_TF4.json",
    "non_dom_params": "sON_TF4.json",
    "size_range": [9, 9],
    "sf_sep": 4.0,
    "tuning_angle": True
  },
  "tON_TF4": {
    "N": 1,
    "subtype": "tON",
    "model_type": "virtual",
    "model_template": "lgnmodel:tON_TF4",
    "dynamics_params": "tON_TF4.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "tON_TF8": {
    "N": 1,
    "subtype": "tON",
    "model_type": "virtual",
    "model_template": "lgnmodel:tON_TF8",
    "dynamics_params": "tON_TF8.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
  "tON_TF15": {
    "N": 1,
    "subtype": "tON",
    "model_type": "virtual",
    "model_template": "lgnmodel:tON_TF15",
    "dynamics_params": "tON_TF15.json",
    "size_range": [2, 10],
    "tuning_angle": False
  },
}


def generate_positions_grids(N, x_len, y_len):
    xs = np.random.uniform(0.0, x_len, size=N)
    ys = np.random.uniform(0.0, y_len, size=N)
    return np.column_stack((xs, ys))
    # exit()
    # width_per_tile = X_len/xs_grids
    # height_per_tile = Y_len/ys_grids

    X = np.zeros(N * xs_grids * ys_grids)
    Y = np.zeros(N * xs_grids * ys_grids)

    counter = 0
    for i in range(xs_grids):
        for j in range(ys_grids):
            X_tile = np.random.uniform(i*width_per_tile,  (i+1) * width_per_tile,  N)
            Y_tile = np.random.uniform(j*height_per_tile, (j+1) * height_per_tile, N)
            X[counter*N:(counter+1)*N] = X_tile
            Y[counter*N:(counter+1)*N] = Y_tile
            counter += 1
    return np.column_stack((X, Y))


# lgn_models = [
#     {
#         'N': 8,
#         'ei': 'e',
#         'model_type': 'virtual',
#         'model_template': 'lgnmodel:tOFF_TF15',
#         'size_range': [2, 10],
#         'dynamics_params': 'tOFF_TF15_3.44215357_-2.11509939_8.27421573_20.0_0.0_ic.json'
#     },
#     {
#         'N': 8,
#         'ei': 'e',
#         'model_type': 'virtual',
#         'model_template': 'lgnmodel:sONsOFF_001',
#         'size_range': [6],
#         'dynamics_params': 'sOFF_TF4_3.5_-2.0_10.0_60.0_15.0_ic.json',
#         'non_dom_params': 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.json',
#         'sf_sep': 6.0
#     },
#     {
#         'N': 5,
#         'ei': 'e',
#         'model_type': 'virtual',
#         'model_template': 'lgnmodel:sONtOFF_001',
#         'size_range': [9],
#         'dynamics_params': 'tOFF_TF8_4.222_-2.404_8.545_23.019_0.0_ic.json',
#         'non_dom_params': 'sON_TF4_3.5_-2.0_30.0_60.0_25.0_ic.json',
#         'sf_sep': 4.0
#     }
# ]

lgn = NetworkBuilder('lgn')
for _, params in lgn_models.items():
    n_cells = params['N']
    size_range = params['size_range']
    # Get position of lgn cells and keep track of the averaged location
    # For now, use randomly generated values
    # print(params)
    # total_N = params['N'] * x_grids * y_grids

    # Get positional coordinates of cells
    positions = generate_positions_grids(n_cells, x_len, y_len)

    # Get spatial filter size of cells
    # filter_sizes = get_filter_spatial_size(params['N'], x_grids, y_grids, params['size_range'])

    lgn.add_nodes(
        N=n_cells,
        # ei=params['ei'],
        subtypex=params['subtype'],
        model_type=params['model_type'],
        model_template=params['model_template'],
        x=[100.0], # positions[:, 0],
        y=[100.0], # positions[:, 1],
        dynamics_params=params['dynamics_params'],
        non_dom_params=params.get('non_dom_params', None),
        spatial_size=np.random.uniform(size_range[0], size_range[1], n_cells),

        # NOTE: If tuning angle is not defined, then it will be randomly generated during the simulation. But
        #  when evaluating a large network many times it will be more efficent to store it in the nodes file.
        # tuning_angle=np.random.uniform(0.0, 360.0, n_cells),

        # TODO: Can sf-sperator be stored in the params json file.
        sf_sep=params.get('sf_sep', None)
    )

lgn.build()
lgn.save(output_dir='network')
