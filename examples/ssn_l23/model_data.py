""" This file defines necessary parameters (constants?) for the model.
The order of the cell types is [EXC, PV, SST, VIP]

"""

# %%
import numpy as np
import pickle
import os
import pandas as pd
import itertools

# fmt: off
""" 11/23/2021: The final influence matrix is recalculated based on the mean values.
(previously they were medians.), and the errors are evaluated as well. To see the
calculation pathway, refer to this excel sheet.
https://alleninstitute-my.sharepoint.com/:x:/g/personal/shinya_ito_alleninstitute_org/ERgemlvA6cRLi4acAty1OjkBaqeyzi6zrEG5uqzglXuEEg?e=6A8Zs8
In the new fit method, these values will be used to make an additional term in the loss function.
"""
l23_infl_matrix_mean    = [[ 0.02884,  0.40241,  0.19445,  0.28137],
                           [-0.01013, -0.01554, -0.00361,  0.00000],
                           [-0.00237, -0.00200, -0.00061, -0.00746],
                           [-0.00046, -0.00049, -0.00657, -0.00100]]

l23_infl_df = pd.DataFrame({
    'lu_key': [f'{t}_to_{s}' for t, s in itertools.product(['e', 'p', 's', 'v'], repeat=2)],
    'mean': np.array(l23_infl_matrix_mean).flatten()
}).set_index('lu_key')


l23_infl_matrix_fracerr =  [[0.460, 0.368, 0.410, 0.415],
                            [0.372, 0.323, 0.598, 1.000],
                            [0.374, 0.377, 0.547, 0.339],
                            [0.623, 1.089, 0.467, 0.845]]

""" 1/20/2022: Additional influence matrix from E4 to L23 population. These values
will be used to provide inputs to L23 population from E4 activity. Again, refer to
the excel file above for the detailed calculations of these values. """
e4_l23_infl_matrix_mean =    [0.02356, 0.27821, 0.00342, 0.02988]
e4_l23_infl_matrix_fracerr = [0.571,   0.473,   1.056,   1.146]

""" L4 excitatory neuron activity will be used for stimulaion.
"""
# if the file exists, load it.
if os.path.exists('./neuropixels_data/l4_mean_traces.pkl'):
    with open('./neuropixels_data/l4_mean_traces.pkl', 'rb') as f:
        l4_excitatory_activity = pickle.load(f)


""" These connection probabilities were taken from Campagnola et al. 2021
Figure Supplement 2 Connectivity Matrix
https://www.biorxiv.org/content/10.1101/2021.03.31.437553v2.full
"""
connection_prob = {}
connection_prob['L23'] = [[ 0.12,  0.81,  0.78,  0.51],
                          [ 0.67,  0.78,  0.22,  0.00],
                          [ 0.44,  0.37,  0.11,  1.00],
                          [ 0.13,  0.05,  0.51,  0.07]]

connection_prob['L4']  = [[ 0.22,  0.38,  0.10,  0.00],
                          [ 0.54,  1.00,  0.00,  0.29],
                          [ 0.54,  0.00,  0.00,  1.00],
                          [ 0.00,  0.05,  0.60,  0.11]]

# L5 have ET and IT. As IT is the majority, I'm using the value for IT.
connection_prob['L5']  = [[ 0.06,  0.21,  0.28,  0.11],
                          [ 0.34,  0.50,  0.14,  0.17],
                          [ 0.13,  0.29,  0.11,  0.36],
                          [ 0.00,  0.03,  0.21,  0.17]]

connection_prob['L6']  = [[ 0.01,  0.44,  0.32,  0.00],
                          [ 0.53,  0.76,  0.05,  0.08],
                          [ 0.15,  0.34,  0.17,  0.74],
                          [ 0.00,  0.17,  0.25,  0.00]]


""" Connection strength is also taken from the same paper. Figure 3D, E
There are resting state PSP and 90th percentile PSP.
There is no distinction for inhibitory layers, so the values are copied for them
"""
connection_strength = {}
connection_strength['L23'] = [[ 0.10,  0.27,  0.09,  0.56],
                              [-0.44, -0.47, -0.25, -0.27],
                              [-0.16, -0.18, -0.17, -0.17],
                              [-0.06, -0.13, -0.15, -0.14]]

connection_strength['L4']  = [[ 0.15,  0.45,  0.05,  0.66],
                              [-0.48, -0.47, -0.25, -0.27],
                              [-0.18, -0.18, -0.17, -0.17],
                              [-0.00, -0.13, -0.15, -0.14]]

connection_strength['L5']  = [[ 0.19,  0.51,  0.08,  0.08],
                              [-0.53, -0.47, -0.25, -0.27],
                              [-0.17, -0.18, -0.17, -0.17],
                              [-0.00, -0.13, -0.15, -0.14]]

connection_strength['L6']  = [[ 0.00,  0.49,  0.20,  0.00],
                              [-0.36, -0.47, -0.25, -0.27],
                              [-0.18, -0.18, -0.17, -0.17],
                              [-0.00, -0.13, -0.15, -0.14]]

connection_strength_90 = {}
connection_strength_90['L23'] = [[ 0.29,  0.60,  0.94,  1.85],
                                 [-0.39, -0.54, -0.61, -0.54],
                                 [-0.34, -0.41, -0.42, -0.76],
                                 [-0.21, -0.30, -0.43, -0.32]]

connection_strength_90['L4']  = [[ 0.32,  0.61,  0.50,  2.31],
                                 [-0.62, -0.54, -0.61, -0.54],
                                 [-0.35, -0.41, -0.42, -0.76],
                                 [-0.00, -0.30, -0.43, -0.32]]

connection_strength_90['L5']  = [[ 0.33,  0.70,  0.85,  0.56],
                                 [-0.56, -0.54, -0.61, -0.54],
                                 [-0.27, -0.41, -0.42, -0.76],
                                 [-0.00, -0.30, -0.43, -0.32]]

connection_strength_90['L6']  = [[ 0.00,  1.24,  0.72,  0.00],
                                 [-0.51, -0.54, -0.61, -0.54],
                                 [-0.46, -0.41, -0.42, -0.76],
                                 [-0.00, -0.30, -0.43, -0.32]]

""" relative population in each layer also matters for how much influence each cell
type has. The data were based on MICrON Minnie dataset and Lee et al. (2010)
"""

relative_population = {}
relative_population['L23'] = [0.90,  0.10 * 0.30,  0.10 * 0.21,  0.10 * 0.49]
relative_population['L4']  = [0.89,  0.11 * 0.55,  0.11 * 0.30,  0.11 * 0.15]
relative_population['L5']  = [0.81,  0.19 * 0.48,  0.19 * 0.43,  0.19 * 0.09]
relative_population['L6']  = [0.93,  0.07 * 0.46,  0.07 * 0.46,  0.07 * 0.08]


""" Firing rate range is derived fron the Neuropixels dataset.
These values are used to convert the calcium signal to firing rates.
(See neuropixels_analysis directory for details.)
"""
firing_rate_ranges = [[ 1.44,  2.79],  # EXC
                      [ 7.44, 14.69],  # PV
                      [11.55, 13.32],  # SST
                      [10.79, 13.18]]  # VIP

# fmt: on

""" Now, calculate influence matrix for each layer.
Influence matrix is net effect considering all probability, strengh and
relative population
"""

layers = ["L23", "L4", "L5", "L6"]
influence_matrix = [
    np.array(connection_prob[layer])
    * np.array(connection_strength[layer])
    * np.expand_dims(np.array(relative_population[layer]), axis=1)
    for layer in layers
]

average_influence_matrix = np.mean(influence_matrix, axis=0)

# Also calculate 90th percentile version
influence_matrix_90 = [
    np.array(connection_prob[layer])
    * np.array(connection_strength_90[layer])
    * np.expand_dims(np.array(relative_population[layer]), axis=1)
    for layer in layers
]

average_influence_matrix_90 = np.mean(influence_matrix_90, axis=0)
