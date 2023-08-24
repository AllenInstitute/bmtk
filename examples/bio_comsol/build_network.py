import numpy as np
import sys
sys.path.append('..')
sys.path.append('../..')

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random
from bio_components.hdf5 import HDF5
from bmtk.builder.auxi.edge_connectors import distance_connector

import logging
logger = logging.getLogger(__name__)

np.random.seed(10)
n_nodes = 100

column = NetworkBuilder('column')
column.add_nodes(
    N=n_nodes,
    pop_name='Scnn1a',
    positions=positions_columinar(N=n_nodes, center=[0, 0, 0], min_radius = 1, max_radius=100, height=100, plot=True),
    rotation_angle_yaxis=xiter_random(N=n_nodes, min_x=0.0, max_x=2*np.pi),
    potental='exc',
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='472363762_fit.json',
    morphology='Scnn1a_473845048_m.swc'
)

column.add_edges(
    source={'pop_name': 'Scnn1a'}, target={'pop_name': 'Scnn1a'},
    connection_rule=distance_connector,
    connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 50.0, 'nsyn_min': 0, 'nsyn_max': 10},
    syn_weight=2.0e-04,
    distance_range=[5.0, 50.0],
    target_sections=['basal', 'apical', 'soma'],
    delay=2.0,
    dynamics_params='AMPA_ExcToExc.json',
    model_template='exp2syn'
)

column.build()
column.save_nodes(output_dir='network')
column.save_edges(output_dir='network')