from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.aux.node_params import positions_columinar, xiter_random
from bmtk.builder.aux.edge_connectors import distance_connector

import math
import numpy as np
import random

"""
def columinar(N=1, center=[0.0, 50.0, 0.0], height=100.0, min_radius=0.0, max_radius=1.0, distribution='uniform'):
    phi = 2.0 * math.pi * np.random.random([N])
    r = np.sqrt((min_radius**2 - max_radius**2) * np.random.random([N]) + max_radius**2)
    x = center[0] + r * np.cos(phi)
    z = center[2] + r * np.sin(phi)
    y = center[1] + height * (np.random.random([N]) - 0.5)

    return np.column_stack((x, y, z))


def xiter_random(N=1, min_x=0.0, max_x=1.0):
    return np.random.uniform(low=min_x, high=max_x, size=(N,))
"""

"""
def distance_connection_handler(source, target, d_weight_min, d_weight_max, d_max,
                                nsyn_min, nsyn_max):
    # Avoid self-connections.
    sid = source.node_id
    tid = target.node_id
    if sid == tid:
        return None

    # first create weights by euclidean distance between cells
    r = np.linalg.norm(np.array(source['positions']) - np.array(target['positions']))
    if r > d_max:
        dw = 0.0
    else:
        t = r / d_max
        dw = d_weight_max * (1.0 - t) + d_weight_min * t

    # drop the connection if the weight is too low
    if dw <= 0:
        return None

    # filter out nodes by treating the weight as a probability of connection
    if random.random() > dw:
        return None

    # Add the number of synapses for every connection.
    tmp_nsyn = random.randint(nsyn_min, nsyn_max)
    return tmp_nsyn
"""

cortex = NetworkBuilder('mcortex')
cortex.add_nodes(N=100,
                 pop_name='Scnn1a',
                 positions=positions_columinar(N=100, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
                 rotation_angle_yaxis=xiter_random(N=100, min_x=0.0, max_x=2*np.pi),
                 rotation_angle_zaxis=3.646878266,
                 potental='exc',
                 level_of_detail='biophysical',
                 params_file='472363762_fit.json',
                 morphology_file='Scnn1a.swc',
                 set_params_function='Biophys1')

cortex.add_edges(source={'pop_name': 'Scnn1a'}, target={'pop_name': 'Scnn1a'},
                 connection_rule=distance_connector,
                 connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 50.0, 'nsyn_min': 0,
                                    'nsyn_max': 10},
                 weight_max=6.4e-05,
                 weight_function='wmax',
                 distance_range=[30.0, 150.0],
                 target_sections=['basal', 'apical'],
                 delay=2.0,
                 params_file='AMPA_ExcToExc.json',
                 set_params_function='exp2syn')

cortex.build()
cortex.save_nodes(output_dir='network')
cortex.save_edges(output_dir='network')