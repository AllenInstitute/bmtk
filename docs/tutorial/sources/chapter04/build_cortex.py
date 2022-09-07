import numpy as np
import math
import random

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random
from bmtk.builder.auxi.edge_connectors import distance_connector

net = NetworkBuilder("V1")
net.add_nodes(N=80, pop_name='Scnn1a',
              positions=positions_columinar(N=80, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
              rotation_angle_yaxis=xiter_random(N=80, min_x=0.0, max_x=2*np.pi),
              rotation_angle_zaxis=xiter_random(N=80, min_x=0.0, max_x=2*np.pi),
              tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
              location='VisL4',
              ei='e',
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='472363762_fit.json',
              morphology='Scnn1a.swc')

net.add_nodes(N=20, pop_name='PV',
              positions=positions_columinar(N=20, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
              rotation_angle_yaxis=xiter_random(N=20, min_x=0.0, max_x=2*np.pi),
              rotation_angle_zaxis=xiter_random(N=20, min_x=0.0, max_x=2*np.pi),
              location='VisL4',
              ei='i',
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='472912177_fit.json',
              morphology='Pvalb.swc')

net.add_nodes(N=200, pop_name='LIF_exc',
              positions=positions_columinar(N=200, center=[0, 50.0, 0], min_radius=30.0, max_radius=60.0, height=100.0),
              tuning_angle=np.linspace(start=0.0, stop=360.0, num=200, endpoint=False),
              location='VisL4',
              ei='e',
              model_type='point_process',
              model_template='nrn:IntFire1',
              dynamics_params='IntFire1_exc_1.json')

net.add_nodes(N=100, pop_name='LIF_inh',
              positions=positions_columinar(N=100, center=[0, 50.0, 0], min_radius=30.0, max_radius=60.0, height=100.0),
              location='VisL4',
              ei='i',
              model_type='point_process',
              model_template='nrn:IntFire1',
              dynamics_params='IntFire1_inh_1.json')


## Generating E-to-E connections
def dist_tuning_connector(source, target, d_weight_min, d_weight_max, d_max, t_weight_min, t_weight_max, nsyn_min,
                          nsyn_max):
    if source['node_id'] == target['node_id']:
        # Avoid self-connections.n_nodes
        return None

    r = np.linalg.norm(np.array(source['positions']) - np.array(target['positions']))
    if r > d_max:
        dw = 0.0
    else:
        t = r / d_max
        dw = d_weight_max * (1.0 - t) + d_weight_min * t

    if dw <= 0:
        # drop the connection if the weight is too low
        return None

    # next create weights by orientation tuning [ aligned, misaligned ] --> [ 1, 0 ], Check that the orientation
    # tuning property exists for both cells; otherwise, ignore the orientation tuning.
    if 'tuning_angel' in source and 'tuning_angle' in target:

        # 0-180 is the same as 180-360, so just modulo by 180
        delta_tuning = math.fmod(abs(source['tuning_angle'] - target['tuning_angle']), 180.0)

        # 90-180 needs to be flipped, then normalize to 0-1
        delta_tuning = delta_tuning if delta_tuning < 90.0 else 180.0 - delta_tuning

        t = delta_tuning / 90.0
        tw = t_weight_max * (1.0 - t) + t_weight_min * t
    else:
        tw = dw

    # drop the connection if the weight is too low
    if tw <= 0:
        return None

    # filter out nodes by treating the weight as a probability of connection
    if random.random() > tw:
        return None

    # Add the number of synapses for every connection.
    # It is probably very useful to take this out into a separate function.
    return random.randint(nsyn_min, nsyn_max)

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
              connection_rule=dist_tuning_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 300.0, 't_weight_min': 0.5,
                                 't_weight_max': 1.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=6.4e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
              connection_rule=dist_tuning_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 300.0, 't_weight_min': 0.5,
                                 't_weight_max': 1.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.0019,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='exp2syn')


### Generating I-to-I connections
net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'model_type': 'biophysical'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.0002,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='GABA_InhToInh.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'model_type': 'point_process'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.00225,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instantaneousInh.json',
              model_template='exp2syn')

### Generating I-to-E connections
net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'biophysical'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.00018,
              weight_function='wmax',
              distance_range=[0.0, 50.0],
              target_sections=['somatic', 'basal', 'apical'],
              delay=2.0,
              dynamics_params='GABA_InhToExc.json',
              model_template='exp2syn')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'point_process'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.009,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instantaneousInh.json',
              model_template='exp2syn')

### Generating E-to-I connections
net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.26, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.00035,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='AMPA_ExcToInh.json',
              model_template='exp2syn')


net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.26, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=0.0043,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='exp2syn')

net.build()
net.save_nodes(output_dir='network')
net.save_edges(output_dir='network')