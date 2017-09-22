import numpy as np
import random
import math

from bmtk.builder.networks import NetworkBuilder


def lerp(v0, v1, t):
    return v0 * (1.0 - t) + v1 * t


def distance_weight(delta_p, w_min, w_max, r_max):
    r = np.linalg.norm(delta_p)

    if r >= r_max:
        return 0.0
    else:
        return lerp(w_max, w_min, r / r_max)


def orientation_tuning_weight(tuning1, tuning2, w_min, w_max):

    # 0-180 is the same as 180-360, so just modulo by 180
    delta_tuning = math.fmod(abs(tuning1 - tuning2), 180.0)

    # 90-180 needs to be flipped, then normalize to 0-1
    delta_tuning = delta_tuning if delta_tuning < 90.0 else 180.0 - delta_tuning

    # t = delta_tuning / 90.0
    return lerp(w_max, w_min, delta_tuning / 90.0)


def distance_tuning_connection_handler(source, target, d_weight_min, d_weight_max, d_max, t_weight_min,
                                       t_weight_max, nsyn_min, nsyn_max):

    # Avoid self-connections.n_nodes
    sid = source.node_id
    tid = target.node_id
    if sid == tid:
        if sid % 1000 == 0:
            print "processing connections for node",  sid
        return None

    # first create weights by euclidean distance between cells
    # DO NOT use PERIODIC boundary conditions in x and y!
    dw = distance_weight(np.array(source['position'][0::2]) - np.array(target['position'][0::2]), d_weight_min,
                         d_weight_max, d_max)

    # drop the connection if the weight is too low
    if dw <= 0:
        return None

    # next create weights by orientation tuning [ aligned, misaligned ] --> [ 1, 0 ]
    # Check that the orientation tuning property exists for both cells; otherwise,
    # ignore the orientation tuning.
    if source['tuning_angle'] != 'NA' and target['tuning_angle'] != 'NA':
        tw = dw * orientation_tuning_weight(source['tuning_angle'],
                                            target['tuning_angle'],
                                            t_weight_min, t_weight_max)
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

    tmp_nsyn = random.randint(nsyn_min, nsyn_max)
    return tmp_nsyn

def distance_connection_handler(source, target, d_weight_min, d_weight_max, d_max, nsyn_min, nsyn_max):
    # Avoid self-connections.
    sid = source.node_id
    tid = target.node_id

    if sid == tid:
        if sid % 1000 == 0:
            print "processing connections for node", sid
        return None

    # first create weights by euclidean distance between cells
    # DO NOT use PERIODIC boundary conditions in x and y!
    dw = distance_weight(np.array(source['position'][0::2]) - np.array(target['position'][0::2]),
                               d_weight_min, d_weight_max, d_max)

    # drop the connection if the weight is too low
    if dw <= 0:
        return None

    # filter out nodes by treating the weight as a probability of connection
    if random.random() > dw:
        return None

    # Add the number of synapses for every connection.
    # It is probably very useful to take this out into a separate function.

    tmp_nsyn = random.randint(nsyn_min, nsyn_max)
    # print('{} ({}) --> {} ({})'.format(sid, source['name'], tid, target['name']))
    return tmp_nsyn

cell_models = {
    'Scnn1a': {
        'N': 3700,
        'props': {
            'ei': 'e',
            'level_of_detail': 'biophysical',
            'morphology': 'Scnn1a-Tg3-Cre_Ai14_IVSCC_-177300.01.02.01_473845048_m.swc',
            'electrophysiology': '472363762_fit.json',
            'rotation_angle_zaxis': -3.646878266,
            'hoc_template': 'Biophys1'
        }

    },
    'Rorb': {
        'N': 3300,
        'props': {
            'ei': 'e',
            'level_of_detail': 'biophysical',
            'morphology': 'Rorb-IRES2-Cre-D_Ai14_IVSCC_-168053.05.01.01_325404214_m.swc',
            'electrophysiology': '473863510_fit.json',
            'rotation_angle_zaxis': -4.159763785,
            'hoc_template': 'Biophys1'
        }
    },
    'Nr5a1': {
        'N': 1500,
        'props': {
            'ei': 'e',
            'level_of_detail': 'biophysical',
            'morphology': 'Nr5a1-Cre_Ai14_IVSCC_-169250.03.02.01_471087815_m.swc',
            'electrophysiology': '473863035_fit.json',
            'rotation_angle_zaxis': -2.639275277,
            'hoc_template': 'Biophys1'
        }
    },
    'PV1': {
        'N': 800,
        'props': {
            'ei': 'i',
            'level_of_detail': 'biophysical',
            'morphology': 'Pvalb-IRES-Cre_Ai14_IVSCC_-176847.04.02.01_470522102_m.swc',
            'electrophysiology': '472912177_fit.json',
            'rotation_angle_zaxis': -2.539551891,
            'hoc_template': 'Biophys1'
        }
    },
    'PV2': {
        'N': 700,
        'props': {
            'ei': 'i',
            'level_of_detail': 'biophysical',
            'morphology': 'Pvalb-IRES-Cre_Ai14_IVSCC_-169125.03.01.01_469628681_m.swc',
            'electrophysiology': '473862421_fit.json',
            'rotation_angle_zaxis': -3.684439949,
            'hoc_template': 'Biophys1'
        }
    },
    'LIF_exc': {
        'N': 29750,
        'props': {
            'ei': 'e',
            'level_of_detail': 'intfire',
            'electrophysiology': 'IntFire1_exc_1.json',
            'hoc_template': 'IntFire1'
        }
    },
    'LIF_inh': {
        'N': 5250,
        'props': {
            'ei': 'i',
            'level_of_detail': 'intfire',
            'electrophysiology': 'IntFire1_inh_1.json',
            'hoc_template': 'IntFire1'
        }
    }
}


net = SynNetwork('V1/L4')
for name, model_params in cell_models.items():
    N = model_params['N']
    cell_props = {'position': np.random.rand(N, 3)*[100.0, -300.0, 100.0],
                  'rotation_angle': np.random.uniform(0.0, 2*np.pi, (N,))}
    if model_params['props']['ei'] == 'e':
        cell_props['tuning_angle'] = np.linspace(0, 360.0, N, endpoint=False)
    else:
        cell_props['tuning_angle'] = ['NA']*N

    cell_props.update(model_params['props'])

    net.add_nodes(N=N,
                  pop_name=name,
                  location='VisL4',
                  **cell_props)

'''
net.build()
net.save_nodes('tmp_nodes.h5', 'tmp_node_types.csv')
exit()
'''

cparameters = {'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7}
net.add_edges(sources={'ei': 'i'}, targets={'ei': 'i', 'level_of_detail': 'biophysical'},
              func=distance_connection_handler, func_params=cparameters,
              weight_max=0.0002,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              params_file='GABA_InhToInh.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'i'}, targets={'ei': 'i', 'level_of_detail': 'intfire'},
              func=distance_connection_handler,
              func_params=cparameters,
              weight_max=0.00225,
              weight_function='wmax',
              delay=2.0,
              params_file='instanteneousInh.json',
              set_params_function='exp2syn')

cparameters = {'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7}
net.add_edges(sources={'ei': 'i'}, targets={'ei': 'e', 'level_of_detail': 'biophysical'},
              func=distance_connection_handler,
              func_params=cparameters,
              weight_max=0.00018,
              weight_function='wmax',
              distance_range=[0.0, 50.0],
              target_sections=['somatic', 'basal', 'apical'],
              delay=2.0,
              params_file='GABA_InhToExc.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'i'}, targets={'ei': 'e', 'level_of_detail': 'intfire'},
              func=distance_connection_handler,
              func_params=cparameters,
              weight_max=0.009,
              weight_function='wmax',
              delay=2.0,
              params_file='instanteneousInh.json',
              set_params_function='exp2syn')

cparameters = {'d_weight_min': 0.0, 'd_weight_max': 0.26, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7}
net.add_edges(sources={'ei': 'e'}, targets={'name': 'PV1'},
              func=distance_connection_handler,
              func_params=cparameters,
              weight_max=0.00035,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              params_file='AMPA_ExcToInh.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'e'}, targets={'name': 'PV2'},
              func=distance_connection_handler,
              func_params=cparameters,
              weight_max=0.00027,
              weight_function='wmax',
              distance_range=[0.0, 1e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              params_file='AMPA_ExcToInh.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'e'}, targets={'name': 'LIF_inh'},
              func=distance_connection_handler,
              func_params=cparameters,
              weight_max=0.0043,
              weight_function='wmax',
              delay=2.0,
              params_file='instanteneousExc.json',
              set_params_function='exp2syn')

cparameters = {'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 300.0, 't_weight_min': 0.5,
               't_weight_max': 1.0, 'nsyn_min': 3, 'nsyn_max': 7}
net.add_edges(sources={'ei': 'e'}, targets={'name': 'Scnn1a'},
              func=distance_tuning_connection_handler,
              func_params=cparameters,
              weight_max=6.4e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'e'}, targets={'name': 'Rorb'},
              func=distance_tuning_connection_handler,
              func_params=cparameters,
              weight_max=5.5e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'e'}, targets={'name': 'Nr5a1'},
              func=distance_tuning_connection_handler,
              func_params=cparameters,
              weight_max=7.2e-05,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              distance_range=[30.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

net.add_edges(sources={'ei': 'e'}, targets={'name': 'LIF_exc'},
              func=distance_tuning_connection_handler,
              func_params=cparameters,
              weight_max=0.0019,
              weight_function='gaussianLL',
              weight_sigma=50.0,
              delay=2.0,
              params_file='instanteneousExc.json',
              set_params_function='exp2syn')

net.build()
net.save_nodes('v1_nodes.h5', 'v1_node_types.csv')
net.save_edges('v1_edges.h5', 'v1_edge_types.csv')
