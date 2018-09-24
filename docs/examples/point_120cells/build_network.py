import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.aux.node_params import positions_columinar


def random_connections(source, target, p=0.1):
    sid = source['node_id']  # Get source id
    tid = target['node_id']  # Get target id

    # Avoid self-connections.
    if sid == tid:
        return None

    return np.random.binomial(1, p)  # nsyns


LIF_models = {
    'LIF_exc': {
        'N': 80,
        'ei': 'e',
        'pop_name': 'LIF_exc',
        'model_type': 'point_process',
        'model_template': 'nest:iaf_psc_delta',
        'dynamics_params': 'iaf_psc_delta_exc.json'
    },
    'LIF_inh': {
        'N': 40,
        'ei': 'i',
        'pop_name': 'LIF_inh',
        'model_type': 'point_process',
        'model_template': 'nest:iaf_psc_delta',
        'dynamics_params': 'iaf_psc_delta_inh.json'
    }
}


net = NetworkBuilder('cortex')
for model in LIF_models:
    params = LIF_models[model].copy()
    positions = positions_columinar(N=LIF_models[model]['N'], center=[0, 10.0, 0], max_radius=50.0, height=200.0)
    net.add_nodes(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                  **params)


net.add_edges(source={'ei': 'e'},
              connection_rule=random_connections,
              connection_params={'p': 0.1},
              syn_weight=2.0,
              delay=1.5,
              dynamics_params='ExcToInh.json',
              model_template='static_synapse')

net.add_edges(source={'ei': 'i'},
              connection_rule=random_connections,
              connection_params={'p': 0.1},
              syn_weight=-1.5,
              delay=1.5,
              dynamics_params='InhToExc.json',
              model_template='static_synapse')

net.build()
net.save_nodes(output_dir='network')
net.save_edges(output_dir='network')



input_network_model = {
    'input_network': {
        'N': 100,
        'ei': 'e',
        'pop_name': 'input_network',
        'model_type': 'virtual'
    }
}


inputNetwork = NetworkBuilder("thalamus")
inputNetwork.add_nodes(**input_network_model['input_network'])

inputNetwork.add_edges(target=net.nodes(),
                       connection_rule=random_connections,
                       connection_params={'p': 0.1},
                       syn_weight=4.2,
                       delay=1.5,
                       dynamics_params='ExcToExc.json',
                       model_template='static_synapse')
inputNetwork.build()
inputNetwork.save_nodes(output_dir='network')
inputNetwork.save_edges(output_dir='network')
