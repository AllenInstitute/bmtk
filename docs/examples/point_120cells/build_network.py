import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar


lif_models = {
    'LIF_exc': {
        'N': 80,
        'ei': 'e',
        'pop_name': 'LIF_exc',
        'model_type': 'point_neuron',
        'model_template': 'nest:iaf_psc_delta',
        'dynamics_params': 'iaf_psc_delta_exc.json'
    },
    'LIF_inh': {
        'N': 40,
        'ei': 'i',
        'pop_name': 'LIF_inh',
        'model_type': 'point_neuron',
        'model_template': 'nest:iaf_psc_delta',
        'dynamics_params': 'iaf_psc_delta_inh.json'
    }
}

input_network_model = {
    'input_network': {
        'N': 100,
        'ei': 'e',
        'pop_name': 'input_network',
        'model_type': 'virtual'
    }
}


def random_connections(source, target, p=0.1):
    sid = source['node_id']  # Get source id
    tid = target['node_id']  # Get target id

    # Avoid self-connections.
    if sid == tid:
        return None

    return np.random.binomial(1, p)  # nsyns


def build_cortex_network():
    cortex = NetworkBuilder('cortex')
    for model in lif_models:
        params = lif_models[model].copy()
        positions = positions_columinar(N=lif_models[model]['N'], center=[0, 10.0, 0], max_radius=50.0, height=200.0)
        cortex.add_nodes(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            **params
        )

    cortex.add_edges(
        source={'ei': 'e'},
        connection_rule=random_connections,
        connection_params={'p': 0.1},
        syn_weight=2.0,
        delay=1.5,
        dynamics_params='ExcToInh.json',
        model_template='static_synapse'
    )

    cortex.add_edges(
        source={'ei': 'i'},
        connection_rule=random_connections,
        connection_params={'p': 0.1},
        syn_weight=-1.5,
        delay=1.5,
        dynamics_params='InhToExc.json',
        model_template='static_synapse'
    )

    cortex.build()
    cortex.save(output_dir='network')

    return cortex


def build_thalamus_network(cortex):
    thalamus = NetworkBuilder('thalamus')
    thalamus.add_nodes(**input_network_model['input_network'])

    thalamus.add_edges(
        target=cortex.nodes(),
        connection_rule=random_connections,
        connection_params={'p': 0.1},
        syn_weight=4.2,
        delay=1.5,
        dynamics_params='ExcToExc.json',
        model_template='static_synapse'
    )
    thalamus.build()
    thalamus.save(output_dir='network')

    return thalamus


if __name__ == '__main__':
    cortex_net = build_cortex_network()
    thalamus_net = build_thalamus_network(cortex_net)
