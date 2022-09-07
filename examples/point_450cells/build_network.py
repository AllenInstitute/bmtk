import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random

#build_recurrent_edges = True

# List of non-virtual cell models
core_cells = [
    {
        'model_name': 'Scnn1a',
        'ei': 'e',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': '472363762_point.json'
    },
    {
        'model_name': 'Rorb',
        'ei': 'e',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': '473863510_point.json'
    },
    {
        'model_name': 'Nr5a1',
        'ei': 'e',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': '473863035_point.json'
    },
    {
        'model_name': 'PV1',
        'ei': 'i',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': '472912177_point.json'
    },
    {
        'model_name': 'PV2',
        'ei': 'i',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': '473862421_point.json'
    }
]

outer_cells = [
    {
        'model_name': 'LIF_exc',
        'ei': 'e',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': 'IntFire1_exc_point.json'
    },
    {
        'model_name': 'LIF_inh',
        'ei': 'i',
        'model_template': 'nest:iaf_psc_alpha',
        'dynamics_params': 'IntFire1_inh_point.json'
    }
]


def build_internal_network():
    print('Building internal network.')

    internal = NetworkBuilder('internal')
    for i, model_props in enumerate(core_cells):
        n_cells = 80 if model_props['ei'] == 'e' else 30  # 80% excitatory, 20% inhib

        # Randomly get positions uniformly distributed in a column
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)

        internal.add_nodes(
            N=n_cells,
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            model_type='point_neuron',
            orig_model='biophysical',
            **model_props
        )

    # Build intfire type cells
    for model_props in outer_cells:
        n_cells = 75  # Just assume 75 cells for both point inhibitory and point excitatory
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)
        internal.add_nodes(
            N=n_cells,
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            model_type='point_neuron',
            orig_model='intfire',
            **model_props
        )

    def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=5):
        """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
        pair will connect the two with a probability prob (excludes self-connections)"""
        if src.node_id == trg.node_id:
            return 0

        return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)

    # Connections onto biophysical components, use the connection map to save section and position of every synapse
    # exc --> exc connections
    internal.add_edges(
        source={'ei': 'e'}, target={'ei': 'e', 'orig_model': 'biophysical'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='ExcToExc.json',
        model_template='static_synapse',
        syn_weight=2.5,
        delay=2.0
    )

    # exc --> inh connections
    internal.add_edges(
        source={'ei': 'e'}, target={'ei': 'i', 'orig_model': 'biophysical'},
        connection_rule=n_connections,
        dynamics_params='ExcToInh.json',
        model_template='static_synapse',
        syn_weight=5.0,
        delay=2.0
    )

    # inh --> exc connections
    internal.add_edges(
        source={'ei': 'i'}, target={'ei': 'e', 'orig_model': 'biophysical'},
        connection_rule=n_connections,
        dynamics_params='InhToExc.json',
        model_template='static_synapse',
        syn_weight=-6.5,
        delay=2.0
    )

    # inh --> inh connections
    internal.add_edges(
        source={'ei': 'i'}, target={'ei': 'i', 'orig_model': 'biophysical'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='InhToInh.json',
        model_template='static_synapse',
        syn_weight=-3.0,
        delay=2.0
    )

    # For connections on point neurons it doesn't make sense to save syanpatic location
    internal.add_edges(
        source={'ei': 'e'}, target={'orig_model': 'intfire'},
        connection_rule=n_connections,
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        syn_weight=3.0,
        delay=2.0
        )

    internal.add_edges(
        source={'ei': 'i'}, target={'orig_model': 'intfire'},
        connection_rule=n_connections,
        dynamics_params='instantaneousInh.json',
        model_template='static_synapse',
        syn_weight=-4.0,
        delay=2.0
    )

    # Build and save internal network
    print('   saving network.')
    internal.build()
    internal.save(output_dir='network')
    print('   done.')

    return internal


def build_external_network(internal):
    # Build a network of 100 virtual cells that will connect to and drive the simulation of the internal network
    print('Building external network')
    external = NetworkBuilder("external")

    external.add_nodes(N=100, model_type='virtual', ei='e')

    # Targets all biophysical excitatory cells
    external.add_edges(
        target=internal.nodes(ei='e', orig_model='biophysical'), source=external.nodes(),
        connection_rule=lambda *_: np.random.randint(0, 5),
        dynamics_params='ExcToExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=11.0
    )

    # Targets all biophysical inhibitory cells
    external.add_edges(
        target=internal.nodes(ei='i', orig_model='biophysical'), source=external.nodes(),
        connection_rule=lambda *_: np.random.randint(0, 5),
        dynamics_params='ExcToInh.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=14.0
    )

    # Targets all intfire1 cells (exc and inh)
    external.add_edges(
        target=internal.nodes(orig_model='intfire'), source=external.nodes(),
        connection_rule=lambda *_: np.random.randint(0, 5),
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=13.0
    )

    print('   saving network.')
    external.build()
    external.save(output_dir='network')
    print('   one.')

    return external


if __name__ == '__main__':
    internal_net = build_internal_network()
    external_net = build_external_network(internal_net)
