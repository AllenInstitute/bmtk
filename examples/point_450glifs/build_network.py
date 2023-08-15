import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random

# This model is a NEST equivelent to the bio_450cells network

# List of non-virtual cell models
glif_models = [
    {
        'model_name': 'Scnn1a',
        'ei': 'e',
        'model_template': 'nest:glif_psc',
        'dynamics_params': '593618144_glif_lif_asc_psc.json'
    },
    {
        'model_name': 'Rorb',
        'ei': 'e',
        'model_template': 'nest:glif_psc',
        'dynamics_params': '480124551_glif_lif_asc_psc.json'
    },
    {
        'model_name': 'Nr5a1',
        'ei': 'e',
        'model_template': 'nest:glif_psc',
        'dynamics_params': '318808427_glif_lif_asc_psc.json'
    },
    {
        'model_name': 'PV1',
        'ei': 'i',
        'model_template': 'nest:glif_psc',
        'dynamics_params': '478958894_glif_lif_asc_psc.json'
    },
    {
        'model_name': 'PV2',
        'ei': 'i',
        'model_template': 'nest:glif_psc',
        'dynamics_params': '487667205_glif_lif_asc_psc.json'
    }
]

intfire_models = [
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


def build_v1_network():
    print('Building v1 network')

    v1 = NetworkBuilder('v1')
    for i, model_props in enumerate(glif_models):
        n_cells = 80 if model_props['ei'] == 'e' else 30  # 80% excitatory, 20% inhib

        # Randomly get positions uniformly distributed in a column
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)

        v1.add_nodes(
            N=n_cells,
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            rotation_angle_yaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  # randomly rotate y axis
            rotation_angle_zaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  #
            model_type='point_neuron',
            placement='inner',
            **model_props
        )

    # Build intfire type cells
    for model_props in intfire_models:
        n_cells = 75  # Just assume 75 cells for both point inhibitory and point excitatory
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)
        v1.add_nodes(
            N=n_cells,
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            model_type='point_neuron',
            placement='outer',
            **model_props
        )

    # The function is called during the build() processes for every source/target pair specified by
    #  add_edges(source_filter, target_filter, connection_rule=n_connections, ...)
    def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=5):
        """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
        pair will connect the two with a probability prob (excludes self-connections)"""
        if src.node_id == trg.node_id:
            return 0

        return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)

    # Connections onto glif components, use the connection map to save section and position of every synapse
    # exc --> exc connections
    v1.add_edges(
        source={'ei': 'e'}, target={'ei': 'e', 'placement': 'inner'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='e2e.json',
        model_template='static_synapse',
        syn_weight=2.5,
        delay=2.0
    )

    # exc --> inh connections
    v1.add_edges(
        source={'ei': 'e'}, target={'ei': 'i', 'placement': 'inner'},
        connection_rule=n_connections,
        dynamics_params='e2i.json',
        model_template='static_synapse',
        syn_weight=5.0,
        delay=2.0
    )

    # inh --> exc connections
    v1.add_edges(
        source={'ei': 'i'}, target={'ei': 'e', 'placement': 'inner'},
        connection_rule=n_connections,
        dynamics_params='i2e.json',
        model_template='static_synapse',
        syn_weight=-6.5,
        delay=2.0
    )

    # inh --> inh connections
    v1.add_edges(
        source={'ei': 'i'}, target={'ei': 'i', 'placement': 'inner'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='i2i.json',
        model_template='static_synapse',
        syn_weight=-3.0,
        delay=2.0
    )

    # For connections on point neurons it doesn't make sense to save syanpatic location
    v1.add_edges(
        source={'ei': 'e'}, target={'placement': 'outer'},
        connection_rule=n_connections,
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        syn_weight=3.0,
        delay=2.0
    )

    v1.add_edges(
        source={'ei': 'i'}, target={'placement': 'outer'},
        connection_rule=n_connections,
        dynamics_params='instantaneousInh.json',
        model_template='static_synapse',
        syn_weight=-4.0,
        delay=2.0
    )

    # Build and save internal network
    print('   saving network.')
    v1.build()
    v1.save(output_dir='network')
    print('   done.')

    return v1


def build_lgn_network(v1):
    # Build a network of 100 virtual cells that will connect to and drive the simulation of the internal network
    print('Building lgn network')
    lgn = NetworkBuilder("lgn")

    lgn.add_nodes(N=100, model_type='virtual', ei='e')

    # Targets all glif excitatory cells
    lgn.add_edges(
        target=v1.nodes(ei='e', placement='inner'), source=lgn.nodes(),
        connection_rule=lambda *_: np.random.randint(0, 5),
        dynamics_params='LGN_to_GLIF.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=11.0
    )

    # Targets all glif inhibitory cells
    lgn.add_edges(
        target=v1.nodes(ei='i', placement='inner'), source=lgn.nodes(),
        connection_rule=lambda *_: np.random.randint(0, 5),
        dynamics_params='LGN_to_GLIF.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=14.0
    )

    # Targets all intfire1 cells (exc and inh)
    lgn.add_edges(
        target=v1.nodes(placement='outer'), source=lgn.nodes(),
        connection_rule=lambda *_: np.random.randint(0, 5),
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=13.0
    )

    print('   saving network.')
    lgn.build()
    lgn.save(output_dir='network')
    print('   done.')

    return lgn


if __name__ == '__main__':
    v1 = build_v1_network()
    lgn = build_lgn_network(v1)
