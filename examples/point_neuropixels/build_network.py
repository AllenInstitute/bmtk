import numpy as np
import pandas as pd

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar


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


def random_connections(source, target, p=0.1):
    sid = source['node_id']  # Get source id
    tid = target['node_id']  # Get target id

    # Avoid self-connections.
    if sid == tid:
        return None

    return np.random.binomial(1, p)  # nsyns


def connect_lgn(src, trgs, dist_cutoff=30.0, max_trgs=10, max_syns=12):
    """Connects the external cells to the internal cells. Each source cell in the external population connects to
    "max_trgs" cells in the target internal population, based on the distance between cells in the xy and xz planes.

    :param src:
    :param trgs:
    :param dist_cutoff:
    :param max_trgs:
    :param max_syns:
    :return:
    """

    src_x, src_y = src['x'], src['y']
    selected_trgs = []
    for idx, trg in enumerate(trgs):
        st_dist = np.sqrt((trg['x']-src_x)**2 + (trg['z']-src_y)**2)
        if st_dist < dist_cutoff:
            selected_trgs.append(idx)

    if len(selected_trgs) == 0:
        return [0 for _ in trgs]

    selected_trgs = np.random.choice(selected_trgs, size=np.min((max_trgs, len(selected_trgs))), replace=False)
    selected_trgs = np.sort(selected_trgs)

    n_syns = np.zeros(len(trgs), dtype=int)
    n_syns[selected_trgs] = np.random.randint(0, max_syns, size=len(selected_trgs))
    return n_syns


def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=5):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    dist = np.sqrt((src['x']-trg['x'])**2 + (src['y']-trg['y'])**2 + (src['z']-trg['z'])**2)
    if dist > 100.0:
        return 0

    return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)


def build_visp():
    print('Creating network VISp')
    visp = NetworkBuilder('VISp')
    for i, model_props in enumerate(core_cells):
        n_cells = 80 if model_props['ei'] == 'e' else 30  # 80% excitatory, 20% inhib

        # Randomly get positions uniformly distributed in a column
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)

        visp.add_nodes(
            N=n_cells,
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            model_type='point_neuron',
            placement='inner',
            **model_props
        )

    # Build intfire type cells
    for model_props in outer_cells:
        n_cells = 75  # Just assume 75 cells for both point inhibitory and point excitatory
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)
        visp.add_nodes(
            N=n_cells,
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            model_type='point_neuron',
            placement='outer',
            **model_props
        )

    # Connections onto biophysical components, use the connection map to save section and position of every synapse
    # exc --> exc connections
    visp.add_edges(
        source={'ei': 'e'}, target={'ei': 'e', 'placement': 'inner'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='ExcToExc.json',
        model_template='static_synapse',
        syn_weight=2.5,
        delay=2.0
    )

    # exc --> inh connections
    visp.add_edges(
        source={'ei': 'e'}, target={'ei': 'i', 'placement': 'inner'},
        connection_rule=n_connections,
        dynamics_params='ExcToInh.json',
        model_template='static_synapse',
        syn_weight=5.0,
        delay=2.0
    )

    # inh --> exc connections
    visp.add_edges(
        source={'ei': 'i'}, target={'ei': 'e', 'placement': 'inner'},
        connection_rule=n_connections,
        dynamics_params='InhToExc.json',
        model_template='static_synapse',
        syn_weight=-6.5,
        delay=2.0
    )

    # inh --> inh connections
    visp.add_edges(
        source={'ei': 'i'}, target={'ei': 'i', 'placement': 'inner'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='InhToInh.json',
        model_template='static_synapse',
        syn_weight=-3.0,
        delay=2.0
    )

    # For connections on point neurons it doesn't make sense to save syanpatic location
    visp.add_edges(
        source={'ei': 'e'}, target={'placement': 'outer'},
        connection_rule=n_connections,
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        syn_weight=3.0,
        delay=2.0
        )

    visp.add_edges(
        source={'ei': 'i'}, target={'placement': 'outer'},
        connection_rule=n_connections,
        dynamics_params='instantaneousInh.json',
        model_template='static_synapse',
        syn_weight=-4.0,
        delay=2.0
    )

    # Build and save internal network
    print('   saving network.')
    visp.build()
    visp.save(output_dir='network')
    print('   done.')

    return visp


def build_lgn(visp):
    # Build a network of 100 virtual cells that will connect to and drive the simulation of the VISp network
    print('Building LGN network')
    lgn = NetworkBuilder('LGN')

    lgn.add_nodes(
        N=100,
        x=np.random.uniform(-40.0, 40.0, size=100),
        y=np.random.uniform(-40.0, 40.0, size=100),
        model_type='virtual',
        ei='e'
    )

    # Targets the core VISp excitatory cells
    lgn.add_edges(
        target=visp.nodes(ei='e', placement='core'), source=lgn.nodes(),
        connection_rule=connect_lgn,
        iterator='one_to_all',
        dynamics_params='ExcToExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=11.0
    )

    # Targets all core VISp inhibitory cells
    lgn.add_edges(
        target=visp.nodes(ei='i', placement='core'), source=lgn.nodes(),
        connection_rule=connect_lgn,
        iterator='one_to_all',
        dynamics_params='ExcToInh.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=14.0
    )

    # Targets all outer ring cells (exc and inh)
    lgn.add_edges(
        target=visp.nodes(placement='outer'), source=lgn.nodes(),
        connection_rule=connect_lgn,
        iterator='one_to_all',
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=13.0
    )

    print('   creating connections.')
    lgn.build()
    print('   saving network.')
    lgn.save(output_dir='network')
    print('   done.')



def build_visl(visp):
    # Build a network of 100 virtual cells that will connect to and drive the simulation of the VISp network
    print('Building VISl network')
    
    visl_units_df = pd.read_csv('units_maps/unit_ids.VISl.valid_units.csv', sep=' ')
    # print(visl_units_df)

    visal = NetworkBuilder('VISl')
    visal.add_nodes(
        N=len(visl_units_df),
        node_id=visl_units_df['node_ids'].values,
        model_type='virtual',
        ei='e'
    )

    # Targets all biophysical excitatory cells
    visal.add_edges(
        target=visp.nodes(ei='e', placement='core'), source=visal.nodes(),
        connection_rule=lambda *_: np.random.uniform(0, 10),
        dynamics_params='ExcToExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=23.0
    )

    # Targets all biophysical inhibitory cells
    visal.add_edges(
        target=visp.nodes(ei='i', placement='core'), source=visal.nodes(),
        connection_rule=lambda *_: np.random.uniform(0, 10),
        dynamics_params='ExcToInh.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=23.0
    )

    visal.add_edges(
        target=visp.nodes(placement='outer'), source=visal.nodes(),
        connection_rule=lambda *_: np.random.uniform(0, 10),
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        delay=1.5,
        syn_weight=15.0
    )

    print('   creating connections.')
    visal.build()
    print('   saving network.')
    visal.save(output_dir='network')
    print('   done.')


def build_hippocampus(visp):
    # Build a network of 100 virtual cells that will connect to and drive the simulation of the VISp network
    print('Building hippocampus network')
    hippo_units_df = pd.read_csv('units_maps/unit_ids.hippocampus.valid_units.csv', sep=' ')
    hippo = NetworkBuilder('hippocampus')
    hippo.add_nodes(
        N=len(hippo_units_df),
        node_id=hippo_units_df['node_ids'].values,
        model_type='virtual',
        ei='e'
    )

    # Targets all biophysical excitatory cells
    hippo.add_edges(
        target=visp.nodes(ei='e', placement='core'), source=hippo.nodes(),
        connection_rule=lambda *_: np.random.uniform(0, 5),
        dynamics_params='ExcToExc.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=3.0
    )

    # Targets all biophysical inhibitory cells
    hippo.add_edges(
        target=visp.nodes(ei='i', placement='core'), source=hippo.nodes(),
        connection_rule=lambda *_: np.random.uniform(0, 5),
        dynamics_params='ExcToInh.json',
        model_template='static_synapse',
        delay=2.0,
        syn_weight=3.0
    )

    hippo.add_edges(
        target=visp.nodes(placement='outer'), source=hippo.nodes(),
        connection_rule=lambda *_: np.random.uniform(2, 6),
        dynamics_params='instantaneousExc.json',
        model_template='static_synapse',
        delay=0.8,
        syn_weight=3.5
    )

    print('   creating connections.')
    hippo.build()
    print('   saving network.')
    hippo.save(output_dir='network')
    print('   done.')


if __name__ == '__main__':
    np.random.seed(100)

    visp = build_visp()
    build_lgn(visp)
    build_visl(visp)
    build_hippocampus(visp)
