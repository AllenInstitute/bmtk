import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.bionet.swc_reader import get_swc
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random

build_recurrent_edges = True

# List of non-virtual cell models
bio_models = [
    {
        'model_name': 'Scnn1a',
        'ei': 'e',
        'morphology': 'Scnn1a_473845048_m.swc',
        'model_template': 'nml:Cell_472363762.cell.nml'
    },
    {
        'model_name': 'Rorb',
        'ei': 'e',
        'morphology': 'Rorb_325404214_m.swc',
        'model_template': 'nml:Cell_473863510.cell.nml'
    },
    {
        'model_name': 'Nr5a1',
        'ei': 'e',
        'morphology': 'Nr5a1_471087815_m.swc',
        'model_template': 'nml:Cell_473863035.cell.nml'
    },
    {
        'model_name': 'PV1',
        'ei': 'i',
        'morphology': 'Pvalb_470522102_m.swc',
        'model_template': 'nml:Cell_472912177.cell.nml'
    },
    {
        'model_name': 'PV2',
        'ei': 'i',
        'morphology': 'Pvalb_469628681_m.swc',
        'model_template': 'nml:Cell_473862421.cell.nml'
    }
]

point_models = [
    {
        'model_name': 'LIF_exc',
        'ei': 'e',
        'dynamics_params': 'IntFire1_exc_1.json'
    },
    {
        'model_name': 'LIF_inh',
        'ei': 'i',
        'dynamics_params': 'IntFire1_inh_1.json'
    }
]


def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=5):
    if src.node_id == trg.node_id:
        return 0

    dist = np.sqrt((src['x']-trg['x'])**2 + (src['y']-trg['y'])**2 + (src['z']-trg['z'])**2)
    if dist > 100.0:
        return 0

    return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)


def connect_external(src, trgs, dist_cutoff=30.0, max_trgs=10, max_syns=12):
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

    n_syns = np.zeros(len(trgs), dtype=np.int)
    n_syns[selected_trgs] = np.random.randint(0, max_syns, size=len(selected_trgs))
    return n_syns


def set_synapses(src, trg, section_names=('soma', 'apical', 'basal'), distance_range=(0.0, 1.0e20)):
    trg_swc = get_swc(trg, morphology_dir='../bio_components/morphologies/', use_cache=True)

    sec_ids, seg_xs = trg_swc.choose_sections(section_names, distance_range, n_sections=1)
    sec_id, seg_x = sec_ids[0], seg_xs[0]
    swc_id, swc_dist = trg_swc.get_swc_id(sec_id, seg_x)
    # coords = trg_swc.get_coords(sec_id, seg_x)

    return [sec_id, seg_x, swc_id, swc_dist]  # coords[0], coords[y], coords[z]


def build_internal_network():
    print('Creating network internal')
    internal = NetworkBuilder('internal')
    for i, model_props in enumerate(bio_models):
        n_cells = 80 if model_props['ei'] == 'e' else 30  # 80% excitatory, 20% inhib

        # Randomly get positions uniformly distributed in a column
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)

        internal.add_nodes(
            N=n_cells,
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            rotation_angle_yaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  # randomly rotate y axis
            rotation_angle_zaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  #
            model_type='biophysical',
            model_processing='aibs_perisomatic',
            **model_props
        )

    for model_props in point_models:
        n_cells = 75  # Just assume 75 cells for both point inhibitory and point excitatory
        positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)
        internal.add_nodes(
            N=n_cells,
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
            model_type='point_neuron',
            model_template='nrn:IntFire1',
            **model_props
        )

    # Connections onto biophysical components, use the connection map to save section and position of every synapse
    # exc --> exc connections
    cm = internal.add_edges(
        source={'ei': 'e'}, target={'ei': 'e', 'model_type': 'biophysical'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='AMPA_ExcToExc.json',
        model_template='Exp2Syn',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=6.0e-05, dtypes=np.float)
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        rule_params={'section_names': ['basal', 'apical'], 'distance_range': [30.0, 150.0]},
        dtypes=[np.int, np.float, np.int, np.float]
    )

    # exc --> inh connections
    cm = internal.add_edges(
        source={'ei': 'e'}, target={'ei': 'i', 'model_type': 'biophysical'},
        connection_rule=n_connections,
        dynamics_params='AMPA_ExcToInh.json',
        model_template='Exp2Syn',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.0006, dtypes=np.float)
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        rule_params={'section_names': ['somatic', 'basal'], 'distance_range': [0.0, 1.0e+20]},
        dtypes=[np.int, np.float, np.int, np.float]
    )

    # inh --> exc connections
    cm = internal.add_edges(
        source={'ei': 'i'}, target={'ei': 'e', 'model_type': 'biophysical'},
        connection_rule=n_connections,
        dynamics_params='GABA_InhToExc.json',
        model_template='Exp2Syn',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.002, dtypes=np.float)
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        rule_params={'section_names': ['somatic', 'basal', 'apical'], 'distance_range': [0.0, 50.0]},
        dtypes=[np.int, np.float, np.int, np.float]
    )

    # inh --> inh connections
    cm = internal.add_edges(
        source={'ei': 'i'}, target={'ei': 'i', 'model_type': 'biophysical'},
        connection_rule=n_connections,
        connection_params={'prob': 0.2},
        dynamics_params='GABA_InhToInh.json',
        model_template='Exp2Syn',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.00015, dtypes=np.float)
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        rule_params={'section_names': ['somatic', 'basal'], 'distance_range': [0.0, 1.0e+20]},
        dtypes=[np.int, np.float, np.int, np.float]
    )

    # For connections on point neurons it doesn't make sense to save syanpatic location
    cm = internal.add_edges(
        source={'ei': 'e'}, target={'model_type': 'point_neuron'},
        connection_rule=n_connections,
        dynamics_params='instantaneousExc.json',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.0019, dtypes=np.float)

    cm = internal.add_edges(
        source={'ei': 'i'}, target={'model_type': 'point_neuron'},
        connection_rule=n_connections,
        dynamics_params='instantaneousInh.json',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.0019, dtypes=np.float)

    print('   creating connections.')
    internal.build()
    print('   saving network.')
    internal.save(output_dir='network')
    print('   done.')

    return internal


def build_external_network(internal):
    print('Creating external (virtual) connections')
    external = NetworkBuilder("external")
    external.add_nodes(
        N=100,
        x=np.random.uniform(-40.0, 40.0, size=100),
        y=np.random.uniform(-40.0, 40.0, size=100),
        model_type='virtual',
        ei='e'
    )

    cm = external.add_edges(
        target=internal.nodes(ei='e', model_type='biophysical'), source=external.nodes(),
        connection_rule=connect_external,
        iterator='one_to_all',
        dynamics_params='AMPA_ExcToExc.json',
        model_template='Exp2Syn',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.00041, dtypes=np.float)
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        dtypes=[np.int, np.float, np.int, np.float]
    )

    cm = external.add_edges(
        target=internal.nodes(ei='i', model_type='biophysical'), source=external.nodes(),
        connection_rule=connect_external,
        iterator='one_to_all',
        dynamics_params='AMPA_ExcToInh.json',
        model_template='Exp2Syn',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.00095, dtypes=np.float)
    cm.add_properties(
        ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
        rule=set_synapses,
        dtypes=[np.int, np.float, np.int, np.float]
    )

    cm = external.add_edges(
        target=internal.nodes(model_type='point_neuron'), source=external.nodes(),
        connection_rule=connect_external,
        iterator='one_to_all',
        dynamics_params='instantaneousExc.json',
        delay=2.0
    )
    cm.add_properties('syn_weight', rule=0.045, dtypes=np.float)

    print('   creating connections.')
    external.build()
    print('   saving network.')
    external.save(output_dir='network')
    print('   done.')

    return external


if __name__ == '__main__':
    internal_net = build_internal_network()
    external_net = build_external_network(internal_net)
