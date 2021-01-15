import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random


# List of non-virtual cell models
bio_models = [
    # {
    #     'pop_name': 'Scnn1a', 'ei': 'e',
    #     'morphology': 'Scnn1a_473845048_m.swc',
    #     'model_template': 'ctdb:Biophys1.hoc',
    #     'dynamics_params': '472363762_fit.json'
    # },
    {
        'pop_name': 'Rorb', 'ei': 'e',
        'morphology': 'Rorb_325404214_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863510_fit.json'
    },
    {
        'pop_name': 'Nr5a1', 'ei': 'e',
        'morphology': 'Nr5a1_471087815_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473863035_fit.json'
    },
    {
        'pop_name': 'PV', 'ei': 'i',
        'morphology': 'Pvalb_469628681_m.swc',
        'model_template': 'ctdb:Biophys1.hoc',
        'dynamics_params': '473862421_fit.json'
    }
]


# Build a network of 300 biophysical cells to simulate
print('Build internal "V1" network')
v1 = NetworkBuilder("V1")

# for i, model_props in enumerate(bio_models):
#     n_cells = 80 if model_props['ei'] == 'e' else 60  # 80% excitatory, 20% inhib
#
#     # Randomly get positions uniformly distributed in a column
#     positions = positions_columinar(N=n_cells, center=[0, 10.0, 0], max_radius=50.0, height=200.0)
#
#     v1.add_nodes(N=n_cells,
#                        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
#                        rotation_angle_yaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  # randomly rotate y axis
#                        rotation_angle_zaxis=xiter_random(N=n_cells, min_x=0.0, max_x=2 * np.pi),  #
#                        model_type='biophysical',
#                        model_processing='aibs_perisomatic',
#                        **model_props)

v1.add_nodes(
    N=80,

    # Reserved SONATA keywords used during simulation
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='472363762_fit.json',
    morphology='Scnn1a_473845048_m.swc',
    model_processing='aibs_perisomatic',

    # The x, y, z locations of each cell in a column
    x=np.random.normal(0.0, 20.0, size=80),
    y=np.random.uniform(400.0, 500.0, size=80),
    z=np.random.normal(0.0, 20.0, size=80),

    # Euler rotations of the cells
    rotation_angle_xaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
    rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=80),
    rotation_angle_zaxis=-3.646878266,

    # Optional parameters
    tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
    pop_name='Scnn1a',
    location='L4',
    ei='e',
)

v1.add_nodes(
    # Rorb excitatory cells
    N=80, pop_name='Rorb', location='L4', ei='e',
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='473863510_fit.json',
    morphology='Rorb_325404214_m.swc',
    model_processing='aibs_perisomatic',
    x=np.random.normal(0.0, 20.0, size=80),
    y=np.random.uniform(400.0, 500.0, size=80),
    z=np.random.normal(0.0, 20.0, size=80),
    rotation_angle_xaxis=np.random.uniform(0.0, 2*np.pi, size=80),
    rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=80),
    rotation_angle_zaxis=-4.159763785,
    tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
)


v1.add_nodes(
    # Nr5a1 excitatory cells
    N=80, pop_name='Nr5a1', location='L4', ei='e',
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='473863035_fit.json',
    morphology='Nr5a1_471087815_m.swc',
    model_processing='aibs_perisomatic',
    x=np.random.normal(0.0, 20.0, size=80),
    y=np.random.uniform(400.0, 500.0, size=80),
    z=np.random.normal(0.0, 20.0, size=80),
    rotation_angle_xaxis=np.random.uniform(0.0, 2*np.pi, size=80),
    rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=80),
    rotation_angle_zaxis=-4.159763785,
    tuning_angle=np.linspace(start=0.0, stop=360.0, num=80, endpoint=False),
)

v1.add_nodes(
    # Parvalbuim inhibitory cells, note these don't have a tuning angle and ei=i
    N=60, pop_name='PV1', location='L4', ei='i',
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    dynamics_params='473862421_fit.json',
    morphology='Pvalb_469628681_m.swc',
    model_processing='aibs_perisomatic',
    x=np.random.normal(0.0, 20.0, size=60),
    y=np.random.uniform(400.0, 500.0, size=60),
    z=np.random.normal(0.0, 20.0, size=60),
    rotation_angle_xaxis=np.random.uniform(0.0, 2*np.pi, size=60),
    rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=60),
    rotation_angle_zaxis=-2.539551891
)


def n_connections(src, trg, prob=0.1, min_syns=1, max_syns=5):
    """Referenced by add_edges() and called by build() for every source/target pair. For every given target/source
    pair will connect the two with a probability prob (excludes self-connections)"""
    if src.node_id == trg.node_id:
        return 0

    return 0 if np.random.uniform() > prob else np.random.randint(min_syns, max_syns)

v1.add_edges(
    # Exc --> Inh connections
    source={'ei': 'e'},
    target={'ei': 'i'},

    connection_rule=1, # n_connections,
    dynamics_params='ExcToInh.json',
    model_template='Exp2Syn',
    syn_weight=0.0006,
    delay=2.0,
    target_sections=['somatic', 'basal'],
    distance_range=[0.0, 1.0e+20])

v1.add_edges(
    # Inh --> Exc connections
    source={'ei': 'i'},
    target={'ei': 'e'},

    connection_rule=1, # n_connections,
    dynamics_params='InhToExc.json',
    model_template='Exp2Syn',
    syn_weight=0.0002,
    delay=2.0,
    target_sections=['somatic', 'basal', 'apical'],
    distance_range=[0.0, 50.0]
)


def ignore_autopases(source, target):
    # No synapses if source == target, otherwise randomize
    if source['node_id'] == target['node_id']:
        return 0
    else:
        return np.random.randint(1, 5)


v1.add_edges(
    # Inh --> Inh connections
    source={'ei': 'i'},
    target={'ei': 'i'},

    # connection_rule=n_connections,
    # connection_params={'prob': 0.2},
    # syn_weight=0.00015,
    connection_rule=ignore_autopases,
    syn_weight=0.00015,
    delay=2.0,
    dynamics_params='InhToInh.json',
    model_template='Exp2Syn',
    target_sections=['somatic', 'basal'],
    distance_range=[0.0, 1.0e+20]
)


def tunning_angle(source, target, max_syns):
    # ignore autoapses
    if source['node_id'] == target['node_id']:
        return 0

    # num of synapses is higher the closer the tuning_angles
    src_tuning = source['tuning_angle']
    trg_tuning = target['tuning_angle']
    dist = np.abs((src_tuning - trg_tuning + 180) % 360 - 180)
    p_dist = 1.0 - (np.max((dist, 10.0)) / 180.0)
    return np.random.binomial(n=max_syns, p=p_dist)


v1.add_edges(
    # Exc --> Exc connections
    source={'ei': 'e'},
    target={'ei': 'e'},

    # connection_rule=n_connections,
    # connection_params={'prob': 0.2},
    # syn_weight=6.0e-05,
    connection_rule=tunning_angle,
    connection_params={'max_syns': 5},
    syn_weight=3.0e-05,
    delay=2.0,
    dynamics_params='ExcToExc.json',
    model_template='Exp2Syn',
    target_sections=['basal', 'apical'],
    distance_range=[30.0, 150.0],
    weight_function='set_syn_weight'
)


# Build and save internal network
v1.build()
print('Saving V1')
v1.save(output_dir='network')


# Build a network of 100 virtual cells that will connect to and drive the simulation of the internal network
print('Building external "LGN" connections')
lgn = NetworkBuilder("LGN")

lgn.add_nodes(
    N=50,
    model_type='virtual',
    ei='e',
    model_template='lgnmodel:sOFF_TF8',
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    dynamics_params='sOFF_TF8.json'
)

lgn.add_nodes(
    N=50,
    model_type='virtual',
    model_template='lgnmodel:sON_TF8',
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    dynamics_params='sON_TF8.json'
)

lgn.add_edges(
    # Targets all V1 excitatory cells
    source=lgn.nodes(),
    target=v1.nodes(ei='e'),

    connection_rule=lambda *_: np.random.randint(0, 5),
    dynamics_params='LGN_ExcToExc.json',
    model_template='Exp2Syn',
    delay=2.0,
    syn_weight=0.0003,
    target_sections=['basal', 'apical', 'somatic'],
    distance_range=[0.0, 50.0]
)

lgn.add_edges(
    # Targets all biophysical inhibitory cells
    source=lgn.nodes(),
    target=v1.nodes(ei='i'),

    connection_rule=lambda *_: np.random.randint(0, 5),
    dynamics_params='LGN_ExcToInh.json',
    model_template='Exp2Syn',
    delay=2.0,
    syn_weight=0.002,
    target_sections=['basal', 'apical'],
    distance_range=[0.0, 1e+20]
)

lgn.build()
print('Saving LGN')
lgn.save(output_dir='network')
