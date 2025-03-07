import numpy as np

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.bionet import rand_syn_locations

# L2/3 - 150 - 300
# L4 - 300 - 400
# L5 - 400 - 550
def get_coords(N, radius_min=0.0, radius_max=400.0, y_range=[300, 400]):
    phi = 2.0 * np.pi * np.random.random([N])
    r = np.sqrt((radius_min ** 2 - radius_max ** 2) * np.random.random([N]) + radius_max ** 2)
    x = r * np.cos(phi)
    y = np.random.uniform(y_range[0], y_range[1], size=N)
    z = r * np.sin(phi)
    return x, y, z


# def generate_coords_plane(ncells, size_x=240.0, size_y=120.0):
#     xs = np.random.uniform(0.0, size_x, ncells)
#     ys = np.random.uniform(0.0, size_y, ncells)
#     return xs, ys


# def exc_exc_rule(source, target, max_syns):
#     """Connect rule for exc-->exc neurons, should return an integer 0 or greater"""
#     if source['node_id'] == target['node_id']:
#         # prevent a cell from synapsing with itself
#         return 0

#     # calculate the distance between tuning angles and use it to choose
#     # number of connections using a binomial distribution.
#     src_tuning = source['tuning_angle']
#     trg_tuning = target['tuning_angle']
#     tuning_dist = np.abs((src_tuning - trg_tuning + 180) % 360 - 180)
#     probs = 1.0 - (np.max((tuning_dist, 10.0)) / 180.0)
#     return np.random.binomial(n=max_syns, p=probs)


# def others_conn_rule(source, target, max_syns, max_distance=300.0, sigma=60.0):
#     if source['node_id'] == target['node_id']:
#         return 0

#     dist = np.sqrt((source['x'] - target['x']) ** 2 + (source['z'] - target['z']) ** 2)
#     if dist > max_distance:
#         return 0

#     prob = np.exp(-(dist / sigma) ** 2)
#     return np.random.binomial(n=max_syns, p=prob)


# def connect_lgn_cells(source, targets, max_targets, min_syns=1, max_syns=15, lgn_size=(240, 120),
#                       l4_radius=400.0, ellipse=(100.0, 500.0)):
#     # Map the lgn cells from the plane to the circle
#     x, y = source['x'], source['y']
#     x = x / lgn_size[0] - 0.5
#     y = y / lgn_size[1] - 0.5
#     src_x = x * np.sqrt(1.0 - (y**2/2.0)) * l4_radius
#     src_y = y * np.sqrt(1.0 - (x**2/2.0)) * l4_radius

#     # Find (the indices) of all the target cells that are within the given ellipse, if there are more than max_targets
#     # then randomly choose them
#     a, b = ellipse[0]**2, ellipse[1]**2
#     dists = [(src_x-t['x'])**2/a + (src_y-t['y'])**2/b for t in targets]
#     valid_targets = np.argwhere(np.array(dists) <= 1.0).flatten()
#     if len(valid_targets) > max_targets:
#         valid_targets = np.random.choice(valid_targets, size=max_targets, replace=False)

#     # Create an array of all synapse count. Most targets with have 0 connection, except for the "valid_targets" which
#     # will have between [min_syns, max_syns] number of connections.
#     nsyns_arr = np.zeros(len(targets), dtype=int)
#     for idx in valid_targets:
#         nsyns_arr[idx] = np.random.randint(min_syns, max_syns)

#     return nsyns_arr


cortcol = NetworkBuilder('cortcol')

x, y, z = get_coords(20, y_range=[-300, -100])
cortcol.add_nodes(
    N=20,

    # Reserved SONATA keywords used during simulation
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='Scnn1a_485510712_params.json',
    morphology='Scnn1a_485510712_morphology.swc',

    # The x, y, z locations and orientations (in Euler angles) of each cell
    # Here, rotation around the pia-to-white-matter axis is randomized
    x=x,
    y=y,
    z=z,
    rotation_angle_xaxis=[0]*20,
    rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
    rotation_angle_zaxis=[3.646878266]*20,

    tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
    layer='L23',
    model_name='Scnn1a',
    ei='e'
)


x, y, z = get_coords(20, y_range=[-400, -300])
cortcol.add_nodes(
    N=20,
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='Scnn1a_485510712_params.json',
    morphology='Scnn1a_485510712_morphology.swc',

    # The x, y, z locations and orientations (in Euler angles) of each cell
    # Here, rotation around the pia-to-white-matter axis is randomized
    x=x,
    y=y,
    z=z,
    rotation_angle_xaxis=[0]*20,
    rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
    rotation_angle_zaxis=[3.646878266]*20,

    tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
    layer='L4',
    model_name='Scnn1a',
    ei='e'
)


x, y, z = get_coords(20, y_range=[-600, -400])
cortcol.add_nodes(
    N=20,
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='Scnn1a_485510712_params.json',
    morphology='Scnn1a_485510712_morphology.swc',
    x=x,
    y=y,
    z=z,
    rotation_angle_xaxis=[0]*20,
    rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
    rotation_angle_zaxis=[3.646878266]*20,

    tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
    layer='L5',
    model_name='Scnn1a',
    ei='e'
)

x, y, z = get_coords(20, y_range=[-600, -100])
cortcol.add_nodes(
    N=20,
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_perisomatic',
    dynamics_params='Pvalb_473862421_params.json',
    morphology='Pvalb_473862421_morphology.swc',
    x=x,
    y=y,
    z=z,
    rotation_angle_xaxis=np.random.uniform(0.0, 2*np.pi, size=20),
    rotation_angle_yaxis=np.random.uniform(0.0, 2*np.pi, size=20),
    rotation_angle_zaxis=[2.539551891]*20,
    model_name='Pvalb',
    ei='i'
)

cortcol.add_edges(
    source={'ei': 'e'}, 
    target={'ei': 'e'},
    connection_rule=lambda *_: np.random.randint(10, 15),
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    syn_weight=6.0e-03,
    delay=0.5,
    target_sections=['somatic', 'basal', 'apical'],
    distance_range=[2.0, 1.0e20]
)

cortcol.add_edges(
    # Exc --> Inh connections
    source={'ei': 'e'},
    target={'ei': 'i'},
    connection_rule=lambda *_: np.random.randint(1, 10),
    dynamics_params='AMPA_ExcToInh.json',
    model_template='Exp2Syn',
    syn_weight=0.002,
    delay=3.0,
    target_sections=['somatic', 'basal'],
    distance_range=[0.0, 1.0e+20])

cortcol.add_edges(
    # Inh --> Exc connections
    source={'ei': 'i'},
    target={'ei': 'e'},

    connection_rule=lambda *_: np.random.randint(0, 6),
    dynamics_params='GABA_InhToExc.json',
    model_template='Exp2Syn',
    syn_weight=0.0002,
    delay=2.0,
    target_sections=['somatic', 'basal', 'apical'],
    distance_range=[0.0, 50.0]
)

cortcol.build()
cortcol.save(output_dir='network_lfp')


thalamus = NetworkBuilder('thalamus')
thalamus.add_nodes(
    N=50,
    model_type='virtual',
    model_template='lgnmodel:tOFF_TF15',
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name='tOFF',
    dynamics_params='tOFF_TF15.json'
)
thalamus.add_nodes(
    N=50,
    model_type='virtual',
    model_template='lgnmodel:tON',
    x=np.random.uniform(0.0, 240.0, 50),
    y=np.random.uniform(0.0, 120.0, 50),
    spatial_size=1.0,
    pop_name='tON',
    dynamics_params='tON_TF8.json'
)
thalamus.add_edges(
    source=thalamus.nodes(),
    target=cortcol.nodes(ei='e', layer='L4'), 
    connection_rule=20,
    # iterator='one_to_all',
    dynamics_params='AMPA_ExcToExc.json',
    model_template='Exp2Syn',
    delay=2.0,
    syn_weight=0.00041,
    target_sections=['basal', 'apical', 'somatic'],
    distance_range=[0.0, 50.0]
)
# lgn.add_edges(
#     target=visp.nodes(ei='i'), source=lgn.nodes(),
#     connection_rule=10,
#     dynamics_params='AMPA_ExcToInh.json',
#     model_template='Exp2Syn',
#     delay=2.0,
#     syn_weight=0.00095,
#     target_sections=['basal', 'apical', 'somatic'],
#     distance_range=[0.0, 1e+20]
# )


thalamus.build()
thalamus.save(output_dir='network_lfp')

# def create_l4(network_dir):
#     l4 = NetworkBuilder('l4')

#     x, y, z = get_coords(20)
#     l4.add_nodes(
#         N=20,

#         # Reserved SONATA keywords used during simulation
#         model_type='biophysical',
#         model_template='ctdb:Biophys1.hoc',
#         model_processing='aibs_perisomatic',
#         dynamics_params='Scnn1a_485510712_params.json',
#         morphology='Scnn1a_485510712_morphology.swc',

#         # The x, y, z locations and orientations (in Euler angles) of each cell
#         # Here, rotation around the pia-to-white-matter axis is randomized
#         x=x,
#         y=y,
#         z=z,
#         rotation_angle_xaxis=0,
#         rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
#         rotation_angle_zaxis=3.646878266,

#         # Optional parameters
#         tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
#         model_name='Scnn1a',
#         ei_type='e'
#     )
#     x, y, z = get_coords(20)
#     l4.add_nodes(
#         # Rorb excitatory cells
#         N=20,
#         model_type='biophysical',
#         model_template='ctdb:Biophys1.hoc',
#         dynamics_params='Rorb_486509958_params.json',
#         morphology='Rorb_486509958_morphology.swc',
#         model_processing='aibs_perisomatic',

#         x=x, y=y, z=z,
#         rotation_angle_xaxis=0,
#         rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
#         rotation_angle_zaxis=4.159763785,

#         model_name='Rorb',
#         ei_type='e',
#         tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
#     )

#     x, y, z = get_coords(20)
#     l4.add_nodes(
#         # Nr5a1 excitatory cells
#         N=20,
#         model_type='biophysical',
#         model_template='ctdb:Biophys1.hoc',
#         dynamics_params='Nr5a1_485507735_params.json',
#         morphology='Nr5a1_485507735_morphology.swc',
#         model_processing='aibs_perisomatic',

#         x=x, y=y, z=z,
#         rotation_angle_xaxis=0,
#         rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=20),
#         rotation_angle_zaxis=4.159763785,

#         model_name='Nr5a1',
#         ei_type='e',
#         tuning_angle=np.linspace(start=0.0, stop=360.0, num=20, endpoint=False),
#     )

#     x, y, z = get_coords(15)
#     l4.add_nodes(
#         # Parvalbumin inhibitory cells, note these aren't assigned a tuning angle and ei=i
#         N=15,

#         model_type='biophysical',
#         model_template='ctdb:Biophys1.hoc',
#         dynamics_params='Pvalb_473862421_params.json',
#         morphology='Pvalb_473862421_morphology.swc',
#         model_processing='aibs_perisomatic',

#         x=x, y=y, z=z,
#         rotation_angle_xaxis=0,
#         rotation_angle_yaxis=np.random.uniform(0.0, 2 * np.pi, size=15),
#         rotation_angle_zaxis=2.539551891,

#         model_name='PValb',
#         ei_type='i',
#     )

#     conns = l4.add_edges(
#         # filter for subpopulation or source and target nodes
#         source=l4.nodes(ei_type='e'),
#         target=l4.nodes(ei_type='e'),

#         # connection function + any required parameters
#         connection_rule=exc_exc_rule,
#         connection_params={'max_syns': 15},

#         # edge-type parameters
#         syn_weight=3.0e-05,
#         delay=2.0,
#         dynamics_params='AMPA_ExcToExc.json',
#         model_template='Exp2Syn',
#     )
#     conns.add_properties(
#         ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
#         rule=rand_syn_locations,
#         rule_params={
#             'sections': ['basal', 'apical'],
#             'distance_range': [30.0, 150.0],
#             'morphology_dir': 'components/morphologies'
#         },
#         dtypes=[int, float, int, float]
#     )

#     ## Create e --> i connections
#     conns = l4.add_edges(
#         source=l4.nodes(ei_type='e'),
#         target=l4.nodes(ei_type='i'),
#         connection_rule=others_conn_rule,
#         connection_params={'max_syns': 8},
#         syn_weight=0.0006,
#         delay=2.0,
#         dynamics_params='AMPA_ExcToInh.json',
#         model_template='Exp2Syn',
#     )
#     conns.add_properties(
#         ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
#         rule=rand_syn_locations,
#         rule_params={
#             'sections': ['somatic', 'basal'],
#             'distance_range': [0.0, 1.0e+20],
#             'morphology_dir': 'components/morphologies'
#         },
#         dtypes=[int, float, int, float]
#     )

#     ## Create i --> e connections
#     conns = l4.add_edges(
#         source=l4.nodes(ei_type='i'),
#         target=l4.nodes(ei_type='e'),
#         connection_rule=others_conn_rule,
#         connection_params={'max_syns': 4},
#         syn_weight=0.07,
#         delay=2.0,
#         dynamics_params='GABA_InhToExc.json',
#         model_template='Exp2Syn',
#     )
#     conns.add_properties(
#         ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
#         rule=rand_syn_locations,
#         rule_params={
#             'sections': ['somatic', 'basal', 'apical'],
#             'distance_range': [0.0, 50.0],
#             'morphology_dir': 'components/morphologies'
#         },
#         dtypes=[int, float, int, float]
#     )

#     ## Create i --> i connections
#     conns = l4.add_edges(
#         source=l4.nodes(ei_type='i'),
#         target=l4.nodes(ei_type='i'),
#         connection_rule=others_conn_rule,
#         connection_params={'max_syns': 4},
#         syn_weight=0.00015,
#         delay=2.0,
#         dynamics_params='GABA_InhToInh.json',
#         model_template='Exp2Syn',
#     )
#     conns.add_properties(
#         ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
#         rule=rand_syn_locations,
#         rule_params={
#             'sections': ['somatic', 'basal', 'apical'],
#             'distance_range': [0.0, 1.0e+20],
#             'morphology_dir': 'components/morphologies'
#         },
#         dtypes=[int, float, int, float]
#     )

#     l4.build()
#     l4.save(output_dir='network')

#     return l4


# def create_lgn(l4):
#     lgn = NetworkBuilder('lgn')

#     x, y = generate_coords_plane(100)
#     lgn.add_nodes(
#         N=100,
#         x=x,
#         y=y,
#         model_type='virtual',
#         model_template='lgnmodel:tON',
#         dynamics_params='tON_TF15.json',
#         ei_type='e'
#     )

#     conns = lgn.add_edges(
#         source=lgn.nodes(),
#         target=l4.nodes(ei_type='e'),
#         connection_rule=connect_lgn_cells,
#         connection_params={'max_targets': 6},
#         iterator='one_to_all',
#         model_template='Exp2Syn',
#         dynamics_params='AMPA_ExcToExc.json',
#         delay=2.0,
#         syn_weight=0.0019
#     )
#     conns.add_properties(
#         ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
#         rule=rand_syn_locations,
#         rule_params={
#             'sections': ['somatic', 'basal', 'apical'],
#             'distance_range': [0.0, 1.0e+20],
#             'morphology_dir': 'components/morphologies'
#         },
#         dtypes=[int, float, int, float]
#     )

#     conns = lgn.add_edges(
#         source=lgn.nodes(),
#         target=l4.nodes(ei_type='i'),
#         connection_rule=connect_lgn_cells,
#         connection_params={'max_targets': 12, 'ellipse': (400.0, 400.0)},
#         iterator='one_to_all',
#         model_template='Exp2Syn',
#         dynamics_params='AMPA_ExcToInh.json',
#         delay=2.0,
#         syn_weight=0.003
#     )
#     conns.add_properties(
#         ['afferent_section_id', 'afferent_section_pos', 'afferent_swc_id', 'afferent_swc_pos'],
#         rule=rand_syn_locations,
#         rule_params={
#             'sections': ['somatic', 'basal', 'apical'],
#             'distance_range': [0.0, 1.0e+20],
#             'morphology_dir': 'components/morphologies'
#         },
#         dtypes=[int, float, int, float]
#     )

#     lgn.build()
#     lgn.save(output_dir='network')


# def create_lifs(l4):
#     lif = NetworkBuilder('lif')

#     # place neurons on outer ring
#     x, y, z = get_coords(80, radius_min=400.0, radius_max=800.0)
#     lif.add_nodes(
#         N=80,
#         x=x, y=y, z=z,
#         model_type='point_neuron',
#         model_template='nrn:IntFire1',
#         dynamics_params='IntFire1_exc_1.json',
#         model_name='LIF_exc',
#         ei_type='e',
#     )

#     x, y, z = get_coords(20, radius_min=400.0, radius_max=800.0)
#     lif.add_nodes(
#         N=20,
#         x=x, y=y, z=z,
#         model_type='point_neuron',
#         model_template='nrn:IntFire1',
#         dynamics_params='IntFire1_inh_1.json',
#         model_name='LIF_inh',
#         ei_type='i',
#     )

#     # Connect LIFs --> L4
#     lif.add_edges(
#         source=lif.nodes(ei_type='e'), target=l4.nodes(ei_type='e'),
#         connection_rule=8,
#         syn_weight=0.015,
#         delay=2.0,
#         distance_range=[30.0, 150.0],
#         target_sections=['somatic', 'basal', 'apical'],
#         dynamics_params='AMPA_ExcToExc.json',
#         model_template='Exp2Syn'
#     )

#     lif.add_edges(
#         source=lif.nodes(ei_type='e'), target=l4.nodes(ei_type='i'),
#         connection_rule=5,
#         syn_weight=0.015,
#         delay=2.0,
#         distance_range=[30.0, 150.0],
#         target_sections=['somatic', 'basal'],
#         dynamics_params='AMPA_ExcToInh.json',
#         model_template='Exp2Syn'
#     )

#     lif.add_edges(
#         source=lif.nodes(ei_type='i'), target=l4.nodes(ei_type='e'),
#         connection_rule=5,
#         syn_weight=0.095,
#         delay=2.0,
#         distance_range=[0.0, 50.0],
#         target_sections=['somatic', 'basal', 'apical'],
#         dynamics_params='GABA_InhToExc.json',
#         model_template='Exp2Syn'
#     )

#     lif.add_edges(
#         source=lif.nodes(ei_type='i'), target=l4.nodes(ei_type='i'),
#         connection_rule=5,
#         syn_weight=0.095,
#         delay=2.0,
#         distance_range=[0.0, 1e+20],
#         target_sections=['somatic', 'basal'],
#         dynamics_params='GABA_InhToInh.json',
#         model_template='Exp2Syn'
#     )

#     # Connect L4 --> LIFs
#     lif.add_edges(
#         source=l4.nodes(ei_type='e'), target=lif.nodes(),
#         connection_rule=lambda *_: np.random.randint(0, 12),
#         syn_weight=0.015,
#         delay=2.0,
#         dynamics_params='instantaneousExc.json',
#         model_template='Exp2Syn'
#     )

#     lif.add_edges(
#         source=l4.nodes(ei_type='i'), target=lif.nodes(),
#         connection_rule=lambda *_: np.random.randint(0, 12),
#         syn_weight=0.05,
#         delay=2.0,
#         dynamics_params='instantaneousInh.json',
#         model_template='Exp2Syn'
#     )

#     # Connect LIFs --> LIFs
#     lif.add_edges(
#         source=lif.nodes(ei_type='e'), target=lif.nodes(),
#         connection_rule=lambda *_: np.random.randint(0, 12),
#         syn_weight=0.005,
#         delay=2.0,
#         dynamics_params='instantaneousExc.json',
#         model_template='Exp2Syn'
#     )

#     lif.add_edges(
#         source=lif.nodes(ei_type='i'), target=lif.nodes(),
#         connection_rule=lambda *_: np.random.randint(0, 12),
#         syn_weight=0.020,
#         delay=2.0,
#         dynamics_params='instantaneousInh.json',
#         model_template='Exp2Syn'
#     )


# if __name__ == '__main__':
#     print('Building l4')
#     l4 = create_l4('network')

#     print('Building lgn')
#     create_lgn(l4)

#     print('Building lifs')
#     create_lifs(l4)