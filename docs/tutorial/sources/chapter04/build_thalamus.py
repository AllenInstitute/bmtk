import numpy as np

from bmtk.builder.networks import NetworkBuilder

lgn = NetworkBuilder('LGN')
lgn.add_nodes(N=500,
              pop_name='tON',
              potential='exc',
              level_of_detail='filter')

v1 = NetworkBuilder('V1')
v1.import_nodes(nodes_file_name='network/V1_nodes.h5', node_types_file_name='network/V1_node_types.csv')


def select_source_cells(sources, target, nsources_min=10, nsources_max=30, nsyns_min=3, nsyns_max=12):
    total_sources = len(sources)
    nsources = np.random.randint(nsources_min, nsources_max)
    selected_sources = np.random.choice(total_sources, nsources, replace=False)
    syns = np.zeros(total_sources)
    syns[selected_sources] = np.random.randint(nsyns_min, nsyns_max, size=nsources)
    return syns
lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='Scnn1a'),
              iterator='all_to_one',
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 10, 'nsources_max': 25},
              syn_weight=4e-03,
              weight_function='wmax',
              distance_range=[0.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              dynamics_params='AMPA_ExcToExc.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='PV1'),
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 15, 'nsources_max': 30},
              iterator='all_to_one',
              syn_weight=0.001,
              weight_function='wmax',
              distance_range=[0.0, 1.0e+20],
              target_sections=['somatic', 'basal'],
              delay=2.0,
              dynamics_params='AMPA_ExcToInh.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(),  target=net.nodes(pop_name='LIF_exc'),
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 10, 'nsources_max': 25},
              iterator='all_to_one',
              syn_weight= 0.045,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='exp2syn')

lgn.add_edges(source=lgn.nodes(),  target=net.nodes(pop_name='LIF_inh'),
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 15, 'nsources_max': 30},
              iterator='all_to_one',
              syn_weight=0.02,
              weight_function='wmax',
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='exp2syn')


lgn.build()
lgn.save_nodes(output_dir='network')
lgn.save_edges(output_dir='network')