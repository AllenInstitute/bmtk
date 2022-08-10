import numpy as np

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.auxi.node_params import positions_columinar
from bmtk.builder.auxi.edge_connectors import distance_connector

"""Create Nodes"""
net = NetworkBuilder("V1")
net.add_nodes(N=80,  # Create a population of 80 neurons
              positions=positions_columinar(N=80, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
              pop_name='Scnn1a', location='VisL4', ei='e',  # optional parameters
              model_type='point_process',  # Tells the simulator to use point-based neurons
              model_template='nest:iaf_psc_alpha',  # tells the simulator to use NEST iaf_psc_alpha models
              dynamics_params='472363762_point.json'  # File containing iaf_psc_alpha mdoel parameters
             )

net.add_nodes(N=20, pop_name='PV', location='VisL4', ei='i',
              positions=positions_columinar(N=20, center=[0, 50.0, 0], max_radius=30.0, height=100.0),
              model_type='point_process',
              model_template='nest:iaf_psc_alpha',
              dynamics_params='472912177_point.json')

net.add_nodes(N=200, pop_name='LIF_exc', location='L4', ei='e',
              positions=positions_columinar(N=200, center=[0, 50.0, 0], min_radius=30.0, max_radius=60.0, height=100.0),
              model_type='point_process',
              model_template='nest:iaf_psc_alpha',
              dynamics_params='IntFire1_exc_point.json')

net.add_nodes(N=100, pop_name='LIF_inh', location='L4', ei='i',
              positions=positions_columinar(N=100, center=[0, 50.0, 0], min_radius=30.0, max_radius=60.0, height=100.0),
              model_type='point_process',
              model_template='nest:iaf_psc_alpha',
              dynamics_params='IntFire1_inh_point.json')



"""Create edges"""
## E-to-E connections
net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=5.0,
              delay=2.0,
              dynamics_params='ExcToExc.json',
              model_template='static_synapse')

net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.34, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=10.0,
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='static_synapse')


### Generating I-to-I connections
net.add_edges(source={'ei': 'i'}, target={'pop_name': 'PV'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=-1.0,
              delay=2.0,
              dynamics_params='InhToInh.json',
              model_template='static_synapse')

net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'pop_name': 'LIF_inh'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=-1.0,
              delay=2.0,
              dynamics_params='instantaneousInh.json',
              model_template='static_synapse')

### Generating I-to-E connections
net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'pop_name': 'Scnn1a'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=-15.0,
              delay=2.0,
              dynamics_params='InhToExc.json',
              model_template='static_synapse')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'pop_name': 'LIF_exc'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 1.0, 'd_max': 160.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=-15.0,
              delay=2.0,
              dynamics_params='instantaneousInh.json',
              model_template='static_synapse')

### Generating E-to-I connections
net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.26, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=15.0,
              delay=2.0,
              dynamics_params='ExcToInh.json',
              model_template='static_synapse')


net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
              connection_rule=distance_connector,
              connection_params={'d_weight_min': 0.0, 'd_weight_max': 0.26, 'd_max': 300.0, 'nsyn_min': 3, 'nsyn_max': 7},
              syn_weight=5.0,
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='static_synapse')

net.build()
net.save_nodes(output_dir='network')
net.save_edges(output_dir='network')



lgn = NetworkBuilder('LGN')
lgn.add_nodes(N=500,
              pop_name='tON',
              potential='exc',
              model_type='virtual')


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
              syn_weight=20.0,
              delay=2.0,
              dynamics_params='ExcToExc.json',
              model_template='static_synapse')

lgn.add_edges(source=lgn.nodes(), target=net.nodes(pop_name='PV1'),
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 15, 'nsources_max': 30},
              iterator='all_to_one',
              syn_weight=20.0,
              delay=2.0,
              dynamics_params='ExcToInh.json',
              model_template='static_synapse')

lgn.add_edges(source=lgn.nodes(),  target=net.nodes(pop_name='LIF_exc'),
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 10, 'nsources_max': 25},
              iterator='all_to_one',
              syn_weight=10.0,
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='static_synapse')

lgn.add_edges(source=lgn.nodes(),  target=net.nodes(pop_name='LIF_inh'),
              connection_rule=select_source_cells,
              connection_params={'nsources_min': 15, 'nsources_max': 30},
              iterator='all_to_one',
              syn_weight=10.0,
              delay=2.0,
              dynamics_params='instantaneousExc.json',
              model_template='static_synapse')


lgn.build()
lgn.save_nodes(output_dir='network')
lgn.save_edges(output_dir='network')