import numpy as np

from bmtk.builder.networks import NetworkBuilder
from bmtk.builder.auxi.edge_connectors import connect_random

#print np.random.geometric(p=0.5)
#print np.sqrt(1-0.005)/0.005

"""
def connect_random(source, target, nsyn_min=0, nsyn_max=10, distribution=None):
    return np.random.randint(nsyn_min, nsyn_max)
"""


thalamus = NetworkBuilder('mthalamus')
thalamus.add_nodes(N=100,
                   pop_name='tON',
                   potential='exc',
                   level_of_detail='filter')

cortex = NetworkBuilder('mcortex')
cortex.import_nodes(nodes_file_name='network/mcortex_nodes.h5', node_types_file_name='network/mcortex_node_types.csv')
thalamus.add_edges(source=thalamus.nodes(), target=cortex.nodes(),
                   connection_rule=connect_random,
                   connection_params={'nsyn_min': 0, 'nsyn_max': 12},
                   syn_weight=1.0e-04,
                   distance_range=[0.0, 150.0],
                   target_sections=['basal', 'apical'],
                   delay=2.0,
                   dynamics_params='AMPA_ExcToExc.json',
                   model_template='exp2syn')

thalamus.build()
thalamus.save_nodes(output_dir='network')
thalamus.save_edges(output_dir='network')

