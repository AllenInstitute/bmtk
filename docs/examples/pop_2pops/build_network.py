from bmtk.builder import NetworkBuilder

net = NetworkBuilder('brunel')
net.add_nodes(pop_name='excitatory',
              ei='e',
              model_type='population',
              model_template='dipde:Internal',
              dynamics_params='exc_model.json')

net.add_nodes(pop_name='inhibitory',
              ei='i',
              model_type='population',
              model_template='dipde:Internal',
              dynamics_params='inh_model.json')

net.add_edges(source={'ei': 'e'}, target={'ei': 'i'},
              syn_weight=0.0019,
              nsyns=5,
              delay=0.002,
              dynamics_params='ExcToInh.json')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e'},
              syn_weight=-0.0015,
              nsyns=5,
              delay=0.002,
              dynamics_params='InhToExc.json')

net.build()
net.save_nodes(nodes_file_name='brunel_nodes.h5', node_types_file_name='brunel_node_types.csv', output_dir='network')
net.save_edges(edges_file_name='brunel_edges.h5', edge_types_file_name='brunel_edge_types.csv', output_dir='network')


input_net = NetworkBuilder('inputs')
input_net.add_nodes(pop_name='tON',
                    ei='e',
                    model_type='virtual')

input_net.add_edges(target=net.nodes(ei='e'),
                    syn_weight=0.0015,
                    nsyns=5,
                    delay=0.002,
                    dynamics_params='input_ExcToExc.json')

input_net.add_edges(target=net.nodes(ei='i'),
                    syn_weight=0.002,
                    nsyns=5,
                    delay=0.002,
                    dynamics_params='input_ExcToInh.json')

input_net.build()
input_net.save_nodes(nodes_file_name='input_nodes.h5', node_types_file_name='input_node_types.csv',
                     output_dir='network')
input_net.save_edges(edges_file_name='input_edges.h5', edge_types_file_name='input_edge_types.csv',
                     output_dir='network')