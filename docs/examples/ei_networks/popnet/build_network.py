from bmtk.builder import NetworkBuilder

delay = 0.0015
g = 3

# scaling factor
sc = 1e-3
#
# e to i and i to e connections
je = 0.1
ji = -g*je

# nsyns 
NE = 1e4
NI = NE/4
Next = NE

net = NetworkBuilder('internal')
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
              syn_weight=sc*je,
              nsyns=NE,
              delay=delay,
              dynamics_params='ExcToInh.json')

net.add_edges(source={'ei': 'i'}, target={'ei': 'e'},
              syn_weight=sc*ji,
              nsyns=NI,
              delay=delay,
              dynamics_params='InhToExc.json')

net.add_edges(source={'ei': 'e'}, target={'ei': 'e'},
              syn_weight= sc*je,
              nsyns=NE,
              delay=delay,
              dynamics_params='ExcToExc.json')

net.add_edges(source={'ei': 'i'}, target={'ei': 'i'},
              syn_weight= sc*ji,
              nsyns=NI,
              delay=delay,
              dynamics_params='InhToInh.json')

net.build()
net.save(output_dir='network')

input_net = NetworkBuilder('external')
input_net.add_nodes(pop_name='tON',
                    ei='e',
                    model_type='virtual')

input_net.add_edges(target=net.nodes(ei='e'),
                    syn_weight=sc*je,
                    nsyns=Next,
                    delay=delay,
                    dynamics_params='input_ExcToExc.json')

input_net.add_edges(target=net.nodes(ei='i'),
                    syn_weight=sc*.13,
                    nsyns=Next,
                    delay=delay,
                    dynamics_params='input_ExcToInh.json')

input_net.build()
input_net.save(output_dir='network')
