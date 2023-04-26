from bmtk.builder import NetworkBuilder

"""Create Nodes"""
net = NetworkBuilder('V1')
net.add_nodes(
    pop_name='excitatory',  # name of specific population optional
    ei='e',  # Optional
    location='VisL4',  # Optional
    model_type='population',  # Required, indicates what types of cells are being model
    model_template='dipde:Internal',  # Required, instructs what DiPDE objects will be created
    dynamics_params='exc_model.json'  # Required, contains parameters used by DiPDE during initialization of object
)

net.add_nodes(
    pop_name='inhibitory',
    ei='i',
    model_type='population',
    model_template='dipde:Internal',
    dynamics_params='inh_model.json'
)

"""Create edges"""
net.add_edges(
    source={'ei': 'e'}, target={'ei': 'i'},
    syn_weight=0.005,
    nsyns=20,
    delay=0.002,
    dynamics_params='ExcToInh.json'
)

net.add_edges(
    source={'ei': 'i'}, target={'ei': 'e'},
    syn_weight=-0.002,
    nsyns=10,
    delay=0.002,
    dynamics_params='InhToExc.json'
)

net.build()
net.save(output_dir='network')


input_net = NetworkBuilder('LGN')
input_net.add_nodes(
    pop_name='tON',
    ei='e',
    model_type='virtual'
)

input_net.add_edges(
    target=net.nodes(ei='e'),
    syn_weight=0.0025,
    nsyns=10,
    delay=0.002,
    dynamics_params='input_ExcToExc.json'
)

input_net.build()
input_net.save(output_dir='network')
