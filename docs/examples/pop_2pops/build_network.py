from bmtk.builder import NetworkBuilder

# Create internal/recurrent populations of nodes
internal_net = NetworkBuilder('internal')
internal_net.add_nodes(
    pop_name='excitatory',
    ei='e',
    model_type='population',
    model_template='dipde:Internal',
    dynamics_params='exc_model.json'
)

internal_net.add_nodes(
    pop_name='inhibitory',
    ei='i',
    model_type='population',
    model_template='dipde:Internal',
    dynamics_params='inh_model.json'
)

# Recurrently connect internal populations
internal_net.add_edges(
    source={'ei': 'e'}, target={'ei': 'i'},
    syn_weight=0.005,
    nsyns=20,
    delay=0.002,
    dynamics_params='ExcToInh.json'
)

internal_net.add_edges(
    source={'ei': 'i'}, target={'ei': 'e'},
    syn_weight=-0.002,
    nsyns=10,
    delay=0.002,
    dynamics_params='InhToExc.json'
)

internal_net.build()
internal_net.save(output_dir='network')


# Create external pops
external_net = NetworkBuilder('external')
external_net.add_nodes(
    pop_name='tON',
    ei='e',
    model_type='virtual'
)

# Connect external pop to internal pops
external_net.add_edges(
    target=internal_net.nodes(ei='e'),
    syn_weight=0.0025,
    nsyns=10,
    delay=0.002,
    dynamics_params='input_ExcToExc.json'
)

external_net.build()
external_net.save(output_dir='network')
