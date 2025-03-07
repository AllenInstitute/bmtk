import numpy as np
from bmtk.builder import NetworkBuilder

# nest_model = 'nest:iaf_psc_delta'
nest_model = 'nest:izhikevich'
# nest_model = 'nest:aeif_cond_alpha'
# nest_model = 'nest:glif_psc'
dynamics_params = 'custom_model_params.default.json'


net = NetworkBuilder('net')
net.add_nodes(
    N=100,
    model_type='point_neuron',
    model_template=nest_model,
    dynamics_params=dynamics_params
)

net.add_edges(
    source=net.nodes(), target=net.nodes(),
    connection_rule=1,
    syn_weight=2.0,
    delay=1.5,
    dynamics_params='ExcToInh.json',
    model_template='static_synapse'
)

net.build()
net.save(output_dir='network_built_in')


virt_exc = NetworkBuilder('virt_exc')
virt_exc.add_nodes(
    N=10,
    model_type='virtual'
)

virt_exc.add_edges(
    target=net.nodes(),
    connection_rule=lambda *_: np.random.randint(0, 10),
    syn_weight=2.0,
    delay=1.0,
    dynamics_params='ExcToInh.json',
    model_template='static_synapse'
)

virt_exc.build()
virt_exc.save(output_dir='network_built_in')