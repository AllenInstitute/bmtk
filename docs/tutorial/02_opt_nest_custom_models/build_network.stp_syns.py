import numpy as np
from bmtk.builder import NetworkBuilder

nest_model = 'nest:izhikevich'
dynamics_params = 'custom_model_params.izhikevich.json'

syn_model = 'tsodyks2_synapse'
syn_dynamic_params = 'tsodyks2.syn_exc.json'


net = NetworkBuilder('net')
net.add_nodes(
    N=10,
    model_type='point_neuron',
    model_template=nest_model,
    dynamics_params=dynamics_params,
    jitter=np.random.uniform(0.5, 1.5, 10)
)

net.add_edges(
    source=net.nodes(), target=net.nodes(),
    connection_rule=1,
    syn_weight=2.0,
    delay=1.5,
    dynamics_params=syn_dynamic_params,
    model_template=syn_model
)

net.build()
net.save(output_dir='network_stp_syns')


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
virt_exc.save(output_dir='network_stp_syns')