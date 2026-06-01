import numpy as np

from bmtk.builder import NetworkBuilder

n_exc = 200
n_inh = 100

rnet = NetworkBuilder('glifs')
rnet.add_nodes(
    N=n_exc,
    ei='e',
    pop_name='e4Nr5a1',
    model_type='point_neuron',
    model_template='glif_psc_double_alpha',
    dynamics_params='354833767_glif_lif_asc_config.json'
)

rnet.add_nodes(
    N=n_inh,
    ei='i',
    pop_name='i4Vip',
    model_type='point_neuron',
    model_template='glif_psc_double_alpha',
    dynamics_params='501570114_glif_lif_asc_config.json'
)

cm = rnet.add_edges(
    source={'ei': 'e'}, target={'ei': 'e'},
    connection_rule=lambda s, t: 0 if s.node_id == t.node_id else 1,
    dynamics_params='glif_e2e.json',
    model_template='static_synapse',
    delay=2.0
)
cm.add_properties(
    names='syn_weight',
    rule=lambda *_: np.random.uniform(1.0, 10.0)
)

cm = rnet.add_edges(
    source={'ei': 'e'}, target={'ei': 'i'},
    connection_rule=1,
    dynamics_params='glif_e2i.json',
    model_template='static_synapse',
    delay=2.0
)
cm.add_properties(
    names='syn_weight',
    rule=lambda *_: np.random.uniform(1.0, 10.0)
)

cm = rnet.add_edges(
    source={'ei': 'i'}, target={'ei': 'e'},
    connection_rule=1,
    dynamics_params='glif_i2e.json',
    model_template='static_synapse',
    delay=2.0
)
cm.add_properties(
    names='syn_weight',
    rule=lambda *_: np.random.uniform(-1.0, -10.0)
)

cm = rnet.add_edges(
    source={'ei': 'i'}, target={'ei': 'i'},
    connection_rule=lambda s, t: 0 if s.node_id == t.node_id else 1,
    dynamics_params='glif_i2i.json',
    model_template='static_synapse',
    delay=2.0
)
cm.add_properties(
    names='syn_weight',
    rule=lambda *_: np.random.uniform(-1.0, -10.0)
)

rnet.build()
rnet.save('network')


vnet = NetworkBuilder('virts')
vnet.add_nodes(
    N=100,
    ei='e',
    model_type='virtual'
)

cm = vnet.add_edges(
    target=rnet.nodes(ei='e'),
    connection_rule=1,
    dynamics_params='virt_e2e.json',
    model_template='static_synapse',
    delay=2.0
)
cm.add_properties(
    names='syn_weight',
    rule=lambda *_: np.random.uniform(5.0, 30.0)
)

cm = vnet.add_edges(
    target=rnet.nodes(ei='i'),
    connection_rule=1,
    dynamics_params='virt_e2i.json',
    model_template='static_synapse',
    delay=2.0
)
cm.add_properties(
    names='syn_weight',
    rule=lambda *_: np.random.uniform(5.0, 30.0)
)

vnet.build()
vnet.save('network')