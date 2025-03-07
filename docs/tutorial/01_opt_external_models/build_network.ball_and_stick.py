import numpy as np
from bmtk.builder.networks import NetworkBuilder


net = NetworkBuilder('BAS')
net.add_nodes(
    N=5,
    model_type='biophysical',
    model_template='python:loadBAS',
    gnabar=np.random.uniform(0.10, 0.15, size=5),
    gkbar=np.random.uniform(0.02, 0.04, size=5),
    gl=np.random.uniform(0.0001, 0.0005, size=5),
    el=np.random.uniform(-80.0, -50.0, size=5),
    g_pas = 0.001,
    e_pas = -65
)


net.build()
net.save(output_dir='network_BAS')


virt_exc = NetworkBuilder('virt_exc')
virt_exc.add_nodes(
    N=20,
    model_type='virtual',
    ei_type='exc'
)
conns = virt_exc.add_edges(
    source=virt_exc.nodes(),
    target=net.nodes(),
    connection_rule=12,
    model_template='Exp2Syn',
    dynamics_params='AMPA_ExcToExc.json',
    distance_range=[0.0, 1.0e20],
    target_sections=['soma', 'dend'],
    delay=1.0,
    syn_weight=0.05
)

virt_exc.build()
virt_exc.save(output_dir='network_BAS')