import os
import numpy as np

from bmtk.builder.networks import NetworkBuilder


np.random.seed(100)

net = NetworkBuilder('purkinje')
net.add_nodes( 
    pop_name='Scnn1a',
    model_type='biophysical',
    model_template='python:Purkinje_morph_1',
    spines_on=0
)
net.add_nodes( 
    pop_name='Scnn1a',
    model_type='biophysical',
    model_template='python:Purkinje_morph_1',
    spines_on=1
)


net.build()
net.save(output_dir='network_Purkinje')


virt = NetworkBuilder('virt_exc')
virt.add_nodes(
    N=5,
    model_type='virtual'
)
virt.add_edges(
    source=virt.nodes(), target=net.nodes(),
    connection_rule=12,
    syn_weight=6.4e-05,
    distance_range=[50.0, 10.0e20],
    target_sections=['dend'],
    delay=1.0,
    dynamics_params='AMPA_ExcToExc.json',
    model_template='exp2syn'
)

virt.build()
virt.save(output_dir='network_Purkinje')
