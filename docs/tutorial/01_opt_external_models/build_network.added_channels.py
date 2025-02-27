from bmtk.builder.networks import NetworkBuilder

net = NetworkBuilder('L5')
net.add_nodes(
    N=1,
    model_type='biophysical',
    model_template='hoc:L5PCtemplate',
    morphology='cell3.asc',
    model_processing='add_channels'
)


net.build()
net.save(output_dir='network_added_channels')


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
    target_sections=['soma', 'basal', 'apical'],
    delay=2.0,
    syn_weight=0.01
)

virt_exc.build()
virt_exc.save(output_dir='network_added_channels')