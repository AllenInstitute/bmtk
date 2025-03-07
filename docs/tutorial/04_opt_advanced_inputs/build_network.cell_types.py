from bmtk.builder.networks import NetworkBuilder

net = NetworkBuilder(f'bio_488683425_fullstub')
net.add_nodes(
    N=1,
    model_type='biophysical',
    model_template='ctdb:Biophys1.hoc',
    model_processing='aibs_allactive_fullaxon',
    morphology=f'491766131_m.swc',
    dynamics_params=f'491766131_fit.json',
    specimen_id=488683425
)

net.build()
net.save(output_dir='network_cell_types')
