from bmtk.builder.networks import NetworkBuilder


# First thing is to create a network builder object
net = NetworkBuilder('mcortex')
net.add_nodes(cell_name='Scnn1a_473845048',
              potental='exc',
              model_type='biophysical',
              model_template='ctdb:Biophys1.hoc',
              model_processing='aibs_perisomatic',
              dynamics_params='472363762_fit.json',
              morphology='Scnn1a_473845048_m.swc')

net.build()
net.save_nodes(output_dir='network')