from bmtk.builder.networks import NetworkBuilder


# First thing is to create a network builder object
net = NetworkBuilder('mcortex')
net.add_nodes(cell_name='Scnn1a',
              positions=[(0.0, 0.0, 0.0)],
              potental='exc',
              level_of_detail='biophysical',
              params_file='472363762_fit.json',
              morphology_file='Scnn1a.swc',
              set_params_function='Biophys1')

net.build()
net.save_nodes(output_dir='network')