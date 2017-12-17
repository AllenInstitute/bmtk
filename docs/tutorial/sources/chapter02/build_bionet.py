from bmtk.builder.networks import NetworkBuilder


cortex = NetworkBuilder('mcortex')
cortex.add_nodes(cell_name='Scnn1a',
              potental='exc',
              level_of_detail='biophysical',
              params_file='472363762_fit.json',
              morphology_file='Scnn1a.swc',
              set_params_function='Biophys1')

cortex.build()
cortex.save_nodes(output_dir='network')


thalamus = NetworkBuilder('mthalamus')
thalamus.add_nodes(N=10,
                   pop_name='tON',
                   potential='exc',
                   level_of_detail='filter')

thalamus.add_edges(source={'pop_name': 'tON'}, target=cortex.nodes(),
              connection_rule=5,
              weight_max=5e-05,
              weight_function='wmax',
              distance_range=[0.0, 150.0],
              target_sections=['basal', 'apical'],
              delay=2.0,
              params_file='AMPA_ExcToExc.json',
              set_params_function='exp2syn')

thalamus.build()
thalamus.save_nodes(output_dir='network')
thalamus.save_edges(output_dir='network')
#thalamus.save_edges(edges_file_name='thalamus_cortex_edges.h5', edge_types_file_name='thalamus_cortex_edge_types.h5')
