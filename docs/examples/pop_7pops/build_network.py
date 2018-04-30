from bmtk.builder.networks import NetworkBuilder

def build_v1():
    net = NetworkBuilder("v1")
    net.add_nodes(pop_name='Scnn1a',
                  location='VisL4',
                  ei='e',
                  level_of_detail='biophysical',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='472363762_pop.json')

    net.add_nodes(pop_name='Rorb',
                  location='VisL4',
                  ei='e',
                  level_of_detail='biophysical',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='473863510_pop.json')

    net.add_nodes(pop_name='Nr5a1',
                  location='VisL4',
                  ei='e',
                  level_of_detail='biophysical',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='473863035_pop.json')

    net.add_nodes(pop_name='PV1',
                  location='VisL4',
                  ei='i',
                  level_of_detail='biophysical',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='472912177_pop.json')

    net.add_nodes(pop_name='PV2',
                  location='VisL4',
                  ei='i',
                  level_of_detail='biophysical',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='473862421_pop.json')

    net.add_nodes(pop_name='LIF_exc',
                  location='VisL4',
                  ei='e',
                  level_of_detail='intfire',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='IntFire1_exc_pop.json')

    net.add_nodes(pop_name='LIF_inh',
                  location='VisL4',
                  ei='i',
                  level_of_detail='intfire',
                  model_type='population',
                  model_template='dipde:Internal',
                  dynamics_params='IntFire1_inh_pop.json')


    # Add edges
    net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'level_of_detail': 'biophysical'},
                  connection_rule=5,
                  syn_weight=0.0002,
                  delay=2.0,
                  dynamics_params='InhToInh.json')

    net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'level_of_detail': 'intfire'},
                  connection_rule=5,
                  syn_weight=0.00225,
                  weight_function='wmax',
                  delay=2.0,
                  dynamics_params='instanteneousInh.json')

    net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'level_of_detail': 'biophysical'},
                  connection_rule=lambda trg, src: 5,
                  syn_weight=0.00018,
                  delay=2.0,
                  dynamics_params='InhToExc.json')

    net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'level_of_detail': 'intfire'},
                  connection_rule=5,
                  syn_weight=0.009,
                  weight_function='wmax',
                  delay=2.0,
                  dynamics_params='instanteneousInh.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV1'},
                  connection_rule=5,
                  syn_weight=0.00035,
                  delay=2.0,
                  dynamics_params='ExcToInh.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV2'},
                  connection_rule=5,
                  syn_weight=0.00027,
                  delay=2.0,
                  dynamics_params='ExcToInh.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
                  connection_rule=5,
                  syn_weight=0.0043,
                  weight_function='wmax',
                  delay=2.0,
                  dynamics_params='instanteneousExc.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
                  connection_rule=5,
                  syn_weight=6.4e-05,
                  delay=2.0,
                  dynamics_params='ExcToExc.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Rorb'},
                  connection_rule=5,
                  syn_weight=5.5e-05,
                  delay=2.0,
                  dynamics_params='ExcToExc.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Nr5a1'},
                  connection_rule=5,
                  syn_weight=7.2e-05,
                  delay=2.0,
                  dynamics_params='ExcToExc.json')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
                  connection_rule=5,
                  syn_weight=0.0019,
                  delay=2.0,
                  dynamics_params='instanteneousExc.json')

    net.build()
    net.save_nodes(nodes_file_name='network/v1_nodes.h5', node_types_file_name='network/v1_node_types.csv')
    net.save_edges(edges_file_name='network/v1_v1_edges.h5', edge_types_file_name='network/v1_v1_edge_types.csv')


def build_lgn():
    LGN = NetworkBuilder("lgn")
    LGN.add_nodes(location='LGN',
                  model_type='virtual',
                  pop_name='tON',
                  ei='e')

    LGN.add_nodes(location='LGN',
                  model_type='virtual',
                  pop_name='tOFF',
                  ei='e')

    LGN.add_nodes(location='LGN',
                  model_type='virtual',
                  pop_name='tONOFF',
                  ei='e')

    VL4 = NetworkBuilder('v1')
    VL4.import_nodes(nodes_file_name='network/v1_nodes.h5', node_types_file_name='network/v1_node_types.csv')
    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'Rorb'},
                  connection_rule=10,
                  syn_weight=5e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'Nr5a1'},
                  connection_rule=10,
                  syn_weight=5e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'Scnn1a'},
                  connection_rule=10,
                  syn_weight=4e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'PV1'},
                  connection_rule=10,
                  syn_weight=0.0001,
                  weight_function='wmax',
                  distance_range=[0.0, 1.0e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='AMPA_ExcToInh.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'PV2'},
                  connection_rule=10,
                  syn_weight=9e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 1.0e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='AMPA_ExcToInh.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'LIF_exc'},
                  connection_rule=10,
                  syn_weight=0.0045,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'LIF_inh'},
                  connection_rule=10,
                  syn_weight=0.002,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousExc.json',
                  set_params_function='exp2syn')

    VL4.build()
    LGN.save_nodes(nodes_file_name='network/lgn_nodes.h5', node_types_file_name='network/lgn_node_types.csv')
    VL4.save_edges(edges_file_name='network/lgn_v1_edges.h5', edge_types_file_name='network/lgn_v1_edge_types.csv')


if __name__ == '__main__':
    build_v1()
    build_lgn()
