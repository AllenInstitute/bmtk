import os
import sys
from optparse import OptionParser
import numpy as np
import pandas as pd
import h5py

from bmtk.builder.networks import NetworkBuilder


def build_l4():
    net = NetworkBuilder("V1/L4")
    net.add_nodes(N=2, pop_name='Scnn1a',
                  positions=[(28.753, -364.868, -161.705), (48.753, -344.868, -141.705)],
                  tuning_angle=[0.0, 25.0],
                  rotation_angle_yaxis=[3.55501, 3.55501],
                  location='VisL4',
                  ei='e',
                  level_of_detail='biophysical',
                  params_file='472363762_fit.json',
                  morphology_file='Scnn1a-Tg3-Cre_Ai14_IVSCC_-177300.01.02.01_473845048_m.swc',
                  rotation_angle_zaxis=-3.646878266,
                  set_params_function='Biophys1')

    net.add_nodes(N=2, pop_name='Rorb',
                  positions=[(241.092, -349.263, 146.916), (201.092, -399.263, 126.916)],
                  tuning_angle=[50.0, 75.0],
                  rotation_angle_yaxis=[3.50934, 3.50934],
                  location='VisL4',
                  ei='e',
                  level_of_detail='biophysical',
                  params_file='473863510_fit.json',
                  morphology_file='Rorb-IRES2-Cre-D_Ai14_IVSCC_-168053.05.01.01_325404214_m.swc',
                  rotation_angle_zaxis=-4.159763785,
                  set_params_function='Biophys1')

    net.add_nodes(N=2, pop_name='Nr5a1',
                  positions=[(320.498, -351.259, 20.273), (310.498, -371.259, 10.273)],
                  tuning_angle=[100.0, 125.0],
                  rotation_angle_yaxis=[0.72202, 0.72202],
                  location='VisL4',
                  ei='e',
                  level_of_detail='biophysical',
                  params_file='473863035_fit.json',
                  morphology_file='Nr5a1-Cre_Ai14_IVSCC_-169250.03.02.01_471087815_m.swc',
                  rotation_angle_zaxis=-2.639275277,
                  set_params_function='Biophys1')

    net.add_nodes(N=2, pop_name='PV1',
                  positions=[(122.373, -352.417, -216.748), (102.373, -342.417, -206.748)],
                  tuning_angle=['NA', 'NA'],
                  rotation_angle_yaxis=[2.92043, 2.92043],
                  location='VisL4',
                  ei='i',
                  level_of_detail='biophysical',
                  params_file='472912177_fit.json',
                  morphology_file='Pvalb-IRES-Cre_Ai14_IVSCC_-176847.04.02.01_470522102_m.swc',
                  rotation_angle_zaxis=-2.539551891,
                  set_params_function='Biophys1')

    net.add_nodes(N=2, pop_name='PV2',
                  positions=[(350.321, -372.535, -18.282), (360.321, -371.535, -12.282)],
                  tuning_angle=['NA', 'NA'],
                  rotation_angle_yaxis=[5.043336, 5.043336],
                  location='VisL4',
                  ei='i',
                  level_of_detail='biophysical',
                  params_file='473862421_fit.json',
                  morphology_file='Pvalb-IRES-Cre_Ai14_IVSCC_-169125.03.01.01_469628681_m.swc',
                  rotation_angle_zaxis=-3.684439949,
                  set_params_function='Biophys1')

    net.add_nodes(N=2, pop_name='LIF_exc',
                  positions=[(-243.04, -342.352, -665.666), (-233.04, -332.352, -675.666)],
                  tuning_angle=['NA', 'NA'],
                  #rotation_angle_yaxis=[5.11801, 5.11801],
                  location='VisL4',
                  ei='e',
                  level_of_detail='intfire',
                  params_file='IntFire1_exc_1.json',
                  set_params_function='IntFire1')

    net.add_nodes(N=2, pop_name='LIF_inh',
                  positions=[(211.04, -321.333, -631.593), (218.04, -327.333, -635.593)],
                  tuning_angle=[150.0, 175.0],
                  #rotation_angle_yaxis=[4.566091, 4.566091],
                  location='VisL4',
                  ei='i',
                  level_of_detail='intfire',
                  params_file='IntFire1_inh_1.json',
                  set_params_function='IntFire1')

    print("Setting connections...")
    print("Generating I-to-I connections.")
    net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'level_of_detail': 'biophysical'},
                  connection_rule=5,
                  weight_max=0.0002,
                  weight_function='wmax',
                  distance_range=[0.0, 1e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='GABA_InhToInh.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'i'}, target={'ei': 'i', 'level_of_detail': 'intfire'},
                  connection_rule=5,
                  weight_max=0.00225,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousInh.json',
                  set_params_function='exp2syn')

    print("Generating I-to-E connections.")
    net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'level_of_detail': 'biophysical'},
                  connection_rule=lambda trg, src: 5,
                  weight_max=0.00018,
                  weight_function='wmax',
                  distance_range=[0.0, 50.0],
                  target_sections=['somatic', 'basal', 'apical'],
                  delay=2.0,
                  params_file='GABA_InhToExc.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'i'}, target={'ei': 'e', 'level_of_detail': 'intfire'},
                  connection_rule=5,
                  weight_max=0.009,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousInh.json',
                  set_params_function='exp2syn')

    print("Generating E-to-I connections.")
    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV1'},
                  connection_rule=5,
                  weight_max=0.00035,
                  weight_function='wmax',
                  distance_range=[0.0, 1e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='AMPA_ExcToInh.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'PV2'},
                  connection_rule=5,
                  weight_max=0.00027,
                  weight_function='wmax',
                  distance_range=[0.0, 1e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='AMPA_ExcToInh.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
                  connection_rule=5,
                  weight_max=0.0043,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousExc.json',
                  set_params_function='exp2syn')

    print("Generating E-to-E connections.")
    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
                  connection_rule=5,
                  weight_max=6.4e-05,
                  weight_function='gaussianLL',
                  weight_sigma=50.0,
                  distance_range=[30.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Rorb'},
                  connection_rule=5,
                  weight_max=5.5e-05,
                  weight_function='gaussianLL',
                  weight_sigma=50.0,
                  distance_range=[30.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'Nr5a1'},
                  connection_rule=5,
                  weight_max=7.2e-05,
                  weight_function='gaussianLL',
                  weight_sigma=50.0,
                  distance_range=[30.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    net.add_edges(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
                  connection_rule=5,
                  weight_max=0.0019,
                  weight_function='gaussianLL',
                  weight_sigma=50.0,
                  delay=2.0,
                  params_file='instanteneousExc.json',
                  set_params_function='exp2syn')


    net.build()
    net.save_nodes(nodes_file_name='output/v1_nodes.h5', node_types_file_name='output/v1_node_types.csv')
    net.save_edges(edges_file_name='output/v1_v1_edges.h5', edge_types_file_name='output/v1_v1_edge_types.csv')

    assert(os.path.exists('output/v1_node_types.csv'))
    node_types_csv = pd.read_csv('output/v1_node_types.csv', sep=' ')
    assert(len(node_types_csv) == 7)
    assert(set(node_types_csv.columns) == {'node_type_id', 'location', 'ei', 'level_of_detail', 'params_file',
                                           'pop_name', 'set_params_function', 'rotation_angle_zaxis',
                                           'morphology_file'})

    assert(os.path.exists('output/v1_nodes.h5'))
    nodes_h5 = h5py.File('output/v1_nodes.h5')
    assert(len(nodes_h5['/nodes/node_gid']) == 14)
    assert(len(nodes_h5['/nodes/node_type_id']) == 14)
    assert(len(nodes_h5['/nodes/node_group']) == 14)
    assert(len(nodes_h5['/nodes/node_group_index']) == 14)
    assert(set(nodes_h5['/nodes/0'].keys()) == {'positions', 'rotation_angle_yaxis', 'tuning_angle'})
    assert(len(nodes_h5['/nodes/0/positions']) == 10)
    assert(len(nodes_h5['/nodes/1/positions']) == 4)

    assert(os.path.exists('output/v1_v1_edge_types.csv'))
    edge_types_csv = pd.read_csv('output/v1_v1_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 11)
    assert(set(edge_types_csv.columns) == {'weight_max', 'edge_type_id', 'target_query', 'params_file',
                                           'set_params_function', 'delay', 'target_sections', 'weight_function',
                                           'weight_sigma', 'source_query', 'distance_range'})

    assert(os.path.exists('output/v1_v1_edges.h5'))
    edges_h5 = h5py.File('output/v1_v1_edges.h5')
    assert(len(edges_h5['/edges/index_pointer']) == 14+1)
    assert(len(edges_h5['/edges/target_gid']) == 14*14)
    assert(len(edges_h5['/edges/source_gid']) == 14*14)
    assert(len(edges_h5['/edges/0/nsyns']) == 14*14)

    return net

def build_lgn():
    def generate_positions(N, x0=0.0, x1=300.0, y0=0.0, y1=100.0):
        X = np.random.uniform(x0, x1, N)
        Y = np.random.uniform(y0, y1, N)
        return np.column_stack((X, Y))

    def select_source_cells(src_cells, trg_cell, n_syns):
        if trg_cell['tuning_angle'] is not None:
            synapses = [n_syns if src['pop_name'] == 'tON' or src['pop_name'] == 'tOFF' else 0 for src in src_cells]
        else:
            synapses = [n_syns if src['pop_name'] == 'tONOFF' else 0 for src in src_cells]

        return synapses

    if not os.path.exists('output'):
        os.makedirs('output')

    LGN = NetworkBuilder("LGN")
    LGN.add_nodes(N=3000,
                  positions=generate_positions(3000),
                  location='LGN',
                  level_of_detail='filter',
                  pop_name='tON',
                  ei='e')

    LGN.add_nodes(N=3000,
                  positions=generate_positions(3000),
                  location='LGN',
                  level_of_detail='filter',
                  pop_name='tOFF',
                  ei='e')

    LGN.add_nodes(N=3000,
                  positions=generate_positions(3000),
                  location='LGN',
                  level_of_detail='filter',
                  pop_name='tONOFF',
                  ei='e')

    VL4 = NetworkBuilder('V1/L4')
    VL4.import_nodes(nodes_file_name='output/v1_nodes.h5', node_types_file_name='output/v1_node_types.csv')
    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'Rorb'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=5e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'Nr5a1'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=5e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'Scnn1a'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=4e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 150.0],
                  target_sections=['basal', 'apical'],
                  delay=2.0,
                  params_file='AMPA_ExcToExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'PV1'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=0.0001,
                  weight_function='wmax',
                  distance_range=[0.0, 1.0e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='AMPA_ExcToInh.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'PV2'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=9e-05,
                  weight_function='wmax',
                  distance_range=[0.0, 1.0e+20],
                  target_sections=['somatic', 'basal'],
                  delay=2.0,
                  params_file='AMPA_ExcToInh.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'LIF_exc'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=0.0045,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousExc.json',
                  set_params_function='exp2syn')

    VL4.add_edges(source=LGN.nodes(), target={'pop_name': 'LIF_inh'},
                  iterator='all_to_one',
                  connection_rule=select_source_cells,
                  connection_params={'n_syns': 10},
                  weight_max=0.002,
                  weight_function='wmax',
                  delay=2.0,
                  params_file='instanteneousExc.json',
                  set_params_function='exp2syn')

    VL4.build()
    LGN.save_nodes(nodes_file_name='output/lgn_nodes.h5', node_types_file_name='output/lgn_node_types.csv')
    VL4.save_edges(edges_file_name='output/lgn_v1_edges.h5', edge_types_file_name='output/lgn_v1_edge_types.csv')

    assert(os.path.exists('output/lgn_node_types.csv'))
    node_types_csv = pd.read_csv('output/lgn_node_types.csv', sep=' ')
    assert(len(node_types_csv) == 3)
    assert(set(node_types_csv.columns) == {'node_type_id', 'location', 'ei', 'level_of_detail', 'pop_name'})


    assert(os.path.exists('output/lgn_nodes.h5'))
    nodes_h5 = h5py.File('output/lgn_nodes.h5')
    assert(len(nodes_h5['/nodes/node_gid']) == 9000)
    assert(len(nodes_h5['/nodes/node_type_id']) == 9000)
    assert(len(nodes_h5['/nodes/node_group']) == 9000)
    assert(len(nodes_h5['/nodes/node_group_index']) == 9000)
    assert(set(nodes_h5['/nodes/0'].keys()) == {'positions'})
    assert(len(nodes_h5['/nodes/0/positions']) == 9000)

    assert(os.path.exists('output/lgn_v1_edge_types.csv'))
    edge_types_csv = pd.read_csv('output/lgn_v1_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 7)
    assert(set(edge_types_csv.columns) == {'weight_max', 'edge_type_id', 'target_query', 'params_file',
                                           'set_params_function', 'delay', 'target_sections', 'weight_function',
                                           'source_query', 'distance_range'})

    assert(os.path.exists('output/lgn_v1_edges.h5'))
    edges_h5 = h5py.File('output/lgn_v1_edges.h5')
    assert(len(edges_h5['/edges/index_pointer']) == 14+1)
    assert(len(edges_h5['/edges/target_gid']) == 6000*14)
    assert(len(edges_h5['/edges/source_gid']) == 6000*14)
    assert(len(edges_h5['/edges/0/nsyns']) == 6000*14)


def build_tw():
    if not os.path.exists('output'):
        os.makedirs('output')

    TW = NetworkBuilder("TW")
    TW.add_nodes(N=3000, pop_name='TW', ei='e', location='TW', level_of_detail='filter')

    # Save cells.csv and cell_types.csv
    TW.save_nodes(nodes_file_name='output/tw_nodes.h5', node_types_file_name='output/tw_node_types.csv')

    VL4 = NetworkBuilder('V1/L4')
    VL4.import_nodes(nodes_file_name='output/v1_nodes.h5', node_types_file_name='output/v1_node_types.csv')
    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'Rorb'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.00015, 'weight_function': 'wmax', 'distance_range': [30.0, 150.0],
                      'target_sections': ['basal', 'apical'], 'delay': 2.0, 'params_file': 'AMPA_ExcToExc.json',
                      'set_params_function': 'exp2syn'}))

    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'Scnn1a'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.00019, 'weight_function': 'wmax', 'distance_range': [30.0, 150.0],
                      'target_sections': ['basal', 'apical'], 'delay': 2.0, 'params_file': 'AMPA_ExcToExc.json',
                      'set_params_function': 'exp2syn'}))

    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'Nr5a1'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.00019, 'weight_function': 'wmax', 'distance_range': [30.0, 150.0],
                      'target_sections': ['basal', 'apical'], 'delay': 2.0, 'params_file': 'AMPA_ExcToExc.json',
                      'set_params_function': 'exp2syn'}))

    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'PV1'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.0022, 'weight_function': 'wmax', 'distance_range': [0.0, 1.0e+20],
                      'target_sections': ['somatic', 'basal'], 'delay': 2.0, 'params_file': 'AMPA_ExcToInh.json',
                      'set_params_function': 'exp2syn'}))

    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'PV2'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.0013, 'weight_function': 'wmax', 'distance_range': [0.0, 1.0e+20],
                      'target_sections': ['somatic', 'basal'], 'delay': 2.0, 'params_file': 'AMPA_ExcToInh.json',
                      'set_params_function': 'exp2syn'}))

    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'LIF_exc'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.015, 'weight_function': 'wmax', 'delay': 2.0,
                      'params_file': 'instanteneousExc.json', 'set_params_function': 'exp2syn'}))

    VL4.add_edges(source=TW.nodes(), target={'pop_name': 'LIF_inh'},
                  connection_rule=lambda trg, src: 5,
                  **({'weight_max': 0.05, 'weight_function': 'wmax', 'delay': 2.0,
                      'params_file': 'instanteneousExc.json', 'set_params_function': 'exp2syn'}))

    VL4.build()
    VL4.save_edges(edges_file_name='output/tw_v1_edges.h5', edge_types_file_name='output/tw_v1_edge_types.csv')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--force-overwrite", dest="force_overwrite", action="store_true", default=False)
    parser.add_option("--out-dir", dest="out_dir", default='./output/')
    parser.add_option("--percentage", dest="percentage", type="float", default=1.0)
    parser.add_option("--with-stats", dest="with_stats", action="store_true", default=False)
    (options, args) = parser.parse_args()

    if not os.path.exists('output'):
        os.makedirs('output')

    l4_network = build_l4()
    build_lgn()
    build_tw()

