import os
import sys
from optparse import OptionParser
import numpy as np

from bmtk.builder.networks import NetworkBuilder


def build_l4():
    if not os.path.exists('output/network/VisL4'):
        os.makedirs('output/network/VisL4')

    net = NetworkBuilder("V1/L4")
    net.add_nodes(N=2, pop_name='Scnn1a', node_type_id='395830185',
                  position='points',
                  position_params={'location': [(28.753, -364.868, -161.705), (48.753, -344.868, -141.705)]},
                  array_params={"tuning_angle": [0.0, 25.0]},
                  location='VisL4',
                  ei='e',
                  gaba_synapse='y',
                  params_file='472363762_point.json',
                  model_type='iaf_psc_alpha')

    net.add_nodes(N=2, pop_name='Rorb', node_type_id='314804042',
                  position='points',
                  position_params={'location': [(241.092, -349.263, 146.916), (201.092, -399.263, 126.916)]},
                  array_params={"tuning_angle": [50.0, 75.0]},
                  location='VisL4',
                  ei='e',
                  gaba_synapse='y',
                  params_file='473863510_point.json',
                  model_type='iaf_psc_alpha')

    net.add_nodes(N=2, pop_name='Nr5a1', node_type_id='318808427',
                  position='points',
                  position_params={'location': [(320.498, -351.259, 20.273), (310.498, -371.259, 10.273)]},
                  array_params={"tuning_angle": [100.0, 125.0]},
                  location='VisL4',
                  ei='e',
                  gaba_synapse='y',
                  params_file='473863035_point.json',
                  model_type='iaf_psc_alpha')

    net.add_nodes(N=2, pop_name='PV1', node_type_id='330080937',
                  position='points',
                  position_params={'location': [(122.373, -352.417, -216.748), (102.373, -342.417, -206.748)]},
                  array_params={'tuning_angle': ['NA', 'NA']},
                  location='VisL4',
                  ei='i',
                  gaba_synapse='y',
                  params_file='472912177_fit.json',
                  model_type='iaf_psc_alpha')

    net.add_nodes(N=2, pop_name='PV2', node_type_id='318331342',
                  position='points',
                  position_params={'location': [(350.321, -372.535, -18.282), (360.321, -371.535, -12.282)]},
                  array_params={'tuning_angle': ['NA', 'NA']},
                  location='VisL4',
                  ei='i',
                  gaba_synapse='y',
                  params_file='473862421_point.json',
                  model_type='iaf_psc_alpha')

    net.add_nodes(N=2, pop_name='LIF_exc', node_type_id='100000101',
                  position='points',
                  position_params={'location': [(-243.04, -342.352, -665.666), (-233.04, -332.352, -675.666)]},
                  array_params={'tuning_angle': ['NA', 'NA']},
                  location='VisL4',
                  ei='e',
                  gaba_synapse='n',
                  params_file='IntFire1_exc_1.json',
                  model_type='iaf_psc_alpha')

    net.add_nodes(N=2, pop_name='LIF_inh', node_type_id='100000102',
                  position='points',
                  position_params={'location': [(211.04, -321.333, -631.593), (218.04, -327.333, -635.593)]},
                  array_params={'tuning_angle': [150.0, 175.0]},
                  location='VisL4',
                  ei='i',
                  gaba_synapse='n',
                  params_file='IntFire1_inh_1.json',
                  model_type='iaf_psc_alpha')



    print("Setting connections...")
    net.connect(source={'ei': 'i'}, target={'ei': 'i', 'gaba_synapse': 'y'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': -1.8, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'InhToInh.json', 'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'i'}, target={'ei': 'e', 'gaba_synapse': 'y'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': -12.6, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'InhToExc.json',
                             'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'i'}, target={'pop_name': 'LIF_inh'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': -1.125, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'InhToInh.json',
                             'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'i'}, target={'pop_name': 'LIF_exc'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': -6.3, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json',
                             'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'e'}, target={'pop_name': 'PV1'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 7.7, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'e'}, target={'pop_name': 'PV2'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 5.4, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'e'}, target={'pop_name': 'LIF_inh'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 3.44, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    print("Generating E-to-E connections.")
    net.connect(source={'ei': 'e'}, target={'pop_name': 'Scnn1a'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 8.448, 'weight_function': 'gaussianLL', 'weight_sigma': 50.0,
                             'delay': 2.0, 'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'e'}, target={'pop_name': 'Rorb'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 4.292, 'weight_function': 'gaussianLL', 'weight_sigma': 50.0,
                             'delay': 2.0, 'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'e'}, target={'pop_name': 'Nr5a1'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 5.184, 'weight_function': 'gaussianLL', 'weight_sigma': 50.0,
                             'delay': 2.0, 'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    net.connect(source={'ei': 'e'}, target={'pop_name': 'LIF_exc'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 1.995, 'weight_function': 'gaussianLL', 'weight_sigma': 50.0,
                             'delay': 2.0, 'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    net.build()
    net.save_cells(filename='output/network/VisL4/nodes.csv',
                   columns=['node_id', 'node_type_id', 'position', 'tuning_angle'],
                   position_labels=['x', 'y', 'z'])

    net.save_types(filename='output/network/VisL4/node_types.csv',
                   columns=['node_type_id', 'pop_name', 'ei', 'gaba_synapse', 'location', 'model_type', 'params_file'])

    net.save_edge_types('output/network/VisL4/edge_types.csv', opt_columns=['weight_max', 'weight_function', 'weight_sigma',
                                                              'delay', 'params_file', 'synapse_model'])
    net.save_edges(filename='output/network/VisL4/edges.h5')
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

    if not os.path.exists('output/network/LGN'):
        os.makedirs('output/network/LGN')

    LGN = NetworkBuilder("LGN")
    LGN.add_nodes(N=3000,
                  position='points', position_params={'location': generate_positions(3000)},
                  node_type_id='tON_001',
                  location='LGN',
                  model_type='spike_generator',
                  pop_name='tON',
                  ei='e',
                  params_file='filter_point.json')

    LGN.add_nodes(N=3000,
                  position='points', position_params={'location': generate_positions(3000)},
                  node_type_id='tOFF_001',
                  location='LGN',
                  model_type='spike_generator',
                  pop_name='tOFF',
                  ei='e',
                  params_file='filter_point.json')

    LGN.add_nodes(N=3000,
                  position='points', position_params={'location': generate_positions(3000)},
                  node_type_id='tONOFF_001',
                  location='LGN',
                  model_type='spike_generator',
                  pop_name='tONOFF',
                  ei='e',
                  params_file='filter_point.json')

    LGN.save_cells(filename='output/network/LGN/lgn_nodes.csv',
                   columns=['node_id', 'node_type_id', 'position'],
                   position_labels=['x', 'y'])
    LGN.save_types(filename='output/network/LGN/lgn_node_types.csv',
                   columns=['node_type_id', 'ei', 'location', 'model_type', 'params_file'])

    VL4 = NetworkBuilder.load("V1/L4", nodes='output/network/VisL4/nodes.csv', node_types='output/network/VisL4/node_types.csv')

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'Rorb'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 4.125, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'Nr5a1'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 4.5, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'Scnn1a'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 5.6, 'weight_function': 'wmax', 'distance_range': [0.0, 150.0],
                             'delay': 2.0, 'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'PV1'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 1.54, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'PV2'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 1.26, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'LIF_exc'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 4.41, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=LGN.nodes(), target={'pop_name': 'LIF_inh'},
                iterator='all_to_one',
                connector=select_source_cells,
                connector_params={'n_syns': 10},
                edge_params={'weight_max': 2.52, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.build()
    VL4.save_edge_types('output/network/LGN/lgn_edge_types.csv',
                        opt_columns=['weight_max', 'weight_function', 'delay', 'params_file', 'synapse_model'])
    VL4.save_edges(filename='output/network/LGN/lgn_edges.h5')


def build_tw():
    if not os.path.exists('output/network/TW'):
        os.makedirs('output/network/TW')

    TW = NetworkBuilder("TW")
    TW.add_nodes(N=3000, node_type_id='TW_001', pop_name='TW', ei='e', location='TW', level_of_detail='filter')

    # Save cells.csv and cell_types.csv
    TW.save_cells(filename='output/network/TW/tw_nodes.csv',
                  columns=['node_id', 'node_type_id', 'pop_name', 'ei', 'location'])

    TW.save_types(filename='output/network/TW/tw_node_types.csv', columns=['node_type_id', 'level_of_detail'])

    VL4 = NetworkBuilder.load("V1/L4", nodes='output/network/VisL4/nodes.csv', node_types='output/network/VisL4/node_types.csv')
    VL4.connect(source=TW.nodes(), target={'pop_name': 'Rorb'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 12.75, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=TW.nodes(), target={'pop_name': 'Scnn1a'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 33.25, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToExc.json',
                             'synapse_model': 'static_synapse'})

    VL4.connect(source=TW.nodes(), target={'pop_name': 'Nr5a1'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 19.0, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToExc.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=TW.nodes(), target={'pop_name': 'PV1'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 83.6, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=TW.nodes(), target={'pop_name': 'PV2'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 32.5, 'weight_function': 'wmax','delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=TW.nodes(), target={'pop_name': 'LIF_exc'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 22.5, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.connect(source=TW.nodes(), target={'pop_name': 'LIF_inh'},
                connector=lambda trg, src: 5,
                edge_params={'weight_max': 55.0, 'weight_function': 'wmax', 'delay': 2.0,
                             'params_file': 'ExcToInh.json', 'synapse_model': 'static_synapse'})

    VL4.build()
    VL4.save_edge_types('output/network/TW/tw_edge_types.csv',
                        opt_columns=['weight_max', 'weight_function', 'delay', 'params_file', 'synapse_model'])

    VL4.save_edges(filename='output/network/TW/tw_edges.h5')



# Call main function
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--force-overwrite", dest="force_overwrite", action="store_true", default=False)
    parser.add_option("--out-dir", dest="out_dir", default='./output/')
    parser.add_option("--percentage", dest="percentage", type="float", default=1.0)
    parser.add_option("--with-stats", dest="with_stats", action="store_true", default=False)
    (options, args) = parser.parse_args()

    if not os.path.exists('output/network'):
        os.makedirs('output/network')

    build_l4()
    build_lgn()
    build_tw()

