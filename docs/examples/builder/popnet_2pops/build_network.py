import os

from bmtk.builder.networks import NetworkBuilder


def build_l4():
    if not os.path.exists('output/network/VisL4'):
        os.makedirs('output/network/VisL4')

    net = NetworkBuilder("V1/L4")
    net.add_nodes(node_type_id=0, pop_name='excitatory', params_file='excitatory_pop.json')
    net.add_nodes(node_type_id=1, pop_name='inhibitory', params_file='inhibitory_pop.json')

    net.connect(target={'pop_name': 'excitatory'}, source={'pop_name': 'inhibitory'},
                edge_params={'weight': -0.001, 'delay': 0.002, 'nsyns': 2, 'params_file': 'ExcToInh.json'})

    net.connect(target={'pop_name': 'inhibitory'}, source={'pop_name': 'excitatory'},
                edge_params={'weight': 0.001, 'delay': 0.002, 'nsyns': 5, 'params_file': 'ExcToInh.json'})

    net.save_types(filename='output/network/VisL4/node_types.csv',
                   columns=['node_type_id', 'pop_name', 'params_file'])

    net.save_edge_types('output/network/VisL4/edge_types.csv',
                        opt_columns=['weight', 'delay', 'nsyns', 'params_file'])


def build_lgn():
    if not os.path.exists('output/network/LGN'):
        os.makedirs('output/network/LGN')

    net = NetworkBuilder("LGN")
    net.add_nodes(N=3000, node_type_id='tON_001', ei='e', location='LGN', pop_name='tON_001', params_file='filter_pop.json')
    net.add_nodes(N=3000, node_type_id='tOFF_001', ei='e', location='LGN', pop_name='tOFF_001', params_file='filter_pop.json')
    net.add_nodes(N=3000, node_type_id='tONOFF_001', ei='e', location='LGN', pop_name='tONOFF_001',
                  params_file='filter_pop.json')

    net.save_cells(filename='output/network/LGN/nodes.csv',
                   columns=['node_id', 'node_type_id'])
    net.save_types(filename='output/network/LGN/node_types.csv',
                   columns=['node_type_id', 'ei', 'location', 'pop_name', 'params_file'])

    net.connect(target={'pop_name': 'excitatory'},
                edge_params={'weight': 0.0015, 'delay': 0.002, 'params_file': 'ExcToExc.json', 'nsyns': 10})

    net.connect(target={'pop_name': 'inhibitory'},
                edge_params={'weight': 0.0019, 'delay': 0.002, 'params_file': 'ExcToInh.json', 'nsyns': 12})

    net.save_edge_types('output/network/LGN/edge_types.csv',
                        opt_columns=['weight', 'delay', 'nsyns', 'params_file'])



if __name__ == '__main__':
    build_l4()
    build_lgn()