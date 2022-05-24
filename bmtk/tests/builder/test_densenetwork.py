import os
import shutil
import pytest
import numpy as np
import pandas as pd
import h5py
import tempfile

from bmtk.builder.network_adaptors.dm_network import DenseNetwork
from bmtk.builder.network_adaptors.dm_network_orig import DenseNetworkOrig


@pytest.mark.parametrize('network_cls', [
    (DenseNetwork),
    (DenseNetworkOrig)
])
def test_save_nsyn_table(network_cls):
    net = network_cls('NET1')
    net.add_nodes(N=10, position=[(0.0, 1.0, -1.0)]*10, cell_type='Scnna1', ei='e')
    net.add_nodes(N=10, position=[(0.0, 1.0, -1.0)]*10, cell_type='PV1', ei='i')
    net.add_nodes(N=10, position=[(0.0, 1.0, -1.0)]*10, tags=np.linspace(0, 100, 10), cell_type='PV2', ei='i')
    net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=lambda s, t: 1,
                  p1='e2i', p2='e2i')
    net.add_edges(source=net.nodes(cell_type='Scnna1'), target=net.nodes(cell_type='PV1'),
                  connection_rule=lambda s, t: 2, p1='s2p')
    net.build()
    nodes_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    nodes_csv = tempfile.NamedTemporaryFile(suffix='.csv')
    edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    edges_csv = tempfile.NamedTemporaryFile(suffix='.csv')

    net.save_nodes(nodes_h5.name, nodes_csv.name)
    net.save_edges(edges_h5.name, edges_csv.name)

    assert(os.path.exists(nodes_h5.name) and os.path.exists(nodes_csv.name))
    node_types_df = pd.read_csv(nodes_csv.name, sep=' ')
    assert(len(node_types_df) == 3)
    assert('cell_type' in node_types_df.columns)
    assert('ei' in node_types_df.columns)
    assert('positions' not in node_types_df.columns)

    nodes_h5 = h5py.File(nodes_h5.name, 'r')
    assert('node_id' in nodes_h5['/nodes/NET1'])
    assert(len(nodes_h5['/nodes/NET1/node_id']) == 30)
    assert(len(nodes_h5['/nodes/NET1/node_type_id']) == 30)
    assert(len(nodes_h5['/nodes/NET1/node_group_id']) == 30)
    assert(len(nodes_h5['/nodes/NET1/node_group_index']) == 30)

    node_groups = {nid: grp for nid, grp in nodes_h5['/nodes/NET1'].items() if isinstance(grp, h5py.Group)}
    for grp in node_groups.values():
        if len(grp) == 1:
            assert('position' in grp and len(grp['position']) == 20)

        elif len(grp) == 2:
            assert('position' in grp and len(grp['position']) == 10)
            assert('tags' in grp and len(grp['tags']) == 10)

        else:
            assert False

    assert(os.path.exists(edges_h5.name) and os.path.exists(edges_csv.name))
    edge_types_df = pd.read_csv(edges_csv.name, sep=' ')
    assert (len(edge_types_df) == 2)
    assert ('p1' in edge_types_df.columns)
    assert ('p2' in edge_types_df.columns)

    edges_h5 = h5py.File(edges_h5.name, 'r')
    assert('source_to_target' in edges_h5['/edges/NET1_to_NET1/indices'])
    assert('target_to_source' in edges_h5['/edges/NET1_to_NET1/indices'])
    assert(len(edges_h5['/edges/NET1_to_NET1/target_node_id']) == 300)
    assert(len(edges_h5['/edges/NET1_to_NET1/source_node_id']) == 300)

    assert(edges_h5['/edges/NET1_to_NET1/target_node_id'][0] == 0)
    assert(edges_h5['/edges/NET1_to_NET1/source_node_id'][0] == 10)
    assert(edges_h5['/edges/NET1_to_NET1/edge_group_index'][0] == 0)
    assert(edges_h5['/edges/NET1_to_NET1/edge_type_id'][0] == 100)
    assert(edges_h5['/edges/NET1_to_NET1/0/nsyns'][0] == 1)

    assert(edges_h5['/edges/NET1_to_NET1/target_node_id'][299] == 19)
    assert(edges_h5['/edges/NET1_to_NET1/source_node_id'][299] == 9)
    assert(edges_h5['/edges/NET1_to_NET1/edge_group_id'][299] == 0)
    assert(edges_h5['/edges/NET1_to_NET1/edge_type_id'][299] == 101)
    assert(edges_h5['/edges/NET1_to_NET1/0/nsyns'][299] == 2)


@pytest.mark.parametrize('network_cls', [
    (DenseNetwork),
    (DenseNetworkOrig)
])
def test_save_weights(network_cls):
    net = network_cls('NET1')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='Scnna1', ei='e')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='PV1', ei='i')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, tags=np.linspace(0, 100, 100), cell_type='PV2', ei='i')
    cm = net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=lambda s, t: 3,
                       p1='e2i', p2='e2i')  # 200*100 = 60000 edges
    cm.add_properties(names=['segment', 'distance'], rule=lambda s, t: [1, 0.5], dtypes=[int, float])

    net.add_edges(source=net.nodes(cell_type='Scnna1'), target=net.nodes(cell_type='PV1'),
                  connection_rule=lambda s, t: 2, p1='s2p')  # 100*100 = 20000'

    net.build()
    net_dir = tempfile.mkdtemp()
    net.save_nodes('tmp_nodes.h5', 'tmp_node_types.csv', output_dir=net_dir)
    net.save_edges('tmp_edges.h5', 'tmp_edge_types.csv', output_dir=net_dir)

    edges_h5 = h5py.File('{}/tmp_edges.h5'.format(net_dir), 'r')
    assert(net.nedges == 80000)
    assert(len(edges_h5['/edges/NET1_to_NET1/0/distance']) == 60000)
    assert(len(edges_h5['/edges/NET1_to_NET1/0/segment']) == 60000)
    assert(len(edges_h5['/edges/NET1_to_NET1/1/nsyns']) == 10000)
    assert(edges_h5['/edges/NET1_to_NET1/0/distance'][0] == 0.5)
    assert(edges_h5['/edges/NET1_to_NET1/0/segment'][0] == 1)
    assert(edges_h5['/edges/NET1_to_NET1/1/nsyns'][0] == 2)


@pytest.mark.parametrize('network_cls', [
    (DenseNetwork),
    (DenseNetworkOrig)
])
def test_save_multinetwork(network_cls):
    net1 = DenseNetwork('NET1')
    net1.add_nodes(N=100, position=[(0.0, 1.0, -1.0)] * 100, cell_type='Scnna1', ei='e')
    net1.add_edges(source={'ei': 'e'}, target={'ei': 'e'}, connection_rule=5, ctype_1='n1_rec')
    net1.build()

    net2 = DenseNetwork('NET2')
    net2.add_nodes(N=10, position=[(0.0, 1.0, -1.0)] * 10, cell_type='PV1', ei='i')
    net2.add_edges(connection_rule=10, ctype_1='n2_rec')
    net2.add_edges(source=net1.nodes(), target={'ei': 'i'}, connection_rule=1, ctype_2='n1_n2')
    net2.add_edges(target=net1.nodes(cell_type='Scnna1'), source={'cell_type': 'PV1'}, connection_rule=2,
                   ctype_2='n2_n1')
    net2.build()

    net_dir = tempfile.mkdtemp()
    net1.save_edges(output_dir=net_dir)
    net2.save_edges(output_dir=net_dir)

    n1_n1_fname = '{}/{}_{}'.format(net_dir, 'NET1', 'NET1')
    edges_h5 = h5py.File(n1_n1_fname + '_edges.h5', 'r')
    assert(len(edges_h5['/edges/NET1_to_NET1/target_node_id']) == 100*100)
    assert(len(edges_h5['/edges/NET1_to_NET1/0/nsyns']) == 100*100)
    assert(edges_h5['/edges/NET1_to_NET1/0/nsyns'][0] == 5)
    edge_types_csv = pd.read_csv(n1_n1_fname + '_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 1)
    assert('ctype_2' not in edge_types_csv.columns.values)
    assert(edge_types_csv['ctype_1'].iloc[0] == 'n1_rec')

    n1_n2_fname = '{}/{}_{}'.format(net_dir, 'NET1', 'NET2')
    edges_h5 = h5py.File(n1_n2_fname + '_edges.h5', 'r')
    assert(len(edges_h5['/edges/NET1_to_NET2/target_node_id']) == 100*10)
    assert(len(edges_h5['/edges/NET1_to_NET2/0/nsyns']) == 100*10)
    assert(edges_h5['/edges/NET1_to_NET2/0/nsyns'][0] == 1)
    edge_types_csv = pd.read_csv(n1_n2_fname + '_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 1)
    assert('ctype_1' not in edge_types_csv.columns.values)
    assert(edge_types_csv['ctype_2'].iloc[0] == 'n1_n2')

    n2_n1_fname = '{}/{}_{}'.format(net_dir, 'NET2', 'NET1')
    edges_h5 = h5py.File(n2_n1_fname + '_edges.h5', 'r')
    assert(len(edges_h5['/edges/NET2_to_NET1/target_node_id']) == 100*10)
    assert(len(edges_h5['/edges/NET2_to_NET1/0/nsyns']) == 100*10)
    assert(edges_h5['/edges/NET2_to_NET1/0/nsyns'][0] == 2)
    edge_types_csv = pd.read_csv(n2_n1_fname + '_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 1)
    assert('ctype_1' not in edge_types_csv.columns.values)
    assert(edge_types_csv['ctype_2'].iloc[0] == 'n2_n1')

    n2_n2_fname = '{}/{}_{}'.format(net_dir, 'NET2', 'NET2')
    edges_h5 = h5py.File(n2_n2_fname + '_edges.h5', 'r')
    assert(len(edges_h5['/edges/NET2_to_NET2/target_node_id']) == 10*10)
    assert(len(edges_h5['/edges/NET2_to_NET2/0/nsyns']) == 10*10)
    assert(edges_h5['/edges/NET2_to_NET2/0/nsyns'][0] == 10)
    edge_types_csv = pd.read_csv(n2_n2_fname + '_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 1)
    assert('ctype_2' not in edge_types_csv.columns.values)
    assert(edge_types_csv['ctype_1'].iloc[0] == 'n2_rec')


if __name__ == '__main__':
    # test_save_weights(DenseNetwork)
    test_save_multinetwork(DenseNetwork)
