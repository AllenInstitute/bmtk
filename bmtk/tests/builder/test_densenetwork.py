import os
import shutil
import pytest
import numpy as np
import pandas as pd
import h5py
import tempfile

from bmtk.builder import NetworkBuilder


def test_create_network():
    net = NetworkBuilder('NET1')
    assert(net.name == 'NET1')
    assert(net.nnodes == 0)
    assert(net.nedges == 0)
    assert(net.nodes_built is False)
    assert(net.edges_built is False)


def test_no_name():
    with pytest.raises(Exception):
        NetworkBuilder('')


def test_build_nodes():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100,
                  position=[(100.0, -50.0, 50.0)]*100,
                  tunning_angle=np.linspace(0, 365.0, 100, endpoint=False),
                  cell_type='Scnna1',
                  model_type='Biophys1',
                  location='V1',
                  ei='e')

    net.add_nodes(N=25,
                  position=np.random.rand(25, 3)*[100.0, 50.0, 100.0],
                  model_type='intfire1',
                  location='V1',
                  ei='e')

    net.add_nodes(N=150,
                  position=np.random.rand(150, 3)*[100.0, 50.0, 100.0],
                  tunning_angle=np.linspace(0, 365.0, 150, endpoint=False),
                  cell_type='SST',
                  model_type='Biophys1',
                  location='V1',
                  ei='i')

    net.build()
    assert(net.nodes_built is True)
    assert(net.nnodes == 275)
    assert(net.nedges == 0)
    assert(len(net.nodes()) == 275)
    assert(len(net.nodes(ei='e')) == 125)
    assert(len(net.nodes(model_type='Biophys1')) == 250)
    assert(len(net.nodes(location='V1', model_type='Biophys1')))

    intfire_nodes = list(net.nodes(model_type='intfire1'))
    assert(len(intfire_nodes) == 25)
    node1 = intfire_nodes[0]
    assert(node1['model_type'] == 'intfire1' and 'cell_type' not in node1)


def test_build_nodes1():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=3, node_id=[100, 200, 300], node_type_id=101, name=['one', 'two', 'three'])
    node_one = list(net.nodes(name='one'))[0]
    assert(node_one['name'] == 'one')
    assert(node_one['node_id'] == 100)
    assert(node_one['node_type_id'] == 101)

    node_three = list(net.nodes(name='three'))[0]
    assert(node_three['name'] == 'three')
    assert(node_three['node_id'] == 300)
    assert(node_three['node_type_id'] == 101)


def test_build_nodes_fail1():
    net = NetworkBuilder('NET1')
    with pytest.raises(Exception):
        net.add_nodes(N=100, list1=[100]*99)


def test_build_nodes_fail2():
    net = NetworkBuilder('NET1')
    with pytest.raises(Exception):
        net.add_nodes(N=2, node_type_id=0)
        net.add_nodes(N=2, node_type_id=0)


def test_nsyn_edges():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, cell_type='Scnna1', ei='e')
    net.add_nodes(N=100, cell_type='PV1', ei='i')
    net.add_nodes(N=100, cell_type='PV2', ei='i')
    net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=lambda s, t: 1)  # 200*100 = 20000 edges
    net.add_edges(source=net.nodes(cell_type='Scnna1'), target=net.nodes(cell_type='PV1'),
                  connection_rule=lambda s, t: 2)  # 100*100*2 = 20000
    net.build()
    assert(net.nedges == 20000 + 20000)
    assert(net.edges_built is True)
    #print list(net.edges())


def test_save_nsyn_table():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='Scnna1', ei='e')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='PV1', ei='i')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, tags=np.linspace(0, 100, 100), cell_type='PV2', ei='i')
    net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=lambda s, t: 1,
                  p1='e2i', p2='e2i')  # 200*100 = 20000 edges
    net.add_edges(source=net.nodes(cell_type='Scnna1'), target=net.nodes(cell_type='PV1'),
                  connection_rule=lambda s, t: 2, p1='s2p')  # 100*100*2 = 20000
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
    assert ('node_id' in nodes_h5['/nodes/NET1'])
    assert (len(nodes_h5['/nodes/NET1/node_id']) == 300)
    assert (len(nodes_h5['/nodes/NET1/node_type_id']) == 300)
    assert (len(nodes_h5['/nodes/NET1/node_group_id']) == 300)
    assert (len(nodes_h5['/nodes/NET1/node_group_index']) == 300)

    node_groups = {nid: grp for nid, grp in nodes_h5['/nodes/NET1'].items() if isinstance(grp, h5py.Group)}
    for grp in node_groups.values():
        if len(grp) == 1:
            assert ('position' in grp and len(grp['position']) == 200)

        elif len(grp) == 2:
            assert ('position' in grp and len(grp['position']) == 100)
            assert ('tags' in grp and len(grp['tags']) == 100)

        else:
            assert False

    assert(os.path.exists(edges_h5.name) and os.path.exists(edges_csv.name))
    edge_types_df = pd.read_csv(edges_csv.name, sep=' ')
    assert (len(edge_types_df) == 2)
    assert ('p1' in edge_types_df.columns)
    assert ('p2' in edge_types_df.columns)

    edges_h5 = h5py.File(edges_h5.name, 'r')
    assert('source_to_target' in edges_h5['/edges/NET1_to_NET1/indicies'])
    assert ('target_to_source' in edges_h5['/edges/NET1_to_NET1/indicies'])
    assert (len(edges_h5['/edges/NET1_to_NET1/target_node_id']) == 30000)
    assert (len(edges_h5['/edges/NET1_to_NET1/source_node_id']) == 30000)

    assert (edges_h5['/edges/NET1_to_NET1/target_node_id'][0] == 0)
    assert (edges_h5['/edges/NET1_to_NET1/source_node_id'][0] == 100)
    assert (edges_h5['/edges/NET1_to_NET1/edge_group_index'][0] == 0)
    assert (edges_h5['/edges/NET1_to_NET1/edge_type_id'][0] == 100)
    assert (edges_h5['/edges/NET1_to_NET1/0/nsyns'][0] == 1)

    assert (edges_h5['/edges/NET1_to_NET1/target_node_id'][29999] == 199)
    assert (edges_h5['/edges/NET1_to_NET1/source_node_id'][29999] == 99)
    assert (edges_h5['/edges/NET1_to_NET1/edge_group_id'][29999] == 0)
    assert (edges_h5['/edges/NET1_to_NET1/edge_type_id'][29999] == 101)
    assert (edges_h5['/edges/NET1_to_NET1/0/nsyns'][29999] == 2)

    #try:
    #    os.remove('tmp_nodes.h5')
    #    os.remove('tmp_node_types.csv')
    #    os.remove('tmp_edges.h5')
    #    os.remove('tmp_edge_types.csv')
    #except:
    #    pass


def test_save_weights():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='Scnna1', ei='e')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='PV1', ei='i')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, tags=np.linspace(0, 100, 100), cell_type='PV2', ei='i')
    cm = net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=lambda s, t: 3,
                       p1='e2i', p2='e2i')  # 200*100 = 60000 edges
    cm.add_properties(names=['segment', 'distance'], rule=lambda s, t: [1, 0.5], dtypes=[np.int, np.float])

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

    #try:
    #    os.remove('tmp_nodes.h5')
    #    os.remove('tmp_node_types.csv')
    #    os.remove('tmp_edges.h5')
    #    os.remove('tmp_edge_types.csv')
    #except:
    #    pass


def test_save_multinetwork():
    net1 = NetworkBuilder('NET1')
    net1.add_nodes(N=100, position=[(0.0, 1.0, -1.0)] * 100, cell_type='Scnna1', ei='e')
    net1.add_edges(source={'ei': 'e'}, target={'ei': 'e'}, connection_rule=5, ctype_1='n1_rec')
    net1.build()

    net2 = NetworkBuilder('NET2')
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


@pytest.mark.xfail
def test_save_multinetwork_1():
    net1 = NetworkBuilder('NET1')
    net1.add_nodes(N=100, position=[(0.0, 1.0, -1.0)] * 100, cell_type='Scnna1', ei='e')
    net1.add_edges(source={'ei': 'e'}, target={'ei': 'e'}, connection_rule=5, ctype_1='n1_rec')
    net1.build()

    net2 = NetworkBuilder('NET2')
    net2.add_nodes(N=10, position=[(0.0, 1.0, -1.0)] * 10, cell_type='PV1', ei='i')
    net2.add_edges(connection_rule=10, ctype_1='n2_rec')
    net2.add_edges(source=net1.nodes(), target={'ei': 'i'}, connection_rule=1, ctype_2='n1_n2')
    net2.add_edges(target=net1.nodes(cell_type='Scnna1'), source={'cell_type': 'PV1'}, connection_rule=2,
                   ctype_2='n2_n1')
    net2.build()
    net_dir = tempfile.mkdtemp()
    net2.save_edges(edges_file_name='NET2_NET1_edges.h5', edge_types_file_name='NET2_NET1_edge_types.csv',
                    output_dir=net_dir, src_network='NET2')

    n1_n2_fname = '{}/{}_{}'.format(net_dir, 'NET2', 'NET1')
    edges_h5 = h5py.File(n1_n2_fname + '_edges.h5', 'r')
    assert(len(edges_h5['/edges/NET2_to_NET1/target_node_id']) == 100*10)
    assert(len(edges_h5['/edges/NET2_to_NET1/0/nsyns']) == 100*10)
    assert(edges_h5['/edges/NET2_to_NET1/0/nsyns'][0] == 2)
    edge_types_csv = pd.read_csv(n1_n2_fname + '_edge_types.csv', sep=' ')
    assert(len(edge_types_csv) == 1)
    assert('ctype_1' not in edge_types_csv.columns.values)
    assert(edge_types_csv['ctype_2'].iloc[0] == 'n2_n1')


if __name__ == '__main__':
    #test_save_weights()
    test_save_multinetwork_1()
