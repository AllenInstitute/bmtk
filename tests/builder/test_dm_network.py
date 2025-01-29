import os
import pytest
import tempfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd

# from bmtk.builder import NetworkBuilder
from bmtk.builder.network_adaptors import DenseNetwork


def test_empty_network():
    net = DenseNetwork(name='test')
    assert(net.name == 'test')
    assert(net.nodes_built == False)
    assert(net.edges_built == False)
    net.build()
    net.save(output_dir=tempfile.gettempdir())

    assert(net.nodes_built == True)
    assert(net.edges_built == True)
    assert(net.nnodes == 0)
    assert(net.nedges == 0)
    assert(net.get_connections() == [])

    with pytest.raises(ValueError):
        net = DenseNetwork(name='')

    with pytest.raises(ValueError):
        net = DenseNetwork(name=None)

    net.clear()
    assert(net.nodes_built == False)
    assert(net.edges_built == False)


def test_set_nsyns_const():
    net = DenseNetwork(name='test')
    net.add_nodes(N=10)
    net.add_edges(
        source=net.nodes(), 
        target=net.nodes(),
        nsyns=3
    )
    net.build()
    edges = net.edges()
    assert(len(edges) == 100)
    for edge in edges:
        assert(edge['nsyns'] == 3)

def test_set_nsyns_list():
    net = DenseNetwork(name='test')
    net.add_nodes(N=10)
    net.add_edges(
        source=net.nodes(), 
        target=net.nodes(),
        nsyns=list(range(1, 101))
    )
    net.build()
    edges = net.edges()
    assert(len(edges) == 100)
    nsyns_table = [e['nsyns'] for e in edges]
    nsyns_table.sort()
    assert(nsyns_table == list(range(1, 101)))

    # for edge in edges:
    #     assert(edge['nsyns'])


def test_add_edges_basic():
    net = DenseNetwork(name='test')
    net.add_nodes(N=10, attr1=np.arange(0, 10), prop1='A')
    net.add_nodes(N=10, attr2=np.arange(0, 10), prop1='B')
    net.add_edges(
        source={'node_id': 1},
        target=net.nodes(node_id=2),
        connection_rule=2
    )
    net.build()
    assert(len(net.edges()) == 1)

    net = DenseNetwork(name='test')
    net.add_nodes(N=10, attr1=np.arange(0, 10), prop1='A')
    net.add_nodes(N=10, attr2=np.arange(0, 10), prop1='B')
    net.build()
    assert(len(net.edges()) == 0)
    net.add_edges(
        source={'node_id': [0, 1, 2, 3]},
        target={'node_id': [4, 5, 6]},
        connection_rule=2
    )
    
    net.build(force=True)
    assert(len(net.edges()) == 12)

    net = DenseNetwork(name='test')
    net.add_nodes(N=10, attr1=np.arange(0, 10), prop1='A')
    net.add_nodes(N=10, attr2=np.arange(0, 10), prop1='B')
    net.add_edges(
        source={'prop1': 'A'},
        target=net.nodes(prop1='B'),
        connection_rule=2
    )
    net.build()
    assert(len(net.edges()) == 100)


def test_bad_add_nodes():
    # net = DenseNetwork(name='test')
    # net.add_nodes(N=100, prop1=np.arange(101))

    net = DenseNetwork(name='test')
    with pytest.raises(Exception):
        net.add_nodes(N=100, prop1=np.arange(101))

    net = DenseNetwork(name='test')
    with pytest.raises(Exception):
        net.add_nodes(N=100, prop1=range(101))

    net = DenseNetwork(name='test')
    net.add_nodes(node_type_id=100)
    with pytest.raises(Exception):
        net.add_nodes(node_type_id=100)

    net = DenseNetwork(name='test')
    with pytest.raises(Exception):
        net.add_nodes(N=10, node_type_id=range(10))

    net = DenseNetwork(name='test')
    net.add_nodes(N=10, node_id=range(10))
    with pytest.raises(Exception):
        net.add_nodes(N=10, node_id=[0]*10)

def test_bad_add_edges():
    net = DenseNetwork(name='test')
    net.add_nodes(N=10, node_id=range(10))
    net.add_edges(edge_type_id=100)
    with pytest.raises(Exception):
        net.add_edges(edge_type_id=100)


def test_edges_itr():
    net = DenseNetwork(name='test')
    net.add_nodes(N=10, attr1=range(10), prop1='A')
    net.add_nodes(N=10, attr1=range(10, 20), prop1='B')
    net.add_edges(
        source=net.nodes(),
        target=net.nodes(),
        connection_rule=5,
        eattr='A'
    )
    net.add_edges(
        source={'prop1': 'A'},
        target={'prop1': 'B'},
        connection_rule=10,
        eattr='B'
    )
    # net.build() # By not calling build() explcity we can test to make sure it will get built anyways
    assert(len(net.edges()) == 500)
    assert(len(net.edges(target_nodes=[0])) == 20)
    assert(len(net.edges(source_nodes=0)) == 30)
    assert(len(net.edges(target_nodes={'prop1': 'B'})) == 300)
    assert(len(net.edges(target_nodes=net.nodes(prop1='B'))) == 300)
    assert(len(net.edges(target_nodes={'prop1': 'C'})) == 0)
    assert(net.edges(target_nodes=net.nodes(), target_network='test2') is not None)

    with pytest.raises(Exception):
        net.edges(source_nodes='Nonsense')

    # print(len(net.edges()) == 500)

    assert(len(net.edges(eattr='A')) == 400)
    assert(len(net.edges(nonsense='A')) == 0)
    assert(len(net.edges(eattr='C')) == 0)


def test_save_opts():
    net_dir = tempfile.mkdtemp()
    net = DenseNetwork('test')
    net.add_nodes(N=10, attr1=[1.0]*10, attr2=range(10))
    
    # net.build()  # save_nodes() should be able to build nodes automatically when first called
    net.save_nodes(output_dir=net_dir)  
    net.save_nodes(output_dir=Path(net_dir) / 'non_compressed', compression='none')

    net.save_nodes(output_dir=Path(net_dir) / 'brand_new_dir')
    assert((Path(net_dir) / 'test_nodes.h5').exists())
    assert((Path(net_dir) / 'test_node_types.csv').exists()) 
    
    net.save_nodes(output_dir=Path(net_dir), nodes_file_name='h5/nodes.h5', node_types_file_name='csv/node_types.csv')
    assert((Path(net_dir) / 'h5' / 'nodes.h5').exists())
    assert((Path(net_dir) / 'csv' / 'node_types.csv').exists())

    net.save_nodes(nodes_file_name='a.h5', node_types_file_name='a.csv', output_dir=net_dir)
    assert((Path(net_dir) / 'a.h5').exists())
    assert((Path(net_dir) / 'a.csv').exists())

    net.save_nodes(nodes_file_name=Path(net_dir) / 'b.h5', node_types_file_name=Path(net_dir) / 'b.csv')
    assert((Path(net_dir) / 'b.h5').exists())
    assert((Path(net_dir) / 'b.csv').exists())

    with pytest.raises(Exception):
        net.save_nodes(
            nodes_file_name=Path(net_dir) / 'b.h5', 
            node_types_file_name=Path(net_dir) / 'c.csv',
            force_overwrite=False
        )

    with pytest.raises(Exception):
        net.save_nodes(
            nodes_file_name=Path(net_dir) / 'c.h5', 
            node_types_file_name=Path(net_dir) / 'b.csv',
            force_overwrite=False
        )

    net.save_nodes(nodes_file_name=Path(net_dir) / 'b.h5', node_types_file_name=Path(net_dir) / 'b.csv', force_overwrite=True)

    net = DenseNetwork('test')
    net.add_nodes(N=10, attr1=[1.0]*10, attr2=range(10))
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=1
    )
    net.save_edges(output_dir=net_dir, force_build=False)
    assert(not (Path(net_dir) / 'test_test_edges.h5').exists())
    assert(not (Path(net_dir) / 'test_test_edge_types.csv').exists())

    net.save_edges(output_dir=net_dir, force_build=True)
    assert((Path(net_dir) / 'test_test_edges.h5').exists())
    assert((Path(net_dir) / 'test_test_edge_types.csv').exists()) 

    net = DenseNetwork('test3')
    net.add_nodes(N=10, attr1=[1.0]*10, attr2=range(10))
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=1,
    )
    net.build()
    net.save_edges(output_dir=Path(net_dir) / 'edges', src_network='test3', edges_file_name='myedges.h5')
    assert((Path(net_dir) / 'edges' / 'myedges.h5').exists())
    net.save_edges(output_dir=net_dir, trg_network='test4', force_overwrite=True)


@pytest.mark.parametrize('compression', ['gzip', 'none', None])
def test_compression_opts(compression):
    net_dir = tempfile.mkdtemp()
    
    net = DenseNetwork('test')
    net.add_nodes(N=10, attr1=[1.0]*10, attr2=range(10))
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=1,
    )
    net.save_edges(output_dir=Path(net_dir), compression=compression)
    with h5py.File(Path(net_dir) / 'test_test_edges.h5', 'r') as edges_h5:
        assert(edges_h5['/edges/test_to_test/source_node_id'].shape == (100, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (100, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (100, ))
        assert(edges_h5['/edges/test_to_test/edge_type_id'].shape == (100, ))



def test_add_gap_junction():
    net_dir = tempfile.mkdtemp()

    net = DenseNetwork('test')
    net.add_nodes(N=2)
    net.add_gap_junctions(source=net.nodes(node_id=0), target=net.nodes(node_id=1))

    net2 = DenseNetwork('test2')
    net2.add_nodes(N=1)
    with pytest.raises(Exception):
        net2.add_gap_junctions(source=net2.nodes(node_id=0), target=net.nodes(node_id=1))

    net.save(output_dir=net_dir)


def test_dm_base():
    net = DenseNetwork('test')
    net.add_nodes(N=10, attr1=[1.0]*10, attr2=range(10))
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=1,
    )
    print(len(net.edges_table()) == 1)
    # net.build()
    # net.save_nodes()


def test_basic_one_to_one():
    net_dir = tempfile.mkdtemp()
    net = DenseNetwork('test')
    net.add_nodes(N=10, attr1=[1.0]*10, attr2=range(10))
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=lambda s, t: s.node_id*t.node_id+1,
        iterator='one_to_one'
    )
    net.build()
    net.save(output_dir=net_dir)

    assert(Path(net_dir).exists())
    assert((Path(net_dir) / 'test_nodes.h5').exists())
    assert((Path(net_dir) / 'test_node_types.csv').exists())
    assert((Path(net_dir) / 'test_test_edges.h5').exists())
    assert((Path(net_dir) / 'test_test_edge_types.csv').exists())
    
    with h5py.File(Path(net_dir) / 'test_test_edges.h5', 'r') as edges_h5:
        # assert(list(edges_h5['/edges/test_to_test'].keys()))
        assert(edges_h5['/edges/test_to_test/source_node_id'].shape == (10*10, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (10*10, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (10*10, ))
        assert(edges_h5['/edges/test_to_test/edge_type_id'].shape == (10*10, ))


def test_basic_one_to_all():
    net_dir = tempfile.mkdtemp()

    net = DenseNetwork('test')
    net.add_nodes(N=10)
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=lambda s, tt: [s.node_id*t.node_id+1 for t in tt],
        iterator='one_to_all'
    )
    net.build()   
    net.save(output_dir=net_dir)

    assert(Path(net_dir).exists())
    assert((Path(net_dir) / 'test_nodes.h5').exists())
    assert((Path(net_dir) / 'test_node_types.csv').exists())
    assert((Path(net_dir) / 'test_test_edges.h5').exists())
    assert((Path(net_dir) / 'test_test_edge_types.csv').exists())

    with h5py.File(Path(net_dir) / 'test_test_edges.h5', 'r') as edges_h5:
        assert(edges_h5['/edges/test_to_test/source_node_id'].shape == (10*10, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (10*10, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (10*10, ))
        assert(edges_h5['/edges/test_to_test/edge_type_id'].shape == (10*10, ))


def test_save_nsyn_table():
    np.random.seed(100)
    net = DenseNetwork('NET1')
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

    # Check edges and node ids match up
    # warning, builder may not build edges in sequential order
    nid_idxs = np.sort(np.argwhere(edges_h5['/edges/NET1_to_NET1/target_node_id'][()] == 0).flatten())
    trg_ids = edges_h5['/edges/NET1_to_NET1/source_node_id'][nid_idxs]
    assert(np.all(trg_ids >= 10))
    assert(np.all(30 > trg_ids))
    edge_0 = nid_idxs[0]
    assert(edges_h5['/edges/NET1_to_NET1/edge_type_id'][edge_0] == 100)
    edge_id = edges_h5['/edges/NET1_to_NET1/edge_group_id'][edge_0]
    edge_idx = edges_h5['/edges/NET1_to_NET1/edge_group_index'][edge_0]
    assert(edges_h5['/edges/NET1_to_NET1'][str(edge_id)]['nsyns'][edge_idx] == 1)

    nid_idxs = np.sort(np.argwhere(edges_h5['/edges/NET1_to_NET1/target_node_id'][()] == 19).flatten())
    trg_ids = edges_h5['/edges/NET1_to_NET1/source_node_id'][nid_idxs]
    print(trg_ids)
    assert(np.all(trg_ids >= 0))
    assert(np.all(10 > trg_ids))
    edge_0 = nid_idxs[0]
    assert(edges_h5['/edges/NET1_to_NET1/edge_type_id'][edge_0] == 101)
    edge_id = edges_h5['/edges/NET1_to_NET1/edge_group_id'][edge_0]
    edge_idx = edges_h5['/edges/NET1_to_NET1/edge_group_index'][edge_0]
    assert(edges_h5['/edges/NET1_to_NET1'][str(edge_id)]['nsyns'][edge_idx] == 2)


def test_save_weights():
    net = DenseNetwork('NET1')
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


def test_save_multinetwork():
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


def test_multigroup_one_to_one():
    net_dir = tempfile.mkdtemp()

    net = DenseNetwork('test')
    net.add_nodes(N=10, attr1=[1]*10, prop1='A')
    net.add_nodes(N=10, attr1=[2]*10, prop1='B')
    net.add_nodes(N=10, attr1=[3]*10, attr2=[4.0]*10, prop1='C', prop2='C')
    net.add_edges(
        source={'prop1': 'A'}, target={'prop1': 'B'},
        connection_rule=lambda src, trg: 2,
        iterator='one_to_one',
        prop1='EP1'
    )
    net.add_edges(
        source={'prop1': 'A'}, target={'prop1': 'C'},
        connection_rule=lambda src, trg: 2,
        iterator='one_to_one',
        prop1='EP2'
    )
    cm = net.add_edges(
        source={'prop1': 'B'}, target={'prop1': 'C'},
        connection_rule=lambda src, trg: 2,
        iterator='one_to_one',
        prop1='EP3'
    )
    cm.add_properties(names='attr1', rule=lambda *_: 2.0, dtypes=float)

    net.build()
    net.save(output_dir=net_dir)
    
    with h5py.File(Path(net_dir) / 'test_nodes.h5', 'r') as nodes_h5:
        assert(list(nodes_h5['/nodes/test'].keys()))
        assert(nodes_h5['/nodes/test/node_id'][()].shape == (30, ))
        assert(nodes_h5['/nodes/test/node_id'][()].shape == (30, ))
        assert(np.unique(nodes_h5['/nodes/test/node_group_id'][()]).shape == (2, ))

    node_types_df = pd.read_csv(Path(net_dir) / 'test_node_types.csv', sep=' ', na_values='NULL')
    assert(node_types_df.shape == (3, 3))

    with h5py.File(Path(net_dir) / 'test_test_edges.h5', 'r') as edges_h5:
        assert(edges_h5['/edges/test_to_test/source_node_id'].shape == (400,))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (400, ))
        assert(edges_h5['/edges/test_to_test/target_node_id'].shape == (400, ))
        assert(edges_h5['/edges/test_to_test/edge_type_id'].shape == (400, ))
        assert(np.unique(edges_h5['/edges/test_to_test/edge_group_id'][()]).shape == (2, ))

    edge_types_df = pd.read_csv(Path(net_dir) / 'test_test_edge_types.csv', sep=' ', na_values='NULL')
    assert(edge_types_df.shape == (3, 4))


def test_import_nodes():
    net_dir = tempfile.mkdtemp()
    
    net_out = DenseNetwork('test')
    net_out.add_nodes(N=10, attr1=[1]*10, prop1='A')
    net_out.add_edges(source=net_out.nodes(), target=net_out.nodes(), connection_rule=1)
    net_out.build()
    net_out.save(output_dir=net_dir)

    net_in = DenseNetwork('test_in')
    net_in.import_nodes(
        nodes_file_name=Path(net_dir) / 'test_nodes.h5',
        node_types_file_name=Path(net_dir) / 'test_node_types.csv'
    )
    assert(len(net_in.nodes()) == 10)

    net_out = DenseNetwork('test2')
    net_out.add_nodes(N=10, attr1=[1]*10, prop1='A')
    net_out.save_nodes(output_dir=net_dir, nodes_file_name='multi_nodes.h5', mode='a')
    net_out = DenseNetwork('test3')
    net_out.add_nodes(N=5, attr1=[1]*5, prop1='A')
    net_out.save_nodes(output_dir=net_dir, nodes_file_name='multi_nodes.h5', mode='a')

    net_in = DenseNetwork('test_in')

    with pytest.raises(Exception):
        net_in.import_nodes(
            nodes_file_name=Path(net_dir) / 'test_test_edges.h5',
            node_types_file_name=Path(net_dir) / 'test_test_edge_types.csv'
        )

    with pytest.raises(Exception):
        net_in.import_nodes(
            nodes_file_name=Path(net_dir) / 'multi_nodes.h5',
            node_types_file_name=Path(net_dir) / 'test2_node_types.csv'
        )

    with pytest.raises(Exception):
        net_in.import_nodes(
            nodes_file_name=Path(net_dir) / 'multi_nodes.h5',
            node_types_file_name=Path(net_dir) / 'test2_node_types.csv',
            population='test_bad_pop'
        )

    net_in.import_nodes(
        nodes_file_name=Path(net_dir) / 'multi_nodes.h5',
        node_types_file_name=Path(net_dir) / 'test2_node_types.csv',
        population='test3'
    )
    assert(len(net_in.nodes()) == 5)

@pytest.mark.parametrize('on_disk', [True, False])
def test_sort_edges(on_disk):
    net_dir = tempfile.mkdtemp()
    net = DenseNetwork('test')
    net.add_nodes(N=10)
    net.add_edges(
        sources=net.nodes(),
        targets=net.nodes(),
        connection_rule=10,
        iterator='one_to_one'
    )
    net.build()
    net.save(output_dir=Path(net_dir) / 'mem_sorted', sort_on_disk=on_disk)

    # net.save(output_dir=Path(net_dir) / 'disk_sorted', sort_on_disk=True)

def test_no_edges():
    net_dir = tempfile.mkdtemp()
    
    net = DenseNetwork('test')
    net.add_nodes(N=10, prop1='A')
    net.add_edges(
        source={'prop1': 'B'}, target={'prop1': 'C'},
        connection_rule=1
    )
    net.build()
    net.save(output_dir=net_dir)


if __name__ == '__main__':
    # test_empty_network()
    # test_bad_add_nodes()
    # test_bad_add_edges()
    # test_set_nsyns_const()
    # test_set_nsyns_list()
    # test_basic()
    # test_edges_itr()
    # test_basic_one_to_one()
    # test_add_edges_basic()
    test_save_opts()
    # test_dm_base()
    # test_multigroup_one_to_one()
    # test_import_nodes()
    # test_sort_edges()