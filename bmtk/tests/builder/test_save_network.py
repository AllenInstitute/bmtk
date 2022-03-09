import pytest
import os
import tempfile
import h5py
import numpy as np
import pandas as pd

from bmtk.builder import NetworkBuilder

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    bcast = comm.bcast
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    barrier = comm.Barrier
except ImportError:
    mpi_rank = 0
    mpi_size = 1
    barrier = lambda: None


def make_tmp_dir():
    tmp_dir = tempfile.mkdtemp() if mpi_rank == 0 else None
    if mpi_size > 1:
        tmp_dir = comm.bcast(tmp_dir, 0)
    return tmp_dir


def make_tmp_file(suffix):
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix).name if mpi_rank == 0 else None
    if mpi_size > 1:
        tmp_file = comm.bcast(tmp_file, 0)
    return tmp_file


def test_basic():
    tmp_dir = make_tmp_dir()
    nodes_file = make_tmp_file(suffix='.h5')
    node_types_file = make_tmp_file(suffix='.csv')
    edges_file = make_tmp_file(suffix='.h5')
    edge_types_file = make_tmp_file(suffix='.csv')

    net = NetworkBuilder('test')
    net.add_nodes(N=100, a=np.arange(100), b='B')
    net.add_edges(
        source={'a': 0},
        target=net.nodes(),
        connection_rule=2,
        x='X'
    )

    net.build()
    net.save_nodes(
        nodes_file_name=nodes_file,
        node_types_file_name=node_types_file,
        output_dir=tmp_dir
    )
    net.save_edges(
        edges_file_name=edges_file,
        edge_types_file_name=edge_types_file,
        output_dir=tmp_dir,
        name='test_test'
    )

    nodes_h5_path = os.path.join(tmp_dir, nodes_file)
    assert(os.path.exists(nodes_h5_path))
    with h5py.File(nodes_h5_path, 'r') as h5:
        assert('/nodes/test' in h5)
        assert(len(h5['/nodes/test/node_id']) == 100)
        assert(len(h5['/nodes/test/node_type_id']) == 100)
        assert('/nodes/test/node_group_id' in h5)
        assert('/nodes/test/node_group_index' in h5)
        assert(len(h5['/nodes/test/0/a']) == 100)

    node_types_csv_path = os.path.join(tmp_dir, node_types_file)
    assert(os.path.exists(node_types_csv_path))
    node_types_df = pd.read_csv(node_types_csv_path, sep=' ')
    assert(len(node_types_df) == 1)
    assert('node_type_id' in node_types_df.columns)
    assert('b' in node_types_df.columns)

    edges_h5_path = os.path.join(tmp_dir, edges_file)
    assert(os.path.exists(edges_h5_path))
    with h5py.File(edges_h5_path, 'r') as h5:
        assert('/edges/test_test' in h5)
        assert(len(h5['/edges/test_test/target_node_id']) == 100)
        assert(h5['/edges/test_test/target_node_id'].attrs['node_population'] == 'test')
        assert(set(h5['/edges/test_test/target_node_id'][()]) == set(range(100)))

        assert(len(h5['/edges/test_test/source_node_id']) == 100)
        assert (h5['/edges/test_test/source_node_id'].attrs['node_population'] == 'test')
        assert(all(np.unique(h5['/edges/test_test/source_node_id'][()] == [0])))

        assert (h5['/edges/test_test/source_node_id'].attrs['node_population'] == 'test')
        assert(len(h5['/edges/test_test/edge_type_id']) == 100)
        assert('/edges/test_test/edge_group_id' in h5)
        assert('/edges/test_test/edge_group_index' in h5)
        assert(len(h5['/edges/test_test/0/nsyns']) == 100)

    edge_type_csv_path = os.path.join(tmp_dir, edge_types_file)
    assert(os.path.exists(edge_type_csv_path))
    edge_types_df = pd.read_csv(edge_type_csv_path, sep=' ')
    assert(len(edge_types_df) == 1)
    assert('edge_type_id' in edge_types_df.columns)
    assert('x' in edge_types_df.columns)

    barrier()


def test_multi_node_models():
    tmp_dir = make_tmp_dir()
    nodes_file = make_tmp_file(suffix='.h5')
    node_types_file = make_tmp_file(suffix='.csv')

    net = NetworkBuilder('test')
    net.add_nodes(N=10, x=np.arange(10), common=range(10), model='A', p='X')
    net.add_nodes(N=10, x=np.arange(10), common=range(10), model='B')
    net.add_nodes(N=20, y=np.arange(20), common=range(20), model='C', p='X')
    net.add_nodes(N=20, y=np.arange(20), common=range(20), model='D')
    net.add_nodes(N=30, z=np.arange(30), common=range(30), model='E')
    net.build()
    net.save_nodes(
        nodes_file_name=nodes_file,
        node_types_file_name=node_types_file,
        output_dir=tmp_dir
    )

    nodes_h5_path = os.path.join(tmp_dir, nodes_file)
    assert(os.path.exists(nodes_h5_path))
    with h5py.File(nodes_h5_path, 'r') as h5:
        assert('/nodes/test' in h5)
        assert(len(h5['/nodes/test/node_id']) == 90)
        assert(len(h5['/nodes/test/node_type_id']) == 90)
        assert(len(np.unique(h5['/nodes/test/node_type_id'])) == 5)
        assert(len(h5['/nodes/test/node_group_id']) == 90)
        assert(len(np.unique(h5['/nodes/test/node_group_id'])) == 3)
        assert(len(h5['/nodes/test/node_group_index']) == 90)

        for grp_id, grp in h5['/nodes/test'].items():
            if not isinstance(grp, h5py.Group):
                continue
            assert('common' in grp)
            assert(int('x' in grp) + int('y' in grp) + int('z' in grp) == 1)

    node_types_csv_path = os.path.join(tmp_dir, node_types_file)
    assert (os.path.exists(node_types_csv_path))
    node_types_df = pd.read_csv(node_types_csv_path, sep=' ')
    assert(len(node_types_df) == 5)
    assert('node_type_id' in node_types_df.columns)
    assert('model' in node_types_df.columns)
    assert('p' in node_types_df.columns)

    barrier()


def test_edge_models():
    tmp_dir = tempfile.mkdtemp()
    edges_file = make_tmp_file(suffix='.h5')
    edge_types_file = make_tmp_file(suffix='.csv')

    net = NetworkBuilder('test')
    net.add_nodes(N=100, x=range(100), model='A')
    net.add_nodes(N=100, x=range(100, 200), model='B')
    net.add_edges(source={'model': 'A'}, target={'model': 'B'}, connection_rule=1, model='A')
    net.add_edges(source={'model': 'A'}, target={'x': 0}, connection_rule=2, model='B')
    net.add_edges(source={'model': 'A'}, target={'x': [1, 2, 3]}, connection_rule=3, model='C')
    net.add_edges(source={'model': 'A', 'x': 0}, target={'model': 'B', 'x': 100}, connection_rule=4, model='D')
    net.build()
    net.save_edges(
        edges_file_name=edges_file,
        edge_types_file_name=edge_types_file,
        output_dir=tmp_dir,
        name='test_test'
    )

    edges_h5_path = os.path.join(tmp_dir, edges_file)
    assert(os.path.exists(edges_h5_path))
    with h5py.File(edges_h5_path, 'r') as h5:
        n_edges = 100*100 + 100*1 + 100*3 + 1
        assert('/edges/test_test' in h5)
        assert(len(h5['/edges/test_test/target_node_id']) == n_edges)
        assert(h5['/edges/test_test/target_node_id'].attrs['node_population'] == 'test')
        assert(len(h5['/edges/test_test/source_node_id']) == n_edges)
        assert(h5['/edges/test_test/source_node_id'].attrs['node_population'] == 'test')
        assert(len(h5['/edges/test_test/edge_type_id']) == n_edges)
        assert(len(h5['/edges/test_test/edge_group_id']) == n_edges)
        assert(len(h5['/edges/test_test/edge_group_index']) == n_edges)

        assert(len(np.unique(h5['/edges/test_test/edge_type_id'])) == 4)
        assert(len(np.unique(h5['/edges/test_test/edge_group_id'])) == 1)
        grp_id = str(h5['/edges/test_test/edge_group_id'][0])
        assert(len(h5['/edges/test_test'][grp_id]['nsyns']) == n_edges)

    edge_type_csv_path = os.path.join(tmp_dir, edge_types_file)
    assert(os.path.exists(edge_type_csv_path))
    edge_types_df = pd.read_csv(edge_type_csv_path, sep=' ')
    assert(len(edge_types_df) == 4)
    assert('edge_type_id' in edge_types_df.columns)
    assert('model' in edge_types_df.columns)

    barrier()


def test_connection_map():
    tmp_dir = tempfile.mkdtemp()
    edges_file = make_tmp_file(suffix='.h5')
    edge_types_file = make_tmp_file(suffix='.csv')

    net = NetworkBuilder('test')
    net.add_nodes(N=10, x=range(10), model='A')
    net.add_nodes(N=20, x=range(10, 30), model='B')

    net.add_edges(source={'model': 'A'}, target={'model': 'B'}, connection_rule=1, edge_model='A')

    cm = net.add_edges(source={'model': 'B'}, target={'model': 'B'}, connection_rule=2, edge_model='B')
    cm.add_properties(names='a', rule=5, dtypes=int)

    cm = net.add_edges(source={'model': 'B'}, target={'x': 0}, connection_rule=3, edge_model='C')
    cm.add_properties(names='b', rule=0.5, dtypes=float)
    cm.add_properties(names='c', rule=lambda *_: 2, dtypes=int)

    net.build()
    net.save_edges(
        edges_file_name=edges_file,
        edge_types_file_name=edge_types_file,
        output_dir=tmp_dir,
        name='test_test'
    )

    edges_h5_path = os.path.join(tmp_dir, edges_file)
    assert(os.path.exists(edges_h5_path))
    with h5py.File(edges_h5_path, 'r') as h5:
        n_edges = 10*20*1 + 20*20*2 + 20*1*3
        assert('/edges/test_test' in h5)
        assert(len(h5['/edges/test_test/target_node_id']) == n_edges)
        assert(h5['/edges/test_test/target_node_id'].attrs['node_population'] == 'test')
        assert(len(h5['/edges/test_test/source_node_id']) == n_edges)
        assert(h5['/edges/test_test/source_node_id'].attrs['node_population'] == 'test')
        assert(len(h5['/edges/test_test/edge_type_id']) == n_edges)
        assert(len(h5['/edges/test_test/edge_group_id']) == n_edges)
        assert(len(h5['/edges/test_test/edge_group_index']) == n_edges)
        assert(len(np.unique(h5['/edges/test_test/edge_type_id'])) == 3)
        assert(len(np.unique(h5['/edges/test_test/edge_group_id'])) == 3)

        for grp_id, grp in h5['/edges/test_test'].items():
            if not isinstance(grp, h5py.Group) or grp_id in ['indicies', 'indices']:
                continue
            assert(int('nsyns' in grp) + int('a' in grp) + int('c' in grp and 'c' in grp) == 1)

    edge_type_csv_path = os.path.join(tmp_dir, edge_types_file)
    assert(os.path.exists(edge_type_csv_path))
    edge_types_df = pd.read_csv(edge_type_csv_path, sep=' ')
    assert(len(edge_types_df) == 3)
    assert('edge_type_id' in edge_types_df.columns)
    assert('edge_model' in edge_types_df.columns)

    barrier()


def test_cross_population_edges():
    tmp_dir = make_tmp_dir()
    edges_file = make_tmp_file(suffix='.h5')
    edge_types_file = make_tmp_file(suffix='.csv')

    net_a1 = NetworkBuilder('A1')
    net_a1.add_nodes(N=100, model='A')
    net_a1.build()

    net_a2 = NetworkBuilder('A2')
    net_a2.add_nodes(N=100, model='B')
    net_a2.add_edges(
        source=net_a1.nodes(),
        target=net_a2.nodes(),
        connection_rule=lambda s, t: 1 if s.node_id == t.node_id else 0
    )
    net_a2.build()
    net_a2.save_edges(
        edges_file_name=edges_file,
        edge_types_file_name=edge_types_file,
        output_dir=tmp_dir,
        name='A1_A2'
    )

    edges_h5_path = os.path.join(tmp_dir, edges_file)
    assert(os.path.exists(edges_h5_path))
    with h5py.File(edges_h5_path, 'r') as h5:
        assert('/edges/A1_A2' in h5)
        assert(len(h5['/edges/A1_A2/source_node_id']) == 100)
        assert(h5['/edges/A1_A2/source_node_id'].attrs['node_population'] == 'A1')
        assert(len(h5['/edges/A1_A2/target_node_id']) == 100)
        assert(h5['/edges/A1_A2/target_node_id'].attrs['node_population'] == 'A2')

    barrier()


if __name__ == '__main__':
    # test_basic()
    # test_multi_node_models()
    # test_edge_models()
    # test_connection_map()
    test_cross_population_edges()
