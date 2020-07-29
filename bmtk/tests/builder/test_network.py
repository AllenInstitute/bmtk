import pytest
import numpy as np

from bmtk.builder import NetworkBuilder


def test_basic():
    net = NetworkBuilder('CA1')
    assert(net.name == 'CA1')
    assert(net.nnodes == 0)
    assert(net.nedges == 0)
    assert(net.nodes_built is False)
    assert(net.edges_built is False)

    assert(len(net.nodes()) == 0)
    assert(len(net.edges()) == 0)
    assert(net.nodes_built is True)
    assert(net.edges_built is True)


def test_no_name():
    # Fail if network name is invalid
    with pytest.raises(ValueError):
        NetworkBuilder(name='')

    with pytest.raises(ValueError):
        NetworkBuilder(name=None)


def test_add_nodes():
    # Tests that mutliple models of nodes can be added to network
    net = NetworkBuilder('V1')
    net.add_nodes(N=10, arg_list=range(10), arg_const='pop1', arg_shared='global')
    net.add_nodes(N=1, arg_list=[11], arg_const='pop2', arg_shared='global')
    net.add_nodes(N=5, arg_unique=range(12, 17), arg_const='pop3', arg_shared='global')  # diff param signature
    net.build()

    assert(net.nodes_built is True)
    assert(net.nnodes == 16)
    assert(net.nedges == 0)
    assert(len(net.nodes()) == 16)
    assert(len(net.nodes(arg_const='pop1')) == 10)
    assert(len(net.nodes(arg_const='pop2')) == 1)
    assert(len(net.nodes(arg_shared='global')) == 16)
    assert(len(net.nodes(arg_shared='invalid')) == 0)

    node_set = net.nodes(arg_list=2)
    assert(len(node_set) == 1)
    node = list(node_set)[0]
    assert(node['arg_const'] == 'pop1')
    assert(node['arg_shared'] == 'global')

    node_set = net.nodes(arg_unique=12)
    assert(len(node_set) == 1)
    node = list(node_set)[0]
    assert(node['arg_const'] == 'pop3')
    assert(node['arg_shared'] == 'global')
    assert('arg_list' not in node)


def test_add_nodes_tuples():
    # Should be able to store tuples of values in single parameters for a given node
    net = NetworkBuilder('V1')
    net.add_nodes(N=10,
                  arg_list=range(10),
                  arg_tuples=[(r, r+1) for r in range(10)],
                  arg_const=('a', 'b'))
    net.build()

    assert(net.nodes_built is True)
    assert(net.nnodes == 10)
    for node in net.nodes():
        assert(len(node['arg_tuples']) == 2)
        assert(node['arg_tuples'][0] == node['arg_list'] and node['arg_tuples'][1] == node['arg_list']+1)
        assert(len(node['arg_const']) == 2)
        assert(node['arg_const'][0] == 'a' and node['arg_const'][1] == 'b')


def test_add_nodes_ids():
    # Special case if parameters node_id and node_type_id are explicitly defined by the user
    net = NetworkBuilder('V1')
    net.add_nodes(N=3, node_id=[100, 200, 300], node_type_id=101, name=['one', 'two', 'three'])
    node_one = list(net.nodes(name='one'))[0]
    assert(node_one['name'] == 'one')
    assert(node_one['node_id'] == 100)
    assert(node_one['node_type_id'] == 101)

    node_three = list(net.nodes(name='three'))[0]
    assert(node_three['name'] == 'three')
    assert(node_three['node_id'] == 300)
    assert(node_three['node_type_id'] == 101)


def test_add_nodes_mismatch_params():
    # Should fail if a parameter list does not equal to the number of nodes in a group
    net = NetworkBuilder('V1')
    with pytest.raises(Exception):
        net.add_nodes(N=100, list1=[100]*99)


def test_build_nodes_id_clash():
    # Should fail if their is a node_type_id clash between two groups
    net = NetworkBuilder('V1')
    with pytest.raises(Exception):
        net.add_nodes(N=2, node_type_id=0)
        net.add_nodes(N=2, node_type_id=0)


def test_add_edges():
    net = NetworkBuilder('V1')
    net.add_nodes(N=10, cell_type='Scnna1', ei='e')
    net.add_nodes(N=10, cell_type='PV1', ei='i')
    net.add_nodes(N=10, cell_type='PV2', ei='i')
    net.add_edges(source={'ei': 'i'}, target={'ei': 'e'},
                  connection_rule=lambda s, t: 1,
                  edge_arg='i2e')
    net.add_edges(source=net.nodes(cell_type='Scnna1'), target=net.nodes(cell_type='PV1'),
                  connection_rule=2,
                  edge_arg='e2i')
    net.build()
    assert(net.nedges == 200 + 200)
    assert(net.edges_built is True)

    for e in net.edges(target_nodes=net.nodes(cell_type='Scnna1')):
        assert(e['edge_arg'] == 'i2e')
        assert(e['nsyns'] == 1)

    for e in net.edges(target_nodes=net.nodes(cell_type='PV1')):
        assert(e['edge_arg'] == 'e2i')
        assert(e['nsyns'] == 2)


def test_add_edges_custom_params():
    # Uses connection map functionality to create edges with unique parameters
    net = NetworkBuilder('V1')
    net.add_nodes(N=10, arg_list=range(10), arg_ctype='e')
    net.add_nodes(N=5, arg_list=range(10, 15), arg_ctype='i')

    cm = net.add_edges(source={'arg_ctype': 'e'}, target={'arg_ctype': 'i'}, connection_rule=1)
    cm.add_properties('syn_weight', rule=0.5, dtypes=np.float)
    cm.add_properties(['src_num', 'trg_num'],
                      rule=lambda s, t: [s['node_id'], t['node_id']],
                      dtypes=[np.int, np.int])
    net.build()

    assert(net.nedges == 50)
    assert(net.edges_built is True)

    for e in net.edges():
        assert(e['syn_weight'] == 0.5)
        assert(e['src_num'] == e.source_node_id)
        assert(e['trg_num'] == e.target_node_id)


def test_cross_pop_edges():
    # Uses connection map functionality to create edges with unique parameters
    net1 = NetworkBuilder('V1')
    net1.add_nodes(N=10, arg_list=range(10), arg_ctype='e')
    net1.build()

    net2 = NetworkBuilder('V2')
    net2.add_nodes(N=5, arg_list=range(10, 15), arg_ctype='i')

    net2.add_edges(source={'arg_ctype': 'i'}, target=net1.nodes(arg_ctype='e'),
                  connection_rule=lambda s, t: 1,
                  edge_arg='i2e')
    net2.build()
    assert(net2.nedges == 50)


if __name__ == '__main__':
    # test_basic()
    # test_no_name()
    # test_add_nodes()
    # test_add_nodes_tuples()
    # test_add_nodes_ids()
    # test_add_nodes_mismatch_params()
    # test_build_nodes_id_clash()
    # test_add_edges()
    # test_add_edges_custom_params()
    test_cross_pop_edges()

