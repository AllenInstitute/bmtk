import pytest

from bmtk.builder import NetworkBuilder


def test_single_node():
    net = NetworkBuilder('NET1')
    net.add_nodes(prop1='prop1', prop2='prop2', param1=['param1'])
    nodes = list(net.nodes())
    assert(len(nodes) == 1)
    assert(nodes[0]['param1'] == 'param1')
    assert(nodes[0]['prop1'] == 'prop1')
    assert(nodes[0]['prop2'] == 'prop2')


def test_node_set():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, prop1='prop1', param1=range(100))
    node_pool = net.nodes()
    assert(node_pool.filter_str == '*')

    nodes = list(node_pool)
    assert(len(nodes) == 100)
    assert(nodes[0]['prop1'] == 'prop1')
    assert(nodes[0]['param1'] == 0)
    assert(nodes[99]['prop1'] == 'prop1')
    assert(nodes[99]['param1'] == 99)
    assert(nodes[0]['node_type_id'] == nodes[99]['node_type_id'])
    assert(nodes[0]['node_id'] != nodes[99]['node_id'])


def test_node_sets():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, prop_n='prop1', pool1='p1', sp='sp', param1=range(100))
    net.add_nodes(N=100, prop_n='prop2', pool2='p2', sp='sp', param1=range(100))
    net.add_nodes(N=100, prop_n='prop3', pool3='p3', sp='sp', param1=range(100))
    node_pool_1 = net.nodes(prop_n='prop1')
    assert(len(node_pool_1) == 100)
    assert(node_pool_1.filter_str == "prop_n=='prop1'")
    for n in node_pool_1:
        assert('pool1' in n and n['prop_n'] == 'prop1')

    node_pool_2 = net.nodes(sp='sp')
    assert(node_pool_2.filter_str == "sp=='sp'")
    assert(len(node_pool_2) == 300)
    for n in node_pool_2:
        assert(n['sp'] == 'sp')

    node_pool_3 = net.nodes(param1=10)
    assert(len(node_pool_3) == 3)
    assert(node_pool_3.filter_str == "param1=='10'")
    nodes = list(node_pool_3)
    assert(nodes[0]['node_id'] == 10)
    assert(nodes[1]['node_id'] == 110)
    assert(nodes[2]['node_id'] == 210)
    assert(nodes[0]['node_type_id'] != nodes[1]['node_type_id'] != nodes[2]['node_type_id'])


def test_multi_search():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=10, prop_n='prop1', sp='sp1', param1=range(0, 10))
    net.add_nodes(N=10, prop_n='prop1', sp='sp2', param1=range(5, 15))
    net.add_nodes(N=20, prop_n='prop2', sp='sp2', param1=range(20))
    node_pool = net.nodes(prop_n='prop1', param1=5)
    assert(len(node_pool) == 2)
    nodes = list(node_pool)
    assert(nodes[0]['node_id'] == 5)
    assert(nodes[1]['node_id'] == 10)


def test_failed_search():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, p1='p1', q1=range(100))
    node_pool = net.nodes(p1='p2')
    assert(len(node_pool) == 0)

    node_pool = net.nodes(q2=10)
    assert(len(node_pool) == 0)


if __name__ == '__main__':
    test_multi_search()