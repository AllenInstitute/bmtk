import pytest
from bmtk.builder.node_set import NodeSet
from bmtk.builder.node import Node
from bmtk.builder.id_generator import IDGenerator


def test_node_set():
    generator = IDGenerator()
    node_set = NodeSet(N=100,
                       node_params={'p1': range(100), 'p2': range(0, 1000, 100)},
                       node_type_properties={'prop1': 'prop1', 'node_type_id': 1})
    nodes = node_set.build(generator)
    assert(len(nodes) == 100)
    assert(nodes[1]['p1'] == 1)
    assert(nodes[1]['p2'] == 100)
    assert(nodes[1]['prop1'] == 'prop1')
    assert(nodes[1]['node_type_id'] == 1)


def test_set_hash():
    node_set1 = NodeSet(N=100,
                        node_params={'param1': range(100)},
                        node_type_properties={'prop1': 'prop1', 'node_type_id': 1})
    node_set2 = NodeSet(N=100,
                        node_params = {'p1': range(100)},
                        node_type_properties={'prop1': 'prop2', 'node_type_id': 2})
    node_set3 = NodeSet(N=10,
                        node_params={'p1': ['hello']*10},
                        node_type_properties={'prop1': 'prop3', 'node_type_id': 3})

    assert(node_set1.params_hash != node_set2.params_hash)
    assert(node_set2.params_hash == node_set3.params_hash)


def test_node():
    node_set1 = NodeSet(N=100,
                        node_params={'param1': range(100)},
                        node_type_properties={'prop1': 'prop1', 'node_type_id': 1})
    nodes = node_set1.build(IDGenerator())
    node_1 = nodes[0]
    assert(node_1.node_id == 0)
    assert(node_1['node_id'] == 0)
    assert(node_1.node_type_id == 1)
    assert(node_1['node_type_id'] == 1)
    assert('prop1' in node_1.node_type_properties)
    assert('param1' in node_1.params)
    assert('node_id' in node_1.params)
    assert('param1' in node_set1.params_keys)
    assert(node_1.params_hash == node_set1.params_hash)


if __name__ == '__main__':
    test_node_set()
    #test_node()