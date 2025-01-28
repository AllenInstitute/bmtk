import pytest
from itertools import product

from bmtk.builder.connection_map import ConnectionMap
from bmtk.builder import NetworkBuilder


@pytest.fixture
def net():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, x=range(100), ei='i')
    net.add_nodes(N=50, x=range(50), y='y', ei='e')
    return net


def test_connection_map_fnc(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'),
                       connector=lambda s, t, a, b: s['node_id']*t['node_id'],
                       connector_params={'a': 1, 'b': 0}, iterator='one_to_one',
                       edge_type_properties={'prop1': 'prop1', 'edge_type_id': 101})
    assert(len(cm.source_nodes) == 100)
    assert(len(cm.target_nodes) == 50)
    assert(cm.params == [])
    assert(cm.iterator == 'one_to_one')
    assert(len(cm.edge_type_properties.keys()) == 2)
    assert(cm.edge_type_id == 101)
    for v in cm.connection_itr():
        src_id, trg_id, val = v
        assert(val == src_id*trg_id)


def test_connection_map_num(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'), connector=10)
    count = 0
    for v in cm.connection_itr():
        src_id, trg_id, val = v
        assert(val == 10)
        count += 1
    assert(count == 5000)


def test_connection_map_list(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'),
                       connector=[s.node_id*t.node_id for s, t in product(net.nodes(ei='i'), net.nodes(ei='e'))])
    count = 0
    for v in cm.connection_itr():
        src_id, trg_id, val = v
        assert(val == src_id*trg_id)
        count += 1
    assert(count == 5000)


def test_connection_map_dict(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'), connector={'nsyn': 10})
    for v in cm.connection_itr():
        src_id, trg_id, val = v
        assert('nsyn' in val and val['nsyn'] == 10)


def test_cm_params1(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'),
                       connector=lambda s, t: 3,
                       edge_type_properties={'prop1': 'prop1', 'edge_type_id': 101})
    cm.add_properties(names='syn_weight', rule=lambda a: a+0.15, rule_params={'a': 0.20}, dtypes=float)

    assert(len(cm.params) == 1)
    edge_props_1 = cm.params[0]
    assert(edge_props_1.names == 'syn_weight')
    assert(edge_props_1.get_prop_dtype('syn_weight') == float)
    for v in cm.connection_itr():
        src_id, trg_id, nsyn = v
        assert(nsyn == 3)
        assert(edge_props_1.rule() == 0.35)


def test_cm_params2(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'),
                       connector=lambda s, t: 3,
                       edge_type_properties={'prop1': 'prop1', 'edge_type_id': 101})
    cm.add_properties(names=['w', 'c'], rule=0.15, dtypes=[float, str])

    assert(len(cm.params) == 1)
    edge_props_1 = cm.params[0]
    assert(edge_props_1.names == ['w', 'c'])
    assert(edge_props_1.get_prop_dtype('w'))
    assert (edge_props_1.get_prop_dtype('c'))
    for v in cm.connection_itr():
        src_id, trg_id, nsyn = v
        assert(nsyn == 3)
        assert(edge_props_1.rule() == 0.15)


def test_cm_params3(net):
    cm = ConnectionMap(sources=net.nodes(ei='i'), targets=net.nodes(ei='e'),
                       connector=lambda s, t: 3,
                       edge_type_properties={'prop1': 'prop1', 'edge_type_id': 101})
    cm.add_properties(names=['w', 'c'], rule=0.15, dtypes=[float, str])
    cm.add_properties(names='a', rule=(1, 2, 3), dtypes=dict)

    assert(len(cm.params) == 2)
    edge_props_1 = cm.params[0]
    assert(edge_props_1.names == ['w', 'c'])
    assert(edge_props_1.get_prop_dtype('w'))
    assert(edge_props_1.get_prop_dtype('c'))
    for v in cm.connection_itr():
        src_id, trg_id, nsyn = v
        assert(nsyn == 3)
        assert(edge_props_1.rule() == 0.15)

    edge_props_2 = cm.params[1]
    assert(edge_props_2.names == 'a')
    assert(edge_props_2.get_prop_dtype('a'))
    assert(edge_props_2.rule() == (1, 2, 3))


if __name__ == '__main__':
    test_connection_map_fnc(net())