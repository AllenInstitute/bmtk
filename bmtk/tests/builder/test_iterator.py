import pytest
import itertools

from bmtk.builder import connector, iterator
from bmtk.builder import NetworkBuilder
from bmtk.builder.node import Node


@pytest.fixture
def net():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, x=range(100), ei='i')
    net.add_nodes(N=50, x=range(50), y='y', ei='e')
    return net


def test_one2one_fnc(net):
    def connector_fnc(s, t):
        assert(s['ei'] == 'i')
        assert(t['ei'] == 'e')
        return '100'

    conr = connector.create(connector_fnc)
    itr = iterator.create('one_to_one', conr)
    count = 0
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id < 100)
        assert(trg_id >= 100)
        assert(val == '100')
        count += 1
    assert(count == 100*50)


def test_one2all_fnc(net):
    def connector_fnc(s, ts):
        assert(isinstance(s, Node))
        assert(s['ei'] == 'i')
        assert(len(ts) == 50)
        return [100]*50

    conr = connector.create(connector_fnc)
    itr = iterator.create('one_to_all', conr)
    count = 0
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id < 100)
        assert(trg_id >= 100)
        assert(val == 100)
        count += 1
    assert(count == 5000)


def test_all2one_fnc(net):
    def connector_fnc(ss, t):
        assert(isinstance(t, Node))
        assert(t['ei'] == 'e')
        assert(len(ss) == 100)
        return [100]*100

    conr = connector.create(connector_fnc)
    itr = iterator.create('all_to_one', conr)
    count = 0
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id < 100)
        assert(trg_id >= 100)
        assert(val == 100)
        count += 1
    assert(count == 5000)


def test_literal(net):
    conr = connector.create(100)
    itr = iterator.create('one_to_one', conr)
    count = 0
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id < 100)
        assert(trg_id >= 100)
        assert(val == 100)
        count += 1

    assert(count == 5000)


def test_dict(net):
    conr = connector.create({'nsyn': 10, 'target': 'axon'})
    itr = iterator.create('one_to_one', conr)
    count = 0
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id < 100)
        assert(trg_id >= 100)
        assert(val['nsyn'] == 10)
        assert(val['target'] == 'axon')
        count += 1

    assert (count == 5000)


def test_one2one_list(net):
    vals = [s.node_id*t.node_id for s,t in itertools.product(net.nodes(ei='i'), net.nodes(ei='e'))]
    conr = connector.create(vals)
    itr = iterator.create('one_to_one', conr)
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id*trg_id == val)


def test_one2all_list(net):
    vals = [v.node_id for v in net.nodes(ei='e')]
    conr = connector.create(vals)
    itr = iterator.create('one_to_all', conr)
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(trg_id == val)


def test_all2one_list(net):
    vals = [v.node_id for v in net.nodes(ei='i')]
    conr = connector.create(vals)
    itr = iterator.create('all_to_one', conr)
    for v in itr(net.nodes(ei='i'), net.nodes(ei='e'), conr):
        src_id, trg_id, val = v
        assert(src_id == val)
