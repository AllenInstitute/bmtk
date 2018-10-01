import pytest

from bmtk.builder import NetworkBuilder

def test_itr_basic():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='Scnna1', ei='e')
    net.add_nodes(N=100, position=[(0.0, 1.0, -1.0)]*100, cell_type='PV1', ei='i')
    net.add_edges(source={'ei': 'e'}, target={'ei': 'i'}, connection_rule=5, syn_type='e2i')
    net.add_edges(source={'cell_type': 'PV1'}, target={'cell_type': 'Scnna1'}, connection_rule=5, syn_type='i2e')
    net.build()

    edges = net.edges()
    assert(len(edges) == 100*100*2)
    assert(edges[0]['nsyns'] == 5)


def test_itr_advanced_search():
    net = NetworkBuilder('NET1')
    net.add_nodes(N=1, cell_type='Scnna1', ei='e')
    net.add_nodes(N=50, cell_type='PV1', ei='i')
    net.add_nodes(N=100, cell_type='PV2', ei='i')
    net.add_edges(source={'ei': 'e'}, target={'ei': 'i'}, connection_rule=5, syn_type='e2i', nm='A')
    net.add_edges(source={'cell_type': 'PV1'}, target={'cell_type': 'PV2'}, connection_rule=5, syn_type='i2i', nm='B')
    net.add_edges(source={'cell_type': 'PV2'}, target={'ei': 'i'}, connection_rule=5, syn_type='i2i', nm='C')
    net.build()

    edges = net.edges(target_nodes=net.nodes(cell_type='Scnna1'))
    assert(len(edges) == 0)

    edges = net.edges(source_nodes={'ei': 'e'}, target_nodes={'ei': 'i'})
    assert(len(edges) == 50 + 100)

    edges = net.edges(source_nodes=[n.node_id for n in net.nodes(ei='e')])
    assert(len(edges) == 50 + 100)

    edges = net.edges(source_nodes={'ei': 'i'})
    assert(len(edges) == 100 * 100 * 2)
    for e in edges:
        assert(e['syn_type'] == 'i2i')

    edges = net.edges(syn_type='i2i')
    print(len(edges) == 100 * 100 * 2)
    for e in edges:
        assert(e['nm'] != 'A')

    edges = net.edges(syn_type='i2i', nm='C')
    assert(len(edges) == 100 * 150)


def test_mulitnet_iterator():
    net1 = NetworkBuilder('NET1')
    net1.add_nodes(N=50, cell_type='Rorb', ei='e')
    net1.build()

    net2 = NetworkBuilder('NET2')
    net2.add_nodes(N=100, cell_type='Scnna1', ei='e')
    net2.add_nodes(N=100, cell_type='PV1', ei='i')
    net2.add_edges(source={'ei': 'e'}, target={'ei': 'i'}, connection_rule=5, syn_type='e2i', net_type='rec')
    net2.add_edges(source=net1.nodes(), target={'ei': 'e'}, connection_rule=1, syn_type='e2e', net_type='fwd')
    net2.build()

    assert(len(net2.edges()) == 50*100 + 100*100)
    assert(len(net2.edges(source_network='NET2', target_network='NET1')) == 0)
    assert(len(net2.edges(source_network='NET1', target_network='NET2')) == 50*100)
    assert(len(net2.edges(target_network='NET2', net_type='rec')) == 100*100)

    edges = net2.edges(source_network='NET1')
    assert(len(edges) == 50*100)
    for e in edges:
        assert(e['net_type'] == 'fwd')

