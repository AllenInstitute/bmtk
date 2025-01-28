import pytest
import os
import json
import tempfile

from .conftest import *
from .pointnet_virtual_files import NodesFile, EdgesFile



@pytest.mark.skip()
def test_add_nodes():

    nodes = NodesFile(N=100)

    net = pointnet.PointNetwork()
    if not os.path.exists('tmp/'):
        os.mkdir('tmp/')
    net.add_component('models_dir', '.')
    with open('iaf_dynamics.json', 'w') as fp:
        json.dump({}, fp)

    with open('iz_dynamics.json', 'w') as fp:
        json.dump({}, fp)

    net.add_nodes(nodes)
    assert(net.networks == [nodes.name])
    assert(net.get_internal_nodes() == net.get_nodes(nodes.name))
    count = 0
    for pointnode in net.get_internal_nodes():
        node_id = pointnode.node_id
        orig_node = nodes[node_id]
        assert(node_id == orig_node.gid)
        assert(pointnode['ei'] == orig_node['ei'])
        assert(pointnode['model_type'] == orig_node['model_type'])
        assert(pointnode['rotation'] == orig_node['rotation'])
        assert(pointnode.model_params == {})
        count += 1
    assert(count == 100)


@pytest.mark.skip()
def test_add_edges():
    nodes = NodesFile(N=100)
    edges = EdgesFile(nodes, nodes)

    net = pointnet.PointNetwork()
    net.add_component('models_dir', '.')
    net.add_component('synaptic_models_dir', '.')

    with open('iaf_dynamics.json', 'w') as fp:
        json.dump({}, fp)

    with open('iz_dynamics.json', 'w') as fp:
        json.dump({}, fp)

    with open('iaf_exc.json', 'w') as fp:
        json.dump({}, fp)

    with open('iaf_inh.json', 'w') as fp:
        json.dump({}, fp)

    with open('izh_exc.json', 'w') as fp:
        json.dump({}, fp)

    with open('izh_inh.json', 'w') as fp:
        json.dump({}, fp)

    net.add_nodes(nodes)
    net.add_edges(edges)

    count = 0
    for trg_node in net.get_internal_nodes():
        for e in net.edges_iterator(trg_node.node_id, nodes.name):
            _, src_node, edge = e
            assert(edge['syn_weight'] == trg_node['weight'])
            count += 1
    assert(count == 10000)

