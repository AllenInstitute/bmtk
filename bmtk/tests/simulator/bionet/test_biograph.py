import pytest
import os
import json

from .conftest import *
from . import bionet_virtual_files as bvf


@pytest.mark.skip()
def test_add_nodes():

    nodes = bvf.NodesFile(N=100)

    net = bionet.BioNetwork()
    net.add_component('morphologies_dir', '.')
    net.add_component('biophysical_neuron_models_dir', '.')
    net.add_component('point_neuron_models_dir', '.')
    net.add_nodes(nodes)

    assert(net.networks == [nodes.name])
    assert(net.get_internal_nodes() == net.get_nodes(nodes.name))
    for bionode in net.get_internal_nodes():
        node_id = bionode.node_id
        orig_node = nodes[node_id]
        assert(node_id == orig_node.gid)
        assert(len(bionode.positions) == 3)
        assert(bionode['ei'] == orig_node['ei'])
        assert(bionode['model_type'] == orig_node['model_type'])
        assert(bionode['rotation'] == orig_node['rotation'])
        assert(os.path.basename(bionode.model_params) == orig_node['dynamics_params'])


@pytest.mark.skip()
def test_add_edges():
    nodes = bvf.NodesFile(N=100)
    edges = bvf.EdgesFile(nodes, nodes)

    net = bionet.BioNetwork()
    net.add_component('morphologies_dir', '.')
    net.add_component('biophysical_neuron_models_dir', '.')
    net.add_component('point_neuron_models_dir', '.')
    net.add_component('synaptic_models_dir', '.')

    with open('biophys_exc.json', 'w') as fp:
        json.dump({}, fp)

    with open('biophys_inh.json', 'w') as fp:
        json.dump({}, fp)

    with open('point_exc.json', 'w') as fp:
        json.dump({}, fp)

    with open('point_inh.json', 'w') as fp:
        json.dump({}, fp)

    net.add_nodes(nodes)
    net.add_edges(edges)

    count = 0
    for trg_node in net.get_internal_nodes():
        #print bionode.node_id
        for e in net.edges_iterator(trg_node.node_id, nodes.name):
            _, src_node, edge = e
            assert(edge['syn_weight'] == trg_node['weight'])
            count += 1
    assert(count == 10000)


if __name__ == '__main__':
    test_add_nodes()