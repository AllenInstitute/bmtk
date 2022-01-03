import pytest
from .conftest import *


@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
@pytest.mark.parametrize('batched', [True, False])
def test_add_nodes(batched):
    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1', batched=batched))
    net.build_nodes()

    assert(len(net.node_populations) == 1)
    assert(len(net.gid_pool) == 100)


@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
@pytest.mark.parametrize('batched', [True, False])
def test_add_multi_nodes(batched):
    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1', nnodes=10, batched=batched))
    net.add_nodes(MockNodePop(name='V2', nnodes=20, batched=batched))
    net.add_nodes(MockNodePop(name='V3', nnodes=30, batched=batched))
    net.build_nodes()

    assert(len(net.node_populations) == 3)
    assert(len(net.gid_pool) == 60)


@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
def test_add_edges():
    # Required to run nest in pytest
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001, "print_time": True})

    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1'))
    net.add_edges(MockEdges(name='V1_to_V1', source_nodes='V1', target_nodes='V1'))
    net.build()

    assert(len(net.node_populations) == 1)
    assert(len(net.gid_pool) == 100)
    assert(len(nest.GetConnections()) == 100)


@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
def test_add_edges_baddelay():
    # Required to run nest in pytest
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 10.00, "print_time": True})

    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1'))
    net.add_edges(MockEdges(name='V1_to_V1', source_nodes='V1', target_nodes='V1', delay=1.0))

    with pytest.raises(Exception):
        net.build()



if __name__ == '__main__':
    # test_add_nodes(batched=False)
    # test_add_multi_nodes(batched=False)
    test_add_edges()
    # test_add_edges_baddelay()
