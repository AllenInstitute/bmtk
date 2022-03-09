import pytest
import numpy as np
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

@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
@pytest.mark.parametrize('spike_times', [
    [],
    np.array([]),
    None,
    [0.01, 0.2, 0.4, 0.6, 0.8],
    np.array([0.01, 0.2, 0.4, 0.6, 0.8]),
    [0.8, 0.2, 0.4, 0.6, 0.01],
    [0.5]
])
def test_add_spike_trains(spike_times):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001, "print_time": True})

    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1', nnodes=100, virtual=False))
    net.add_nodes(MockNodePop(name='LGN', nnodes=100, virtual=True))
    net.add_edges(MockEdges(name='LGN_to_V1', source_nodes='LGN', target_nodes='V1', delay=1.0))
    net.build_nodes()
    net.add_spike_trains(MockSpikes(spike_times), node_set=MockNodeSet(['LGN']))


@pytest.mark.skipif(not nest_installed, reason='NEST is not installed')
@pytest.mark.parametrize('spike_times', [
    [-1.0, 0.0, 1.0],
    np.array([-1.0, 0.0, 1.0]),
    [-1.0],
    [0.0]
])
def test_add_spike_trains_badtimes(spike_times):
    # PointNET should throw error if spike-times contains 0 or negative value.
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001, "print_time": True})

    net = pointnet.PointNetwork()
    net.add_nodes(MockNodePop(name='V1', nnodes=100, virtual=False))
    net.add_nodes(MockNodePop(name='LGN', nnodes=100, virtual=True))
    net.add_edges(MockEdges(name='LGN_to_V1', source_nodes='LGN', target_nodes='V1', delay=1.0))
    net.build_nodes()

    with pytest.raises(Exception):
        net.add_spike_trains(MockSpikes(spike_times), node_set=MockNodeSet(['LGN']))




if __name__ == '__main__':
    # test_add_nodes(batched=False)
    # test_add_multi_nodes(batched=False)
    test_add_edges()
    # test_add_edges_baddelay()
    # test_add_spike_trains(spike_trains=None)
