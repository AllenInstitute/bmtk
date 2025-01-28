import pytest
import numpy as np

from bmtk.builder.auxi.edge_connectors import distance_connector, connect_random


class MockNode(object):
    def __init__(self, node_id, **node_params):
        self.node_id = node_id
        self.node_params = node_params

    def __getitem__(self, item):
        return self.node_params[item]


def test_distance_connector():
    np.random.seed(1)
    # Avoids self connections
    n1 = MockNode(0, positions=[0.0, 0.0, 1.0])
    nsyns = distance_connector(source=n1, target=n1, d_weight_min=1.0, d_weight_max=100.0, d_max=1.0, nsyn_min=5,
                               nsyn_max=10)
    assert(nsyns is None)

    # distance is too great
    n1 = MockNode(0, positions=[1.0, 1.0, 0.0])
    n2 = MockNode(1, positions=[-1.0, -1.0, 0.0])
    nsyns = distance_connector(source=n1, target=n2, d_weight_min=1.0, d_weight_max=100.0, d_max=1.0, nsyn_min=5,
                               nsyn_max=10)
    assert(nsyns is None)

    n1 = MockNode(0, positions=[0.3, 1.0, 0.0])
    n2 = MockNode(1, positions=[0.5, 1.0, 0.0])
    nsyns = distance_connector(source=n1, target=n2, d_weight_min=1.0, d_weight_max=100.0, d_max=1.0, nsyn_min=5,
                               nsyn_max=10)
    assert(10 >= nsyns >= 5)


def test_connect_random():
    np.random.seed(1)
    n1 = MockNode(node_id=0)
    n2 = MockNode(node_id=1)
    nsyns = connect_random(source=n1, target=n2, nsyn_min=5, nsyn_max=10)
    assert (10 >= nsyns >= 5)


if __name__ == '__main__':
    # test_distance_connector()
    test_connect_random()