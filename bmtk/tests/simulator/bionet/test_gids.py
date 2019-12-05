import pytest
from .conftest import *


@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_gid_pool():
    gid_map = GidPool()
    gid_map.add_pool(name='p1', n_nodes=1000)
    gid_map.add_pool(name='p2', n_nodes=10000)
    gid_map.add_pool(name='p3', n_nodes=1)
    gid_map.add_pool(name='p4', n_nodes=500)

    assert(gid_map.get_gid(name='p1', node_id=0) == 0)
    assert(gid_map.get_pool_id(0) == (0, 'p1'))
    assert(gid_map.get_gid('p1', node_id=999) == 999)

    assert(gid_map.get_gid(name='p2', node_id=0) == 1000)
    assert(gid_map.get_pool_id(1000) == (0, 'p2'))

    assert(gid_map.get_gid(name='p3', node_id=0) == 11000)
    assert(gid_map.get_pool_id(11000)  == (0, 'p3'))
    assert(gid_map.get_pool_id(10999) == (9999, 'p2'))
    assert(gid_map.get_pool_id(10998) == (9998, 'p2'))
    assert(gid_map.get_gid(node_id=9998, name='p2') == 10998)
    assert(gid_map.get_pool_id(11001) == (0, 'p4'))

    assert(gid_map.get_pool_id(11500) == (499, 'p4'))
    assert(gid_map.get_gid(node_id=499, name='p4') == 11500)


if __name__ == '__main__':
    test_gid_pool()