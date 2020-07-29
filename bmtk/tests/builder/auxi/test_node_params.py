import pytest
import numpy as np

from bmtk.builder.auxi.node_params import positions_columinar, positions_cuboid, positions_list


def test_positions_columinar():
    np.random.seed(1)
    points = positions_columinar(N=10, center=[0.0, 0.0, 0.0], height=10.0, min_radius=1.0, max_radius=2.0)
    assert(points.shape == (10, 3))

    # check height restriction
    ys = points[:, 1]
    assert(np.min(ys) >= -5.0)
    assert(np.max(ys) <= 5.0)

    # check radius restrictions
    xz_plane = points[:,[0,2]]
    dists = np.linalg.norm(xz_plane, axis=1)
    assert(np.min(dists) >= 1.0)
    assert(np.max(dists) <= 2.0)


def test_positions_cuboid():
    np.random.seed(1)
    points = positions_cuboid(N=10, center=[0.0, 0.0, 0.0], height=5.0, xside_length=2.0, yside_length=2.0,
                              min_dist=1.0)
    assert(points.shape == (10, 3))


def test_positions_list():
    assert(positions_list().shape == (2, 3))


if __name__ == '__main__':
    # test_positions_columinar()
    # test_positions_cuboid()
    test_positions_list()
