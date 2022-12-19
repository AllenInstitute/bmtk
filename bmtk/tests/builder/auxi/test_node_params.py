import pytest
import numpy as np

from bmtk.builder.auxi.node_params import positions_columinar, positions_cuboid, positions_list, positions_rect_prism, positions_ellipsoid, positions_density_matrix, positions_nrrd


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

def test_positions_rect_prism():
    np.random.seed(1)
    points = positions_rect_prism(N=10, center=[10.0, 20.0, 30.0], height=15.0, x_length=3.0, z_length=8.0)
    assert(points.shape == (10, 3))
    
     # check height, width, length restriction
    ys = points[:, 1]
    assert(np.min(ys) >= 20.0-15.0)
    assert(np.max(ys) <= 20.0+15.0)
    xs = points[:, 0]
    assert(np.min(xs) >= 10.0-3.0)
    assert(np.max(xs) <= 10.0+3.0)
    zs = points[:, 2]
    assert(np.min(zs) >= 30.0-8.0)
    assert(np.max(zs) <= 30.0+8.0)

def test_positions_ellipsoid():
    np.random.seed(1)
    points = positions_ellipsoid(N=10, center=[10.0, 20.0, 30.0], height=15.0, x_length=3.0, z_length=8.0)
    assert(points.shape == (10, 3))
    
     # check height, width, length restriction
    dxs = points[:, 0] - 10.0
    dys = points[:, 1] - 20.0
    dzs = points[:, 2] - 30.0
    
    assert(np.max(dxs**2/3.0**2+dys**2/15.0**2+dzs**2/8.0**2)<1)
    
def test_positions_density_matrix():
    np.random.seed(1)
    mat = 100000*np.array([[[0,1,5],[0,0.5,0],[0,0,0]],
                [[0,1,5],[0,0.5,0],[0,0,0]],
                [[0,1,1],[0,1,0],[0,0,5]]])

    position_scale=np.array([[25.0,0,0],[0,25.0,0],[0,0,25.0]])
    
    points = positions_density_matrix(mat, position_scale)
    
    assert(points.shape == (34,3))
    assert(np.max(abs(points[:,0]))<25*3)
    assert(np.max(abs(points[:,1]))<25*3)
    assert(np.max(abs(points[:,2]))<25*3)
    
def test_positions_nrrd():
    np.random.seed(1)
    nrrd_filename = 'docs/tutorial/sources/nrrd/structure_721.nrrd'
    points = positions_nrrd(nrrd_filename, 2000)
    assert(points.shape==(2048, 3))
    
    points = positions_nrrd(nrrd_filename, 2000, split_bilateral='z')
    assert(points.shape==(1024, 3))
    

if __name__ == '__main__':
    # test_positions_columinar()
    # test_positions_cuboid()
    test_positions_list()
    test_positions_rect_prism()
