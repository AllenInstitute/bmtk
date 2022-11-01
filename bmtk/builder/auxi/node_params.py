# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import math
import nrrd
import matplotlib.pyplot as plt


def positions_columinar(N=1, center=[0.0, 50.0, 0.0], height=100.0, min_radius=0.0, max_radius=1.0, plot=False):
    """Returns a set of random x,y,z coordinates within a given cylinder or cylindrical ring.
    Height is given as the y (index 1) coordinates.

    :param N: Number of points to return
    :param center: center of sphere
    :param height: maximum length of sphere (y coord)
    :param min_radius: minimum horizontal radius on x-z plane
    :param max_radius: maximum horizontal radius on x-z plane
    :return: A (N, 3) matrix
    """
    phi = 2.0 * math.pi * np.random.random([N])
    r = np.sqrt((min_radius**2 - max_radius**2) * np.random.random([N]) + max_radius**2)
    x = center[0] + r * np.cos(phi)
    z = center[2] + r * np.sin(phi)
    y = center[1] + height * (np.random.random([N]) - 0.5)

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(9,12), subplot_kw={'projection':'3d'})
        plot_positions(x, z, y, ax[0], labels=['X','Z','Y'])
        ax[0].set_title('Side view')
        ax[0].view_init(elev=5., azim=0)

        plot_positions(x, z, y, ax[1], labels=['X','Z','Y'])
        ax[1].set_title('Bird\'s eye view')
        ax[1].view_init(elev=90., azim=0)

    return np.column_stack((x, y, z))

positions_columnar = positions_columinar

def positions_rect_prism(N=1, center=[0.0, 50.0, 0.0], height=20.0, x_length=100.0, z_length=100.0, plot=False):
    """Returns a set of random x,y,z coordinates within a rectangular prism.
    Height is given as the y (index 1) coordinates.

    :param N: Number of points to return
    :param center: center of rectangular prism
    :param height: height of prism (y-coordinate)
    :param x_length: length of prism in x
    :param z_length: length of prism in z
    :return: A (N, 3) matrix
    """

    x = center[0] + x_length * (2*(np.random.random([N])) - 1)
    z = center[2] + z_length * (2*(np.random.random([N])) - 1)
    y = center[1] + height * (2*(np.random.random([N])) - 1)
    if plot:
        fig, ax = plt.subplots(1,2,figsize=(9,12), subplot_kw={'projection':'3d'})
        plot_positions(x, z, y, ax[0], labels=['X','Z','Y'])
        ax[0].set_title('Side view')
        ax[0].view_init(elev=5., azim=0)

        plot_positions(x, z, y, ax[1], labels=['X','Z','Y'])
        ax[1].set_title('Bird\'s eye view')
        ax[1].view_init(elev=90., azim=0)

    return np.column_stack((x, y, z))

def positions_ellipsoid (N=1, center=[0.0, 50.0, 0.0], height=50, x_length=100.0, z_length=200.0, plot=False):
    """Returns a set of random x,y,z coordinates within an ellipsoid. Height is given as the y (index 1) coordinates.

    :param N: Number of points to return
    :param center: center of ellipsoid
    :param height: height of ellipsoid (y-coordinate)
    :param x_length: length of ellipsoid in x
    :param z_length: length of ellipsoid in z
    :return: A (N, 3) matrix
    """
    # Generate prism bounding the ellipsoid
    positions = positions_rect_prism (3*N, center, height, x_length, z_length, plot=False)
    val = ((positions[:,0]-center[0])**2/x_length**2)\
          +((positions[:,1]-center[1])**2/height**2)\
          +((positions[:,2]-center[2])**2/z_length**2)
    positions = np.squeeze(positions [np.where(val<1),:])
    positions = positions [:N,:]

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(9,12), subplot_kw={'projection':'3d'})
        plot_positions(positions[:,0], positions[:,2], positions[:,1], ax[0], labels=['X','Z','Y'])
        ax[0].set_title('Side view')
        ax[0].view_init(elev=5., azim=0)

        plot_positions(positions[:,0], positions[:,2], positions[:,1], ax[1], labels=['X','Z','Y'])
        ax[1].set_title('Bird\'s eye view')
        ax[1].view_init(elev=90., azim=0)

    return positions
def positions_cuboid(N=1, center=[0.0, 0.0, 0.0], height=100.0, xside_length=100.0, yside_length=100.0, min_dist=20.0,
                     plot = False):
    """This function distributes the cells in a 3D cuboid (x,y,z sides may have different lengths). The method used
    assures cells cannot be placed too close to one another (must be > min_dist apart)
    WARNING: If cell density is high and there is more than 1 population of cells, there is a high chance cells will be
    placed on top of one another. You can use positions_list() to avoid this...

    Written by Ben Latimer at University of Missouri (latimerb@missouri.edu)

    :return: A (N, 3) matrix
    """
    
    # Create the possible x,y,z coordinates
    x_grid = np.arange(center[0], xside_length+min_dist, min_dist)
    y_grid = np.arange(center[1], yside_length+min_dist, min_dist)
    z_grid = np.arange(center[2], height+min_dist, min_dist)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
    positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Pick N indices for the positions matrix randomly, without replacement (so coordinates are unique)
    inds = np.random.choice(np.arange(0, np.size(positions, 0)), N, replace=False)

    # Assign positions
    x = positions[inds][:, 0]
    y = positions[inds][:, 1]
    z = positions[inds][:, 2]

    if plot:
        fig, ax = plt.subplots(1,2,figsize=(9,12), subplot_kw={'projection':'3d'})
        plot_positions(x, y, z, ax[0], labels=['X','Y','Z'])
        ax[0].set_title('Side view')
        ax[0].view_init(elev=5., azim=0)

        plot_positions(x, y, z, ax[1], labels=['X','Y','Z'])
        ax[1].set_title('Bird\'s eye view')
        ax[1].view_init(elev=90., azim=0)

    return np.column_stack((x, y, z))


def positions_list(positions=np.array([(0, 0, 0), (0, 0, 1)])):
    """This function is designed to be used with an externally supplied array of x,y,z coordinates. It is useful to
    avoid cell overlap in high density situations or to make some unique geometry. After you create each population,
    delete those positions from the "master" list of coordinates. This will assure that no cells are placed on top of
    one another.

    Written by Ben Latimer at University of Missouri (latimerb@missouri.edu)

    :param positions:
    :return:
    """
    
    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    return np.column_stack((x, y, z))

def positions_density_matrix(mat, position_scale=np.array([[1,0,0],[0,1,0],[0,0,1]]), plot=False, CCF_orientation=False):
    """This function places random x,y,z coordinates according to a supplied 3D array of densities (cells/mm^3).
    The optional position_scale parameter defines a transformation matrix A to physical space such that:
    [x_phys, y_phys, z_phys] = A * [x_mat, y_mat, z_mat]

    Note: position_scale and output coordinates are in units of microns, while density is specified in mm^3.

    :param mat: A 3-dimensional matrix of densities (cells/mm^3)
    :param position_scale: A (3, 3) matrix (microns)
    :param plot: Generates a plot of the cell locations when set to True
    :param CCF_orientation: if True, plot will be oriented for Allen atlas common coordinate framework
           See https://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas3-DReferenceModels

    :return: A (N, 3) matrix (microns)
    """
    vol_per_voxel = position_scale.astype(float).dot([1, 1, 1]).prod() / (1000) ** 3
    mat = mat.astype(float) * vol_per_voxel

    # Draw Poisson distributed number of cells per voxel based on density
    num = np.random.poisson(lam=mat)

    ncells_tot = np.sum(num)

    # Assign random positions to the number of cells in each voxel
    rand_array = np.random.uniform(size=(3, ncells_tot))
    x = np.empty(ncells_tot, dtype=float)
    y = np.empty(ncells_tot, dtype=float)
    z = np.empty(ncells_tot, dtype=float)

    i = 0
    inds = np.nonzero(num)
    for j in range(len(inds[0])):
        for k in range(num[inds[0][j], inds[1][j], inds[2][j]]):
            pt = position_scale.dot([float(inds[0][j]) + rand_array[0, i],
                                float(inds[1][j]) + rand_array[1, i],
                                float(inds[2][j]) + rand_array[2, i]])

            x[i], y[i], z[i] = pt
            i += 1

    if plot:
        fig, ax = plt.subplots(figsize=(11, 11),  subplot_kw={'projection':'3d'})
        if CCF_orientation:
            plot_positions(x, z, y, ax, labels=['X','Z','Y'])
            ax.invert_zaxis()
            ax.view_init(elev=10., azim=0)
        else:
            plot_positions(x, y, z, ax, labels=['X','Y','Z'])
            ax.view_init(elev=10., azim=0)

    return np.column_stack((x, y, z))

def positions_nrrd (nrrd_filename, max_dens_per_mm3, split_bilateral=None, plot=False):
    '''Generates random cell positions based on a *.nrrd file. Matrix values are interpreted as cell densities
    for each voxel. The maximum density is scaled to max_dens_per_mm3 (cells/mm^3).
    If the *.nrrd file is a structural mask, cells will be placed uniformly at max_dens_per_mm3 within the
    structure geometry.
    By default, only one hemisphere is shown, but this can be disabled by setting bilateral=True.

    :param nrrd_filename: path to *.nrrd file
    :param max_dens_per_mm3: desired density at maximum value of nrrd array (cells/mm^3)
    :param split_bilateral: return only unilateral structure by removing half of the array along the given axis.
            If no splitting is desired, pass in None
    :return: A (N, 3) matrix (microns)
    '''

    readdata, header = nrrd.read(nrrd_filename)
    space_dir = header['space directions']

    readdata_scaled = readdata / np.max(readdata) * max_dens_per_mm3

    if split_bilateral is not None:
        if split_bilateral=='x':
            readdata_scaled = readdata_scaled[:int(readdata_scaled.shape[0]/2), :, :]
        elif split_bilateral=='y':
            readdata_scaled = readdata_scaled[:, :int(readdata_scaled.shape[1]/2), :]
        elif split_bilateral=='z':
            readdata_scaled = readdata_scaled[:, :, :int(readdata_scaled.shape[2]/2)]
        else:
            error("split_bilateral must take values of 'x', 'y', or 'z', or 'None' if no processing is desired")

    positions = positions_density_matrix(readdata_scaled, position_scale=space_dir, plot=plot, CCF_orientation=True)

    return positions

def xiter_random(N=1, min_x=0.0, max_x=1.0):
    return np.random.uniform(low=min_x, high=max_x, size=(N,))

def plot_positions(x,y,z,ax,labels):
    # Do plot a verification for each of these helper functions

    ax.scatter(x,y,z, marker='.', s=3)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])