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
from collections import defaultdict
from sklearn.neighbors import KDTree

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

def positions_density_matrix(mat, position_scale=np.array([[1,0,0],[0,1,0],[0,0,1]]), origin=np.array([0,0,0]),
                             plot=False, CCF_orientation=False, dmin=0.0, method='prog', verbose=False):
    """This function places random x,y,z coordinates according to a supplied 3D array of densities (cells/mm^3).
    The optional position_scale parameter defines a transformation matrix A to physical space such that:
    [x_phys, y_phys, z_phys] = A * [x_mat, y_mat, z_mat]

    Note: position_scale and output coordinates are in units of microns, while density is specified in mm^3.

    :param mat: A 3-dimensional matrix of densities (cells/mm^3)
    :param position_scale: A (3, 3) matrix (microns)
    :param plot: Generates a plot of the cell locations when set to True
    :param CCF_orientation: if True, plot will be oriented for Allen atlas common coordinate framework
           See https://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas3-DReferenceModels
    :param dmin: Minimum distance allowed between cell centers - using this function can severely slow down cell
           placement, especially as we approach the maximal possible density for this minimum distance
    :param verbose: If set to true, prints every 100 units placed for monitoring progress

    :return: A (N, 3) matrix (microns)
    """

    if (dmin<0):
        raise ValueError('Minimum distance between cell centers (dmin) must not be negative')

    if method == 'prog':
        # Use batch progressive sampling
        #positions_density_matrix2_old(mat, position_scale=position_scale,
        #                              origin=origin,
        #                              plot=False, CCF_orientation=False, dmin=dmin, verbose=verbose)
        positions = positions_dmin_prog(mat, position_scale=position_scale, dmin=dmin, verbose=verbose)
    elif method == 'lattice':
        # Use lattice jittering
        # Calc bounding box - TRANSFORM FIRST, THEN GET MIN/MAX

        positions = positions_dmin_lattice(mat, position_scale=position_scale, dmin=dmin)
    else:
        #raise    '\'method' can be either 'prog' for progressive sampling or 'lattice' for lattice jittering
        pass

    # Remove cells to achieve non-uniform density, if applicable

    temp = set(mat.flat)
    temp.discard(0)

    print('positions:', positions.shape)
    inds = np.matmul(np.linalg.inv(position_scale),(positions.T)).T
    inds = np.floor(inds).astype(int)
    indx = inds[:,0]
    indy = inds[:,1]
    indz = inds[:,2]
    max_dens = np.max(mat)

    if len(temp) > 1:
        rand_keep = np.random.uniform(size=positions.shape[0])

        for j in range(len(rand_keep)):
            vox_dens = mat[indx[j], indy[j], indz[j]]
            if rand_keep[j] >= vox_dens / max_dens:
                positions[j,:] = [np.nan, np.nan, np.nan]
        positions = positions[~np.isnan(positions[:,0]),:]

    # Shift re: origin
    positions += origin

    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    if plot:
        fig, ax = plt.subplots(figsize=(11, 11),  subplot_kw={'projection':'3d'})
        if CCF_orientation:
            plot_positions(x, z, y, ax, labels=['X','Z','Y'])
            ax.invert_zaxis()
            ax.view_init(elev=10., azim=0)
        else:
            plot_positions(x, y, z, ax, labels=['X','Y','Z'])
            ax.view_init(elev=10., azim=0)

    return positions

def positions_dmin_prog(mat, position_scale=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                              dmin=0.0, verbose = False):
    """This function places random x,y,z coordinates with a minimal distance according to a supplied 3D array of
    densities (cells/mm^3) using progressive sampling.
    The optional position_scale parameter defines a transformation matrix A to physical space such that:
    [x_phys, y_phys, z_phys] = A * [x_mat, y_mat, z_mat]

    Note: position_scale and output coordinates are in units of microns, while density is specified in mm^3.

    :param mat: A 3-dimensional matrix of densities (cells/mm^3)
    :param position_scale: A (3, 3) matrix (microns)
    :param plot: Generates a plot of the cell locations when set to True
    :param CCF_orientation: if True, plot will be oriented for Allen atlas common coordinate framework
           See https://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas3-DReferenceModels
    :param dmin: Minimum distance allowed between cell centers (microns) - using this function can severely slow down cell
           placement, especially as we approach the maximal possible density for this minimum distance

    :return: A (N, 3) matrix (microns)
    """

    # Can later put this inside check function?
    if (dmin < 0):
        raise ValueError('Minimum distance between cell centers (dmin) must not be negative')

    V_sphere = 4 / 3 * np.pi * (dmin/1000/2) ** 3
    # Number of spheres at FCC/HCP packing: 0.74*V/V_sphere
    # Random close packing: 0.64*V/V_sphere
    dens_lim = 0.63 / V_sphere   # in points per mm^3
    print('Density limit:{:.1f}'.format(dens_lim))
    print('Max density requested:{:.1f}'.format(np.max(mat)))

    vol_per_voxel = position_scale.astype(float).dot([1, 1, 1]).prod() / (1000) ** 3
    mat = mat.astype(float) * vol_per_voxel

    max_dens = np.max(mat)
    n_vox_nz = np.nonzero(mat)[0].shape[0]

    # First distribute uniformly in non zero voxels at max_dens
    ind_nz = np.nonzero(mat)
    ndraws_tot = int(max_dens * n_vox_nz)
    ndraws = ndraws_tot
    positions = np.array([]).reshape(0,3)
    shift = False
    iter = 0
    max_iter = 1
    past_iters = np.full(3, np.nan)
    pi_ind = 0
    favor_new = False
    #n_confl = 0
    #conf_ch = []

    while positions.shape[0] < ndraws_tot:
        position_len = positions.shape[0]
        positions_new = []
        #dists = []
        #d_inds = []

        itercount = 0
        n_pass = 0
        while (itercount < max_iter) and (n_pass < int(ndraws_tot / 5000)):
            rand_pos_new = np.random.uniform(size=(3,ndraws))   # Or for simple geometries, draw over max_dims
            rand_nz_vox_new = np.random.choice(range(ind_nz[0].shape[0]), ndraws)

            positions_new.append(position_scale.dot([ind_nz[0][rand_nz_vox_new].astype(float) + rand_pos_new[0,:],
                             ind_nz[1][rand_nz_vox_new].astype(float) + rand_pos_new[1,:],
                             ind_nz[2][rand_nz_vox_new].astype(float) + rand_pos_new[2,:]]))
            positions_new[-1] = positions_new[-1].T
            #position_len = positions.shape[0]

            # Check against existing tree
            # Find nearest neighbor

            # If after max iter you were not able to find more than the min number of cells (10) - iter is how many rounds to find 10, favor new points, shift.

            if positions.shape[0] > 0:
                tree = KDTree(positions, leaf_size=2)
                dists, d_inds = tree.query(positions_new[-1], k=1)
                conf_inds = np.squeeze(dists < dmin)
                positions_new[-1][conf_inds,:] = [np.nan, np.nan, np.nan]
                notna = ~np.isnan(positions_new[-1][:, 0])
                positions_new[-1] = positions_new[-1][notna, :]

            itercount += 1
            n_pass += positions_new[-1].shape[0]
            #if n_pass > ndraws_tot / 2000:
            #    break
        if pi_ind >= len(past_iters):
            # Get mean number of iterations for past 3 points and use to set new maxiter
            max_iter = 1.5 * np.max(past_iters)
            #print(max_iter)
            pi_ind = 0
        past_iters[pi_ind] = itercount
        pi_ind += 1

        if n_pass < int(ndraws_tot / 2000):
            # Replace some of existing points
            # Draw a small fraction of the total number of points
            #rand_pos_new = np.random.uniform(size=(3, int(0.01*ndraws_tot)))  # Or for simple geometries, draw over max_dims
            #rand_nz_vox_new = np.random.choice(range(ind_nz[0].shape[0]), int(0.01*ndraws_tot))
            rand_pos_new = np.random.uniform(size=(3, ndraws))  # Or for simple geometries, draw over max_dims
            rand_nz_vox_new = np.random.choice(range(ind_nz[0].shape[0]), ndraws)

            positions_new.append(position_scale.dot([ind_nz[0][rand_nz_vox_new].astype(float) + rand_pos_new[0, :],
                                                     ind_nz[1][rand_nz_vox_new].astype(float) + rand_pos_new[1, :],
                                                     ind_nz[2][rand_nz_vox_new].astype(float) + rand_pos_new[2, :]]))
            positions_new[-1] = positions_new[-1].T
            # Remove fraction of existing points in conflict with new points
            inds2 = tree.query_radius(positions_new[-1], r=dmin)
            lens = np.array([len(a) for a in inds2])
            k = np.min([math.ceil(0.005*ndraws_tot), np.count_nonzero(lens <= 1)])
            noconf = np.nonzero(lens == 0)[0]
            swap = np.argpartition(lens, k)[:k]   # k indices with smallest number of conflicts
            #swap = np.setdiff1d(swap, noconf)
            inds3 = np.unique(np.hstack(inds2[swap]))
            positions[inds3, :] = [np.nan, np.nan, np.nan]
            positions = positions[~np.isnan(positions[:, 0]),:]
            positions_new[-1] = positions_new[-1][np.concatenate((swap, noconf)),:]

        positions_new = np.vstack(positions_new)

        # Check new points that are retained against each other
        if positions_new.shape[0] > 1:
            tree = KDTree(positions_new, leaf_size=2)
            dists, d_inds = tree.query(positions_new, k=2)
            # Overremoves by checking against removed cells, but tracking removed cells is not faster
            conf_inds = np.squeeze(dists[:,1] < dmin)
            positions_new[conf_inds, :] = [np.nan, np.nan, np.nan]
            notna = ~np.isnan(positions_new[:, 0])
            positions_new = positions_new[notna, :]

        #if positions_new.shape[0]) < ndraws_tot / 2000:
        if max_iter>20:
            shift = True
        positions = np.concatenate((positions, positions_new), axis=0)  # Add new positions
        #notna = ~np.isnan(positions[:,0])
        #positions = positions[notna,:]     # Existing positions, remove NaN rows
        #print(f'{positions_new.shape[0]}/ {n_pass} points accepted')
        print(f'{positions.shape[0]}/{ndraws_tot} cells placed')

        ndraws = ndraws_tot
    positions = positions[:ndraws,:]
    print(f'{positions.shape[0]} cells')

    return positions

def positions_dmin_lattice(mat, position_scale=np.array([0,0,0]), dmin=0.0):
    '''Packing with a minimum distance, starting from a hexagonal close packing lattice
    :param x_box, y_box, z_box: size of each dimension of lattice box to be generated (microns)
    :param N: number of points to be placed

    '''

    ind_nz = np.nonzero(mat)
    minx = np.min(ind_nz[0])
    miny = np.min(ind_nz[1])
    minz = np.min(ind_nz[2])
    maxx = np.max(ind_nz[0])
    maxy = np.max(ind_nz[1])
    maxz = np.max(ind_nz[2])

    corners = [v.flatten() for v in (np.meshgrid([minx, maxx+1], [miny, maxy+1], [minz, maxz+1]))]
    corners = np.vstack(corners)
    corners_t = (position_scale.astype(float).dot(corners))

    minx2 = np.min(corners_t[0])
    print(minx.shape)
    miny2 = np.min(corners_t[1])
    minz2 = np.min(corners_t[2])
    maxx2 = np.max(corners_t[0])
    maxy2 = np.max(corners_t[1])
    maxz2 = np.max(corners_t[2])

    # max_dims
    # Calc additional origin
    origin2 = [minx2, miny2, minz2]
    x_box = maxx2 - minx2
    y_box = maxy2 - miny2
    z_box = maxz2 - minz2

    # Calc box N
    N_box = math.floor(x_box * y_box * z_box * np.max(mat) / (1000 ** 3))              # N for lattice box
    N = math.floor(N_box * len(ind_nz[0])/((maxx-minx+1)*(maxy-miny+1)*(maxz-minz+1)))     # Final desired N
    print(N)
    vol_box = x_box * y_box * z_box
    r = (vol_box * 0.74 / N_box * 3 / 4 / np.pi) ** (1. / 3)

    x_n = np.ceil(x_box / (2 * r)).astype('int')
    y_n = np.ceil(y_box / (r) / np.sqrt(3)).astype('int')
    z_n = np.ceil(z_box / (2 * r) / np.sqrt(6) * 3).astype('int')

    x, y, z = hcp(x_n, y_n, z_n, r)

    # Randomly remove units in excess of N
    # Set to NaN rather than drop to keep lattice in register

    x_vec = x.flatten()
    y_vec = y.flatten()
    z_vec = z.flatten()
    todrop = np.random.choice(range(len(x_vec)), len(x_vec) - N_box, replace=False)
    x_vec[todrop] = np.nan
    y_vec[todrop] = np.nan
    z_vec[todrop] = np.nan

    positions = np.vstack((x_vec, y_vec, z_vec))
    # Remove unneeded units by mat or geometry function
    position_scale_inv = np.linalg.inv(position_scale)
    inds = np.matmul(position_scale_inv, (positions+np.reshape(origin2, (-1, 1)))).T
    inds = np.floor(inds)
    mask = ~np.isnan(inds)
    mat_vals = np.full(inds.shape[0], fill_value=np.nan)
    mat_vals[mask[:,0]] = mat[inds[mask[:, 0],0].astype(int),inds[mask[:, 0],1].astype(int),inds[mask[:, 0],2].astype(int)]
    mat_vals = mat_vals==0
    mask = ~np.isnan(mat_vals)
    positions = positions.T
    positions [mat_vals[mask],:] = [np.nan, np.nan, np.nan]

    x_vec = positions[:,0]
    y_vec = positions[:,1]
    z_vec = positions[:,2]

    x = x_vec.reshape((x_n, y_n, z_n))
    y = y_vec.reshape((x_n, y_n, z_n))
    z = z_vec.reshape((x_n, y_n, z_n))

    # Jitter by up to 2*r-dmin and check if okay,if not, redraw jitter
    jitr_max = 2 * r - dmin

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                # Don't need it to be uniformly distributed - the outer points are less likely to qualify
                if np.isnan(x[i, j, k]):
                    # Eliminated unit, skip
                    continue
                pt_old = [x[i, j, k], y[i, j, k], z[i, j, k]]
                x[i, j, k] = float('inf')
                y[i, j, k] = float('inf')
                z[i, j, k] = float('inf')

                keep = False

                q = 2  # Depends on jitr_max
                indsi = np.arange(max(i - q, 0), min(i + q, x.shape[0]))
                indsj = np.arange(max(j - q, 0), min(j + q, x.shape[1]))
                indsk = np.arange(max(k - q, 0), min(k + q, x.shape[2]))

                near_inds = np.meshgrid(indsi, indsj, indsk)

                while not keep:
                    jitx = np.random.uniform(-jitr_max, jitr_max)
                    jity = np.random.uniform(-jitr_max, jitr_max)
                    jitz = np.random.uniform(-jitr_max, jitr_max)

                    x_new = pt_old[0] + jitx
                    y_new = pt_old[1] + jity
                    z_new = pt_old[2] + jitz

                    dists = np.sqrt((x_new - x[near_inds[0], near_inds[1], near_inds[2]]) ** 2 +
                                    (y_new - y[near_inds[0], near_inds[1], near_inds[2]]) ** 2 +
                                    (z_new - z[near_inds[0], near_inds[1], near_inds[2]]) ** 2)
                    d = np.nanmin(np.append(dists.flatten(), float('inf')))
                    keep = d > dmin

                # Replace existing
                x[i, j, k] = x_new
                y[i, j, k] = y_new
                z[i, j, k] = z_new

    # Trim any points that have been jittered out of permitted region
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    z = z[~np.isnan(z)]

    before_trim_len = x.shape[0]
    inside = (x <= x_box) & (x >= 0) & (y <= y_box) & (y >= 0) & (z <= z_box) & (z >= 0)
    x = x[inside]
    y = y[inside]
    z = z[inside]
    needed = before_trim_len-x.shape[0]

    positions = np.vstack((x, y, z)).T

    if positions.shape[0] > N:
        # Trim excess N, points are ordered, so pick randomly
        todrop = np.random.choice(range(len(x_vec)), positions.shape[0]-N, replace=False)
        positions[todrop,:] = np.nan

    while positions.shape[0] < N:
        tree = KDTree(positions, leaf_size=2)

        rand_pos_new = np.random.uniform(size=(3, before_trim_len))
        rand_nz_vox_new = np.random.choice(range(ind_nz[0].shape[0]), before_trim_len)

        positions_new = position_scale.dot([(ind_nz[0]-minx)[rand_nz_vox_new].astype(float) + rand_pos_new[0, :],
                                            (ind_nz[1]-miny)[rand_nz_vox_new].astype(float) + rand_pos_new[1, :],
                                            (ind_nz[2]-minz)[rand_nz_vox_new].astype(float) + rand_pos_new[2, :]])
        positions_new = positions_new.T
        dists, d_inds = tree.query(positions_new, k=1)
        conf_inds = np.squeeze(dists < dmin)
        positions_new[conf_inds, :] = [np.nan, np.nan, np.nan]
        positions_new = positions_new[~np.isnan(positions_new[:, 0]), :]
        if positions_new.shape[0] > 1:
            tree = KDTree(positions_new, leaf_size=2)
            dists, d_inds = tree.query(positions_new, k=2)
            conf_inds = np.squeeze(dists[:,1] < dmin)
            positions_new[conf_inds, :] = [np.nan, np.nan, np.nan]
            notna = ~np.isnan(positions_new[:, 0])
            positions_new = positions_new[notna, :]
        positions = np.concatenate((positions, positions_new), axis=0)  # Add new positions

    positions = positions[:N, :]
    # If N rather than density, replace any missing

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    z = z[~np.isnan(z)]

    # Shift re: origin
    x += origin2[0]
    y += origin2[1]
    z += origin2[2]

    return positions


def positions_nrrd (nrrd_filename, max_dens_per_mm3, split_bilateral=None, dmin = 0.0, plot=False, method='prog', verbose=False):
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
    origin = header['space origin']

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

    # For testing non-uniform density
    #t = np.arange(0, readdata_scaled.shape[2], 1)
    #s = 1+np.sin(2*np.pi*0.1*t)
    #readdata_scaled = readdata_scaled * s

    positions = positions_density_matrix(readdata_scaled, position_scale=space_dir, origin = origin, plot=plot,
                                         CCF_orientation=True, dmin=dmin, method=method, verbose=verbose)

    return positions

def xiter_random(N=1, min_x=0.0, max_x=1.0):
    return np.random.uniform(low=min_x, high=max_x, size=(N,))

def plot_positions(x,y,z,ax,labels):
    # Do plot a verification for each of these helper functions

    ax.scatter(x,y,z, marker='.', s=3)

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
def hcp(x_n, y_n, z_n, r):
    # Hexagonal close packing lattice generation
    i = np.arange(x_n)
    j = np.arange(y_n)
    k = np.arange(z_n)

    get_x = lambda i, j, k, r: 2 * r * i + r * ((j + k) % 2)
    x = get_x(i[:, None, None], j[None, :, None], k[None, None, :], r)
    get_y = lambda i, j, k, r: r * np.sqrt(3) * (j + 1 / 3 * (k % 2)) + 0 * i
    y = get_y(i[:, None, None], j[None, :, None], k[None, None, :], r)
    get_z = lambda i, j, k, r: 2 * r * np.sqrt(6) / 3 * k + 0 * i + 0 * j
    z = get_z(i[:, None, None], j[None, :, None], k[None, None, :], r)

    return x, y, z