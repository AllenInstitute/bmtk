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


def positions_columinar(N=1, center=[0.0, 50.0, 0.0], height=100.0, min_radius=0.0, max_radius=1.0, distribution='uniform'):
    phi = 2.0 * math.pi * np.random.random([N])
    r = np.sqrt((min_radius**2 - max_radius**2) * np.random.random([N]) + max_radius**2)
    x = center[0] + r * np.cos(phi)
    z = center[2] + r * np.sin(phi)
    y = center[1] + height * (np.random.random([N]) - 0.5)

    return np.column_stack((x, y, z))

# This function distributes the cells in a 3D cuboid (x,y,z sides may have different lengths). 
# The method used assures cells cannot be placed too close to one another (must be > min_dist apart)
# WARNING: If cell density is high and there is more than 1 population of cells, there is a high chance...
# ... cells will be placed on top of one another. You can use positions_list() to avoid this...
# ... but you must create your own array of cell positions
# Written by Ben Latimer at University of Missouri (latimerb@missouri.edu)
def positions_cuboid(N=1, center=[0.0, 0.0, 0.0], height=100.0, xside_length=100.0, yside_length=100.0, min_dist=20.0):
    
    # Create the possible x,y,z coordinates
    x_grid = np.arange(center[0],xside_length+min_dist,min_dist)
    y_grid = np.arange(center[1],yside_length+min_dist,min_dist)
    z_grid = np.arange(center[2],height+min_dist,min_dist)
    xx, yy, zz = np.meshgrid(x_grid, y_grid, z_grid)
    positions = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # Pick N indices for the positions matrix randomly, without replacement (so coordinates are unique)
    inds = np.random.choice(np.arange(0,np.size(positions,0)),N,replace=False)

    # Assign positions
    x = positions[inds][:,0]
    y = positions[inds][:,1]
    z = positions[inds][:,2]

    return np.column_stack((x, y, z))

# This function is designed to be used with an externally supplied array of x,y,z coordinates. 
# It is useful to avoid cell overlap in high density situations or to make some unique geometry.
# After you create each population, delete those positions from the "master" list of coordinates.
# This will assure that no cells are placed on top of one another. 
# Written by Ben Latimer at University of Missouri (latimerb@missouri.edu)
def positions_list(positions=np.array([(0,0,0),(0,0,1)])):
    
    x = positions[:,0]
    y = positions[:,1]
    z = positions[:,2]

    return np.column_stack((x, y, z))



def xiter_random(N=1, min_x=0.0, max_x=1.0):
    return np.random.uniform(low=min_x, high=max_x, size=(N,))
