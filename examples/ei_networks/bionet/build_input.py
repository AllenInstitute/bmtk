import os
import numpy as np
import h5py

from bmtk.utils.io.spike_trains import PoissonSpikesGenerator
from bmtk.builder.aux.node_params import positions_columinar, xiter_random

#input_ids = [n.node_id for n in inputNetwork.nodes()]

input_h5 = h5py.File('network/external_nodes.h5','r')
node_ids = np.array(input_h5['/nodes/external/node_id'],dtype = np.uint)
input_ids = node_ids.tolist()

from bmtk.utils.io.spike_trains import PoissonSpikesGenerator

# Create a Poisson Spike train for all input nodes that fire at a rate of 0.5Hz.
# The time units below is in milliseconds

psg = PoissonSpikesGenerator(gids=input_ids, firing_rate = 2.2, tstart=0.0, tstop=3000.0)

# Save the spike trains
if not os.path.exists('network/source_input/'):
    os.makedirs('network/source_input/')

psg.to_hdf5(file_name='inputs/poission_r1000_10in2p2.h5')
