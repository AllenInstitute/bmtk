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
import h5py
import numpy as np
import os
import datetime


def load_spikes_ascii(file_name):
    '''
    Load ascii spike file
    '''
    t = os.path.getmtime(file_name)
    print(file_name, "modified on:", datetime.datetime.fromtimestamp(t))
    spk_ts,spk_gids = np.loadtxt(file_name, 
                                 dtype='float32,int',
                                 unpack=True)

    spk_ts=spk_ts*1E-3

    print('loaded spikes from ascii')

    return [spk_ts,spk_gids]


def load_spikes_h5(file_name):
    '''
    Load ascii spike file
    '''

    t = os.path.getmtime(file_name)
    print(file_name, "modified on:", datetime.datetime.fromtimestamp(t))

    with h5py.File(file_name,'r') as h5:

        spk_ts=h5["time"][...]*1E-3
        spk_gids=h5["gid"][...]


    print('loaded spikes from hdf5')

    return [spk_ts,spk_gids]


def load_spikes_nwb(file_name,trial_name):

    '''
    Load spikes from the nwb file
    
    Returns:
    -------
    
    spike_times: list
    spike_gids: list
    '''
    f5 = h5py.File(file_name, 'r')

    
    spike_trains_handle = f5['processing/%s/spike_train' % trial_name] # nwb.SpikeTrain.get_processing(f5,'trial_0')

    spike_times = []
    spike_gids = []

    for gid in spike_trains_handle.keys():
    
        times_gid = spike_trains_handle['%d/data' %int(gid)][:]
        spike_times.extend(times_gid)
        spike_gids.extend([int(gid)]*len(times_gid))
        
    return [np.array(spike_times)*1E-3,np.array(spike_gids)]

