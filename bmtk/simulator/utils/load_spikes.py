import h5py
import numpy as np
import os
import datetime


def load_spikes_ascii(file_name):
    '''
    Load ascii spike file
    '''
    t = os.path.getmtime(file_name)
    print file_name, "modified on:", datetime.datetime.fromtimestamp(t)
    spk_ts,spk_gids = np.loadtxt(file_name, 
                                 dtype='float32,int',
                                 unpack=True)

    spk_ts=spk_ts*1E-3

    print 'loaded spikes from ascii'

    return [spk_ts,spk_gids]


def load_spikes_h5(file_name):
    '''
    Load ascii spike file
    '''

    t = os.path.getmtime(file_name)
    print file_name, "modified on:", datetime.datetime.fromtimestamp(t)

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

