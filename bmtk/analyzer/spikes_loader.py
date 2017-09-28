import h5py
import numpy as np
import os
import datetime



def load_spikes(file_name,trial_name=None):

    """Loads spikes from multiple file formats

    :param file_name: name of spike file
    :param trian_name: name of a trial within a spike file

    :return spikes: dict with spikes

    """


    filename, file_extension = os.path.splitext(file_name)

    if file_extension ==".nwb":
        spikes = load_spikes_from_nwb(file_name,trial_name)

    if file_extension ==".txt":
        spikes = load_spikes_from_txt(file_name)

    if file_extension ==".h5":
        spikes = load_spikes_from_h5(file_name)


    return spikes

    
def load_spikes_from_txt(file_name):
    """
    Load spikes from the txt file
    
    
    :param file_name: name of spike file
    :param trian_name: name of a trial within a spike file

    :return spikes: dict with spikes


    """
    t = os.path.getmtime(file_name)
    print file_name, "modified on:", datetime.datetime.fromtimestamp(t)
    spk_ts,spk_gids = np.loadtxt(file_name, 
                                 dtype='float32,int',
                                 unpack=True)

    spikes = {}
    spikes["time"] =spk_ts
    spikes["gid"] =spk_gids

    print 'loaded spikes from txt'

    return spikes


def load_spikes_from_h5(file_name):
    """
    Load spikes from the hdf5 file
    
    
    :param file_name: name of spike file
    :param trian_name: name of a trial within a spike file

    :return spikes: dict with spikes


    """

    t = os.path.getmtime(file_name)
    print file_name, "modified on:", datetime.datetime.fromtimestamp(t)

    spikes ={}
    with h5py.File(file_name,'r') as h5:

        spikes["time"]=h5["time"][...]
        spikes["gid"]=h5["gid"][...]


    print 'loaded spikes from hdf5'

    return spikes


def load_spikes_from_nwb(file_name,trial_name):

    """
    Load spikes from the nwb file
    
    
    :param file_name: name of spike file
    :param trian_name: name of a trial within a spike file

    :return spikes: dict with spikes


    """
    f5 = h5py.File(file_name, 'r')

    
    spike_trains_handle = f5['processing/%s/spike_train' % trial_name] # nwb.SpikeTrain.get_processing(f5,'trial_0')

    spike_times = []
    spike_gids = []

    for gid in spike_trains_handle.keys():
    
        times_gid = spike_trains_handle['%d/data' %int(gid)][:]
        spike_times.extend(times_gid)
        spike_gids.extend([int(gid)]*len(times_gid))
        
    spikes = {}
    spikes["time"] = np.array(spike_times) 
    spikes["gid"] = np.array(spike_gids)

    return spikes
