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
import pandas as pd
import numpy as np

try:
    from distutils.version import LooseVersion
    use_sort_values = LooseVersion(pd.__version__) >= LooseVersion('0.19.0')

except:
    use_sort_values = False


def spikes2dict(spikes_file):
    spikes_df = pd.read_csv(spikes_file, sep=' ', names=['time', 'gid'])

    if use_sort_values:
        spikes_sorted = spikes_df.sort_values(['gid', 'time'])
    else:
        spikes_sorted = spikes_df.sort(['gid', 'time'])

    spike_dict = {}
    for gid, spike_train in spikes_sorted.groupby('gid'):
        spike_dict[gid] = np.array(spike_train['time'])
    return spike_dict


def spike_files_equal(spikes_txt_1, spikes_txt_2, err=0.0001):
    trial_1 = spikes2dict(spikes_txt_1)
    trial_2 = spikes2dict(spikes_txt_2)
    if set(trial_1.keys()) != set(trial_2.keys()):
        return False

    for gid, spike_train1 in trial_1.items():
        spike_train2 = trial_2[gid]
        if len(spike_train1) != len(spike_train2):
            return False

        for s1, s2 in zip(spike_train1, spike_train2):
            if abs(s1 - s2) > err:
                return False

    return True


def get_mean_firing_rates(spike_gids, node_ids, tstop_msec):

    """
    Compute mean firing rate over the duration of the simulation
    
    :param spike_gids: gids of cells which spiked
    :param node_ids: np.array of node_ids

    :return mean_firing_rate: np.array mean firing rates

    """

    min_gid = np.min(node_ids)
    max_gid = np.max(node_ids)

    gid_bins = np.arange(min_gid-0.5,max_gid+1.5,1)
    hist,bins = np.histogram(spike_gids, bins=gid_bins)

    tstop_sec = tstop_msec*1E-3
    mean_firing_rates = hist/tstop_sec
    
    return mean_firing_rates



def spikes_equal_in_window(spikes1,spikes2,twindow):
    """
    Compare spikes within a time window    
    :param spikes1: dict with "time" and "gid" arrays for raster 1
    :param spikes2: dict with "time" and "gid" arrays for raster 2
    :param twindow: [tstart,tend] time window
    
    :return boolean: True if equal, False if different
    """

    ix1_window0=np.where(spikes1["time"]>twindow[0]) 
    ix1_window1=np.where(spikes1["time"]<twindow[1]) 
    ix1_window = np.intersect1d(ix1_window0,ix1_window1)


    ix2_window0=np.where(spikes2["time"]>twindow[0]) 
    ix2_window1=np.where(spikes2["time"]<twindow[1]) 
    ix2_window = np.intersect1d(ix2_window0,ix2_window1)

    print(len(spikes1["time"][ix1_window]),len(spikes2["time"][ix2_window]))
    if len(spikes1["time"][ix1_window]) != len(spikes2["time"][ix2_window]):
        print("There is a DIFFERENT number of spikes in each file within the window")
        print("No point to compare individual spikes")
        return
    else: 
        print("number of spikes are the same, checking details...")
    ix1_sort = np.argsort(spikes1["time"][ix1_window],kind="mergesort")
    ix2_sort = np.argsort(spikes2["time"][ix2_window],kind="mergesort")


    if (np.array_equal(spikes1["gid"][ix1_window[ix1_sort]],spikes2["gid"][ix2_window[ix2_sort]])) and (np.array_equal(spikes1["time"][ix1_window[ix1_sort]],spikes2["time"][ix2_window[ix2_sort]])):
        print("spikes are IDENTICAL!")
        return True
    else:
        print("spikes are DIFFERENT")
        return False

