import pandas as pd
import numpy as np


def spikes2dict(spikes_file):
    spikes_df = pd.read_csv(spikes_file, sep=' ', names=['time', 'gid'])
    spikes_sorted = spikes_df.sort_values(['gid', 'time'])
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

