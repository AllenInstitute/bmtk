import numpy as np
import pytest
import six

from bmtk.utils.reports.spike_trains import SpikeTrains, sort_order, pop_na


def load_spike_trains(file_path):
    if file_path.endswith('.csv'):
        return SpikeTrains.from_csv(file_path)

    elif file_path.endswith('.h5'):
        return SpikeTrains.from_sonata(file_path)

    elif file_path.endswith(('.nwb')):
        return SpikeTrains.from_nwb(file_path)


def test_spikes_nopopulation(file_path, pop_name):
    spikes = load_spike_trains(file_path) # SpikeTrains.from_csv(file_path)
    # print(spikes.populations)
    assert(len(spikes.populations) == 1)
    assert(spikes.populations[0] == pop_name)
    assert(len(spikes) == 124)

    node_list = spikes.nodes()
    assert(len(node_list) == 14)
    assert(len(node_list[0]) == 2)
    assert(isinstance(node_list[0][0], (six.string_types, np.integer)))  # first value is the name/id of population
    assert(node_list[0][0] == pop_name)
    assert(isinstance(node_list[0][1], np.integer))  # second value is node_id
    assert(len(spikes.nodes(populations='INVALID')) == 0)
    assert(len(spikes.nodes(populations=pop_name)) == 14)
    assert(len(spikes.nodes(populations=[pop_name])) == 14)

    assert(np.all(spikes.time_range() == (136.6, 958.0)))

    spikes_df = spikes.to_dataframe()
    assert(spikes_df.shape == (124, 3))
    assert(np.issubdtype(spikes_df['node_ids'].dtype, np.integer))
    assert(np.issubdtype(spikes_df['timestamps'].dtype, np.floating))

    # check ability to filter to_dataframe method
    assert(len(spikes.to_dataframe(populations=spikes.populations)) == 124)
    assert(spikes.to_dataframe(populations=pop_name).shape == (124, 3))
    assert(spikes.to_dataframe(populations='INVALID_POP').shape == (0, 3))
    assert(spikes.to_dataframe(populations=['INVALID_POP', pop_name]).shape == (124, 3))

    filtered_nodes = spikes.to_dataframe(node_ids=0)
    assert(len(filtered_nodes) == 15)
    assert(np.all(filtered_nodes['node_ids'].unique() == [0]))

    filtered_nodes = spikes.to_dataframe(node_ids=[10, 11, 13, 12, 14])
    assert(len(filtered_nodes) == 44)
    assert(set(filtered_nodes['node_ids'].unique()) == set([10, 11, 12, 13]))

    filtered_nodes = spikes.to_dataframe(node_ids=[10, 11, 13, 12, 14], time_window=(200.0, 800.0))
    assert(len(filtered_nodes) == 36)
    for _, row in filtered_nodes.iterrows():
        assert(row['node_ids'] in [10, 11, 12, 13])
        assert(200 <= row['timestamps'] <= 800.0)

    # Check get_times method
    assert(len(spikes.get_times(0)) == 15)
    assert(isinstance(spikes.get_times(0)[0], np.float))
    assert(len(spikes.get_times(0, population=pop_name)) == 15)
    assert(len(spikes.get_times(10, time_window=(500.0, 1000.0))) == 8)
    assert(len(spikes.get_times(10, population='INVALID', time_window=(500.0, 1000.0))) == 0)

    # Check spikes iterator
    spike_counts = 0
    for spk in spikes.spikes():
        assert(isinstance(spk[0], np.float))
        assert(isinstance(spk[1], six.string_types))
        assert(isinstance(spk[2], (np.int, np.integer)))
        spike_counts += 1
    assert(spike_counts == 124)

    last_nid = -1
    spike_counts = 0
    for spk in spikes.spikes(sort_order=sort_order.by_id):
        assert(isinstance(spk[0], np.float))
        assert(isinstance(spk[1], six.string_types))
        assert(isinstance(spk[2], (np.int, np.integer)))
        nid = spk[2]
        assert(last_nid <= nid)
        last_nid = nid
        spike_counts += 1
    assert(spike_counts == 124)

    last_st = -1.0
    spike_counts = 0
    for spk in spikes.spikes(sort_order=sort_order.by_time):
        assert(isinstance(spk[0], np.float))
        assert(isinstance(spk[1], six.string_types))
        assert(isinstance(spk[2], (np.int, np.integer)))
        spk_time = spk[0]
        assert(last_st <= spk_time)
        last_st = spk_time
        spike_counts += 1
    assert(spike_counts == 124)

if __name__ == '__main__':
    #test_spikes_nopopulation(file_path='spike_files/spikes.noheader.nopop.csv', pop_name=pop_na)
    #test_spikes_nopopulation(file_path='./spike_files/spikes.one_pop.csv', pop_name='v1')
    #test_spikes_nopopulation(file_path='spike_files/spikes.old.h5', pop_name=pop_na)
    #test_spikes_nopopulation(file_path='spike_files/spikes.one_pop.h5', pop_name='v1')
    test_spikes_nopopulation(file_path='spike_files/spikes.onepop.v1.0.nwb', pop_name=pop_na)
