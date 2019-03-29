import numpy as np

from bmtk.utils.reports.spike_trains import SpikeTrains, sort_order, pop_na


def test_spikes_nopopulation(file_path, pop_name):
    spikes = SpikeTrains.from_csv(file_path)
    assert(len(spikes.populations) == 1)
    assert(spikes.populations[0] == pop_name)
    assert(len(spikes) == 124)

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
    assert(len(spikes.get_times(0, population=pop_name)) == 15)
    assert(len(spikes.get_times(10, time_window=(500.0, 1000.0))) == 8)
    assert(len(spikes.get_times(10, population='INVALID', time_window=(500.0, 1000.0))) == 0)

    # Check spikes iterator
    last_nid = -1
    spike_counts = 0
    for spk in spikes.spikes(sort_order=sort_order.by_id):
        nid = spk[2]
        assert(last_nid <= nid)
        last_nid = nid
        spike_counts += 1
    assert(spike_counts == 124)

    last_st = -1.0
    spike_counts = 0
    for spk in spikes.spikes(sort_order=sort_order.by_time):
        spk_time = spk[0]
        assert(last_st <= spk_time)
        last_st = spk_time
        spike_counts += 1
    assert(spike_counts == 124)



test_spikes_nopopulation(file_path='spike_files/spikes.noheader.nopop.csv', pop_name=pop_na)
test_spikes_nopopulation(file_path='spike_files/spikes.one_pop.csv', pop_name='v1')