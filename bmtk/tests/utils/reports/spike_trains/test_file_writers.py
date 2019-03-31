import os
import numpy as np
import pandas as pd
import h5py

from bmtk.utils.reports.spike_trains import SpikeTrains, sort_order, pop_na
from bmtk.utils.reports.spike_trains.csv_adaptors import write_csv
from bmtk.utils.reports.spike_trains.sonata_adaptors import write_sonata

# from bmtk.utils.reports.spike_trains.spike_train_buffer import STBufferedWriter

def load_spike_trains(file_path):
    if file_path.endswith('.csv'):
        return SpikeTrains.from_csv(file_path)

    elif file_path.endswith('.h5'):
        return SpikeTrains.from_sonata(file_path)

    elif file_path.endswith('.nwb'):
        return SpikeTrains.from_nwb(file_path)


def test_csv_writer_onepop(input_path, pop_name):
    spikes = load_spike_trains(input_path)
    output_path = 'output/tmpspikes.csv'
    write_csv(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_time)
    output_df = pd.read_csv(output_path, sep=' ')
    assert(len(output_df) == 124)
    assert(output_df['population'].unique() == [pop_name])
    assert(np.all(np.diff(output_df['timestamps']) >= 0))

    write_csv(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_id)
    output_df = pd.read_csv(output_path, sep=' ')
    assert(len(output_df) == 124)
    assert(np.all(np.diff(output_df['node_ids']) >= 0))


def test_csv_writer_multipop(input_path):
    spikes = load_spike_trains(input_path)
    output_path = 'output/tmpspikes.csv'
    write_csv(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_time)
    output_df = pd.read_csv(output_path, sep=' ')
    assert(len(output_df) == 144434)
    assert(np.all(np.diff(output_df['timestamps']) >= 0))
    os.remove(output_path)

    write_csv(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_id)
    output_df = pd.read_csv(output_path, sep=' ')
    assert(len(output_df) == 144434)
    output_lgn = output_df[output_df['population'] == 'v1']
    assert(np.all(np.diff(output_lgn['node_ids']) >= 0))
    output_tw = output_df[output_df['population'] == 'tw']
    assert(np.all(np.diff(output_tw['node_ids']) >= 0))
    os.remove(output_path)


def test_sonata_writer_onepop(input_path, pop_name):
    spikes = load_spike_trains(input_path)
    output_path = 'output/tmpspikes.h5'
    write_sonata(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_time)
    spikes_h5 = h5py.File(output_path, 'r')
    spikes_grp = spikes_h5['/spikes/{}'.format(pop_name)]
    assert(spikes_grp.attrs['sorting'] == 'by_time')
    timestamps = spikes_grp['timestamps'][()]
    assert(len(timestamps) == 124)
    assert(np.all(np.diff(timestamps) >= 0))
    node_ids = spikes_grp['node_ids'][()]
    assert(len(node_ids) == 124)
    os.remove(output_path)

    write_sonata(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_id)
    spikes_h5 = h5py.File(output_path, 'r')
    spikes_grp = spikes_h5['/spikes/{}'.format(pop_name)]
    assert(spikes_grp.attrs['sorting'] == 'by_id')
    timestamps = spikes_grp['timestamps'][()]
    assert(len(timestamps) == 124)
    node_ids = spikes_grp['node_ids'][()]
    assert(np.all(np.diff(node_ids) >= 0))
    assert(len(node_ids) == 124)
    os.remove(output_path)


def test_sonata_writer_multipop(input_path):
    spikes = load_spike_trains(input_path)
    output_path = 'output/tmpspikes.h5'
    write_sonata(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_time)
    spikes_h5 = h5py.File(output_path, 'r')
    lgn_spikes = spikes_h5['/spikes/lgn']
    lgn_timestamps = lgn_spikes['timestamps'][()]
    assert(len(lgn_timestamps) == 123356)
    assert(np.all(np.diff(lgn_timestamps) >= 0))
    assert(len(lgn_spikes['node_ids']) == 123356)
    assert(len(spikes_h5['/spikes/tw/timestamps']) == 21078)
    assert(len(spikes_h5['/spikes/tw/node_ids']) == 21078)
    os.remove(output_path)

    write_sonata(path=output_path, spiketrain_reader=spikes, sort_order=sort_order.by_id)
    spikes_h5 = h5py.File(output_path, 'r')
    lgn_spikes = spikes_h5['/spikes/lgn']
    lgn_node_ids = lgn_spikes['node_ids'][()]
    assert(len(lgn_node_ids) == 123356)
    assert(np.all(np.diff(lgn_node_ids) >= 0))
    assert(len(lgn_spikes['timestamps']) == 123356)
    assert(len(spikes_h5['/spikes/tw/timestamps']))
    assert(len(spikes_h5['/spikes/tw/node_ids']))
    os.remove(output_path)


if __name__ == '__main__':
    test_csv_writer_onepop('spike_files/spikes.noheader.nopop.csv', pop_name=pop_na)
    test_csv_writer_onepop('spike_files/spikes.one_pop.csv', pop_name='v1')
    test_csv_writer_onepop('spike_files/spikes.old.h5', pop_name=pop_na)
    test_csv_writer_onepop('spike_files/spikes.one_pop.h5', pop_name='v1')
    test_csv_writer_onepop('spike_files/spikes.onepop.v1.0.nwb', pop_name=pop_na)

    test_csv_writer_multipop('spike_files/spikes.multipop.csv')
    test_csv_writer_multipop('spike_files/spikes.multipop.h5')

    test_sonata_writer_onepop('spike_files/spikes.noheader.nopop.csv', pop_name=pop_na)
    test_sonata_writer_onepop('spike_files/spikes.one_pop.csv', pop_name='v1')
    test_sonata_writer_onepop('spike_files/spikes.old.h5', pop_name=pop_na)
    test_sonata_writer_onepop('spike_files/spikes.one_pop.h5', pop_name='v1')
    test_sonata_writer_onepop('spike_files/spikes.onepop.v1.0.nwb', pop_name=pop_na)

    test_sonata_writer_multipop('spike_files/spikes.multipop.csv')
    test_sonata_writer_multipop('spike_files/spikes.multipop.h5')
