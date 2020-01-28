import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import h5py
import shutil

# , STNumpyBuffer, STBuffer2
from bmtk.utils.reports.spike_trains import SpikeTrains, PoissonSpikeGenerator, pop_na
from bmtk.utils.reports.spike_trains import sort_order
from bmtk.utils.reports.spike_trains import spike_train_buffer


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
except:
    MPI_rank = 0
    MPI_size = 1


@pytest.mark.parametrize('adaptor_cls', [spike_train_buffer.STCSVBuffer, spike_train_buffer.STMemoryBuffer])
def test_single_proc(adaptor_cls):
    buffer_dir = tempfile.mkdtemp()
    output_csv = os.path.join(buffer_dir, 'testfile.csv')
    output_h5 = os.path.join(buffer_dir, 'testfile.h5')

    adaptor = adaptor_cls()
    spike_trains = SpikeTrains(read_adaptor=adaptor, write_adaptor=adaptor)
    timestamps = np.linspace(1000.0, 0.0, 1000)
    node_ids = np.arange(0, 1000)
    for node_id, timestamp in zip(node_ids, timestamps):
        spike_trains.add_spike(node_id, timestamp)

    for node_id in range(1000, 2000):
        spike_trains.add_spikes(node_id, np.linspace(0.0, 2000.0, 100))

    for node_id in range(0, 100, 5):
        spike_trains.add_spikes(np.repeat(node_id, 50), np.random.uniform(0.1, 3000.0, 50), population='test')

    spike_trains.to_csv(output_csv, sort_order=sort_order.by_time)
    df = pd.read_csv(output_csv, sep=' ')
    assert(len(df) == 102000)
    assert(len(df['population'].unique()) == 2)
    test_pop = df[df['population'] == 'test']
    assert(len(test_pop) == 20*50)
    assert(all(np.diff(test_pop['timestamps']) >= 0.0))

    default_pop = df[df['population'] == pop_na]
    assert(len(default_pop) == 1000 + 1000*100)
    assert(all(np.diff(default_pop['timestamps']) >= 0.0))

    spike_trains.to_sonata(output_h5, sort_order=sort_order.by_id)
    h5root = h5py.File(output_h5, 'r')
    test_pop = h5root['spikes/test']
    assert(test_pop.attrs['sorting'] == 'by_id')
    assert(test_pop['timestamps'].shape == (1000,))
    assert(test_pop['node_ids'].shape == (1000,))
    assert(len(np.unique(test_pop['node_ids'][()])) == 20)
    assert(all(np.diff(test_pop['node_ids'][()]) >= 0))

    default_pop = h5root['spikes'][pop_na]
    assert(default_pop.attrs['sorting'] == 'by_id')
    assert(default_pop['timestamps'].shape == (1000 + 1000*100,))
    assert(default_pop['node_ids'].shape == (1000 + 1000*100,))
    assert(all(np.diff(default_pop['node_ids'][()]) >= 0))
    assert(all(np.diff(default_pop['node_ids']) >= 0))
    assert(len(np.unique(default_pop['node_ids'][()])) == 2000)

    spike_trains.close()
    shutil.rmtree(buffer_dir)


def test_psg_fixed():
    psg = PoissonSpikeGenerator(population='test', seed=0.0)
    psg.add(node_ids=range(10), firing_rate=10.0, times=(0.0, 10.0))
    assert(psg.populations == ['test'])
    assert(psg.nodes() == list(range(10)))

    time_range = psg.time_range()
    assert(0 <= time_range[0] < 1.0)
    assert(9.0 < time_range[1] <= 10.0)

    # This may fail on certain versions
    assert(psg.get_times(node_id=5).size > 10)
    assert(0 < psg.get_times(node_id=8).size < 300)

    for i in range(10):
        spikes = psg.get_times(i)
        assert(np.max(spikes) > 0.1)


def test_psg_variable():
    times = np.linspace(0.0, 10.0, 1000)
    fr = np.exp(-np.power(times - 5.0, 2) / (2*np.power(.5, 2)))*5

    psg = PoissonSpikeGenerator(population='test', seed=0.0)
    psg.add(node_ids=range(10), firing_rate=fr, times=times)

    assert(psg.populations == ['test'])
    assert(psg.nodes() == list(range(10)))

    for i in range(10):
        spikes = psg.get_times(i)
        assert(len(spikes) > 0)
        assert(1.0 < np.min(spikes))
        assert(np.max(spikes) < 9.0)

'''
def multi_proc():
    # from bmtk.utils.reports.spike_trains.spike_train_buffer import STMemoryBuffer as STBuffer
    from bmtk.utils.reports.spike_trains.spike_train_buffer import STMPIBuffer as STBuffer

    adaptor = STBuffer()
    spike_trains = SpikeTrains(read_adaptor=adaptor, write_adaptor=adaptor)
    timestamps = np.linspace(1000.0, 0.0, 1000)
    node_ids = np.arange(0, 1000)
    for i in range(MPI_rank, len(node_ids), MPI_size):
        spike_trains.add_spike(node_ids[i], timestamps[i])

    for node_id in range(1000 + MPI_rank, 3000, MPI_size):
        spike_trains.add_spikes(node_id, np.linspace(0.0, 2000.0, 500))

    n_nodes = int(100/MPI_size)

    node_ids = range(MPI_rank*n_nodes, (MPI_rank+1)*n_nodes, 5)
    for node_id in node_ids:
        spike_trains.add_spikes(np.repeat(node_id, 50), np.random.uniform(0.1, 3000.0, 50), population='test')

    spike_trains.to_sonata('testfile_mpi.h5', sort_order=sort_order.by_time)
'''

if __name__ == '__main__':
    if MPI_size == 1:
        #single_proc(spike_train_buffer.STCSVBuffer)
        # test_single_proc(spike_train_buffer.STMemoryBuffer)
        #test_psg_fixed()
        test_psg_variable()

    else:
        pass
        #multi_proc()
