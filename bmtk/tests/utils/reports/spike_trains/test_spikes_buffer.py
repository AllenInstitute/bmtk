import pytest
import os
import tempfile
import numpy as np
import pandas as pd
import h5py
import shutil
from six import string_types

# , STNumpyBuffer, STBuffer2
from bmtk.utils.reports.spike_trains import SpikeTrains, PoissonSpikeGenerator, pop_na
from bmtk.utils.reports.spike_trains import sort_order
from bmtk.utils.reports.spike_trains import spike_train_buffer
from bmtk.utils.reports.spike_trains.spike_train_buffer import STMemoryBuffer, STCSVBuffer

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
except:
    MPI_rank = 0
    MPI_size = 1


@pytest.mark.parametrize('spiketrain_buffer', [
    STMemoryBuffer(default_population='V1', store_type='list'),
    STMemoryBuffer(default_population='V1', store_type='array'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_add_spike(spiketrain_buffer): #spiketrain_buffer=STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())):
    st = spiketrain_buffer
    for ts in np.linspace(0.0, 3000.0, 12, endpoint=False):
        st.add_spike(node_id=0, timestamp=ts)
    st.add_spike(node_id=10, timestamp=0)
    st.add_spike(node_id=10, timestamp=1)
    st.add_spike(node_id=11, timestamp=2)
    for ts in np.linspace(250.0, 3000.0, 12, endpoint=True):
        st.add_spike(population='V2', node_id=10, timestamp=ts)

    assert(set(st.populations) == {'V1', 'V2'})
    assert(st.n_spikes() == 15)
    assert(st.n_spikes('V1') == 15)
    assert(set(st.node_ids()) == {0, 10, 11})
    assert(set(st.node_ids('V1')) == {0, 10, 11})
    assert(np.allclose(st.get_times(population='V1', node_id=0), np.linspace(0.0, 3000.0, 12, endpoint=False)))
    assert(np.allclose(st.get_times(node_id=10), [0.0, 1.0]))
    assert(np.allclose(st.get_times(node_id=11), [2.0]))
    assert(len(st.get_times(node_id=1000)) == 0)

    assert(st.n_spikes('V2') == 12)
    assert(set(st.node_ids('V2')) == {10})
    assert(np.allclose(st.get_times(population='V2', node_id=10), np.linspace(250.0, 3000.0, 12, endpoint=True)))


@pytest.mark.parametrize('spiketrain_buffer', [
    STMemoryBuffer(default_population='V1', store_type='list'),
    STMemoryBuffer(default_population='V1', store_type='array'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_add_spikes(spiketrain_buffer):
    st = spiketrain_buffer
    st.add_spikes(node_ids=np.full(10, 0, dtype=np.uint), timestamps=np.linspace(0.1, 1.0, 10, endpoint=True))
    st.add_spikes(node_ids=1, timestamps=np.arange(0, 5))
    st.add_spikes(node_ids=2, timestamps=[0.001])
    st.add_spikes(node_ids=[3, 4, 3], timestamps=[0.001, 0.002, 0.003])
    st.add_spikes(population='V2', node_ids=[1, 1, 1], timestamps=np.array([0.0, 1.01, 2.02]))

    assert(st.n_spikes() == 19)
    assert(set(st.node_ids()) == {0, 1, 2, 3, 4})
    assert(np.allclose(st.get_times(node_id=0), np.linspace(0.1, 1.0, 10, endpoint=True)))
    assert(np.allclose(st.get_times(node_id=1), [0.0, 1.0, 2.0, 3.0, 4.0]))
    assert(np.allclose(st.get_times(node_id=2), [0.001]))
    assert(np.allclose(st.get_times(node_id=3), [0.001, 0.003]))
    assert(np.allclose(st.get_times(population='V2', node_id=1), [0.0, 1.01, 2.02]))


@pytest.mark.parametrize('spiketrain_buffer', [
    STMemoryBuffer(default_population='V1', store_type='list'),
    STMemoryBuffer(default_population='V1', store_type='array'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_to_dataframe(spiketrain_buffer):
    st = spiketrain_buffer
    st.add_spikes(node_ids=0, timestamps=np.linspace(0.1, 1.0, 10, endpoint=True))
    st.add_spikes(node_ids=2, timestamps=np.full(9, 2.0))  # out of order to test by_id sorting
    st.add_spikes(node_ids=1, timestamps=[1.5])
    st.add_spikes(node_ids=5, timestamps=[5.0, 4.5, 6.5, 7.0, 1.5, 0.0])  # for testing by_time
    st.add_spikes(population='V2', node_ids=0, timestamps=np.linspace(0.1, 1.0, 10, endpoint=True))

    all_df = st.to_dataframe(with_population_col=True)
    assert(set(all_df.columns) == {'node_ids', 'timestamps', 'population'})
    assert(set(all_df['population'].unique()) == {'V1', 'V2'})
    assert(len(all_df) == 36)
    assert(len(all_df[all_df['population'] == 'V1']) == 26)
    assert(len(all_df[all_df['population'] == 'V2']) == 10)

    v1_df = st.to_dataframe(populations='V1', with_population_col=True)
    assert(len(v1_df) == 26)
    assert(all(v1_df['population'].unique() == ['V1']))
    assert(np.allclose(np.sort(v1_df[v1_df['node_ids'] == 2]['timestamps']), np.full(9, 2.0)))

    v1_df_id = st.to_dataframe(populations='V1', sort_order=sort_order.by_id, with_population_col=False)
    assert('population' not in v1_df_id.columns)
    assert(np.all(np.diff(v1_df_id['node_ids']) >= 0))

    v1_df_time = st.to_dataframe(populations='V1', sort_order=sort_order.by_time, with_population_col=False)
    assert(np.all(np.diff(v1_df_time['timestamps']) >= 0))


@pytest.mark.parametrize('spiketrain_buffer', [
    STMemoryBuffer(default_population='V1', store_type='list'),
    STMemoryBuffer(default_population='V1', store_type='array'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_iterator(spiketrain_buffer):
    st = spiketrain_buffer
    st.add_spikes(node_ids=0, timestamps=np.linspace(0.1, 1.0, 10, endpoint=True))
    st.add_spikes(node_ids=2, timestamps=np.full(9, 2.0))  # out of order to test by_id sorting
    st.add_spikes(node_ids=1, timestamps=[1.5])
    st.add_spikes(node_ids=5, timestamps=[5.0, 4.5, 6.5, 7.0, 1.5, 0.0])  # for testing by_time
    st.add_spikes(population='V2', node_ids=0, timestamps=np.linspace(0.1, 1.0, 10, endpoint=True))

    all_spikes = list(st.spikes())
    assert(len(all_spikes) == 36)
    #print(all_spikes)
    #print(type(all_spikes[0][0]))
    assert(isinstance(all_spikes[0][0], (np.double, np.float)))
    assert(isinstance(all_spikes[0][1], string_types))
    assert(isinstance(all_spikes[0][2], np.int))

    assert(len(list(st.spikes(populations=['V1', 'V2']))) == 36)
    assert(len(list(st.spikes(populations='V2'))) == 10)

    v1_node_ids = [i[2] for i in st.spikes(sort_order=sort_order.by_id, populations=['V1'])]
    assert(np.all(np.diff(v1_node_ids) >= 0))

    v1_node_times = [i[0] for i in st.spikes(sort_order=sort_order.by_time, populations=['V1'])]
    assert(np.all(np.diff(v1_node_times) >= 0))

@pytest.mark.skip()
@pytest.mark.parametrize('adaptor_cls', [
    spike_train_buffer.STCSVBuffer,
    spike_train_buffer.STMemoryBuffer
])
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


@pytest.mark.skip()
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

@pytest.mark.skip()
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
    # if MPI_size == 1:
    #     #single_proc(spike_train_buffer.STCSVBuffer)
    #     # test_single_proc(spike_train_buffer.STMemoryBuffer)
    #     #test_psg_fixed()
    #     # test_psg_variable()

    # test_add_spike(STMemoryBuffer(default_population='V1', store_type='list'))
    # test_add_spikes(STMemoryBuffer(default_population='V1', store_type='array'))
    # test_add_spikes(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    # test_to_dataframe(STMemoryBuffer(default_population='V1', store_type='array'))
    # test_to_dataframe(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    # test_iterator(STMemoryBuffer(default_population='V1', store_type='list'))
    test_iterator(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    #else:
    #    pass
    #    #multi_proc()
