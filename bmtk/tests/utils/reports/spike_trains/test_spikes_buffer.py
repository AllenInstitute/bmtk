import pytest
import tempfile
import numpy as np
from six import string_types

from bmtk.utils.reports.spike_trains import sort_order
from bmtk.utils.reports.spike_trains.spike_train_buffer import STMemoryBuffer, STCSVBuffer, STMPIBuffer

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
    # STMPIBuffer(default_population='V1'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_add_spike(spiketrain_buffer):
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
    # STMPIBuffer(default_population='V1'),
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
    print(len(all_df))
    assert(set(all_df.columns) == {'node_ids', 'timestamps', 'population'})
    assert(set(all_df['population'].unique()) == {'V1', 'V2'})
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
    # STMPIBuffer(default_population='V1'),
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
    assert(isinstance(all_spikes[0][0], (np.double, np.float)))
    assert(isinstance(all_spikes[0][1], string_types))
    assert(isinstance(all_spikes[0][2], (np.int, np.uint, np.long)))

    assert(len(list(st.spikes(populations=['V1', 'V2']))) == 36)
    assert(len(list(st.spikes(populations='V2'))) == 10)

    v1_node_ids = [i[2] for i in st.spikes(sort_order=sort_order.by_id, populations=['V1'])]
    assert(np.all(np.diff(v1_node_ids) >= 0))

    v1_node_times = [i[0] for i in st.spikes(sort_order=sort_order.by_time, populations=['V1'])]
    assert(np.all(np.diff(v1_node_times) >= 0))


@pytest.mark.parametrize('spiketrain_buffer', [
    STMemoryBuffer(default_population='V1', store_type='list'),
    STMemoryBuffer(default_population='V1', store_type='array'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_invalid_pop(spiketrain_buffer):
    st = spiketrain_buffer
    st.add_spikes(node_ids=np.full(10, 0, dtype=np.uint), timestamps=np.linspace(0.1, 1.0, 10, endpoint=True))
    assert(st.n_spikes(population='INVALID') == 0)
    assert(len(st.node_ids(population='INVALID')) == 0)


@pytest.mark.parametrize('spiketrain_buffer', [
    STMemoryBuffer(default_population='V1', store_type='list'),
    STMemoryBuffer(default_population='V1', store_type='array'),
    STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp())
])
def test_no_spikes(spiketrain_buffer):
    st = spiketrain_buffer

    assert(len(st.populations) == 0)
    assert(st.to_dataframe().shape == (0, 3))
    assert(list(st.spikes()) == [])


if __name__ == '__main__':
    # if MPI_size == 1:
    #     #single_proc(spike_train_buffer.STCSVBuffer)
    #     # test_single_proc(spike_train_buffer.STMemoryBuffer)
    #     #test_psg_fixed()
    #     # test_psg_variable()

    # test_add_spike(STMemoryBuffer(default_population='V1', store_type='list'))
    # test_add_spikes(STMemoryBuffer(default_population='V1', store_type='array'))
    # test_add_spikes(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    # test_add_spikes(STMPIBuffer(default_population='V1'))
    # test_add_spike(STMPIBuffer(default_population='V1'))
    # test_to_dataframe(STMemoryBuffer(default_population='V1', store_type='array'))
    # test_to_dataframe(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    # test_to_dataframe(STMPIBuffer(default_population='V1'))
    # test_iterator(STMemoryBuffer(default_population='V1', store_type='list'))
    # test_iterator(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    # test_iterator(STMPIBuffer(default_population='V1'))

    # test_no_spikes(STMemoryBuffer(default_population='V1', store_type='list'))
    test_no_spikes(STCSVBuffer(default_population='V1', cache_dir=tempfile.mkdtemp()))
    # test_no_spikes(STMemoryBuffer(default_population='V1', store_type='list'))
