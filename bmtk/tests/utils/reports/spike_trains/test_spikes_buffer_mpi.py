import pytest
import numpy as np
import tempfile
from six import string_types


from bmtk.utils.reports.spike_trains.spike_train_buffer import STMPIBuffer, STCSVMPIBufferV2

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
except:
    MPI_rank = 0
    MPI_size = 1


def tmpdir():
    tmp_dir = tempfile.mkdtemp() if MPI_rank == 0 else None
    tmp_dir = comm.bcast(tmp_dir, 0)
    return tmp_dir


# @pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_basic(st):
    st.add_spikes(population='V1', node_ids=MPI_rank, timestamps=[MPI_rank]*5)
    st.add_spike(population='V2', node_id=MPI_size, timestamp=float(MPI_rank))

    assert(set(st.populations) == {'V1', 'V2'})
    assert(set(st.get_populations(on_rank='all')) == {'V1', 'V2'})
    assert(set(st.get_populations(on_rank='local')) == {'V1', 'V2'})

    assert(st.n_spikes('V1') == MPI_size*5)
    assert(st.n_spikes('V1', on_rank='all') == MPI_size*5)
    assert(st.n_spikes('V1', on_rank='local') == 5)
    assert(st.n_spikes('V2') == MPI_size)
    assert(st.n_spikes('V2', on_rank='all') == MPI_size)
    assert(st.n_spikes('V2', on_rank='local') == 1)

    assert(np.all(np.sort(st.node_ids('V1')) == np.arange(MPI_size)))
    assert(np.all(np.sort(st.node_ids('V1', on_rank='all')) == np.arange(MPI_size)))
    assert(np.all(st.node_ids('V1', on_rank='local') == [MPI_rank]))
    assert(np.all(st.node_ids('V2') == [MPI_size]))
    assert(np.all(st.node_ids('V2', on_rank='all') == [MPI_size]))
    assert(np.all(st.node_ids('V2', on_rank='local') == [MPI_size]))

    assert(np.allclose(st.get_times(population='V1', node_id=0), [0.0]*5))
    assert(np.allclose(st.get_times(population='V1', node_id=0, on_rank='all'), [0.0]*5))
    times = st.get_times(population='V1', node_id=0, on_rank='local')
    if MPI_rank == 0:
        assert(np.allclose(times, [0.0, 0.0, 0.0, 0.0, 0.0]))
    else:
        assert(len(times) == 0)
    assert(np.allclose(np.sort(st.get_times(population='V2', node_id=MPI_size, on_rank='all')),
                       np.arange(MPI_size).astype(np.double)))
    assert(np.allclose(st.get_times(population='V2', node_id=MPI_size, on_rank='local'), [float(MPI_rank)]))


# @pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_basic_root(st):
    st.add_spikes(population='V1', node_ids=0, timestamps=[float(MPI_rank)])
    st.add_spike(population='V1', node_id=1, timestamp=1.0)
    st.add_spikes(population='R{}'.format(MPI_rank+1), node_ids=MPI_size+1, timestamps=[0.1, 0.2, 0.3])

    pops = st.get_populations(on_rank='root')
    n_spikes = st.n_spikes('V1', on_rank='root')
    node_ids = st.node_ids('V1', on_rank='root')
    timestamps = st.get_times(population='V1', node_id=0, on_rank='root')
    if MPI_rank == 0:
        assert(set(pops) == {'R{}'.format(r+1) for r in range(MPI_size)} | {'V1'})
        assert(n_spikes == MPI_size*2)
        assert(set(node_ids) == {0, 1})
        assert(np.allclose(np.sort(timestamps), np.arange(MPI_size, dtype=np.double)))
    else:
        assert(pops is None)
        assert(n_spikes is None)
        assert(node_ids is None)
        assert(timestamps is None)


# @pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_split_ids(st):
    st.add_spikes(population='V1', node_ids=MPI_rank, timestamps=[MPI_rank/10.0]*5)

    assert(set(st.populations) == {'V1'})
    assert(st.n_spikes('V1', on_rank='local') == 5)
    assert(st.n_spikes('V1', on_rank='all') == 5*MPI_size)
    assert(np.all(np.sort(st.node_ids(population='V1', on_rank='all')) == np.arange(MPI_size)))

    df = st.to_dataframe(populations='V1', on_rank='all')
    assert(len(df) == MPI_size*5)
    assert(np.all(np.sort(df['node_ids'].unique()) == np.arange(MPI_size)))

    df = st.to_dataframe(populations='V1', on_rank='root')
    if MPI_rank == 0:
        assert(len(df) == MPI_size*5)
    else:
        assert(df is None)


# @pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_to_dataframe(st):
    st.add_spikes(population='V1', node_ids=MPI_rank, timestamps=[MPI_rank / 10.0]*5)
    st.add_spike(population='V1', node_id=MPI_size, timestamp=float(MPI_rank))
    st.add_spikes(population='R{}'.format(MPI_rank), node_ids=MPI_size+1, timestamps=np.linspace(0.0, 1.0, 5))

    df = st.to_dataframe(on_rank='all')
    assert(len(df) == 5*MPI_size + MPI_size + 5*MPI_size)
    assert(set(df['population'].unique()) == {'R{}'.format(r) for r in range(MPI_size)} | {'V1'})
    assert(set(df[df['population'] == 'V1']['node_ids'].unique()) == {i for i in range(MPI_size+1)})

    df = st.to_dataframe(on_rank='local')
    assert(len(df) == 5 + 1 + 5)
    assert(set(df['population'].unique()) == {'R{}'.format(MPI_rank), 'V1'})

    df = st.to_dataframe(on_rank='root')
    if MPI_rank == 0:
        assert(len(df) == 5*MPI_size + MPI_size + 5*MPI_size)
        assert(set(df['population'].unique()) == {'R{}'.format(r) for r in range(MPI_size)} | {'V1'})
        assert(set(df[df['population'] == 'V1']['node_ids'].unique()) == {i for i in range(MPI_size+1)})
    else:
        assert(df is None)


# @pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_iterator(st):
    st.add_spikes(population='V1', node_ids=MPI_rank, timestamps=[MPI_rank / 10.0]*5)
    st.add_spike(population='V1', node_id=MPI_size, timestamp=float(MPI_rank))
    st.add_spikes(population='R{}'.format(MPI_rank), node_ids=MPI_size+1, timestamps=np.linspace(0.0, 1.0, 5))

    all_spikes = list(st.spikes(on_rank='all'))
    assert(len(all_spikes) == 5*MPI_size + MPI_size + 5*MPI_size)
    assert(isinstance(all_spikes[0][0], (np.double, np.float)))
    assert(isinstance(all_spikes[0][1], string_types))
    assert(isinstance(all_spikes[0][2], (np.int, np.uint)))
    assert({s[1] for s in all_spikes} == {'R{}'.format(r) for r in range(MPI_size)} | {'V1'})
    assert({s[2] for s in all_spikes} == {i for i in range(MPI_size+2)})

    local_spikes = list(st.spikes(on_rank='local'))
    assert(len(local_spikes) == 11)
    assert({s[1] for s in local_spikes} == {'R{}'.format(MPI_rank), 'V1'})

    root_spikes = list(st.spikes(on_rank='root'))
    if MPI_rank == 0:
        assert(len(root_spikes) == 5*MPI_size + MPI_size + 5*MPI_size)
        assert({s[1] for s in root_spikes} == {'R{}'.format(r) for r in range(MPI_size)} | {'V1'})
    else:
        assert(len(root_spikes) == 0)


@pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_no_root_spikes(st):
    # An issue we've had before where one of the ranks doesn't have any spiking neurons
    if MPI_rank > 0:
        st.add_spikes(population='R{}'.format(MPI_rank), node_ids=MPI_rank, timestamps=np.linspace(0.0, 1.0, 5))

    assert(set(st.populations)  == {'R{}'.format(r) for r in range(1, MPI_size)})
    assert(st.to_dataframe(on_rank='all').shape == ((MPI_size-1)*5, 3))
    assert(st.to_dataframe(on_rank='local').shape == (0 if MPI_rank == 0 else 5, 3))


# @pytest.mark.skipif(MPI_size < 2, reason='Can only run test using mpi')
@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_root_spikesonly(st):
    # Similar to above except only rank 0 has spikes
    if MPI_rank == 0:
        st.add_spikes(population='R{}'.format(MPI_rank), node_ids=MPI_rank, timestamps=np.linspace(0.0, 1.0, 5))

    assert(st.populations == ['R0'])
    assert(st.to_dataframe(on_rank='all').shape == (5, 3))
    assert(st.to_dataframe(on_rank='local').shape == (0 if MPI_rank != 0 else 5, 3))


@pytest.mark.parametrize('st', [
    STMPIBuffer(default_population='V1'),
    STCSVMPIBufferV2(cache_dir=tmpdir())
])
def test_no_spikes(st):
    # Make sure it still works even if there are no spikes
    assert(len(st.populations) == 0)
    assert(st.to_dataframe(on_rank='all').shape == (0, 3))
    assert(st.to_dataframe(on_rank='local').shape == (0, 3))
    df = st.to_dataframe(on_rank='root')
    assert(df is None if MPI_rank != 0 else df.shape == (0, 3))

    assert(list(st.spikes(on_rank='all')) == [])
    assert(list(st.spikes(on_rank='all')) == [])
    assert(list(st.spikes('all')) == [])


if __name__ == '__main__':
    # test_basic(STMPIBuffer(default_population='V1'))
    # test_basic(STCSVMPIBufferV2(cache_dir=tmpdir()))

    # test_basic_root(STMPIBuffer(default_population='V1'))
    # test_basic_root(STCSVMPIBufferV2(cache_dir=tmpdir()))

    # test_split_ids(STMPIBuffer(default_population='V1'))
    # test_split_ids(STCSVMPIBufferV2(cache_dir=tmpdir()))

    # test_to_dataframe(STMPIBuffer(default_population='V1'))
    # test_to_dataframe(STCSVMPIBufferV2(cache_dir=tmpdir()))

    # test_iterator(STMPIBuffer(default_population='V1'))
    # test_iterator(STCSVMPIBufferV2(cache_dir=tmpdir()))

    # test_no_root_spikes(STMPIBuffer(default_population='V1'))
    # test_root_spikesonly(STCSVMPIBufferV2(cache_dir=tmpdir()))

    test_no_spikes(STMPIBuffer(default_population='V1'))
    # test_no_spikes(STCSVMPIBufferV2(cache_dir=tmpdir()))
