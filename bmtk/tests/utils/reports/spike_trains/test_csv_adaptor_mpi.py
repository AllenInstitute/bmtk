import pytest
import numpy as np
import pandas as pd
import tempfile

from bmtk.utils.reports.spike_trains.spike_train_buffer import STMPIBuffer, STCSVMPIBufferV2
# from bmtk.utils.reports.spike_trains.adaptors.csv_adaptors import write_csv, write_csv_itr
from bmtk.utils.reports.spike_trains.spikes_file_writers import write_csv, write_csv_itr
from bmtk.utils.reports.spike_trains import sort_order

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
except:
    MPI_rank = 0
    MPI_size = 1


def create_st_buffer_mpi(st_cls):
    # Helper for creating spike_trains object
    if issubclass(st_cls, STCSVMPIBufferV2):
        tmp_dir = tempfile.mkdtemp() if MPI_rank == 0 else None
        tmp_dir = comm.bcast(tmp_dir, 0)
        return st_cls(cache_dir=tmp_dir)
    else:
        return st_cls()


def tmpdir():
    tmp_dir = tempfile.mkdtemp() if MPI_rank == 0 else None
    tmp_dir = comm.bcast(tmp_dir, 0)
    return tmp_dir


def tmpfile():
    tmp_file = tempfile.NamedTemporaryFile(suffix='.csv').name if MPI_rank == 0 else None
    tmp_file = comm.bcast(tmp_file, 0)
    return tmp_file


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)
    st.add_spikes(population='V1', node_ids=MPI_rank, timestamps=[MPI_rank]*5)
    st.add_spike(population='V2', node_id=MPI_size, timestamp=float(MPI_rank))
    st.add_spikes(population='R{}'.format(MPI_rank), node_ids=0, timestamps=[0.1, 0.2, 0.3, 0.4])

    tmp_csv = tmpfile()
    write_fnc(tmp_csv, st)

    df = pd.read_csv(tmp_csv, sep=' ')
    assert(set(df['population'].unique()) == {'R{}'.format(r) for r in range(MPI_size)} | {'V1', 'V2'})
    assert(df.shape == (5*MPI_size + MPI_size + 4*MPI_size, 3))
    assert(set(df[df['population'] == 'V1']['node_ids'].unique()) == {i for i in range(MPI_size)})
    assert(set(df[df['population'] == 'V1']['timestamps']) == {float(i) for i in range(MPI_size)})
    for r in range(MPI_size):
        rank_df = df[df['population'] == 'R{}'.format(r)]
        assert(np.all(rank_df['node_ids'] == [0, 0, 0, 0]))
        assert(np.allclose(np.sort(rank_df['timestamps']), [0.1, 0.2, 0.3, 0.4]))


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv_byid(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)
    st.add_spikes(population='V1', node_ids=[MPI_size + MPI_rank, MPI_rank], timestamps=[0.5, 1.0])

    tmp_csv = tmpfile()
    write_fnc(tmp_csv, st, sort_order=sort_order.by_id)

    df = pd.read_csv(tmp_csv, sep=' ')
    assert(df.shape == (MPI_size*2, 3))
    assert(np.all(df['population'].unique() == ['V1']))
    assert(np.all(df['node_ids'] == np.arange(MPI_size*2)))


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv_bytime(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)
    st.add_spikes(population='V1', node_ids=[MPI_rank, MPI_rank],
                  timestamps=np.array([MPI_rank/10.0, (MPI_size + MPI_rank)/10.0], dtype=np.float))

    tmp_csv = tmpfile()
    write_fnc(tmp_csv, st, sort_order=sort_order.by_time)

    df = pd.read_csv(tmp_csv, sep=' ')
    assert(df.shape == (MPI_size*2, 3))
    assert(np.all(df['population'].unique() == ['V1']))
    assert(np.all(np.diff(df['timestamps']) > 0))


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_csv,
    write_csv_itr
])
def test_write_csv_empty(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)

    tmp_csv = tmpfile()
    write_fnc(tmp_csv, st, sort_order=sort_order.by_time)

    df = pd.read_csv(tmp_csv, sep=' ')
    assert(df.shape == (0, 3))

if __name__ == '__main__':
    # test_write_csv(STMPIBuffer, write_csv_itr)
    # test_write_csv(STMPIBuffer, write_csv)

    # test_write_csv_byid(STMPIBuffer, write_csv)
    # test_write_csv_bytime(STMPIBuffer, write_csv)

    test_write_csv_empty(STMPIBuffer, write_csv)

