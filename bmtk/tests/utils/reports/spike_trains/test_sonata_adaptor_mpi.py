import pytest
import numpy as np
import h5py
import tempfile

from bmtk.utils.reports.spike_trains.spike_train_buffer import STMPIBuffer, STCSVMPIBufferV2
# from bmtk.utils.reports.spike_trains.adaptors.sonata_adaptors import write_sonata, write_sonata_itr
from bmtk.utils.reports.spike_trains.spikes_file_writers import write_sonata, write_sonata_itr
from bmtk.utils.sonata.utils import check_magic, get_version
from bmtk.utils.reports.spike_trains import sort_order

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    bcast = comm.bcast
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
    tmp_file = tempfile.NamedTemporaryFile(suffix='.h5').name if MPI_rank == 0 else None
    tmp_file = comm.bcast(tmp_file, 0)
    return tmp_file


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)
    st.add_spikes(population='V1', node_ids=MPI_rank, timestamps=[MPI_rank]*5)
    st.add_spike(population='V2', node_id=MPI_size, timestamp=float(MPI_rank))
    st.add_spikes(population='R{}'.format(MPI_rank), node_ids=0, timestamps=[0.1, 0.2, 0.3, 0.4])

    tmp_h5 = tmpfile()
    write_fnc(tmp_h5, st)

    if MPI_rank == 0:
        # Warnings: some systems creates lock even for reading an hdf5 file
        with h5py.File(tmp_h5, 'r') as h5:
            assert(check_magic(h5))
            assert(get_version(h5) is not None)
            assert(set(h5['/spikes'].keys()) >= {'R{}'.format(r) for r in range(MPI_size)} | {'V1', 'V2'})
            assert(set(h5['/spikes/V1']['node_ids'][()]) == {i for i in range(MPI_size)})
            assert(set(h5['/spikes/V2']['timestamps'][()]) == {float(i) for i in range(MPI_size)})
            for r in range(MPI_size):
                grp = h5['/spikes/R{}'.format(r)]
                assert(np.all(grp['node_ids'][()] == [0, 0, 0, 0]))
                assert(np.allclose(grp['timestamps'][()], [0.1, 0.2, 0.3, 0.4]))


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_byid(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)
    st.add_spikes(population='V1', node_ids=[MPI_size + MPI_rank, MPI_rank], timestamps=[0.5, 1.0])

    tmp_h5 = tmpfile()
    write_fnc(tmp_h5, st, sort_order=sort_order.by_id)

    if MPI_rank == 0:
        with h5py.File(tmp_h5, 'r') as h5:
            assert(check_magic(h5))
            assert(get_version(h5) is not None)
            assert(np.all(h5['/spikes/V1']['node_ids'][()] == list(range(MPI_size*2))))
            assert(len(h5['/spikes/V1']['timestamps'][()]) == MPI_size * 2)


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_bytime(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)
    st.add_spikes(population='V1', node_ids=[MPI_rank, MPI_rank],
                  timestamps=np.array([MPI_rank/10.0, (MPI_size + MPI_rank)/10.0], dtype=np.float))

    tmp_h5 = tmpfile()
    write_fnc(tmp_h5, st, sort_order=sort_order.by_time)

    if MPI_rank == 0:
        with h5py.File(tmp_h5, 'r') as h5:
            assert(check_magic(h5))
            assert(get_version(h5) is not None)
            assert(len(h5['/spikes/V1']['node_ids'][()]) == MPI_size*2)
            assert(np.all(np.diff(h5['/spikes/V1']['timestamps'][()]) > 0))


@pytest.mark.parametrize('st_cls', [
    STMPIBuffer,
    STCSVMPIBufferV2
])
@pytest.mark.parametrize('write_fnc', [
    write_sonata,
    write_sonata_itr
])
def test_write_sonata_empty(st_cls, write_fnc):
    st = create_st_buffer_mpi(st_cls)

    tmp_h5 = tmpfile()
    write_fnc(tmp_h5, st)

    if MPI_rank == 0:
        with h5py.File(tmp_h5, 'r') as h5:
            assert(check_magic(h5))
            assert(get_version(h5) is not None)
            assert('/spikes' in h5)


if __name__ == '__main__':
    # test_write_sonata(STMPIBuffer, write_sonata)
    # test_write_sonata(STMPIBuffer, write_sonata_itr)
    # test_write_sonata_byid(STMPIBuffer, write_sonata)
    # test_write_sonata_bytime(STMPIBuffer, write_sonata)
    test_write_sonata_empty(STMPIBuffer, write_sonata)
