import numpy as np

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    barrier = comm.barrier

except ImportError:
    mpi_rank = 0
    mpi_size = 1
    barrier = lambda: None


def add_hdf5_attrs(hdf5_handle):
    # TODO: move this as a utility function
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)
    hdf5_handle['/'].attrs['version'] = [np.uint32(0), np.uint32(1)]
