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
