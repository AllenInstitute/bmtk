from .compartment_reader import CompartmentReaderVer01 as SonataReaderDefault
from .compartment_writer import CompartmentWriterv01 as SonataWriterDefault


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhosts = comm.Get_size()

except Exception as exc:
    pass


class CompartmentReport(object):
    def __new__(cls, path, mode='r', adaptor=None, *args, **kwargs):
        if adaptor is not None:
            return adaptor
        else:
            if mode == 'r':
                return SonataReaderDefault(path, mode, **kwargs)
            else:
                return SonataWriterDefault(path, mode, **kwargs)
