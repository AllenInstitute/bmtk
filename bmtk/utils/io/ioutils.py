class BMTKWorldComm(object):
    def __init__(self):
        self._comm = None

    @property
    def comm(self):
        if self._comm is None:
            try:
                from mpi4py import MPI
                self._comm = MPI.COMM_WORLD

            except Exception as exc:
                self._comm = None

        return self._comm

    @comm.setter
    def comm(self, comm):
        self._comm = comm

    @property
    def MPI_rank(self):
        if self.comm is None:
            return 0
        else:
            return self.comm.Get_rank()

    @property
    def MPI_size(self):
        if self.comm is None:
            return 1
        else:
            return self.comm.Get_size()

    def barrier(self):
        if self.comm is not None:
            self.comm.Barrier()


bmtk_world_comm = BMTKWorldComm()


def set_world_comm(comm):
    # global bmtk_world_comm
    bmtk_world_comm.comm = comm
