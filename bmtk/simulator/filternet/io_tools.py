# from bmtk.simulator.core.io_tools import iof
from bmtk.simulator.core.io_tools import IOUtils
from bmtk.utils.io.ioutils import bmtk_world_comm


class FilterNetIOUtils(IOUtils):
    def __init__(self):
        super(FilterNetIOUtils, self).__init__()
        self.mpi_rank = bmtk_world_comm.MPI_rank
        self.mpi_size = bmtk_world_comm.MPI_size

    def log_info(self, message, all_ranks=False):
        if all_ranks is False and self.mpi_rank != 0:
            return

        self.logger.info(message)


io = FilterNetIOUtils()
