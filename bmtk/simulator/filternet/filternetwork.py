import numpy as np

from bmtk.simulator.core.simulator_network import SimNetwork
from bmtk.simulator.filternet.cell import Cell
from bmtk.simulator.filternet.pyfunction_cache import py_modules
from bmtk.simulator.filternet.sonata_adaptors import FilterNodeAdaptor
from bmtk.utils.io.ioutils import bmtk_world_comm
from bmtk.simulator.filternet.io_tools import io


class FilterNetwork(SimNetwork):
    def __init__(self):
        super(FilterNetwork, self).__init__()

        self._local_cells = []
        self._network_jitter = (1.0, 1.0)
        self._io = io

    @property
    def jitter(self):
        return self._network_jitter

    @jitter.setter
    def jitter(self, val):
        assert(len(val) == 2)
        assert(val[0] <= val[1])
        self._network_jitter = val

    def _register_adaptors(self):
        super(FilterNetwork, self)._register_adaptors()
        self._node_adaptors['sonata'] = FilterNodeAdaptor

    def cells(self):
        return self._local_cells

    def build(self):
        self.build_nodes()

    def set_default_processing(self, processing_fnc):
        py_modules.add_cell_processor('default', processing_fnc)

    def build_nodes(self):
        rank_msg = '' if bmtk_world_comm.MPI_size < 2 else ' (on rank {})'.format(bmtk_world_comm.MPI_rank)

        for node_pop in self.node_populations:
            nodes = node_pop[bmtk_world_comm.MPI_rank::bmtk_world_comm.MPI_size]
            n_rank_nodes = int(node_pop.n_nodes() / float(bmtk_world_comm.MPI_size))

            ten_percent = int(np.ceil(n_rank_nodes*0.1))
            io.log_debug(' Adding {} cells.'.format(node_pop.name))

            for i, node in enumerate(nodes):
                if i > 0 and i % ten_percent == 0:
                    io.log_debug('  Adding cell {} of {}{}.'.format(i, n_rank_nodes, rank_msg))

                cell = Cell(node, population=node_pop.name)
                cell.default_jitter = self.jitter
                cell.build()
                self._local_cells.append(cell)

        bmtk_world_comm.barrier()
