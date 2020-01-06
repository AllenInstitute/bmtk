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
        for node_pop in self.node_populations:
            for node in node_pop[bmtk_world_comm.MPI_rank::bmtk_world_comm.MPI_size]:
                cell = Cell(node, population=node_pop.name)
                cell.default_jitter = self.jitter
                cell.build()
                self._local_cells.append(cell)
