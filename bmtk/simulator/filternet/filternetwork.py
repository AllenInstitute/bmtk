from bmtk.simulator.core.simulator_network import SimNetwork
from bmtk.simulator.filternet.cell import Cell
from bmtk.simulator.filternet.pyfunction_cache import py_modules
from bmtk.simulator.filternet.sonata_adaptors import FilterNodeAdaptor

class FilterNetwork(SimNetwork):
    def __init__(self):
        super(FilterNetwork, self).__init__()

        self._local_cells = []

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
            for node in node_pop.get_nodes():
                cell = Cell(node)
                cell.build()
                self._local_cells.append(cell)


