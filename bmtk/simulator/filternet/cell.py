from bmtk.simulator.filternet.pyfunction_cache import py_modules


class Cell(object):
    def __init__(self, node):
        self._node = node
        self._gid = node.gid
        self._node_id = node.node_id
        self._lgn_cell_obj = None

    @property
    def gid(self):
        return self._gid

    @property
    def lgn_cell_obj(self):
        return self._lgn_cell_obj

    def build(self):
        cell_loaders = self._node.model_processing
        if len(cell_loaders) > 0:
            raise Exception('Cannot use more than one model_processing method per cell. Exiting.')
        elif len(cell_loaders) == 1:
            model_processing_fnc = py_modules.cell_processor(cell_loaders[0])
        else:
            model_processing_fnc = py_modules.cell_processor('default')

        #print self._node.dynamics_params
        #model_template =
        #print self._node.model_template
        #exit()

        self._lgn_cell_obj = model_processing_fnc(self._node, self._node.model_template, self._node.dynamics_params)
