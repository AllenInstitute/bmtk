from bmtk.simulator.filternet.pyfunction_cache import py_modules


class Cell(object):
    def __init__(self, node, population=None):
        self._node = node
        self._gid = node.gid
        self._node_id = node.node_id
        self._lgn_cell_obj = None
        self._default_jitter = (0.0, 0.0)
        self._population = population

    @property
    def gid(self):
        return self._gid

    @property
    def lgn_cell_obj(self):
        return self._lgn_cell_obj

    @property
    def population(self):
        return self._population

    @property
    def default_jitter(self):
        return self._jitter

    @default_jitter.setter
    def default_jitter(self, val):
        self._default_jitter = val

    def build(self):
        cell_loaders = self._node.model_processing
        if len(cell_loaders) > 1:
            raise Exception('Cannot use more than one model_processing method per cell. Exiting.')
        elif len(cell_loaders) == 1:
            model_processing_fnc = py_modules.cell_processor(cell_loaders[0])
        else:
            model_processing_fnc = py_modules.cell_processor('default')

        if not self._node.predefined_jitter:
            self._node.jitter = self._default_jitter

        self._lgn_cell_obj = model_processing_fnc(self, self._node.model_template, self._node.dynamics_params)

    def get(self, item, default):
        if item in self._node:
            return self._node[item]
        else:
            return default

    def __getitem__(self, item):
        return self._node[item]

    def __contains__(self, item):
        return item in self._node

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return getattr(self._node, name)
