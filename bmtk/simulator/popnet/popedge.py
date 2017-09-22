from bmtk.simulator.utils.graph import SimEdge


class PopEdge(SimEdge):
    def __init__(self, source_pop, target_pop, edge_params, dynamics_params):
        super(PopEdge, self).__init__(edge_params, dynamics_params)
        self.__source_pop = source_pop
        self.__target_pop = target_pop
        self._weight = self.__get_prop('weight', 0.0)
        self._nsyns = self.__get_prop('nsyns', 0)
        self._delay = self.__get_prop('delay', 0.0)

    @property
    def source(self):
        return self.__source_pop

    @property
    def target(self):
        return self.__target_pop

    @property
    def params(self):
        return self._orig_params

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def nsyns(self):
        return self._nsyns

    @nsyns.setter
    def nsyns(self, value):
        self._nsyns = value

    @property
    def delay(self):
        return self._delay

    @delay.setter
    def delay(self, value):
        self._delay = value

    def __get_prop(self, name, default=None):
        if name in self._orig_params:
            return self._orig_params[name]
        elif name in self._dynamics_params:
            return self._dynamics_params[name]
        else:
            return default

    def __repr__(self):
        relevant_params = "weight: {}, delay: {}, nsyns: {}".format(self.weight, self.delay, self.nsyns)
        rstr = "{} --> {} {{{}}}".format(self.source.pop_id, self.target.pop_id, relevant_params)
        return rstr
