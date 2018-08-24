class SimEdge(object):
    @property
    def node_id(self):
        raise NotImplementedError()

    @property
    def gid(self):
        raise NotImplementedError()


class EdgePopulation(object):
    @property
    def source_nodes(self):
        raise NotImplementedError()

    @property
    def target_nodes(self):
        raise NotImplementedError()

    def initialize(self, network):
        raise NotImplementedError()