class SimNode(object):
    @property
    def node_id(self):
        raise NotImplementedError()

    @property
    def gid(self):
        raise NotImplementedError()


class NodePopulation(object):
    def __init__(self):
        self._has_internal_nodes = False
        self._has_virtual_nodes = False

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def internal_nodes_only(self):
        return self._has_internal_nodes and not self._has_virtual_nodes

    @property
    def virtual_nodes_only(self):
        return self._has_virtual_nodes and not self._has_internal_nodes

    @property
    def mixed_nodes(self):
        return self._has_internal_nodes and self._has_virtual_nodes

    def initialize(self, network):
        raise NotImplementedError()

    @classmethod
    def load(cls, **properties):
        raise NotImplementedError()
