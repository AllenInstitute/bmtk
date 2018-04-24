

class NodesReader(object):
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


class EdgesReader(object):
    unknown = 0
    recurrent = 0
    virtual = 1
    mixed = 2

    def __init__(self):
        self._connection_type = -1

    @property
    def recurrent_connections(self):
        return self._connection_type == self.recurrent

    @property
    def virtual_connections(self):
        return self._connection_type == self.virtual

    @property
    def mixed_connections(self):
        return self._connection_type == self.mixed

    @property
    def source_nodes(self):
        raise NotImplementedError()

    @property
    def target_nodes(self):
        raise NotImplementedError()

    def set_connection_type(self, src_pop, trg_pop):
        if src_pop.internal_nodes_only and trg_pop.internal_nodes_only:
            self._connection_type = self.recurrent

        elif src_pop.virtual_nodes_only and trg_pop.internal_nodes_only:
            self._connection_type = self.virtual

        else:
            self._connection_type = self.mixed

    def initialize(self, network):
        raise NotImplementedError()

