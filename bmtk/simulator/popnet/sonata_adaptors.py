from bmtk.simulator.core.sonata_reader import NodeAdaptor, SonataBaseNode, EdgeAdaptor, SonataBaseEdge


class PopNetEdge(SonataBaseEdge):
    @property
    def syn_weight(self):
        return self._edge['syn_weight']


class PopEdgeAdaptor(EdgeAdaptor):
    def get_edge(self, sonata_edge):
        return PopNetEdge(sonata_edge, self)
