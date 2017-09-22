import networkx as nx

from bmtk.builder.network import Network
from bmtk.builder.node import Node


class NxNetwork(Network):
    def __init__(self, name, **network_props):
        super(NxNetwork, self).__init__(name, **network_props or {})

        self.net = nx.MultiDiGraph()
        self.__nodes = []


    def _initialize(self):
        self.net.clear()

    def _add_nodes(self, nodes):
        self.__nodes += nodes
        self.net.add_nodes_from(nodes)

    def _add_edges(self, edge, connections):
        for src, trg, nsyns in connections:
            self.net.add_edge(src, trg, nsyns=nsyns, edge_type_id=edge.edge_type_id)


    def _clear(self):
        self.net.clear()

    def _nodes_iter(self, nids=None):
        if nids is not None:
            return ((nid, d)
                     for nid, d in self.__nodes
                     if nid in nids )
        else:
            return self.__nodes
            #return self.net.nodes_iter(data=True)

    def _edges_iter(self, nids=None, rank=0):
        if nids == None or len(nids) == 0:
            for e in self.net.edges(data=True):
                yield (e[0], e[1], e[2]['nsyns'], e[2]['edge_type_id'])
                #return self.net.edges(data=True)
        elif rank == 0:
            for e in self.net.out_edges(nids, data=True):
                yield (e[0], e[1], e[2]['nsyns'], e[2]['edge_type_id'])
        else:
            for e in self.net.in_edges(nids, data=True):
                yield (e[0], e[1], e[2]['nsyns'], e[2]['edge_type_id'])
            #return self.net.in_edges(nids, data=True)

    @property
    def nnodes(self):
        return nx.number_of_nodes(self.net)

    @property
    def nedges(self):
        return nx.number_of_edges(self.net)