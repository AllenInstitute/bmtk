# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
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