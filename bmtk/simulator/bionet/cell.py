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
from neuron import h
import numpy as np


pc = h.ParallelContext()  # object to access MPI methods
MPI_RANK = int(pc.id())


class Cell(object):
    """A abstract base class for any cell object.

    A base class for implementation of a cell-type objects like biophysical cells, LIF cells, etc. Do not instantiate
    a Cell object directly. Cell classes act as wrapper around HOC cell object with extra functionality for setting
    positions, synapses, and other parameters depending on the desired cell class.
    """
    def __init__(self, node, population_name, network=None):
        self._node = node
        self._network = network

        self._gid = network.gid_pool.get_gid(name=population_name, node_id=node.node_id)
        self._node_id = node.node_id
        self._props = node
        self._netcons = []  # list of NEURON network connection object attached to this cell

        self._pos_soma = []
        self.set_soma_position()

        # register the cell
        pc.set_gid2node(self.gid, MPI_RANK)

        # Load the NEURON HOC object
        self._hobj = node.load_cell()

        self._edge_props = []

    @property
    def node(self):
        return self._node

    @property
    def hobj(self):
        return self._hobj

    @property
    def gid(self):
        return self._gid

    @property
    def node_id(self):
        return self._node_id

    @property
    def group_id(self):
        return self._node.group_id

    @property
    def network_name(self):
        return self._node.network

    @property
    def netcons(self):
        return self._netcons

    @property
    def soma_position(self):
        return self._pos_soma

    def set_soma_position(self):
        positions = self._node.position
        if positions is not None:
            self._pos_soma = positions.reshape(3, 1)

    def init_connections(self):
        self.rand_streams = []
        self.prng = np.random.RandomState(self.gid)  # generate random stream based on gid

    def scale_weights(self, factor):
        for nc in self.netcons:
            weight = nc.weight[0]
            nc.weight[0] = weight*factor

    def get_connection_info(self):
        return []

    def set_syn_connections(self, edge_prop, src_node, stim=None):
        raise NotImplementedError

    def __getitem__(self, node_prop):
        return self._node[node_prop]
