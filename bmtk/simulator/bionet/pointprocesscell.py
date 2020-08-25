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
import six
from bmtk.simulator.bionet.cell import Cell


pc = h.ParallelContext()    # object to access MPI methods


class ConnectionStruct(object):
    def __init__(self, edge_prop, src_node, nc, is_virtual=False):
        self._src_node = src_node
        self._edge_prop = edge_prop
        self._nc = nc
        self._is_virtual = is_virtual

    @property
    def is_virtual(self):
        return self._is_virtual

    @property
    def source_node(self):
        return self._src_node

    @property
    def syn_weight(self):
        return self._nc.weight[0]

    @syn_weight.setter
    def syn_weight(self, val):
        self._nc.weight[0] = val


class PointProcessCell(Cell):
    """Implimentation of a Leaky Integrate-and-file neuron type cell."""
    def __init__(self, node, population_name, bionetwork):
        super(PointProcessCell, self).__init__(node, population_name=population_name, network=bionetwork)
        self.set_spike_detector()
        self._src_gids = []
        self._src_nets = []
        self._edge_type_ids = []
        self._connections = []

    def set_spike_detector(self):
        nc = h.NetCon(self.hobj, None)
        pc.cell(self.gid, nc)

    def set_im_ptr(self):
        pass

    def set_syn_connection(self, edge_prop, src_node, stim=None):
        syn_params = edge_prop.dynamics_params
        nsyns = edge_prop.nsyns
        delay = edge_prop.delay

        syn_weight = edge_prop.syn_weight(src_node, self._node)
        if not edge_prop.preselected_targets:
            # TODO: this is not very robust, need some other way
            syn_weight *= syn_params['sign'] * nsyns

        if stim is not None:
            src_gid = -1
            #src_gid = src_node.node_id
            nc = h.NetCon(stim.hobj, self.hobj)
        else:
            src_gid = src_node.node_id
            nc = pc.gid_connect(src_gid, self.hobj)

        weight = syn_weight
        nc.weight[0] = weight
        nc.delay = delay
        self._netcons.append(nc)
        self._src_gids.append(src_gid)
        self._src_nets.append(-1)
        self._edge_type_ids.append(edge_prop.edge_type_id)
        self._edge_props.append(edge_prop)
        self._connections.append(ConnectionStruct(edge_prop, src_node, nc, stim is not None))

        return nsyns

    def connections(self):
        return self._connections

    def get_connection_info(self):
        # TODO: There should be a more effecient and robust way to return synapse information.
        return [[self.gid, self._src_gids[i], self.network_name, self._src_nets[i], 'NaN', 'NaN',
                 self.netcons[i].weight[0], self.netcons[i].delay, self._edge_type_id[i], 1]
                for i in range(len(self._src_gids))]

    def print_synapses(self):
        rstr = ''
        for i in six.moves.range(len(self._src_gids)):
            rstr += '{}> <-- {} ({}, {})\n'.format(i, self._src_gids[i], self.netcons[i].weight[0],
                                                   self.netcons[i].delay)

        return rstr
