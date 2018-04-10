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
from bmtk.simulator.bionet.cell import Cell
# from bmtk.simulator.bionet import io,nrn

from neuron import h


pc = h.ParallelContext()    # object to access MPI methods


class LIFCell(Cell):
    # TODO: Rename to PointProcessCell
    """Implimentation of a Leaky Integrate-and-file neuron type cell."""
    def __init__(self, node, bionetwork):
        super(LIFCell, self).__init__(node)
        self.set_spike_detector()
        self._src_gids = []
        self._src_nets = []
        self._edge_type_id = []

    def set_spike_detector(self):
        nc = h.NetCon(self.hobj, None)
        pc.cell(self.gid, nc)

    def set_im_ptr(self):
        pass

    def set_syn_connection(self, edge_prop, src_node, stim=None):
        #src_gid = src_node.node_id
        syn_params = edge_prop['dynamics_params']
        nsyns = edge_prop.nsyns
        delay = edge_prop['delay']

        syn_weight = edge_prop.weight(src_node, self._node)
        if not edge_prop.preselected_targets:
            # TODO: this is not very robust, need some other way
            syn_weight *= syn_params['sign'] * nsyns

        if stim is not None:
            src_gid = -1
            nc = h.NetCon(stim.hobj, self.hobj)
        else:
            src_gid = src_node.node_id
            nc = pc.gid_connect(src_gid, self.hobj)

        weight = syn_weight
        nc.weight[0] = weight
        nc.delay = delay
        self._netcons.append(nc)
        self._src_gids.append(src_gid)
        #self._src_nets.append(src_node.network)
        self._src_nets.append(-1)
        #self._edge_type_id.append(edge_prop.edge_type_id)
        self._edge_type_id.append(-1)
        return nsyns

    def set_syn_connections(self, nsyn, syn_weight, edge_type, src_gid, stim=None):
        """Set synaptic connection"""
        syn_params = edge_type['params']
        delay = edge_type['delay']

        if stim:
            nc = h.NetCon(stim.hobj, self.hobj)
        else:
            nc = pc.gid_connect(src_gid, self.hobj)

        # scale weight by the number of synapse the artificial cell receives
        weight = nsyn * syn_weight * syn_params['sign']
        nc.weight[0] = weight
        nc.delay = delay
        self._netcons.append(nc)

    def get_connection_info(self):
        # TODO: There should be a more effecient and robust way to return synapse information.
        return [[self.gid, self._src_gids[i], self.network_name, self._src_nets[i], 'NaN', 'NaN',
                 self.netcons[i].weight[0], self.netcons[i].delay, self._edge_type_id[i], 1]
                for i in range(len(self._src_gids))]

    def print_synapses(self):
        rstr = ''
        for i in xrange(len(self._src_gids)):
            rstr += '{}> <-- {} ({}, {})\n'.format(i, self._src_gids[i], self.netcons[i].weight[0],
                                                   self.netcons[i].delay)

        return rstr
