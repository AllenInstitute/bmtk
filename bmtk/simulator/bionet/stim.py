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

"""
Representation of a Virtual/External/Stim node.

TODO:
 * Rename to Virtual
 * This is the biggest bottleneck when loading a network, find a way to optimize.
"""
class Stim(object):
    def __init__(self, node, spike_train_dataset):
        self._node_id = node.node_id
        self.stim_gid = node.gid
        self.rand_streams = []
        self._spike_train_dataset = spike_train_dataset

        self.set_stim(node, self._spike_train_dataset)
        
    @property
    def node_id(self):
        return self._node_id

    def set_stim(self, stim_prop, spike_train):
        #spike_trains_handle = input_prop["spike_trains_handle"]
        #self.spike_train = spike_trains_handle['%d/data' % self.stim_gid][:]

        self.train_vec = h.Vector(spike_train[:])
        vecstim = h.VecStim()
        vecstim.play(self.train_vec)
        
        self.hobj = vecstim
