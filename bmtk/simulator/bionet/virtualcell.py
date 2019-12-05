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


class VirtualCell(object):
    """Representation of a Virtual/External node"""

    def __init__(self, node, population, spike_train_dataset):
        # VirtualCell is currently not a subclass of bionet.Cell class b/c the parent has a bunch of properties that
        # just don't apply to a virtual cell. May want to make bionet.Cell more generic in the future.
        self._node_id = node.node_id
        self._node = node
        self._population = population
        self._hobj = None
        self._spike_train_dataset = spike_train_dataset
        self._train_vec = []
        self.set_stim(node, self._spike_train_dataset)
        
    @property
    def node_id(self):
        return self._node_id

    @property
    def hobj(self):
        return self._hobj

    def set_stim(self, stim_prop, spike_train):
        """Gets the spike trains for each individual cell."""
        self._train_vec = h.Vector(spike_train.get_times(node_id=self.node_id)) #, population=self._population))
        vecstim = h.VecStim()
        vecstim.play(self._train_vec)
        self._hobj = vecstim

    def __getitem__(self, item):
        return self._node[item]
