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
from bmtk.simulator.utils.graph import SimNode

class PopNode(SimNode):
    def __init__(self, node_id, graph, network, params):
        self._graph = graph
        self._node_id = node_id
        self._network = network
        self._graph_params = params

        self._dynamics_params = {}
        self._updated_params = {'dynamics_params': self._dynamics_params}

        self._gids = set()

    @property
    def node_id(self):
        return self._node_id

    @property
    def pop_id(self):
        return self._node_id

    @property
    def network(self):
        return self._network

    @property
    def dynamics_params(self):
        return self._dynamics_params

    @dynamics_params.setter
    def dynamics_params(self, value):
        self._dynamics_params = value

    @property
    def is_internal(self):
        return False

    def __getitem__(self, item):
        if item in self._updated_params:
            return self._updated_params[item]
        elif item in self._graph_params:
            return self._graph_params[item]
        elif self._model_params is not None:
            return self._model_params[item]

    def add_gid(self, gid):
        self._gids.add(gid)

    def get_gids(self):
        return list(self._gids)


class InternalNode(PopNode):
    """
    def __init__(self, node_id, graph, network, params):
        super(InternalNode, self).__init__(node_id, graph, network, params)
        #self._pop_id = node_id
        #self._graph = graph
        #self._network = network
        #self._graph_params = params
        #self._dynamics_params = {}
        #self._update_params = {'dynamics_params': self._dynamics_params}
    """
    @property
    def tau_m(self):
        return self['tau_m']
        #return self._dynamics_params.get('tau_m', None)

    @tau_m.setter
    def tau_m(self, value):
        #return self['tau_m']
        self._dynamics_params['tau_m'] = value

    @property
    def v_max(self):
        return self._dynamics_params.get('v_max', None)

    @v_max.setter
    def v_max(self, value):
        self._dynamics_params['v_max'] = value

    @property
    def dv(self):
        return self._dynamics_params.get('dv', None)

    @dv.setter
    def dv(self, value):
        self._dynamics_params['dv'] = value

    @property
    def v_min(self):
        return self._dynamics_params.get('v_min', None)

    @v_min.setter
    def v_min(self, value):
        self._dynamics_params['v_min'] = value

    @property
    def is_internal(self):
        return True

    def __repr__(self):
        props = 'pop_id={}, tau_m={}, v_max={}, v_min={}, dv={}'.format(self.pop_id, self.tau_m, self.v_max, self.v_min,
                                                                        self.dv)
        return 'InternalPopulation({})'.format(props)


class ExternalPopulation(PopNode):
    def __init__(self, node_id, graph, network, params):
        super(ExternalPopulation, self).__init__(node_id, graph, network, params)
        self._firing_rate = -1
        if 'firing_rate' in params:
            self._firing_rate = params['firing_rate']

    @property
    def firing_rate(self):
        return self._firing_rate

    @property
    def is_firing_rate_set(self):
        return self._firing_rate >= 0

    @firing_rate.setter
    def firing_rate(self, rate):
        assert(isinstance(rate, float) and rate >= 0)
        self._firing_rate = rate

    @property
    def is_internal(self):
        return False

    def __repr__(self):
        props = 'pop_id={}, firing_rate={}'.format(self.pop_id, self.firing_rate)
        return 'ExternalPopulation({})'.format(props)

