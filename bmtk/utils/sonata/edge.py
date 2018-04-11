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
class EdgeSet(object):
    def __init__(self, edge_ids, population):
        self._edge_ids = edge_ids
        self._population = population
        self._n_edges = len(self._edge_ids)
        self.__itr = 0

    def __iter__(self):
        self.__itr = 0
        return self

    def next(self):
        if self.__itr >= self._n_edges:
            raise StopIteration

        next_edge = self._population.iloc(self._edge_ids[self.__itr])
        self.__itr += 1
        return next_edge


class Edge(object):
    def __init__(self, src_node_id, trg_node_id, source_pop, target_pop, group_id, group_props, edge_types_props):
        self._src_node_id = src_node_id
        self._trg_node_id = trg_node_id
        self._source_population = source_pop
        self._target_population = target_pop
        self._group_props = group_props
        self._group_id = group_id
        self._edge_type_props = edge_types_props

    @property
    def source_node_id(self):
        return self._src_node_id

    @property
    def target_node_id(self):
        return self._trg_node_id

    @property
    def source_population(self):
        return self._source_population

    @property
    def target_population(self):
        return self._target_population

    @property
    def group_id(self):
        return self._group_id

    @property
    def edge_type_id(self):
        return self._edge_type_props['edge_type_id']

    @property
    def dynamics_params(self):
        raise NotImplementedError

    def __getitem__(self, prop_key):
        if prop_key in self._group_props:
            return self._group_props[prop_key]
        elif prop_key in self._edge_type_props:
            return self._edge_type_props[prop_key]
        else:
            raise KeyError('Property {} not found in edge.'.format(prop_key))

    def __contains__(self, prop_key):
        return prop_key in self._group_props or prop_key in self._edge_type_props