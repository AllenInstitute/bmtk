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


class NodeSet(object):
    # TODO: Merge NodeSet and NodePopulation
    def __init__(self, node_indicies, population, **parameters):
        self._indicies = node_indicies
        self._n_nodes = len(self._indicies)
        self._population = population

        self.__itr_index = 0

    @property
    def node_ids(self):
        return self._population.inode_ids(self._indicies)

    @property
    def gids(self):
        return self._population.igids(self._indicies)

    @property
    def node_type_ids(self):
        return self._population.inode_type_ids(self._indicies)

    '''
    @property
    def node_types(self):
        return [self._population._node_types_table[ntid] for ntid in self._node_type_ids]
    '''

    def get_properties(self, property_name):
        raise NotImplementedError

    def __len__(self):
        return self._n_nodes

    def __iter__(self):
        self.__itr_index = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.__itr_index >= self._n_nodes:
            raise StopIteration

        node = self._population.get_row(self._indicies[self.__itr_index])
        self.__itr_index += 1
        return node


class Node(object):
    # TODO: include population name/reference
    # TODO: make a dictionary (or preferably a collections.MutableMap
    def __init__(self, node_id, node_type_id, node_types_props, group_id, group_props, dynamics_params, gid=None):
        self._node_id = node_id
        self._gid = gid
        self._node_type_id = node_type_id
        self._node_type_props = node_types_props
        self._group_id = group_id
        self._group_props = group_props

    @property
    def node_id(self):
        return self._node_id

    @property
    def gid(self):
        return self._gid

    @property
    def group_id(self):
        return self._group_id

    @property
    def node_type_id(self):
        return self._node_type_id

    @property
    def group_props(self):
        return self._group_props

    @property
    def node_type_properties(self):
        return self._node_type_props

    @property
    def dynamics_params(self):
        raise NotImplementedError

    def __getitem__(self, prop_key):
        if prop_key in self._group_props:
            return self._group_props[prop_key]
        elif prop_key in self._node_type_props:
            return self._node_type_props[prop_key]
        elif prop_key == 'node_id':
            return self.node_id
        elif property == 'node_type_id':
            return self.node_type_id
        else:
            raise KeyError('Unknown property {}'.format(prop_key))

    def __contains__(self, prop_key):
        return prop_key in self._group_props or prop_key in self._node_type_props
