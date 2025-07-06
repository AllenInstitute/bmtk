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
from ast import literal_eval
from six import string_types


class NodePool(object):
    """Stores a collection of nodes based off some query of the network.

    Returns the results of a query of nodes from a network using the nodes() method. Nodes are still generated and
    saved by the network, this just stores the query information and provides iterator methods for accessing different
    nodes.

    TODO:
        * Implement a collection-set algebra including | and not operators. ie.
            nodes = net.nodes(type=1) | net.nodes(type=2)
        * Implement operators on properties
            nodes = net.nodes(val) > 100
            nodes = 100 in net.nodes(val)
    """

    def __init__(self, network, slice=None, **properties):
        self.__network = network
        self.__properties = properties
        self.__filter_str = None
        self.__slice = slice
        
        self.__itr_lst = [n for n in self.__network.nodes_iter() if self.__query_object_properties(n, self.__properties)]
        if self.__slice:
            self.__itr_lst[self.__slice]


        # self.__itr_indices = None
        # self.__itr_curr = 0
        # self.__itr_list = None
        # self.__itr_list_end = None
        # self.__itr_cidx = 0


    def __len__(self):
        return len(self.__itr_lst) # sum(1 for _ in self)

    def __iter__(self):
        return (n for n in self.__network.nodes_iter() if self.__query_object_properties(n, self.__properties))


    # def __iter__(self):
        
        
        # return (n for n in self.__network.nodes_iter() if self.__query_object_properties(n, self.__properties))
        # itr_list = [n for n in self.__network.nodes_iter() if self.__query_object_properties(n, self.__properties)]
        # if self.__slice:
        #     itr_list = itr_list[self.__slice]
        # print('--', len(self.__itr_list))
        # exit()

        # self.__itr_list_end = len(self.__itr_list)
        # self.__itr_cidx = 0
        # return iter(self.__itr_lst)
    
    # def __next__(self):
    #     if self.__itr_cidx < self.__itr_list_end:
    #         # print('next')
    #         # print(self.__itr_cidx)
    #         self.__itr_cidx += 1
    #         # print('next', self.__itr_list[self.__itr_cidx-1])
    #         return self.__itr_list[self.__itr_cidx-1]
    #     else:
    #         raise StopIteration
        



    @property
    def network(self):
        return self.__network

    @property
    def network_name(self):
        return self.__network.name

    @property
    def filter_str(self):
        if self.__filter_str is None:
            if len(self.__properties) == 0:
                self.__filter_str = '*'
            else:
                self.__filter_str = ''
                for k, v in self.__properties.items():
                    conditional = "{}=='{}'".format(k, v)
                    self.__filter_str += conditional + '&'
                if self.__filter_str.endswith('&'):
                    self.__filter_str = self.__filter_str[0:-1]

        return self.__filter_str

    @classmethod
    def from_filter(cls, network, filter_str):
        assert(isinstance(filter_str, string_types))
        if len(filter_str) == 0 or filter_str == '*':
            return cls(network, position=None)

        properties = {}
        for condtional in filter_str.split('&'):
            var, val = condtional.split('==')
            properties[var] = literal_eval(val)
        return cls(network, position=None, **properties)

    def __query_object_properties(self, obj, props):
        if props is None:
            return True

        for k, v in props.items():
            ov = obj.get(k, None)
            if ov is None:
                return False

            if hasattr(v, '__call__'):
                if not v(ov):
                    return False
            elif isinstance(v, list):
                if ov not in v:
                    return False
            elif ov != v:
                return False

        return True

    # def __getitem__(self, key):
    #     # print(key, type(key))
    #     if isinstance(key, slice):
    #         return NodePool(self.__network, slice=key, **self.__properties)

    #     else:
    #         raise NotImplementedError()
    #     # exit()
