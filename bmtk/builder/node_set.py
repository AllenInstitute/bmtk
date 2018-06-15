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
import six
from .node import Node


class NodeSet(object):
    def __init__(self, N, node_params, node_type_properties):
        self.__N = N
        self.__node_params = node_params
        self.__node_type_properties = node_type_properties

        assert('node_type_id' in node_type_properties)
        self.__node_type_id = node_type_properties['node_type_id']

        # Used for determining which node_sets share the same params columns
        columns = list(self.__node_params.keys())
        columns.sort()
        self.__params_col_hash = hash(str(columns))

    @property
    def N(self):
        return self.__N

    @property
    def node_type_id(self):
        return self.__node_type_id

    @property
    def params_keys(self):
        return self.__node_params.keys()

    @property
    def params_hash(self):
        return self.__params_col_hash

    def build(self, nid_generator):
        # fetch existing node ids or create new ones
        node_ids = self.__node_params.get('node_id', None)
        if node_ids is None:
            node_ids = [nid for nid in nid_generator(self.N)]

        # turn node_params from dictionary of lists to a list of dictionaries.
        ap_flat = [{} for _ in six.moves.range(self.N)]
        for key, plist in self.__node_params.items():
            for i, val in enumerate(plist):
                ap_flat[i][key] = val

        # create node objects
        return [Node(nid, params, self.__node_type_properties, self.__params_col_hash)
                for (nid, params) in zip(node_ids, ap_flat)]
