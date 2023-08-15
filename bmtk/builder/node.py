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


class Node(dict):
    def __init__(self, node_id, node_params, node_type_properties, params_hash=-1):
        super(Node, self).__init__({})

        self._node_params = node_params
        self._node_params['node_id'] = node_id
        self._node_type_properties = node_type_properties
        self._params_hash = params_hash
        self._node_id = node_id

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type_id(self):
        return self._node_type_properties['node_type_id']

    @property
    def params(self):
        return self._node_params

    @property
    def node_type_properties(self):
        return self._node_type_properties

    @property
    def params_hash(self):
        return self._params_hash

    def get(self, key, default=None):
        if key in self._node_params:
            return self._node_params[key]
        elif key in self._node_type_properties:
            return self._node_type_properties[key]
        else:
            return default

    def __contains__(self, item):
        return item in self._node_type_properties or item in self._node_params

    def __getitem__(self, item):
        if item in self._node_params:
            return self._node_params[item]
        else:
            return self._node_type_properties[item]

    def __hash__(self):
        return hash(self.node_id)

    def __repr__(self):
        tmp_dict = dict(self._node_type_properties)
        tmp_dict.update(self._node_params)
        return tmp_dict.__repr__()
