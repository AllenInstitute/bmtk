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


class Edge(object):
    def __init__(self, src_gid, trg_gid, edge_type_props, syn_props):
        self.__src_gid = src_gid
        self.__trg_gid = trg_gid
        self.__edge_type_props = edge_type_props
        self.__syn_props = syn_props

    @property
    def source_gid(self):
        return self.__src_gid

    @property
    def target_gid(self):
        return self.__trg_gid

    @property
    def edge_type_properties(self):
        return self.__edge_type_props

    @property
    def edge_type_id(self):
        return self.edge_type_properties['edge_type_id']

    @property
    def synaptic_properties(self):
        return self.__syn_props

    def __contains__(self, item):
        return item in self.edge_type_properties or item in self.synaptic_properties

    def __getitem__(self, item):
        if item in self.edge_type_properties:
            return self.edge_type_properties[item]
        elif item in self.synaptic_properties:
            return self.synaptic_properties[item]
        else:
            return None

    def __repr__(self):
        rstr = "{} --> {} ('edge_type_id': {}, ".format(self.source_gid, self.target_gid, self.edge_type_id)
        rstr += "{}: {}" ', '.join("'{}': {}".format(k, v) for k, v in self.synaptic_properties.items())
        return rstr + ")"
