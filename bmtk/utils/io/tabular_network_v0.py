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
import pandas as pd
import h5py

import tabular_network as tn

"""
This is for the original bionet network format developed at the AI in 2016-2017. nodes, node_types, and edge_types
use csv format, while edges use an hdf5 format.

"""
class TabularNetwork(tn.TabularNetwork):
    @staticmethod
    def load_nodes(nodes_file, node_types_file):
        nf = NodesFile()
        nf.load(nodes_file, node_types_file)
        return nf

    @staticmethod
    def load_edges(edges_file, edge_types_file):
        ef = EdgesFile()
        ef.load(edges_file, edge_types_file)
        return ef


class NodeRow(tn.NodeRow):
    def __init__(self, gid, node_props, types_props, columns):
        super(NodeRow, self).__init__(gid, node_props, types_props)
        self._columns = columns

    @property
    def with_dynamics_params(self):
        return False

    @property
    def dynamics_params(self):
        return None


class NodesFile(tn.NodesFile):
    def __init__(self):
        super(NodesFile, self).__init__()
        self._network_name = 'NA'
        self._version = 'v0.0'

        self._nodes_df = None
        self._nodes_columns = None
        self._columns = None

    @property
    def gids(self):
        return list(self._nodes_df.index)

    def load(self, nodes_file, node_types_file):
        self._nodes_df = pd.read_csv(nodes_file, sep=' ', index_col=['node_id'])
        self._node_types_table = tn.TypesTable(node_types_file, 'node_type_id')

        self._nrows = len(self._nodes_df.index)
        self._nodes_columns = tn.ColumnProperty.from_csv(self._nodes_df)
        self._columns = self._nodes_columns + self._node_types_table.columns

    def get_node(self, gid, cache=False):
        nodes_data = self._nodes_df.loc[gid]
        node_type_data = self._node_types_table[nodes_data['node_type_id']]
        return NodeRow(gid, nodes_data, node_type_data, self._columns)

    def __len__(self):
        return self._nrows

    def next(self):
        if self._iter_index >= len(self):
            raise StopIteration
        else:
            gid = self._nodes_df.index.get_loc(self._iter_index)
            self._iter_index += 1
            return self.get_node(gid)


class EdgeRow(tn.EdgeRow):
    def __init__(self, trg_gid, src_gid, nsyns, edge_type_props):
        super(EdgeRow, self).__init__(trg_gid, src_gid, edge_type_props=edge_type_props)
        self._edge_props['nsyns'] = nsyns

    @property
    def with_dynamics_params(self):
        return False

    @property
    def dynamics_params(self):
        return None


class EdgesFile(tn.EdgesFile):
    def __init__(self):
        self._nrows = 0
        self._index_len = 0

        self._edge_ptr_ds = None
        self._num_syns_ds = None
        self._src_gids_ds = None
        self._edge_types_ds = None
        self._edge_types_table = {}

    @property
    def source_network(self):
        return None

    @property
    def target_network(self):
        return None

    def load(self, edges_file, edge_types_file):
        edges_hf = h5py.File(edges_file, 'r')
        self._edge_ptr_ds = edges_hf['edge_ptr']
        self._num_syns_ds = edges_hf['num_syns']
        self._src_gids_ds = edges_hf['src_gids']

        # TODO: validate edge_types dataset keys
        self._edge_types_ds = edges_hf['edge_types']
        self._edge_types_table = tn.TypesTable(edge_types_file, 'edge_type_id')
        self._index_len = len(self._edge_ptr_ds)
        self._nrows = len(self._src_gids_ds)

    def edges_itr(self, target_gid):
        assert(isinstance(target_gid, int))
        if target_gid+1 >= self._index_len:
            raise StopIteration()

        index_begin = self._edge_ptr_ds[target_gid]
        index_end = self._edge_ptr_ds[target_gid+1]
        for iloc in xrange(index_begin, index_end):
            source_gid = self._src_gids_ds[iloc]
            edge_type_id = self._edge_types_ds[iloc]
            edge_type = self._edge_types_table[edge_type_id]
            nsyns = self._num_syns_ds[iloc]
            yield EdgeRow(target_gid, source_gid, nsyns, edge_type)

    def __len__(self):
        return self._nrows
