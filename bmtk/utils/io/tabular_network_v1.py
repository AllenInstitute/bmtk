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
For the initial draft of the network format developed jointly by AI and collaborators in Q2 of 2017.

Edges and nodes files are stored in hdf5, while the edge-types and node-types are stored in csv. In the hd5f files
optional properties are stored in groups assigned to each node/edge. Optionally each property group may include
dynamics_params subgroup to describe the model of each node/row, or dynamics_params may be referenced in the types
metadata file.

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
    def __init__(self, gid, group, group_props, types_props):
        super(NodeRow, self).__init__(gid, group_props, types_props)
        # TODO: use group to determine if dynamics_params are included.

    @property
    def with_dynamics_params(self):
        return False

    @property
    def dynamics_params(self):
        return None


class NodesFile(tn.NodesFile):
    def __init__(self):
        super(NodesFile, self).__init__()

        self._nodes_hf = None
        self._nodes_index = pd.DataFrame()
        self._group_table = {}
        self._nrows = 0

    @property
    def gids(self):
        return list(self._nodes_index.index)

    def load(self, nodes_file, node_types_file):
        nodes_hf = h5py.File(nodes_file, 'r')
        if 'nodes' not in nodes_hf.keys():
            raise Exception('Could not find nodes in {}'.format(nodes_file))
        nodes_group = nodes_hf['nodes']

        self._network_name = nodes_group.attrs['network'] if 'network' in nodes_group.attrs.keys() else 'NA'
        self._version = 'v0.1'  # TODO: get the version number from the attributes

        # Create Indices
        self._nodes_index['node_gid'] = pd.Series(nodes_group['node_gid'], dtype=nodes_group['node_gid'].dtype)
        self._nodes_index['node_type_id'] = pd.Series(nodes_group['node_type_id'],
                                                      dtype=nodes_group['node_type_id'].dtype)
        self._nodes_index['node_group'] = pd.Series(nodes_group['node_group'],
                                                    dtype=nodes_group['node_group'].dtype)
        self._nodes_index['node_group_index'] = pd.Series(nodes_group['node_group_index'],
                                                          dtype=nodes_group['node_group_index'].dtype)
        self._nodes_index.set_index(['node_gid'], inplace=True)
        self._nrows = len(self._nodes_index)

        # Save the node-types
        self._node_types_table = tn.TypesTable(node_types_file, 'node_type_id')

        # save pointers to the groups table
        self._group_table = {grp_id: Group(grp_id, grp_ptr, self._node_types_table)
                             for grp_id, grp_ptr in nodes_group.items() if isinstance(grp_ptr, h5py.Group)}

    def get_node(self, gid, cache=False):
        node_metadata = self._nodes_index.loc[gid]
        ng = node_metadata['node_group']
        ng_idx = node_metadata['node_group_index']

        group_props = self._group_table[str(ng)][ng_idx]
        types_props = self._node_types_table[node_metadata['node_type_id']]

        return NodeRow(gid, self._group_table[str(ng)], group_props, types_props)

    def __len__(self):
        return self._nrows

    def next(self):
        if self._iter_index >= len(self):
            raise StopIteration
        else:
            gid = self._nodes_index.index.get_loc(self._iter_index)
            self._iter_index += 1
            return self.get_node(gid)


class EdgeRow(tn.EdgeRow):
    def __init__(self, trg_gid, src_gid, syn_group, edge_props={}, edge_type_props={}):
        super(EdgeRow, self).__init__(trg_gid, src_gid, edge_props, edge_type_props)
        # TODO: Look in syn_group to see if dynamics_params are included

    @property
    def with_dynamics_params(self):
        return False

    @property
    def dynamics_params(self):
        return None


class EdgesFile(tn.EdgesFile):
    def __init__(self):
        super(EdgesFile, self).__init__()
        self._nedges = 0
        self._source_network = None
        self._target_network = None

        # We'll save the target-index dataset into memory
        self._target_index = None
        self._target_index_len = 0

        # to save memory just keep pointers to datasets and access them as needed.
        self._target_gid_ds = None
        self._source_gid_ds = None
        self._edge_type_ds = None
        self._edge_group_ds = None
        self._edge_group_index_ds = None
        self._edge_types_table = None

        self._group_table = {}  # A table for all subgroups

    @property
    def source_network(self):
        return self._source_network

    @property
    def target_network(self):
        return self._target_network

    def load(self, edges_file, edge_types_file):
        edges_hf = h5py.File(edges_file, 'r')
        if 'edges' not in edges_hf.keys():
            raise Exception('Could not find edges in {}'.format(edges_file))
        edges_group = edges_hf['edges']

        # Preload the target index pointers into memory
        self._target_index = pd.Series(edges_group['index_pointer'], dtype=edges_group['index_pointer'].dtype)
        self._target_index_len = len(self._target_index)

        # For the other index tables we only load in a file pointer
        self._target_gid_ds = edges_group['target_gid']
        if 'network' in self._target_gid_ds.attrs.keys():
            self._target_network = self._target_gid_ds.attrs['network']

        self._source_gid_ds = edges_group['source_gid']
        if 'network' in self._source_gid_ds.attrs.keys():
            self._source_network = self._source_gid_ds.attrs['network']

        self._edge_type_ds = edges_group['edge_type_id']
        self._edge_group_ds = edges_group['edge_group']
        self._edge_group_index_ds = edges_group['edge_group_index']

        self._nedges = len(self._edge_group_index_ds)

        # Load in edge-types table
        self._edge_types_table = tn.TypesTable(edge_types_file, 'edge_type_id')

        # Load in the group properties
        # TODO: look in attributes for group synonyms
        # TODO: HDF5 group name will always be a string, but value in groups dataset will be an int.
        self._group_table = {grp_id: Group(grp_id, grp_ptr, self._edge_types_table)
                             for grp_id, grp_ptr in edges_group.items() if isinstance(grp_ptr, h5py.Group)}

    def edges_itr(self, target_gid):
        assert(isinstance(target_gid, int))
        if target_gid+1 >= self._target_index_len:
            raise StopIteration()

        index_begin = self._target_index.iloc[target_gid]
        index_end = self._target_index.iloc[target_gid+1]
        for iloc in xrange(index_begin, index_end):
            yield self[iloc]

    def __len__(self):
        return self._nedges

    def __getitem__(self, iloc):
        trg_gid = self._target_gid_ds[iloc]
        src_gid = self._source_gid_ds[iloc]

        et_id = self._edge_type_ds[iloc]
        et_props = self._edge_types_table[et_id]

        syn_group = self._edge_group_ds[iloc]
        syn_index = self._edge_group_index_ds[iloc]
        group_props = self._group_table[str(syn_group)][syn_index]

        return EdgeRow(trg_gid, src_gid, syn_group, group_props, et_props)


class Group(object):
    def __init__(self, group_id, h5_group, types_table):
        self._types_table = types_table
        self._group_id = group_id

        self._group_columns = tn.ColumnProperty.from_h5(h5_group)
        self._group_table = [(prop, h5_group[prop.name]) for prop in self._group_columns]

        self._all_columns = self._group_columns + types_table.columns

        # TODO: check to see if dynamics_params exists

    @property
    def columns(self):
        return self._all_columns

    def __getitem__(self, indx):
        group_props = {}
        for cprop, h5_obj in self._group_table:
            group_props[cprop.name] = h5_obj[indx]
        return group_props

    def __repr__(self):
        return "Group('group id': {}, 'properties':{})".format(self._group_id, self._all_columns)
