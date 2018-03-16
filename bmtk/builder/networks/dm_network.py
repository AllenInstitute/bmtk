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
import os
import numpy as np
import h5py
import csv

from ..network import Network
from bmtk.utils.io import TabularNetwork
from bmtk.builder.node import Node
from bmtk.builder.edge import Edge


class DenseNetwork(Network):
    def __init__(self, name, **network_props):
        super(DenseNetwork, self).__init__(name, **network_props or {})

        self.__edges_types = {}
        self.__src_mapping = {}

        self.__networks = {}
        self.__node_count = 0
        self._nodes = []

        self.__edges_tables = []
        self._target_networks = {}

    def _initialize(self):
        self.__id_map = []
        self.__lookup = []

    def _add_nodes(self, nodes):
        self._nodes.extend(nodes)
        self._nnodes = len(self._nodes)

        """
        id_label = 'node_id' if 'node_id' in nodes[0].keys() else 'id'

        start_idx = len(self.__id_map)  #
        self.__id_map += [n[id_label] for n in nodes]
        self.__nodes += [(interal_id, nodes[node_idx])
                         for node_idx, interal_id in enumerate(xrange(start_idx, len(self.__id_map)))]

        assert(len(self.__id_map) == len(self.__nodes))
        """

    def edges_table(self):
        return self.__edges_tables

    def _save_nodes(self, nodes_file_name):
        if not self._nodes_built:
            self._build_nodes()

        # save the node_types file
        # TODO: how do we add attributes to the h5
        group_indx = 0
        groups_lookup = {}
        group_indicies = {}
        group_props = {}
        for ns in self._node_sets:
            if ns.params_hash in groups_lookup:
                continue
            else:
                groups_lookup[ns.params_hash] = group_indx
                group_indicies[group_indx] = 0
                group_props[group_indx] = {k: [] for k in ns.params_keys if k != 'node_id'}
                group_indx += 1

        node_gid_table = np.zeros(self._nnodes)  # todo: set dtypes
        node_type_id_table = np.zeros(self._nnodes)
        node_group_table = np.zeros(self._nnodes)
        node_group_index_tables = np.zeros(self._nnodes)

        for i, node in enumerate(self.nodes()):
            node_gid_table[i] = node.node_id
            node_type_id_table[i] = node.node_type_id
            group_id = groups_lookup[node.params_hash]
            node_group_table[i] = group_id
            node_group_index_tables[i] = group_indicies[group_id]
            group_indicies[group_id] += 1

            group_dict = group_props[group_id]
            for key, prop_ds in group_dict.items():
                prop_ds.append(node.params[key])

        with h5py.File(nodes_file_name, 'w') as hf:
            hf.create_dataset('nodes/node_gid', data=node_gid_table, dtype='uint64')
            hf['nodes/node_gid'].attrs['network'] = self.name
            hf.create_dataset('nodes/node_type_id', data=node_type_id_table, dtype='uint64')
            hf.create_dataset('nodes/node_group', data=node_group_table, dtype='uint32')
            hf.create_dataset('nodes/node_group_index', data=node_group_index_tables, dtype='uint64')


            for grp_id, props in group_props.items():
                #if len(props.items()) == 0:
                hf.create_group('nodes/{}'.format(grp_id))

                for key, dataset in props.items():
                    ds_path = 'nodes/{}/{}'.format(grp_id, key)
                    try:
                        hf.create_dataset(ds_path, data=dataset)
                    except TypeError:
                        str_list = [str(d) for d in dataset]
                        hf.create_dataset(ds_path, data=str_list)

    def nodes_iter(self, node_ids=None):
        if node_ids is not None:
            return [n for n in self._nodes if n.node_id in node_ids]
        else:
            return self._nodes

    def _process_nodepool(self, nodepool):
        return nodepool

    def import_nodes(self, nodes_file_name, node_types_file_name):
        nodes_network = TabularNetwork.load_nodes(nodes_file=nodes_file_name, node_types_file=node_types_file_name)
        for node_type_id, node_type_props in nodes_network.node_types_table.items():
            self._add_node_type(node_type_props)

        nodes = []
        for n in nodes_network:
            self._node_id_gen.remove_id(n.gid)
            self._nodes.append(Node(n.gid, n.node_props, n.node_type_props))

    def _add_edges(self, connection_map, i):
        syn_table = self.EdgeTable(connection_map)
        connections = connection_map.connection_itr()
        for con in connections:
            if con[2] is not None:
                syn_table[con[0], con[1]] = con[2]

        target_net = connection_map.target_nodes
        self._target_networks[target_net.network_name] = target_net.network

        nsyns = np.sum(syn_table.nsyn_table)
        self._nedges += int(nsyns)
        edge_table = {'syn_table': syn_table,
                      'nsyns': nsyns,
                      'edge_types': connection_map.edge_type_properties,
                      'edge_type_id': connection_map.edge_type_properties['edge_type_id'],
                      'source_network': connection_map.source_nodes.network_name,
                      'target_network': connection_map.target_nodes.network_name,
                      'params': {},
                      'params_dtypes': {},
                      'source_query': connection_map.source_nodes.filter_str,
                      'target_query': connection_map.target_nodes.filter_str}


        for param in connection_map.params:
            rule = param.rule
            param_names = param.names
            edge_table['params_dtypes'].update(param.dtypes)
            if isinstance(param_names, list) or isinstance(param_names, tuple):
                tmp_tables = [self.PropertyTable(nsyns) for _ in range(len(param_names))]
                for source in connection_map.source_nodes:
                    src_node_id = source.node_id
                    for target in connection_map.target_nodes:
                        trg_node_id = target.node_id  # TODO: pull this out and put in it's own list
                        for _ in range(syn_table[src_node_id, trg_node_id]):
                            pvals = rule(source, target)
                            for i in range(len(param_names)):
                                tmp_tables[i][src_node_id, trg_node_id] = pvals[i]

                for i, name in enumerate(param_names):
                    # TODO: I think a copy constructor might get called, move this out.
                    edge_table['params'][name] = tmp_tables[i]

            else:
                pt = self.PropertyTable(np.sum(nsyns))
                for source in connection_map.source_nodes:
                    src_node_id = source.node_id
                    for target in connection_map.target_nodes:
                        trg_node_id = target.node_id  # TODO: pull this out and put in it's own list
                        #print('{}, {}: {}'.format(src_node_id, trg_node_id, edge_table[src_node_id, trg_node_id]))
                        for _ in range(syn_table[src_node_id, trg_node_id]):
                            pt[src_node_id, trg_node_id] = rule(source, target)
                edge_table['params'][param_names] = pt

        self.__edges_tables.append(edge_table)

    def _save_edges(self, edges_file_name, src_network, trg_network):
        groups = {}
        group_dtypes = {}  # TODO: this should be stored in PropertyTable
        grp_id_itr = 0
        groups_lookup = {}
        total_syns = 0

        matching_edge_tables = [et for et in self.__edges_tables
                                if et['source_network'] == src_network and et['target_network'] == trg_network]

        for ets in matching_edge_tables:
            params_hash =  str(ets['params'].keys())
            group_id = groups_lookup.get(params_hash, None)
            if group_id is None:
                group_id = grp_id_itr
                groups_lookup[params_hash] = group_id
                grp_id_itr += 1

            ets['group_id'] = group_id
            groups[group_id] = {}
            group_dtypes[group_id] = ets['params_dtypes']
            for param_name in ets['params'].keys():
                groups[group_id][param_name] = []

            total_syns += int(ets['nsyns'])

        group_index_itrs = [0 for _ in range(grp_id_itr)]
        trg_gids = np.zeros(total_syns)  # set dtype to uint64
        src_gids = np.zeros(total_syns)
        edge_groups = np.zeros(total_syns)  # dtype uint16 or uint8
        edge_group_index = np.zeros(total_syns)  # uint32
        edge_type_ids = np.zeros(total_syns)  # uint32

        # TODO: Another potential issue if node-ids don't start with 0
        index_ptrs = np.zeros(len(self._target_networks[trg_network].nodes()) + 1)
        #index_ptrs = np.zeros(len(self._nodes)+1)  # TODO: issue when target nodes come from another network
        index_ptr_itr = 0

        gid_indx = 0
        for trg_node in self._target_networks[trg_network].nodes():
            index_ptrs[index_ptr_itr] = gid_indx
            index_ptr_itr += 1

            for ets in matching_edge_tables:
                edge_group_id = ets['group_id']
                group_table = groups[edge_group_id]

                syn_table = ets['syn_table']
                if syn_table.has_target(trg_node.node_id):
                    if ets['params']:
                        for src_id, nsyns in syn_table.trg_itr(trg_node.node_id):
                            # Add on to the edges index
                            indx_end = gid_indx+nsyns
                            while gid_indx < indx_end:
                                trg_gids[gid_indx] = trg_node.node_id
                                src_gids[gid_indx] = src_id
                                edge_type_ids[gid_indx] = ets['edge_type_id']
                                edge_groups[gid_indx] = edge_group_id
                                edge_group_index[gid_indx] = group_index_itrs[edge_group_id]
                                group_index_itrs[edge_group_id] += 1
                                gid_indx += 1

                            for param_name, param_table in ets['params'].items():
                                param_vals = group_table[param_name]
                                for val in param_table.itr_vals(src_id, trg_node.node_id):
                                    param_vals.append(val)

                    else:
                        # If no properties just print nsyns table.
                        if 'nsyns' not in group_table:
                            group_table['nsyns'] = []
                        group_dtypes[edge_group_id]['nsyns'] = 'uint16'
                        for src_id, nsyns in syn_table.trg_itr(trg_node.node_id):
                            trg_gids[gid_indx] = trg_node.node_id
                            src_gids[gid_indx] = src_id
                            edge_type_ids[gid_indx] = ets['edge_type_id']
                            edge_groups[gid_indx] = edge_group_id
                            edge_group_index[gid_indx] = group_index_itrs[edge_group_id]
                            # group_dtypes
                            group_index_itrs[edge_group_id] += 1
                            gid_indx += 1

                            group_table['nsyns'].append(nsyns)

        trg_gids = trg_gids[:gid_indx]
        src_gids = src_gids[:gid_indx]
        edge_groups = edge_groups[:gid_indx]
        edge_group_index = edge_group_index[:gid_indx]
        edge_type_ids = edge_type_ids[:gid_indx]

        index_ptrs[index_ptr_itr] = gid_indx

        with h5py.File(edges_file_name, 'w') as hf:
            hf.create_dataset('edges/target_gid', data=trg_gids, dtype='uint64')
            hf['edges/target_gid'].attrs['network'] = trg_network
            hf.create_dataset('edges/source_gid', data=src_gids, dtype='uint64')
            hf['edges/source_gid'].attrs['network'] = src_network

            hf.create_dataset('edges/edge_group', data=edge_groups, dtype='uint16')
            hf.create_dataset('edges/edge_group_index', data=edge_group_index, dtype='uint32')
            hf.create_dataset('edges/edge_type_id', data=edge_type_ids, dtype='uint32')
            hf.create_dataset('edges/index_pointer', data=index_ptrs, dtype='uint32')

            for group_id, params_dict in groups.items():
                for params_key, params_vals in params_dict.items():
                    group_path = 'edges/{}/{}'.format(group_id, params_key)
                    dtype = group_dtypes[group_id][params_key]
                    if dtype is not None:
                        hf.create_dataset(group_path, data=list(params_vals), dtype=dtype)
                    else:
                        hf.create_dataset(group_path, data=list(params_vals))

    def _clear(self):
        self._nedges = 0
        self._nnodes = 0

    def edges_iter(self, trg_gids, src_network=None, trg_network=None):
        matching_edge_tables = self.__edges_tables
        if trg_network is not None:
            matching_edge_tables = [et for et in self.__edges_tables if et['target_network'] == trg_network]

        if src_network is not None:
            matching_edge_tables = [et for et in matching_edge_tables if et['source_network'] == src_network]

        for trg_gid in trg_gids:
            for ets in matching_edge_tables:
                syn_table = ets['syn_table']
                if syn_table.has_target(trg_gid):
                    for src_id, nsyns in syn_table.trg_itr(trg_gid):
                        if ets['params']:
                            synapses = [{} for _ in range(nsyns)]
                            for param_name, param_table in ets['params'].items():
                                for i, val in enumerate(param_table[src_id, trg_gid]):
                                    synapses[i][param_name] = val
                            for syn_prop in synapses:
                                yield Edge(src_gid=src_id, trg_gid=trg_gid, edge_type_props=ets['edge_types'],
                                           syn_props=syn_prop)
                        else:
                            yield Edge(src_gid=src_id, trg_gid=trg_gid, edge_type_props=ets['edge_types'],
                                       syn_props={'nsyns': nsyns})

    @property
    def nnodes(self):
        if not self.nodes_built:
            return 0
        return self._nnodes

    @property
    def nedges(self):
        return self._nedges

    class EdgeTable(object):
        def __init__(self, connection_map):
            # TODO: save column and row lengths
            # Create maps between source_node gids and their row in the matrix.
            self.__idx2src = [n.node_id for n in connection_map.source_nodes]
            self.__src2idx = {node_id: i for i, node_id in enumerate(self.__idx2src)}

            # Create maps betwee target_node gids and their column in the matrix
            self.__idx2trg = [n.node_id for n in connection_map.target_nodes]
            self.__trg2idx = {node_id: i for i, node_id in enumerate(self.__idx2trg)}

            self._nsyn_table = np.zeros((len(self.__idx2src), len(self.__idx2trg)), dtype=np.uint8)

        def __getitem__(self, item):
            # TODO: make sure matrix is column oriented, or swithc trg and srcs.
            indexed_pair = (self.__src2idx[item[0]], self.__trg2idx[item[1]])
            return self._nsyn_table[indexed_pair]

        def __setitem__(self, key, value):
            assert(len(key) == 2)
            indexed_pair = (self.__src2idx[key[0]], self.__trg2idx[key[1]])
            self._nsyn_table[indexed_pair] = value

        def has_target(self, node_id):
            return node_id in self.__trg2idx

        @property
        def nsyn_table(self):
            return self._nsyn_table

        @property
        def target_ids(self):
            return self.__idx2trg

        @property
        def source_ids(self):
            return self.__idx2src

        def trg_itr(self, trg_id):
            trg_i = self.__trg2idx[trg_id]
            for src_j, src_id in enumerate(self.__idx2src):
                nsyns = self._nsyn_table[src_j, trg_i]
                if nsyns:
                    yield src_id, nsyns

    class PropertyTable(object):
        # TODO: add support for strings
        def __init__(self, nvalues):
            self._prop_array = np.zeros(nvalues)
            # self._prop_table = np.zeros((nvalues, 1))  # TODO: set dtype
            self._index = np.zeros((nvalues, 2), dtype=np.uint32)
            self._itr_index = 0

        def itr_vals(self, src_id, trg_id):
            indicies = np.where((self._index[:, 0] == src_id) & (self._index[:, 1] == trg_id))
            for val in self._prop_array[indicies]:
                yield val

        def __setitem__(self, key, value):
            self._index[self._itr_index, 0] = key[0]  # src_node_id
            self._index[self._itr_index, 1] = key[1]  # trg_node_id
            self._prop_array[self._itr_index] = value
            self._itr_index += 1

        def __getitem__(self, item):
            indicies = np.where((self._index[:, 0] == item[0]) & (self._index[:, 1] == item[1]))
            return self._prop_array[indicies]


