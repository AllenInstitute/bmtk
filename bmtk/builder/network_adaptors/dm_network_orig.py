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
import six
import csv

from .network import Network
from bmtk.builder.node import Node
from bmtk.builder.edge import Edge
from bmtk.utils import sonata

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    barrier = comm.barrier

except ImportError:
    mpi_rank = 0
    mpi_size = 1
    barrier = lambda: None


class DenseNetworkOrig(Network):
    def __init__(self, name, **network_props):
        super(DenseNetworkOrig, self).__init__(name, **network_props or {})

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

        # TODO: open in append mode
        if mpi_rank == 0:
            with h5py.File(nodes_file_name, 'w') as hf:
                # Add magic and version attribute
                add_hdf5_attrs(hf)

                pop_grp = hf.create_group('/nodes/{}'.format(self.name))
                pop_grp.create_dataset('node_id', data=node_gid_table, dtype='uint64')
                pop_grp.create_dataset('node_type_id', data=node_type_id_table, dtype='uint64')
                pop_grp.create_dataset('node_group_id', data=node_group_table, dtype='uint32')
                pop_grp.create_dataset('node_group_index', data=node_group_index_tables, dtype='uint64')

                for grp_id, props in group_props.items():
                    model_grp = pop_grp.create_group('{}'.format(grp_id))

                    for key, dataset in props.items():
                        # ds_path = 'nodes/{}/{}'.format(grp_id, key)
                        try:
                            model_grp.create_dataset(key, data=dataset)
                        except TypeError:
                            str_list = [str(d) for d in dataset]
                            hf.create_dataset(key, data=str_list)
        barrier()

    def nodes_iter(self, node_ids=None):
        if node_ids is not None:
            return [n for n in self._nodes if n.node_id in node_ids]
        else:
            return self._nodes

    def _process_nodepool(self, nodepool):
        return nodepool

    def import_nodes(self, nodes_file_name, node_types_file_name, population=None):
        sonata_file = sonata.File(data_files=nodes_file_name, data_type_files=node_types_file_name)
        if sonata_file.nodes is None:
            raise Exception('nodes file {} does not have any nodes.'.format(nodes_file_name))

        populations = sonata_file.nodes.populations
        if len(populations) == 1:
            node_pop = populations[0]
        elif population is None:
            raise Exception('The nodes file {} contains multiple populations.'.format(nodes_file_name) +
                            'Please specify population parameter.')
        else:
            for pop in populations:
                if pop.name == population:
                    node_pop = pop
                    break
            else:
                raise Exception('Nodes file {} does not contain population {}.'.format(nodes_file_name, population))

        for node_type_props in node_pop.node_types_table:
            self._add_node_type(node_type_props)

        for node in node_pop:
            self._node_id_gen.remove_id(node.node_id)
            self._nodes.append(Node(node.node_id, node.group_props, node.node_type_properties))

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
                tmp_tables = [self.PropertyTable(nsyns, param.dtypes[n]) for n in param_names]
                # tmp_tables = [self.PropertyTable(nsyns) for _ in range(len(param_names))]
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
                pt = self.PropertyTable(np.sum(nsyns), dtype=param.dtypes[param_names])
                for source in connection_map.source_nodes:
                    src_node_id = source.node_id
                    for target in connection_map.target_nodes:
                        trg_node_id = target.node_id  # TODO: pull this out and put in it's own list
                        #print('{}, {}: {}'.format(src_node_id, trg_node_id, edge_table[src_node_id, trg_node_id]))
                        for _ in range(syn_table[src_node_id, trg_node_id]):
                            pt[src_node_id, trg_node_id] = rule(source, target)
                edge_table['params'][param_names] = pt

        self.__edges_tables.append(edge_table)


    def _save_gap_junctions(self, gj_file_name):
        source_ids = []
        target_ids = []
        src_gap_ids = []
        trg_gap_ids = []

        for et in self.__edges_tables:
            try:
                is_gap = et['edge_types']['is_gap_junction']
            except:
                continue
            if is_gap:
                if et['source_network'] != et['target_network']:
                    raise Exception("All gap junctions must be two cells in the same network builder.")
                
                table = et['syn_table']
                junc_table = table.nsyn_table
                locs = np.where(junc_table > 0)
                for i in range(len(locs[0])):
                    source_ids.append(table.source_ids[locs[0][i]])
                    target_ids.append(table.target_ids[locs[1][i]])
                    src_gap_ids.append(self._gj_id_gen.next())
                    trg_gap_ids.append(self._gj_id_gen.next())
            else:
                continue
        
        if len(source_ids) > 0:
            with h5py.File(gj_file_name, 'w') as f:
                add_hdf5_attrs(f)
                f.create_dataset('source_ids', data=np.array(source_ids))
                f.create_dataset('target_ids', data=np.array(target_ids))
                f.create_dataset('src_gap_ids', data=np.array(src_gap_ids))
                f.create_dataset('trg_gap_ids', data=np.array(trg_gap_ids))


    def _save_edges(self, edges_file_name, src_network, trg_network, name=None):
        groups = {}
        group_dtypes = {}  # TODO: this should be stored in PropertyTable
        grp_id_itr = 0
        groups_lookup = {}
        total_syns = 0

        matching_edge_tables = [et for et in self.__edges_tables
                                if et['source_network'] == src_network and et['target_network'] == trg_network]

        for ets in matching_edge_tables:
            params_hash = str(ets['params'].keys())
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
                            group_index_itrs[edge_group_id] += 1
                            gid_indx += 1

                            group_table['nsyns'].append(nsyns)

        trg_gids = trg_gids[:gid_indx]
        src_gids = src_gids[:gid_indx]
        edge_groups = edge_groups[:gid_indx]
        edge_group_index = edge_group_index[:gid_indx]
        edge_type_ids = edge_type_ids[:gid_indx]

        pop_name = '{}_to_{}'.format(src_network, trg_network) if name is None else name

        index_ptrs[index_ptr_itr] = gid_indx
        with h5py.File(edges_file_name, 'w') as hf:
            add_hdf5_attrs(hf)
            pop_grp = hf.create_group('/edges/{}'.format(pop_name))
            pop_grp.create_dataset('target_node_id', data=trg_gids, dtype='uint64')
            pop_grp['target_node_id'].attrs['node_population'] = trg_network
            pop_grp.create_dataset('source_node_id', data=src_gids, dtype='uint64')
            pop_grp['source_node_id'].attrs['node_population'] = src_network

            pop_grp.create_dataset('edge_group_id', data=edge_groups, dtype='uint16')
            pop_grp.create_dataset('edge_group_index', data=edge_group_index, dtype='uint32')
            pop_grp.create_dataset('edge_type_id', data=edge_type_ids, dtype='uint32')
            # pop_grp.create_dataset('edges/index_pointer', data=index_ptrs, dtype='uint32')

            for group_id, params_dict in groups.items():
                model_grp = pop_grp.create_group(str(group_id))
                for params_key, params_vals in params_dict.items():
                    #group_path = 'edges/{}/{}'.format(group_id, params_key)
                    dtype = group_dtypes[group_id][params_key]
                    if dtype is not None:
                        model_grp.create_dataset(params_key, data=list(params_vals), dtype=dtype)
                    else:
                        model_grp.create_dataset(params_key, data=list(params_vals))

            self._create_index(pop_grp['target_node_id'], pop_grp, index_type='target')
            self._create_index(pop_grp['source_node_id'], pop_grp, index_type='source')

    def _create_index(self, node_ids_ds, output_grp, index_type='target'):
        if index_type == 'target':
            edge_nodes = np.array(node_ids_ds, dtype=np.int64)
            output_grp = output_grp.create_group('indices/target_to_source')
        elif index_type == 'source':
            edge_nodes = np.array(node_ids_ds, dtype=np.int64)
            output_grp = output_grp.create_group('indices/source_to_target')

        edge_nodes = np.append(edge_nodes, [-1])
        n_targets = np.max(edge_nodes)
        ranges_list = [[] for _ in six.moves.range(n_targets + 1)]

        n_ranges = 0
        begin_index = 0
        cur_trg = edge_nodes[begin_index]
        for end_index, trg_gid in enumerate(edge_nodes):
            if cur_trg != trg_gid:
                ranges_list[cur_trg].append((begin_index, end_index))
                cur_trg = int(trg_gid)
                begin_index = end_index
                n_ranges += 1

        node_id_to_range = np.zeros((n_targets + 1, 2))
        range_to_edge_id = np.zeros((n_ranges, 2))
        range_index = 0
        for node_index, trg_ranges in enumerate(ranges_list):
            if len(trg_ranges) > 0:
                node_id_to_range[node_index, 0] = range_index
                for r in trg_ranges:
                    range_to_edge_id[range_index, :] = r
                    range_index += 1
                node_id_to_range[node_index, 1] = range_index

        output_grp.create_dataset('range_to_edge_id', data=range_to_edge_id, dtype='uint64')
        output_grp.create_dataset('node_id_to_range', data=node_id_to_range, dtype='uint64')

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
        def __init__(self, nvalues, dtype):
            self._prop_array = np.zeros(nvalues, dtype=dtype)
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


def add_hdf5_attrs(hdf5_handle):
    # TODO: move this as a utility function
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)
    hdf5_handle['/'].attrs['version'] = [np.uint32(0), np.uint32(1)]
