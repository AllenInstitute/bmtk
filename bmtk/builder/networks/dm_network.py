from ..network import Network
import numpy as np
import h5py
import csv

from bmtk.utils.io import TabularNetwork
from bmtk.builder.node import Node

class DenseNetwork(Network):

    def __init__(self, name, **network_props):
        super(DenseNetwork, self).__init__(name, **network_props or {})

        self.__edges_types = {}
        self.__src_mapping = {}

        self.__networks = {}
        self.__node_count = 0
        self._nodes = []

        self.__edges_tables = []


    def _initialize(self):
        self.__id_map = []
        # self.__nodes = []

        #self.__networks = {}
        #self.__node_count = 0
        self.__lookup = []
        #self.__edges = None
        #self.__edges = np.zeros((0,0), dtype=np.uint8)


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

    # def _save_nodes(self, nodes_file_name, node_types_file_name):
    def save_nodes(self, nodes_file_name, node_types_file_name):
        if not self._nodes_built:
            self._build_nodes()

        # save the node_types file
        # TODO: how do we add attributes to the h5
        node_types_cols = ['node_type_id'] + [col for col in self._node_types_columns if col != 'node_type_id']
        with open(node_types_file_name, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=' ')
            csvw.writerow(node_types_cols)
            for node_type in self._node_types_properties.values():
                csvw.writerow([node_type.get(cname, 'NULL') for cname in node_types_cols])

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
            #print node.node_id, node.node_type_id
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
            hf.create_dataset('nodes/node_type_id', data=node_type_id_table, dtype='uint64')
            hf.create_dataset('nodes/node_group', data=node_group_table, dtype='uint32')
            hf.create_dataset('nodes/node_group_index', data=node_group_index_tables, dtype='uint64')

            for grp_id, props in group_props.items():
                for key, dataset in props.items():
                    ds_path = 'nodes/{}/{}'.format(grp_id, key)
                    try:
                        hf.create_dataset(ds_path, data=dataset)
                    except TypeError:
                        str_list = [str(d) for d in dataset]
                        hf.create_dataset(ds_path, data=str_list)

    def _nodes_iter(self, node_ids=None):
        #print self._nodes
        #exit()

        if node_ids is not None:
            raise NotImplementedError()
        else:
            return self._nodes


    def _process_nodepool(self, nodepool):
        #if nodepool.network not in self.__networks:
        #    offset = self.__node_count
        #    self.__node_count += len(nodepool.network.nodes())
        #    self.__networks[nodepool.network] = offset
        #    self.__edges = np.zeros((self.__node_count, self.__node_count), dtype=np.uint8)
        #    self.__lookup += [n[1]['id'] for n in nodepool.network.nodes()]
        return nodepool


    def import_nodes(self, nodes_file_name, node_types_file_name):
        nodes_network = TabularNetwork.load_nodes(nodes_file=nodes_file_name, node_types_file=node_types_file_name)
        for node_type_id, node_type_props in nodes_network.node_types_table.items():
            self._add_node_type(node_type_props)

        nodes = []
        for n in nodes_network:
            self._node_id_gen.remove_id(n.gid)
            self._nodes.append(Node(n.gid, n.node_props, n.node_type_props))


    def _add_edges(self, connection_map):
        syn_table = self.EdgeTable(connection_map)
        connections = connection_map.connection_itr()
        for con in connections:
            if con[2] is not None:
                syn_table[con[0], con[1]] = con[2]

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

        #params = {}
        for param in connection_map.params:
            rule = param.rule
            param_names = param.names
            edge_table['params_dtypes'].update(param.dtypes)
            if isinstance(param_names, list) or isinstance(param_names, tuple):
                #print 'HERE'
                #print nsyns
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

                #print len(tmp_tables[0]._prop_array)
                #print 'done'
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
        #print edge_table['params_dtypes']
        #print self.__edges_tables[0]['params']['distance']
        #exit()


    def save_edges(self, edges_file_name, edge_types_file_name, src_network=None, trg_network=None):
        # def save_edges(self, edges_file_name, edge_types_file_name):
        #print self.__edges_tables

        #groups = []

        # TODO: needs to handle cases for inter-network connections
        #print self._edge_type_properties

        src_network = self.__edges_tables[0].get('source_network', None) or self.name
        trg_network = self.__edges_tables[0].get('target_network', None) or self.name
        # print src_network
        # print trg_network

        # Create edge-type file


        cols = ['edge_type_id', 'target_query', 'source_query']
        cols += [col for col in self._edge_types_columns if col not in cols]
        with open(edge_types_file_name, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=' ')
            csvw.writerow(cols)
            for edge_type in self._edge_type_properties.values():
                csvw.writerow([edge_type.get(cname, 'NULL') for cname in cols])

        groups = {}
        group_dtypes = {}  # TODO: this should be stored in PropertyTable

        grp_id_itr = 0
        groups_lookup = {}

        #grp_id = 0
        total_syns = 0
        for ets in self.__edges_tables:
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

            # grp_id += 1
            total_syns += int(ets['nsyns'])

        group_index_itrs = [0 for _ in range(grp_id_itr)]
        #print group_index_itrs
        #total_syns = int(total_syns)
        # TODO: merge group that have the same properties.

        trg_gids = np.zeros(total_syns)  # set dtype to uint64
        src_gids = np.zeros(total_syns)
        edge_groups = np.zeros(total_syns)  # dtype uint16 or uint8
        edge_group_index = np.zeros(total_syns)  # uint32
        edge_type_ids = np.zeros(total_syns)  # uint32

        # TODO: Another potential issue if node-ids don't start with 0
        index_ptrs = np.zeros(len(self._nodes)+1)  # TODO: issue when target nodes come from another network
        index_ptr_itr = 0

        """
        distance_table = ets['params']['distance']
        print len(distance_table._prop_array)
        print len(distance_table._index)
        print distance_table._index
        # self._prop_array = np.zeros(nvalues)
        # self._prop_table = np.zeros((nvalues, 1))  # TODO: set dtype
        # self._index = np.zeros((nvalues, 2), dtype=np.uint8)
        exit()
        """

        #print groups
        #exit()
        """
        for trg_node in self._nodes:
            ets = self.__edges_tables[0]
            distance_table = ets['params']['distance']
            #print distance_table
            syn_table = ets['syn_table']
            if syn_table.has_target(trg_node.node_id):
                for src_id, nsyns in syn_table.trg_itr(trg_node.node_id):

                    for v in distance_table.itr_vals(src_id, trg_node.node_id):
                        print '{} --> {} ({})'.format(src_id, trg_node.node_id, v)

        print self.__edges_tables[0]['params']['distance']
        exit()
        """

        gid_indx = 0
        for trg_node in self._nodes:  # TODO: need to order from 0 to N
            index_ptrs[index_ptr_itr] = gid_indx
            index_ptr_itr += 1

            for ets in self.__edges_tables:
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
                                #if param_name == 'distance':
                                #    print len(groups[0]['distance'])
                                param_vals = group_table[param_name]

                                #if param_name == 'distance':
                                #    print param_name
                                #    print len(param_vals)
                                #exit()

                                for val in param_table.itr_vals(src_id, trg_node.node_id):
                                    #print val
                                    param_vals.append(val)

                    else:
                        # TODO: if no properties just print nsyns table.
                        # raise NotImplementedError()
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

        #print len(groups[0]['distance'])
        #exit()

        trg_gids = trg_gids[:gid_indx]
        src_gids = src_gids[:gid_indx]
        edge_groups = edge_groups[:gid_indx]
        edge_group_index = edge_group_index[:gid_indx]
        edge_type_ids = edge_type_ids[:gid_indx]
        #src_gids.reshape((gid_indx,))
        #edge_groups.reshape((gid_indx,))
        #edge_group_index.reshape((gid_indx,))
        #edge_type_ids.reshape((gid_indx,))
        """
        trg_gids = np.zeros(total_syns)  # set dtype to uint64
        src_gids = np.zeros(total_syns)
        edge_groups = np.zeros(total_syns)  # dtype uint16 or uint8
        edge_group_index = np.zeros(total_syns)  # uint32
        edge_type_ids = np.zeros(total_syns)  # uint32
        """

        #print len(groups[0]['nsyns'])

        index_ptrs[index_ptr_itr] = gid_indx

        #print trg_gids
        #exit()


        with h5py.File(edges_file_name, 'w') as hf:
            hf.create_dataset('edges/target_gid', data=trg_gids, dtype='uint64')
            hf['edges/target_gid'].attrs['network'] = trg_network
            #'source_network'
            hf.create_dataset('edges/source_gid', data=src_gids, dtype='uint64')
            hf['edges/source_gid'].attrs['network'] = src_network

            hf.create_dataset('edges/edge_group', data=edge_groups, dtype='uint16')
            hf.create_dataset('edges/edge_group_index', data=edge_group_index, dtype='uint32')
            hf.create_dataset('edges/edge_type_id', data=edge_type_ids, dtype='uint32')
            hf.create_dataset('edges/index_pointer', data=index_ptrs, dtype='uint32')

            for group_id, params_dict in groups.items():
                for params_key, params_vals in params_dict.items():
                    group_path = 'edges/{}/{}'.format(group_id, params_key)
                    #print group_path, len(params_vals)
                    #print group_dtypes
                    dtype = group_dtypes[group_id][params_key]
                    if dtype is not None:
                        hf.create_dataset(group_path, data=list(params_vals), dtype=dtype)
                    else:
                        hf.create_dataset(group_path, data=list(params_vals))


    """
    def _add_edges(self, edge, connections):
        edgetable = self.EdgeTable(edge)
        self.__edges_types[edge.id] = edgetable

        for con in connections:
            if con[2] is not None:
                self._nedges += 1
                edgetable[con[0], con[1]] = con[2]
    """

    def _clear(self):
        self._nedges = 0
        self._nnodes = 0



    def _edges_iter(self, target_gids, source_gids):
        raise NotImplementedError()

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
            #self._source_nodes = connection_map.source_nodes
            #self._target_nodes = connection_map.target_nodes

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
            #print nvalues
            self._prop_array = np.zeros(nvalues)
            #self._prop_table = np.zeros((nvalues, 1))  # TODO: set dtype
            self._index = np.zeros((nvalues, 2), dtype=np.uint32)
            self._itr_index = 0

        def __setitem__(self, key, value):
            self._index[self._itr_index, 0] = key[0]  # src_node_id
            self._index[self._itr_index, 1] = key[1]  # trg_node_id
            #print value
            # print self._prop_array
            self._prop_array[self._itr_index] = value
            self._itr_index += 1

        def itr_vals(self, src_id, trg_id):
            indicies = np.where((self._index[:, 0] == src_id) & (self._index[:, 1] == trg_id))
            for val in self._prop_array[indicies]:
                yield val
                #yield self._prop_array[i]
