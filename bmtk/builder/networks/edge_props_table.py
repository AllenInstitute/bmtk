import numpy as np


class EdgeTypesTable(object):
    """A class for creating and storing the actual connectivity matrix plus all the possible (hdf5 bound) properties
    of an edge - unlike the ConnectionMap class which only stores the unevaluated rules each edge-types. There should
    be one EdgeTypesTable for each call to Network.add_edges(...)
    """
    def __init__(self, connection_map):
        self._connection_map = connection_map

        self.source_network = connection_map.source_nodes.network_name
        self.target_network = connection_map.target_nodes.network_name
        self.edge_type_id = connection_map.edge_type_properties['edge_type_id']
        self.edge_group_id = -1  # This will be assigned later during save_edges

        # Create the nsyns table to store the num of synapses/edges between each possible source/target node pair
        self._nsyns_idx2src = [n.node_id for n in connection_map.source_nodes]
        self._nsyns_src2idx = {node_id: i for i, node_id in enumerate(self._nsyns_idx2src)}
        self._nsyns_idx2trg = [n.node_id for n in connection_map.target_nodes]
        self._nsyns_trg2idx = {node_id: i for i, node_id in enumerate(self._nsyns_idx2trg)}
        self._nsyns_updated = False
        self._n_syns = 0
        self.nsyn_table = np.zeros((len(self._nsyns_idx2src), len(self._nsyns_idx2trg)), dtype=np.uint8)

        self._prop_vals = {}  # used to store the arrays for each property
        self._prop_node_ids = None  # used to save the source_node_id and target_node_id for each edge
        self._source_nodes_map = None  # map source_node_id --> Node object
        self._target_nodes_map = None  # map target_node_id --> Node object

    @property
    def n_syns(self):
        if self._nsyns_updated:
            self._nsyns_updated = False
            self._n_syns = int(np.sum(self.nsyn_table))
        return self._n_syns

    @property
    def n_edges(self):
        if self._prop_vals:
            return self.n_syns
        else:
            return np.count_nonzero(self.nsyn_table)

    @property
    def prop_node_ids(self):
        if self._prop_node_ids is None or self._nsyns_updated:
            self._prop_node_ids = np.zeros((self.n_edges, 2), dtype=np.uint32)
            idx = 0
            for r, src_id in enumerate(self._nsyns_idx2src):
                for c, trg_id in enumerate(self._nsyns_idx2trg):
                    nsyns = self.nsyn_table[r, c]
                    self._prop_node_ids[idx:(idx + nsyns), 0] = src_id
                    self._prop_node_ids[idx:(idx + nsyns), 1] = trg_id
                    idx += nsyns

        return self._prop_node_ids

    @property
    def source_nodes_map(self):
        if self._source_nodes_map is None:
            self._source_nodes_map = {s.node_id: s for s in self._connection_map.source_nodes}
        return self._source_nodes_map

    @property
    def target_nodes_map(self):
        if self._target_nodes_map is None:
            self._target_nodes_map = {t.node_id: t for t in self._connection_map.target_nodes}
        return self._target_nodes_map

    @property
    def hash_key(self):
        prop_keys = ['{}({})'.format(p['name'], p['dtype']) for p in self.get_property_metatadata()]
        prop_keys.sort()
        return hash(':'.join(prop_keys))

    def get_property_metatadata(self):
        if not self._prop_vals:
            return [{'name': 'nsyns', 'dtype': self.nsyn_table.dtype}]
        else:
            return [{'name': pname, 'dtype': pvals.dtype} for pname, pvals in self._prop_vals.items()]

    def set_nsyns(self, source_id, target_id, nsyns):
        assert(nsyns >= 0)
        indexed_pair = (self._nsyns_src2idx[source_id], self._nsyns_trg2idx[target_id])
        self.nsyn_table[indexed_pair] = nsyns
        self._nsyns_updated = True

    def create_property(self, prop_name, prop_type=None):
        assert(prop_name not in self._prop_vals)

        prop_size = self.n_syns
        self._prop_vals[prop_name] = np.zeros(prop_size, dtype=prop_type)

    def iter_edges(self):
        prop_node_ids = self.prop_node_ids
        src_nodes_lu = self.source_nodes_map
        trg_nodes_lu = self.target_nodes_map
        for edge_index in range(self.n_edges):
            src_id = prop_node_ids[edge_index, 0]
            trg_id = prop_node_ids[edge_index, 1]
            source_node = src_nodes_lu[src_id]
            target_node = trg_nodes_lu[trg_id]

            yield source_node, target_node, edge_index

    def set_property_value(self, prop_name, edge_index, prop_value):
        self._prop_vals[prop_name][edge_index] = prop_value

    def get_property_value(self, prop_name):
        if prop_name == 'nsyns':
            # nonzero_idxs = np.argwhere(nsyn_table_flat > 0).flatten()
            nsyns_table_flat = self.nsyn_table.ravel()
            # node_ids_flat = np.array(np.meshgrid(self._nsyns_idx2src, self._nsyns_idx2trg)).T.reshape(-1, 2)
            nonzero_indxs = np.argwhere(nsyns_table_flat > 0).flatten()
            return nsyns_table_flat[nonzero_indxs]
        else:
            return self._prop_vals[prop_name]

    def free_data(self):
        del self.nsyn_table
        del self._prop_vals


class EdgeTypesTableMPI(EdgeTypesTable):
    pass