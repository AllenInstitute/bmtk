import os
import numpy as np
import h5py
import hashlib
import logging
import pandas as pd

from bmtk.builder.builder_utils import mpi_rank, mpi_size, barrier, build_time_uuid


logger = logging.getLogger(__name__)


class EdgeTypesTableIFace(object):
    @property
    def n_syns(self):
        raise NotImplementedError()

    @property
    def n_edges(self):
        raise NotImplementedError()


class EdgeTypesTableMemory(object):
    """A class for creating and storing the actual connectivity matrix plus all the possible (hdf5 bound) properties
    of an edge - unlike the ConnectionMap class which only stores the unevaluated rules each edge-types. There should
    be one EdgeTypesTable for each call to Network.add_edges(...)

    By default edges in the SONATA edges.h5 table are stored in a (sparse) SxT table, S/T the num of source/target
    cells. If individual edge properties (syn_weight, syn_location, etc) and added then it must be stored in a SxTxN
    table, N the avg. number of synapses between each source/target pair. The actually number of edges (ie rows)
    saved in the SONATA file will vary.
    """

    def __init__(self, connection_map, network_name):
        self._connection_map = connection_map
        self._network_name = network_name

        self.source_network = connection_map.source_nodes.network_name
        self.target_network = connection_map.target_nodes.network_name
        self.edge_type_id = connection_map.edge_type_properties['edge_type_id']
        self.edge_group_id = -1  # This will be assigned later during save_edges

        # Create the nsyns table to store the num of synapses/edges between each possible source/target node pair
        # self._nsyns_idx2src = [n.node_id for n in connection_map.source_nodes]
        # self._nsyns_src2idx = {node_id: i for i, node_id in enumerate(self._nsyns_idx2src)}
        # self._nsyns_idx2trg = [n.node_id for n in connection_map.target_nodes]
        # self._nsyns_trg2idx = {node_id: i for i, node_id in enumerate(self._nsyns_idx2trg)}
        # self._nsyns_updated = False
        # self._n_syns = 0
        # self.nsyn_table = np.zeros((len(self._nsyns_idx2src), len(self._nsyns_idx2trg)), dtype=np.uint32)

        max_conns = connection_map.max_connections()
        self.nsyn_table_src_ids = np.zeros(max_conns, dtype=np.uint32)
        self.nsyn_table_trg_ids = np.zeros(max_conns, dtype=np.uint32)
        self.nsyn_table_vals = np.zeros(max_conns, dtype=np.uint16)
        
        # logger.info(f'> Edgetype {self.edge_type_id}: {len(connection_map.source_nodes)}x{len(connection_map.target_nodes)} : {max_conns}')
        self._nsyn_table_idx = 0
        self._nsyns = -1

        self._prop_vals = {}  # used to store the arrays for each property
        self._prop_node_ids = None  # used to save the source_node_id and target_node_id for each edge
        self._source_nodes_map = None  # map source_node_id --> Node object
        self._target_nodes_map = None  # map target_node_id --> Node object

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_source_nodes_map'] = {s.node_id: s for s in self._connection_map.source_nodes}
        del state['_connection_map']
        return state

    @property
    def n_syns(self):
        """Number of synapses."""
        if self._nsyns < 0:
            self._n_syns = np.sum(self.nsyn_table_vals[0:self._nsyn_table_idx])
        
        return int(self._n_syns)

    @property
    def n_edges(self):
        """Number of unque edges/connections (eg rows in SONATA edges file). When multiple synapse can be safely
        represented with just one edge it will have n_edges < n_syns.
        """
        if self._prop_vals:
            return self.n_syns
        else:
            return np.count_nonzero(self.nsyn_table_vals)

    @property
    def edge_type_node_ids(self):
        """Returns a table n_edges x 2, first column containing source_node_ids and second target_node_ids."""
        if self._prop_node_ids is None or self._nsyns_updated:
            if len(self._prop_vals) == 0:
                return self.nsyn_table_src_ids[:self._nsyn_table_idx], self.nsyn_table_trg_ids[:self._nsyn_table_idx]               
                # # Get the source and target node ids from the rows/columns of nsyns table cells that are greater than 0
                # nsyn_table_flat = self.nsyn_table.ravel()
                # src_trg_prods = np.array(np.meshgrid(self._nsyns_idx2src, self._nsyns_idx2trg)).T.reshape(-1, 2)
                # nonzero_idxs = np.argwhere(nsyn_table_flat > 0).flatten()
                # self._prop_node_ids = src_trg_prods[nonzero_idxs, :].astype(np.uint64)

            else:
                # If there are synaptic properties go through each source/target pair and add their node-ids N times,
                # where N is the number of synapses between the two nodes
                repeats = self.nsyn_table_vals[:self._nsyn_table_idx]
                src_ids_unrolled = np.repeat(self.nsyn_table_src_ids[:self._nsyn_table_idx], repeats)
                trg_ids_unrolled = np.repeat(self.nsyn_table_trg_ids[:self._nsyn_table_idx], repeats)
                return src_ids_unrolled, trg_ids_unrolled
        
        #         self._prop_node_ids = np.zeros((self.n_edges, 2), dtype=np.int64)
        #         idx = 0
        #         for r, src_id in enumerate(self._nsyns_idx2src):
        #             for c, trg_id in enumerate(self._nsyns_idx2trg):
        #                 nsyns = self.nsyn_table[r, c]

        #                 self._prop_node_ids[idx:(idx + nsyns), 0] = src_id
        #                 self._prop_node_ids[idx:(idx + nsyns), 1] = trg_id
        #                 idx += nsyns

        # return self._prop_node_ids

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
        """Creates a hash key for edge-types based on their (hdf5) properties, for grouping together properties of
        different edge-types. If two edge-types have the same (hdf5) properties they should have the same hash value.
        """
        prop_keys = ['{}({})'.format(p['name'], p['dtype']) for p in self.get_property_metatadata()]
        prop_keys.sort()

        # WARNING: python's hash() function is randomized which is a problem when using MPI to process different edge
        # types across different ranks.
        prop_keys = ':'.join(prop_keys).encode('utf-8')
        return hashlib.md5(prop_keys).hexdigest()[:9]

    @property
    def edge_type_properties(self):
        return self._connection_map.edge_type_properties


    def get_property_metatadata(self):
        return self.get_property_metadata()
    
    def get_property_metadata(self):
        if not self._prop_vals:
            return [{'name': 'nsyns', 'dtype': self.nsyn_table_vals.dtype}]
        else:
            return [{'name': pname, 'dtype': pvals.dtype} for pname, pvals in self._prop_vals.items()]

    def set_nsyns(self, source_id, target_id, nsyns):
        # assert(nsyns >= 0)
        indexed_pair = (self._nsyns_src2idx[source_id], self._nsyns_trg2idx[target_id])
        self.nsyn_table[indexed_pair] = nsyns
        self._nsyns_updated = True

    def set_nsyn(self, source_id, target_id, nsyn):
        self.nsyn_table_src_ids[self._nsyn_table_idx] = source_id
        self.nsyn_table_trg_ids[self._nsyn_table_idx] = target_id
        self.nsyn_table_vals[self._nsyn_table_idx] = nsyn
        self._nsyn_table_idx += 1

    def create_property(self, prop_name, prop_type=None):
        assert(prop_name not in self._prop_vals)

        prop_size = self.n_syns
        self._prop_vals[prop_name] = np.zeros(prop_size, dtype=prop_type)

    def iter_edges(self):
        edge_index = 0
        for idx in range(self._nsyn_table_idx):
            nsyns = self.nsyn_table_vals[idx]
            src_id = self.nsyn_table_src_ids[idx]
            trg_id = self.nsyn_table_trg_ids[idx]
            for _ in range(nsyns):
                source_node = self.source_nodes_map[src_id]
                target_node = self.target_nodes_map[trg_id]
                yield source_node, target_node, edge_index
                edge_index += 1
        
        # prop_node_ids = self.edge_type_node_ids
        # src_nodes_lu = self.source_nodes_map
        # trg_nodes_lu = self.target_nodes_map
        # for edge_index in range(self.n_edges):
        #     src_id = prop_node_ids[edge_index, 0]
        #     trg_id = prop_node_ids[edge_index, 1]
        #     source_node = src_nodes_lu[src_id]
        #     target_node = trg_nodes_lu[trg_id]

        #     yield source_node, target_node, edge_index

    def set_property_value(self, prop_name, edge_index, prop_value):
        self._prop_vals[prop_name][edge_index] = prop_value

    def get_property_value(self, prop_name):
        if prop_name == 'nsyns':
            nsyns_table_flat = self.nsyn_table_vals.ravel()
            nonzero_indxs = np.argwhere(nsyns_table_flat > 0).flatten()

            return nsyns_table_flat[nonzero_indxs]
        else:
            return self._prop_vals[prop_name]

    def to_dataframe(self, **kwargs):
        src_trg_ids = self.edge_type_node_ids
        ret_df = pd.DataFrame({
            'source_node_id': src_trg_ids[:, 0],
            'target_node_id': src_trg_ids[:, 1],
            # 'edge_type_id': self.edge_type_id
        })
        for edge_prop in self.get_property_metatadata():
            pname = edge_prop['name']
            ret_df[pname] = self.get_property_value(prop_name=pname)

        return ret_df

    def save(self):
        pass

    def free_data(self):
        del self.nsyn_table
        del self._prop_vals


# class EdgeTypesTableMPI(EdgeTypesTableMemory):
#     """Like parent used for storing actualized edges data, but designed for when using MPI and the edges-tables rules
#     are split across different ranks/processors.

#     The data tables are saved in temporary files on the disk, so that the rank which is responsible for writing to the
#     final hdf5 file. This could also be useful for very large networks built on one core.

#     TODO: Look into saving in memory, use MPI Gather/Send.
#     """
#     _tmp_table_valid = False  # Singleton flag to ensure hdf5 temp file is created only once

#     def __init__(self, connection_map, network_name, **opt_args):
#         super(EdgeTypesTableMPI, self).__init__(connection_map, network_name)
#         self.tmp_table_name = EdgeTypesTableMPI.get_tmp_table_path(mpi_rank)

#     @staticmethod
#     def get_tmp_table_path(rank=0, name=None):
#         builder_uuid = build_time_uuid()
#         return '.edge_types_table.{}.{}.h5'.format(rank, builder_uuid)

#     def _open_tmp_table(self):
#         if EdgeTypesTableMPI._tmp_table_handle is None:
#             EdgeTypesTableMPI._tmp_table_handle = h5py.File(self.tmp_table_name, 'w')

#         return EdgeTypesTableMPI._tmp_table_handle

#     def _init_tmp_table(self):
#         # There may/will be multiple EdgeTypeTables objects, but only one .edge_type_table*.h5 file per rank, but don't
#         # overwrite the file everytime save() is called. Use a singleton to ensure hdf5 tmp file is created only once.
#         if not EdgeTypesTableMPI._tmp_table_valid:
#             with h5py.File(self.tmp_table_name, 'w') as h5:
#                 h5.create_group('unprocessed')
#                 EdgeTypesTableMPI._tmp_table_valid = True

#     def save(self):
#         """Saves edges data to hdf5 on the disk so that other ranks can read it (without MPISend)."""
#         self._init_tmp_table()

#         source_node_ids, target_node_ids = super().edge_type_node_ids
#         if len(source_node_ids) == 0:
#             # ignore if no actual edges
#             return

#         with h5py.File(self.tmp_table_name, 'r+') as h5:
#             # Create a new group
#             edge_type_id_str = str(self.edge_type_id)
#             if edge_type_id_str in h5:
#                 del h5[edge_type_id_str]

#             edge_type_grp = h5.create_group('/unprocessed/{}/{}'.format(self._network_name, edge_type_id_str))

#             edge_type_grp.create_dataset('source_node_id', data=source_node_ids)
#             edge_type_grp.create_dataset('target_node_id', data=target_node_ids)

#             for prop_mdata in super().get_property_metatadata():
#                 pname = prop_mdata['name']
#                 ptype = prop_mdata['dtype']
#                 pvals = super().get_property_value(pname)
#                 edge_type_grp.create_dataset(pname, data=pvals, dtype=ptype)
#                 edge_type_grp.attrs['size'] = len(pvals)
#                 edge_type_grp.attrs['hash_key'] = self.hash_key


#             h5.flush()

#     def __del__(self):
#         tmp_h5_path = EdgeTypesTableMPI.get_tmp_table_path(rank=mpi_rank)
#         try:
#             if os.path.exists(tmp_h5_path):
#                 os.remove(tmp_h5_path)
#         except (FileNotFoundError, IOError, Exception) as e:
#             logger.warning('Unable to delete temp edges file {}.'.format(tmp_h5_path))


# class EdgeTypesTableUpdated(EdgeTypesTableMPI):
#     pass


class EdgeTypesTableMPIPickled(EdgeTypesTableMemory):
    def __init__(self, connection_map, network_name, **opt_args):
        super(EdgeTypesTableMPIPickled, self).__init__(connection_map, network_name)

    def save(self):
        pass
    
    def __del__(self):
        pass



first_msg = True
def log_once(msg):
    global first_msg
    if first_msg:
        logger.info(msg)
        first_msg = False


class EdgeTypesTableUpdated(object):
    def __new__(cls, *args, **kwargs):
        rank_passing = kwargs.get('rank_passing', 'h5')
        if mpi_size == 1:
            log_once('>> EdgeTypesTableMemory')
            return EdgeTypesTableMemory(*args, **kwargs)

        elif rank_passing == 'h5':
            log_once('>> EdgeTypesTableMPI')
            return EdgeTypesTableMPI(*args, **kwargs)

        else:
            log_once('>> EdgeTypesTableMPIPickled')
            return EdgeTypesTableMPIPickled(*args, **kwargs)
