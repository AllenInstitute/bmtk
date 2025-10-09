import os
import numpy as np
import logging
import h5py
import pickle
import copy

from ..builder_utils import mpi_rank, mpi_size, barrier, build_time_uuid, comm
from mpi4py import MPI
from .edge_props_table import EdgeTypesTableMPI
from functools import reduce

logger = logging.getLogger(__name__)


class EdgesCollator:
    def __init__(self, edge_types_table, network_name, **opt_args):
        self._edge_type_tables = edge_types_table
        self.collected_edges = []
        self.n_rank_edges = sum(e.n_edges for e in edge_types_table)
        self._nedges_total = None
        self.can_sort = False # mpi_size == 1
        self.group_ids = None
        self.group_ids_lu = None
        self.group_metadata = {}
        self._group_offsets = None

        self.mpi_collection_method = opt_args.get('mpi_collection_method', 'comm')

    def process(self):
        self.group_ids_lu = self.assign_group_ids()
        self.group_ids = self.group_ids_lu.values()

        self._collect_across_ranks()
        # exit()

        for etable in self.collected_edges:
            group_id = self.group_ids_lu[etable.hash_key]
            if group_id not in self.group_metadata:
                self.group_metadata[group_id] = {
                    'columns': etable.get_property_metadata(),
                    'size': 0
                }
            self.group_metadata[group_id]['size'] += etable.n_edges

        self._group_offsets = [(None, None) for _ in self.collected_edges]
        c_indices = {grp_id: 0 for grp_id in self.group_ids}
        for i, etable in enumerate(self.collected_edges):
            grp_id = self.group_ids_lu[etable.hash_key]
            idx_beg = c_indices[grp_id]
            idx_end = idx_beg + etable.n_edges
            c_indices[grp_id] = idx_end
            self._group_offsets[i] = (idx_beg, idx_end)


    def itr_chunks(self):
        # self._group_offsets = {grp_id: 0 for grp_id in self.group_ids}

        idx_beg = 0
        for edge_table_num, edges_table in enumerate(self.collected_edges):
            idx_end = idx_beg + edges_table.n_edges
            yield edge_table_num, idx_beg, idx_end

    def get_source_node_ids(self, chunk_id):
        src_node_ids, _ = self.collected_edges[chunk_id].edge_type_node_ids
        return src_node_ids

    def get_target_node_ids(self, chunk_id):
        _, trg_node_ids = self.collected_edges[chunk_id].edge_type_node_ids
        return trg_node_ids

    def get_edge_type_ids(self, chunk_id):
        return self.collected_edges[chunk_id].edge_type_id

    def get_edge_group_ids(self, chunk_id, as_array=False):
        edge_group_table = self.collected_edges[chunk_id]
        grp_id = self.group_ids_lu[edge_group_table.hash_key]
        if as_array:
            return np.full(edge_group_table.n_edges, grp_id)
        else:
            return grp_id


    def get_edge_group_indices(self, chunk_id):
        idx_beg, idx_end = self._group_offsets[chunk_id]
        return np.arange(idx_beg, idx_end, dtype='int')


    def get_group_data(self, chunk_id):
        idx_beg, idx_end = self._group_offsets[chunk_id]
        edge_group_table = self.collected_edges[chunk_id]
        grp_id = self.group_ids_lu[edge_group_table.hash_key]
        return [(grp_id, prop['name'], idx_beg, idx_end) for prop in edge_group_table.get_property_metadata()]


    def get_group_property(self, prop_name, group_id, chunk_id):
        edges_table = self.collected_edges[chunk_id]
        return edges_table.get_property_value(prop_name)


    def assign_group_ids(self):
        hash_keys = set([e.hash_key for e in self._edge_type_tables])
        if mpi_size > 1:
            recv_data = comm.allgather(hash_keys)
            all_hash_keys = list(reduce(set.union, recv_data))
            all_hash_keys.sort()
        else:
            all_hash_keys = hash_keys

        return {hkey: grp_id for grp_id, hkey in enumerate(all_hash_keys)}


    @property
    def n_total_edges(self):
        if self._nedges_total is None:
            if mpi_size == 1:
                self._nedges_total = self.n_rank_edges
            else:
                snd_buf = np.array(self.n_rank_edges, dtype=np.uint)
                rcv_buf = np.zeros(1, dtype=np.uint)
                comm.Allreduce(snd_buf, rcv_buf, op=MPI.SUM)
                self._nedges_total = rcv_buf[0]
        
        return self._nedges_total


    def get_group_metadata(self, group_id):
        grp_md = self.group_metadata[group_id]
        return [{
            'name': cd['name'],
            'type': cd['dtype'],
            'dim': (grp_md['size'], )
        } for cd in grp_md['columns']]


    def _collect_across_ranks(self):
        if mpi_size == 1 or self.mpi_collection_method == 'none':
            self.collected_edges = self._edge_type_tables
        
        elif self.mpi_collection_method == 'pickle':
            filename = f'.conn_data.rank{mpi_rank}.{build_time_uuid()}.pkl'
            
            with open(filename, 'wb') as fhandle:
                pickle.dump(self._edge_type_tables, fhandle)


        elif self.mpi_collection_method == 'comm':
            recv_data = {}
            if mpi_rank == 0:
                recv_data[0] = self._edge_type_tables
                for s in range(1, mpi_size):
                    recv_data[s] = comm.recv(source=s, tag=222)

                # collected_edges = []
                for rank, recv_edge_tables in recv_data.items():
                    if isinstance(recv_edge_tables, (list, tuple)):
                        self.collected_edges.extend(recv_edge_tables)
                    
                    elif isinstance(recv_edge_tables, str) and os.path.exists(recv_edge_tables):
                        with open(recv_edge_tables, 'rb') as f:
                            try:
                                self.collected_edges.extend(pickle.load(recv_edge_tables))
                                os.remove(recv_edge_tables)
                            except Exception:
                                logger.warning(f'Unable to load pickle file {recv_edge_tables}')

                    else:
                        logger.warning(f'Unable to recieve edges from rank {rank}')

            else:
                comm.send(self._edge_type_tables, dest=0, tag=222)
            
        else:
            raise NotImplementedError(f'Invalid mpi_collection_method option "{self.mpi_collection_method}" (valid: comm, pickle, none)')
