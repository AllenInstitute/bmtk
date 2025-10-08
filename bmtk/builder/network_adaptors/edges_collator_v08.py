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
        self.n_rank_edges = sum(e.n_edges for e in edge_types_table)
        self._nedges_total = None
        self.can_sort = mpi_size == 1
        self.group_ids = None
        self.group_metadata = {}

        self.mpi_collection_method = opt_args.get('mpi_collection_method', 'comm')

    def process(self):
        group_ids_lu = self.assign_group_ids()
        self.group_ids = group_ids_lu.values()

        self._collect_across_ranks()
        exit()

        for etable in self._edge_type_tables:
            group_id = group_ids_lu[etable.hash_key]
            if etable.hash_key not in self.group_metadata:
                self.group_metadata[group_id] = {
                    'columns': etable.get_property_metadata(),
                    'size': 0
                }
            self.group_metadata[group_id]['size'] += etable.n_edges





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
            return self._edge_type_tables
        elif self.mpi_collection_method == 'pickle':
            filename = f'.conn_data.rank{mpi_rank}.{build_time_uuid()}.pkl'
            
            # print(self._edge_type_tables[0])
            # exit()
            with open(filename, 'wb') as fhandle:
                pickle.dump(self._edge_type_tables, fhandle)

            exit()

        elif self.mpi_collection_method == 'comm':
            recv_data = {}
            if mpi_rank == 0:
                recv_data[0] = self._edge_type_tables
                for s in range(1, mpi_size):
                    recv_data[s] = comm.recv(source=s, tag=222)

                print(recv_data)
                print(recv_data[1][0].source_nodes_map)
                exit()

            else:
                comm.send(self._edge_type_tables, dest=0, tag=222)
            

        else:
            raise NotImplementedError(f'Invalid mpi_collection_method option "{self.mpi_collection_method}" (valid: comm, pickle, none)')
