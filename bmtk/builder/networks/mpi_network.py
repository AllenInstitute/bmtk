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
from .dm_network import DenseNetwork
from mpi4py import MPI
from heapq import heappush, heappop
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


class MPINetwork(DenseNetwork):
    def __init__(self, name, **network_props):
        super(MPINetwork, self).__init__(name, **network_props or {})
        self._edge_assignment = None

    def _add_edges(self, connection_map, i):
        if self._assign_to_rank(i):
            super(MPINetwork, self)._add_edges(connection_map, i)

    def save_nodes(self, nodes_file_name, node_types_file_name):
        if rank == 0:
            super(MPINetwork, self).save_nodes(nodes_file_name, node_types_file_name)
        comm.Barrier()

    """
    def save_edges(self, edges_file_name=None, edge_types_file_name=None, output_dir='.', src_network=None,
                   trg_network=None, force_build=True, force_overwrite=False):

        if rank == 0:
            # print rank, len(self.edges_table())
            super(MPINetwork, self).save_edges(edges_file_name, edge_types_file_name, output_dir, src_network,
                                               trg_network, force_build, force_overwrite)

        comm.Barrier()
    """

    def edges_iter(self, trg_gids, src_network=None, trg_network=None):
        for trg_gid in trg_gids:
            edges = list(super(MPINetwork, self).edges_iter([trg_gid], src_network, trg_network))
            collected_edges = comm.gather(edges, root=0)
            if rank == 0:
                for edge_list in collected_edges:
                    for edge in edge_list:
                        # print 'b'
                        yield edge
            else:
                yield None

            comm.Barrier()

    def _save_edges(self, edges_file_name, src_network, trg_network):
        target_gids = [n.node_id for n in self._target_networks[trg_network].nodes()]
        # TODO: make sure target_gids are sorted

        trg_gids_ds = []
        src_gids_ds = []
        edge_type_id_ds = []
        edge_group_ds = []
        edge_group_index_ds = []

        eg_collection = {}
        eg_ids = 0
        eg_lookup = {}
        eg_table = {}
        eg_indices = {}
        for cm in self.get_connections():
            col_key = cm.properties_keys()
            if col_key in eg_collection:
                group_id = eg_collection[col_key]
            else:
                group_id = eg_ids
                eg_collection[col_key] = group_id
                eg_ids += 1
            eg_lookup[cm.edge_type_id] = group_id
            eg_indices[group_id] = 0
            eg_table[group_id] = {k: [] for k in cm.property_names}

        for e in self.edges_iter(target_gids, src_network=src_network, trg_network=trg_network):
            if rank == 0:
                trg_gids_ds.append(e.target_gid)
                src_gids_ds.append(e.source_gid)
                edge_type_id_ds.append(e.edge_type_id)

                group_id = eg_lookup[e.edge_type_id]
                edge_group_ds.append(group_id)
                group_id_index = eg_indices[group_id]
                edge_group_index_ds.append(group_id_index)
                eg_indices[group_id] += 1

                for k, v in e.synaptic_properties.items():
                    eg_table[group_id][k].append(v)

        if rank == 0:
            # Create index from target_gids dataset
            index_pointer_ds = []
            cur_gid = 0
            index = 0
            while index < len(trg_gids_ds):
                if trg_gids_ds[index] == cur_gid:
                    index += 1
                else:
                    cur_gid += 1
                    index_pointer_ds.append(index)
            index_pointer_ds.append(len(trg_gids_ds)+1)


            with h5py.File(edges_file_name, 'w') as hf:
                hf.create_dataset('edges/target_gid', data=trg_gids_ds, dtype='uint64')
                hf['edges/target_gid'].attrs['network'] = trg_network
                hf.create_dataset('edges/source_gid', data=src_gids_ds, dtype='uint64')
                hf['edges/source_gid'].attrs['network'] = src_network

                hf.create_dataset('edges/edge_group', data=edge_group_ds, dtype='uint16')
                hf.create_dataset('edges/edge_group_index', data=edge_group_index_ds, dtype='uint32')
                hf.create_dataset('edges/edge_type_id', data=edge_type_id_ds, dtype='uint32')
                hf.create_dataset('edges/index_pointer', data=index_pointer_ds, dtype='uint32')

                for gid, group in eg_table.items():
                    for col_key, col_ds in group.items():
                        ds_loc = 'edges/{}/{}'.format(gid, col_key)
                        hf.create_dataset(ds_loc, data=col_ds)

        comm.Barrier()

    def _assign_to_rank(self, i):
        if self._edge_assignment is None:
            self._build_rank_assignments()

        return rank == self._edge_assignment[i]

    def _build_rank_assignments(self):
        """Builds the _edge_assignment array.

        Division of connections is decided by the maximum possible edges (i.e. number of source and target nodes). In
        the end assignment should balance the connection matrix sizes need by each rank.
        """
        rank_heap = []  # A heap of tuples (weight, rank #)
        for a in range(nprocs):
            heappush(rank_heap, (0, a))

        # find the rank with the lowest weight, assign that rank to build the i'th connection matrix, update the rank's
        # weight and re-add to the heap.
        # TODO: sort connection_maps in descending order to get better balance
        self._edge_assignment = []
        for cm in self.get_connections():
            r = heappop(rank_heap)
            self._edge_assignment.append(r[1])
            heappush(rank_heap, (r[0] + cm.max_connections(), r[1]))

