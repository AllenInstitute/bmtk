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
import logging

from .network import Network
from bmtk.builder.node import Node
from bmtk.builder.edge import Edge
from bmtk.utils import sonata

from .edges_collator import EdgesCollator
from .edge_props_table import EdgeTypesTable
from ..index_builders import create_index_in_memory, create_index_on_disk
from ..builder_utils import mpi_rank, mpi_size, barrier
from ..edges_sorter import sort_edges


logger = logging.getLogger(__name__)


class DenseNetwork(Network):
    def __init__(self, name, **network_props):
        super(DenseNetwork, self).__init__(name, **network_props or {})
        # self.__edges_types = {}
        # self.__src_mapping = {}
        # self.__networks = {}
        # self.__node_count = 0
        self._nodes = []
        self.__edges_tables = []
        self._target_networks = {}

    def _initialize(self):
        self.__id_map = []
        self.__lookup = []

    def _add_nodes(self, nodes):
        self._nodes.extend(nodes)
        self._nnodes = len(self._nodes)

    def edges_table(self):
        return self.__edges_tables

    def _save_nodes(self, nodes_file_name):
        if not self._nodes_built:
            self._build_nodes()

        # save the node_types file
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
        """

        :param connection_map:
        :param i:
        """
        edge_type_id = connection_map.edge_type_properties['edge_type_id']
        logger.debug('Generating edges data for edge_types_id {}.'.format(edge_type_id))
        edges_table = EdgeTypesTable(connection_map, network_name=self.name)
        connections = connection_map.connection_itr()

        # iterate through all possible SxT source/target pairs and use the user-defined function/list/value to update
        # the number of syns between each pair. TODO: See if this can be vectorized easily.
        for conn in connections:
            if conn[2]:
                edges_table.set_nsyns(source_id=conn[0], target_id=conn[1], nsyns=conn[2])

        target_net = connection_map.target_nodes
        self._target_networks[target_net.network_name] = target_net.network

        # For when the user specified individual edge properties to be put in the hdf5 (syn_weight, syn_location, etc),
        # get prop value and add it to the edge-types table. Need to fetch and store SxTxN value (where N is the avg
        # num of nsyns between each source/target pair) and it is necessary that the nsyns table be finished.
        for param in connection_map.params:
            rule = param.rule
            rets_multiple_vals = isinstance(param.names, (list, tuple, np.ndarray))

            if not rets_multiple_vals:
                prop_name = param.names  # name of property
                prop_type = param.dtypes.get(prop_name, None)
                edges_table.create_property(prop_name=param.names, prop_type=prop_type)  # initialize property array

                for source_node, target_node, edge_index in edges_table.iter_edges():
                    # calls connection map rule and saves value to edge table
                    pval = rule(source_node, target_node)
                    edges_table.set_property_value(prop_name=prop_name, edge_index=edge_index, prop_value=pval)

            else:
                # Same as loop above, but some connection-map 'rules' will return multiple properties for each edge.
                pnames = param.names
                ptypes = [param.dtypes[pn] for pn in pnames]
                for prop_name, prop_type in zip(pnames, ptypes):
                    edges_table.create_property(prop_name=prop_name, prop_type=prop_type)  # initialize property arrays

                for source_node, target_node, edge_index in edges_table.iter_edges():
                    pvals = rule(source_node, target_node)
                    for pname, pval in zip(pnames, pvals):
                        edges_table.set_property_value(prop_name=pname, edge_index=edge_index, prop_value=pval)

        logger.debug('Edge-types {} data built with {} connection ({} synapses)'.format(
            edge_type_id, edges_table.n_edges, edges_table.n_syns)
        )

        edges_table.save()

        # To EdgeTypesTable the number of synaptic/gap connections between all source/target paris, which can be more
        # than the number of actual edges stored (for efficency), may be a better user-representation.
        self._nedges += edges_table.n_syns  # edges_table.n_edges
        self.__edges_tables.append(edges_table)

    def _get_edge_group_id(self, params_hash):
        return int(params_hash)

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

    def _save_edges(self, edges_file_name, src_network, trg_network, pop_name=None, sort_by='target_node_id',
                    index_by=('target_node_id', 'source_node_id')):
        barrier()

        if mpi_rank == 0:
            logger.debug('Saving {} --> {} edges to {}.'.format(src_network, trg_network, edges_file_name))

        filtered_edge_types = [
            # Some edges may not match the source/target population
            et for et in self.__edges_tables
            if et.source_network == src_network and et.target_network == trg_network
        ]

        merged_edges = EdgesCollator(filtered_edge_types, network_name=self.name)
        merged_edges.process()
        n_total_conns = merged_edges.n_total_edges
        barrier()

        if n_total_conns == 0:
            if mpi_rank == 0:
                logger.warning('Was not able to generate any edges using the "connection_rule". Not saving.')
            return

        # Try to sort before writing file, If edges are split across ranks/files for MPI/size issues then we need to
        # write to disk first then sort the hdf5 file
        sort_on_disk = False
        edges_file_name_final = edges_file_name
        if sort_by:
            if merged_edges.can_sort:
                merged_edges.sort(sort_by=sort_by)
            else:
                sort_on_disk = True
                edges_file_name_final = edges_file_name

                edges_file_basename = os.path.basename(edges_file_name)
                edges_file_dirname = os.path.dirname(edges_file_name)
                edges_file_name = os.path.join(edges_file_dirname, '.unsorted.{}'.format(edges_file_basename))
                if mpi_rank == 0:
                    logger.debug('Unable to sort edges in memory, will temporarly save to {}'.format(edges_file_name) +
                                 ' before sorting hdf5 file.')
        barrier()

        if mpi_rank == 0:
            logger.debug('Saving {} edges to disk'.format(n_total_conns))
            pop_name = '{}_to_{}'.format(src_network, trg_network) if pop_name is None else pop_name
            with h5py.File(edges_file_name, 'w') as hf:
                # Initialize the hdf5 groups and datasets
                add_hdf5_attrs(hf)
                pop_grp = hf.create_group('/edges/{}'.format(pop_name))

                pop_grp.create_dataset('source_node_id', (n_total_conns,), dtype='uint64')
                pop_grp['source_node_id'].attrs['node_population'] = src_network
                pop_grp.create_dataset('target_node_id', (n_total_conns,), dtype='uint64')
                pop_grp['target_node_id'].attrs['node_population'] = trg_network
                pop_grp.create_dataset('edge_group_id', (n_total_conns,), dtype='uint16')
                pop_grp.create_dataset('edge_group_index', (n_total_conns,), dtype='uint32')
                pop_grp.create_dataset('edge_type_id', (n_total_conns,), dtype='uint32')

                for group_id in merged_edges.group_ids:
                    # different model-groups will have different datasets/properties depending on what edge information
                    # is being saved for each edges
                    model_grp = pop_grp.create_group(str(group_id))
                    for prop_mdata in merged_edges.get_group_metadata(group_id):
                        model_grp.create_dataset(prop_mdata['name'], shape=prop_mdata['dim'], dtype=prop_mdata['type'])

                # Uses the collated edges (eg combined edges across all edge-types) to actually write the data to hdf5,
                # potentially in multiple chunks. For small networks doing it this way isn't very effiecent, however
                # this has the benefits:
                #  * For very large networks it won't always be possible to store all the data in memory.
                #  * When using MPI/multi-node the chunks can represent data from different ranks.
                for chunk_id, idx_beg, idx_end in merged_edges.itr_chunks():
                    pop_grp['source_node_id'][idx_beg:idx_end] = merged_edges.get_source_node_ids(chunk_id)
                    pop_grp['target_node_id'][idx_beg:idx_end] = merged_edges.get_target_node_ids(chunk_id)
                    pop_grp['edge_type_id'][idx_beg:idx_end] = merged_edges.get_edge_type_ids(chunk_id)
                    pop_grp['edge_group_id'][idx_beg:idx_end] = merged_edges.get_edge_group_ids(chunk_id)
                    pop_grp['edge_group_index'][idx_beg:idx_end] = merged_edges.get_edge_group_indices(chunk_id)

                    for group_id, prop_name, grp_idx_beg, grp_idx_end in merged_edges.get_group_data(chunk_id):
                        prop_array = merged_edges.get_group_property(prop_name, group_id, chunk_id)
                        pop_grp[str(group_id)][prop_name][grp_idx_beg:grp_idx_end] = prop_array

            if sort_on_disk:
                logger.debug('Sorting {} by {} to {}'.format(edges_file_name, sort_by, edges_file_name_final))
                sort_edges(
                    input_edges_path=edges_file_name,
                    output_edges_path=edges_file_name_final,
                    edges_population='/edges/{}'.format(pop_name),
                    sort_by=sort_by,
                    # sort_on_disk=True,
                )
                try:
                    logger.debug('Deleting intermediate edges file {}.'.format(edges_file_name))
                    os.remove(edges_file_name)
                except OSError as e:
                    logger.warning('Unable to remove intermediate edges file {}.'.format(edges_file_name))

            if index_by:
                index_by = index_by if isinstance(index_by, (list, tuple)) else [index_by]
                for index_type in index_by:
                    logger.debug('Creating index {}'.format(index_type))
                    create_index_in_memory(
                        edges_file=edges_file_name_final,
                        edges_population='/edges/{}'.format(pop_name),
                        index_type=index_type
                    )

        barrier()
        del merged_edges

        if mpi_rank == 0:
            logger.debug('Saving completed.')

    def _clear(self):
        self._nedges = 0
        self._nnodes = 0

    def edges_iter(self, trg_gids, src_network=None, trg_network=None):
        matching_edge_tables = self.__edges_tables
        if trg_network is not None:
            matching_edge_tables = [et for et in self.__edges_tables if et.target_network == trg_network]

        if src_network is not None:
            matching_edge_tables = [et for et in matching_edge_tables if et.source_network == src_network]

        for edge_type_table in matching_edge_tables:
            et_df = edge_type_table.to_dataframe()
            et_df = et_df[et_df['target_node_id'].isin(trg_gids)]
            if len(et_df) == 0:
                continue

            edge_type_props = edge_type_table.edge_type_properties
            for row in et_df.to_dict(orient='records'):
                yield Edge(
                    src_gid=row['source_node_id'],
                    trg_gid=row['target_node_id'],
                    edge_type_props=edge_type_props,
                    syn_props=row
                )

    @property
    def nnodes(self):
        if not self.nodes_built:
            return 0
        return self._nnodes

    @property
    def nedges(self):
        return self._nedges


def add_hdf5_attrs(hdf5_handle):
    # TODO: move this as a utility function
    hdf5_handle['/'].attrs['magic'] = np.uint32(0x0A7A)
    hdf5_handle['/'].attrs['version'] = [np.uint32(0), np.uint32(1)]
