import os
import numpy as np
import logging
import h5py

from ..builder_utils import mpi_rank, mpi_size, barrier
from .edge_props_table import EdgeTypesTableMPI


logger = logging.getLogger(__name__)


class EdgesCollatorSingular(object):
    """Used to collect all the edges data-tables created and stored in the EdgeTypesTable to simplify the process
    of saving into a SONATA edges file. All the actual edges may be stored across diffrent edge-type-tables/mpi-ranks
    and needs to be merged together (and possibly sorted) before writing to HDF5 file.
    """

    def __init__(self, edge_types_tables, network_name):
        self._edge_types_tables = edge_types_tables
        self._network_name = network_name
        self._model_groups_md = {}
        self._group_ids_lu = {}
        self._grp_id_itr = 0

        self.n_total_edges = sum(et.n_edges for et in edge_types_tables)

        self.assign_groups()

        self.can_sort = True
        self.source_ids = None
        self.target_ids = None
        self.edge_type_ids = None
        self.edge_group_ids = None
        self.edge_group_index = None
        self._prop_data = {}

    def process(self):
        logger.debug('Processing and collating {:,} edges.'.format(self.n_total_edges))

        self.source_ids = np.zeros(self.n_total_edges, dtype=np.uint)
        self.target_ids = np.zeros(self.n_total_edges, dtype=np.uint)
        self.edge_type_ids = np.zeros(self.n_total_edges, dtype=np.uint32)
        self.edge_group_ids = np.zeros(self.n_total_edges, dtype=np.uint32)
        self.edge_group_index = np.zeros(self.n_total_edges, dtype=np.uint32)

        self._prop_data = {
            g_id: {
                n: np.zeros(g_md['prop_size'], dtype=t) for n, t in zip(g_md['prop_names'], g_md['prop_type'])
            } for g_id, g_md in self._model_groups_md.items()
        }

        idx_beg = 0
        group_idx = {g_id: 0 for g_id in self._model_groups_md.keys()}
        for et in self._edge_types_tables:
            idx_end = idx_beg + et.n_edges

            src_trg_ids = et.edge_type_node_ids
            self.source_ids[idx_beg:idx_end] = src_trg_ids[:, 0]
            self.target_ids[idx_beg:idx_end] = src_trg_ids[:, 1]
            self.edge_type_ids[idx_beg:idx_end] = et.edge_type_id
            self.edge_group_ids[idx_beg:idx_end] = et.edge_group_id

            group_idx_beg = group_idx[et.edge_group_id]
            group_idx_end = group_idx_beg + et.n_edges
            self.edge_group_index[idx_beg:idx_end] = np.arange(group_idx_beg, group_idx_end, dtype=np.uint32)
            for pname, pdata in self._prop_data[et.edge_group_id].items():
                pdata[group_idx_beg:group_idx_end] = et.get_property_value(pname)

            idx_beg = idx_end
            group_idx[et.edge_group_id] = group_idx_end

            et.free_data()

    @property
    def group_ids(self):
        return list(self._prop_data.keys())

    def get_group_metadata(self, group_id):
        grp_md = self._model_groups_md[group_id]
        grp_dim = (grp_md['prop_size'], )
        return [{'name': n, 'type': t, 'dim': grp_dim} for n, t in zip(grp_md['prop_names'], grp_md['prop_type'])]

    def assign_groups(self):
        for et in self._edge_types_tables:
            # Assign each edge type a group_id based on the edge-type properties. When two edge-types tables use the
            # same model properties (in the hdf5) they should be put into the same group
            edge_types_hash = et.hash_key
            if edge_types_hash not in self._group_ids_lu:
                self._group_ids_lu[edge_types_hash] = self._grp_id_itr

                group_metadata = et.get_property_metatadata()
                self._model_groups_md[self._grp_id_itr] = {
                    'prop_names': [p['name'] for p in group_metadata],
                    'prop_type': [p['dtype'] for p in group_metadata],
                    'prop_size': 0
                }
                self._grp_id_itr += 1

            group_id = self._group_ids_lu[edge_types_hash]
            et.edge_group_id = group_id

            # number of rows in each model group
            self._model_groups_md[group_id]['prop_size'] += et.n_edges

    def itr_chunks(self):
        chunk_id = 0
        idx_beg = 0
        idx_end = self.n_total_edges

        yield chunk_id, idx_beg, idx_end

    def get_target_node_ids(self, chunk_id):
        return self.target_ids

    def get_source_node_ids(self, chunk_id):
        return self.source_ids

    def get_edge_type_ids(self, chunk_id):
        return self.edge_type_ids

    def get_edge_group_ids(self, chunk_id):
        return self.edge_group_ids

    def get_edge_group_indices(self, chunk_id):
        return self.edge_group_index

    def get_group_data(self, chunk_id):
        ret_val = []
        for group_id in self._prop_data.keys():
            for group_name in self._prop_data[group_id].keys():

                idx_end = len(self._prop_data[group_id][group_name])
                ret_val.append((group_id, group_name, 0, idx_end))

        return ret_val

    def get_group_property(self, group_name, group_id, chunk_id):
        return self._prop_data[group_id][group_name]

    def sort(self, sort_by, sort_group_properties=True):
        """In memory sort of the dataset

        :param sort_by:
        :param sort_group_properties:
        """
        # Find the edges hdf5 column to sort by
        if sort_by == 'target_node_id':
            sort_ds = self.target_ids
        elif sort_by == 'source_node_id':
            sort_ds = self.source_ids
        elif sort_by == 'edge_type_id':
            sort_ds = self.edge_type_ids
        elif sort_by == 'edge_group_id':
            sort_ds = self.edge_group_ids
        else:
            logger.warning('Unable to sort dataset, unrecognized column {}.'.format(sort_by))
            return

        # check if dataset is already sorted
        if np.all(np.diff(sort_ds) <= 0):
            return

        # Find order of arguments of sorted arrays, and sort main columns
        sort_idx = np.argsort(sort_ds)
        self.source_ids = self.source_ids[sort_idx]
        self.target_ids = self.target_ids[sort_idx]
        self.edge_type_ids = self.edge_type_ids[sort_idx]
        self.edge_group_ids = self.edge_group_ids[sort_idx]
        self.edge_group_index = self.edge_group_index[sort_idx]

        if sort_group_properties:
            # For sorting group properties, so the "edge_group_index" column is sorted (wrt each edge_group_id). Note
            # that it is not strictly necessary to sort the group properties, as sonata edge_group_index keeps the
            # reference, but doing the sorting may improve memory/efficency during setting up a simulation

            for grp_id, grp_props in self._prop_data.items():
                # Filter out edge_group_index array for each group_id, get the new order and apply to each property.
                grp_id_filter = np.argwhere(self.edge_group_ids == grp_id).flatten()
                updated_order = self.edge_group_index[grp_id_filter]
                for prop_name, prop_data in grp_props.items():
                    grp_props[prop_name] = prop_data[updated_order]

                # reorder the edge_group_index (for values with corresponding group_id)
                self.edge_group_index[grp_id_filter] = np.arange(0, len(grp_id_filter), dtype=np.uint32)


class EdgesCollatorMPI(object):
    """For collecting all the different edge-types tables to make writing edges and iterating over the entire network
    easier. Similar to above but for when edge-rules data are split across multiple MPI ranks/processors. Can also
    be utlized for single core building when the network is too big to store in memory at once.

    TODO: Consider saving tables to memory on each rank, and using MPI Gather/Send.
    """
    def __init__(self, edge_types_tables, network_name):
        self._edge_types_tables = edge_types_tables
        self._network_name = network_name

        self.n_total_edges = 0  # total number of edges across all ranks
        self.n_local_edges = 0  # total number of edges for just those edge-types saved on current rank
        self._edges_by_rank = {r: 0 for r in range(mpi_size)}  # number of edges per .edge_types_table*h5 file
        self._rank_offsets = [0]

        self._model_groups_md = {}  # keep track of all the edge-types metdata/properties across all ranks
        self._group_ids = set()  # keep track of all group_ids
        self._group_offsets = {}
        self._proc_fhandles = {}  # reference to open readable hdf5 handles.

        self.can_sort = False

    def process(self):
        barrier()

        # Iterate through all the temp hdf5 edge-type files on the disk and gather information about all the
        # different edge-types and their properties.
        # NOTE: Assumes that each .edge_type_table*h5 file contains unique edge_type_ids
        for rank in range(mpi_size):
            rank_edge_table_path = EdgeTypesTableMPI.get_tmp_table_path(rank=rank, name=self._network_name)
            if os.path.exists(rank_edge_table_path):  # possible .tmp file doesn't exist
                with h5py.File(rank_edge_table_path, 'r') as edge_table_h5:
                    for et_id, et_grp in edge_table_h5['unprocessed'][self._network_name].items():
                        et_size = et_grp.attrs['size']
                        self.n_total_edges += et_size   # et_grp.attrs['size']
                        self.n_local_edges += et_size if rank == mpi_rank else 0
                        edge_type_id = int(et_id)

                        self._edges_by_rank[rank] += et_size
                        self._rank_offsets.append(self._rank_offsets[-1] + et_size)

                        # Save metadata about the edge-type-id
                        group_hash_key = et_grp.attrs['hash_key']
                        self._model_groups_md[edge_type_id] = {
                            'hash_key': group_hash_key,
                            'size': et_size,
                            'rank': rank,
                            'properties': []
                        }
                        et_props = [(n, d) for n, d in et_grp.items() if n not in ['source_node_id', 'target_node_id']]
                        for pname, pdata in et_props:
                            self._model_groups_md[edge_type_id]['properties'].append({
                                'name': pname,
                                'type': pdata.dtype
                            })

        # Assign the group_ids for each edge-type. If two or more edge-types contain the same property name+types they
        # should be assigned to the same group_id. Must also take care to unify group_id's across multiple
        # .edge_type_table*h5 files
        group_hashes = set([mg['hash_key'] for mg in self._model_groups_md.values()])
        ordred_group_hashes = list(group_hashes)
        ordred_group_hashes.sort()  # should ensure order will be uniform across all MPI ranks, maybe use AllGather?
        grp_id_map = {}  # keep track of what hash keys match what group_ids
        grp_id_itr = 0
        for edge_type_id, mg in self._model_groups_md.items():
            hash_key = mg['hash_key']
            if hash_key not in grp_id_map:
                # assign hash to the next group_id
                grp_id_map[hash_key] = grp_id_itr
                self._group_ids.add(grp_id_itr)
                grp_id_itr += 1

            mg['edge_group_id'] = grp_id_map[hash_key]

        # For model-group, figure out where the offset for that group occurs on each rank. This is so we can align
        # edge_group_index across multiple mpi ranks below.
        group_rank_sizes = {group_id: np.zeros(mpi_size+1, dtype=np.uint32) for group_id in self._group_ids}
        for edge_type_id, mg in self._model_groups_md.items():
            rank = mg['rank']
            group_id = mg['edge_group_id']
            group_rank_sizes[group_id][rank+1] += mg['size']
        group_rank_offsets = {}
        for group_id, group_sizes in group_rank_sizes.items():
            group_rank_offsets[group_id] = np.cumsum(group_sizes)

        # collect info on local edge-group-properties to simplify things when building /processed data on rank
        local_group_props = {}
        et_count = 0
        for edge_type_id, mg_grp in self._model_groups_md.items():
            if mg_grp['rank'] != mpi_rank:
                continue

            group_id = mg_grp['edge_group_id']

            if group_id not in local_group_props:
                local_group_props[group_id] = {
                    'size': 0,
                    'pnames': [p['name'] for p in mg_grp['properties']],
                    'ptypes': [p['type'] for p in mg_grp['properties']]
                }
            local_group_props[group_id]['size'] += mg_grp['size']
            et_count += 1

        # Try to take all the edge-types-tables on the current rank and combine them into one "processed" table (and
        # multiple model groups). There will be a penalty for doing more disk writing, and doing this part is not
        # strictly necessary. But for very large and complicated network it will make the process more parallizable
        # (since the writing is only done on one rank).
        unprocessed_h5_path = EdgeTypesTableMPI.get_tmp_table_path(rank=mpi_rank, name=self._network_name)

        if os.path.exists(unprocessed_h5_path):
            unprocessed_h5 = h5py.File(unprocessed_h5_path.format(mpi_rank), 'r')
        else:
            # It is possible a .edge_types_table.<rank>.h5 file doesn't exist because there are no edges on that rank.
            # As such hack together a fake hdf5 format with /unprocessed/network_name/ path but no data.
            unprocessed_h5 = {
                'unprocessed': {self._network_name: {}}
            }

        with h5py.File('.edge_types_table.processed.{}.h5'.format(mpi_rank), 'w') as local_h5:
            # WARNING: Trying to process the data back into the .edge_types_table*h5 being read from will randomly cause
            #          resource unavailble errors.
            local_grp_sizes = {}  # count the size of each property model group for all edge-types on this rank
            for edge_id, edge_grp in unprocessed_h5['unprocessed'][self._network_name].items():
                group_id = self._model_groups_md[int(edge_id)]['edge_group_id']
                if group_id not in local_grp_sizes:
                    local_grp_sizes[group_id] = 0
                local_grp_sizes[group_id] += edge_grp.attrs['size']

            if 'processed' in local_h5:
                del local_h5['edges']

            # initialize data-sets
            proc_grp = local_h5.create_group('processed')
            proc_grp.create_dataset('source_node_id', shape=(self.n_local_edges,), dtype=np.uint)
            proc_grp.create_dataset('target_node_id', shape=(self.n_local_edges,), dtype=np.uint)
            proc_grp.create_dataset('edge_type_id', shape=(self.n_local_edges,), dtype=np.uint32)
            proc_grp.create_dataset('edge_group_id', shape=(self.n_local_edges,), dtype=np.uint32)
            proc_grp.create_dataset('edge_group_index', shape=(self.n_local_edges,), dtype=np.uint32)
            for group_id, grp_props in local_group_props.items():
                grp = proc_grp.create_group(str(group_id))
                for pname, ptype in zip(grp_props['pnames'], grp_props['ptypes']):
                    grp.create_dataset(pname, shape=(grp_props['size'],), dtype=ptype)

            # iterate through all edge-types, write src/trg ids, edge-type-ids, and model properties to the correct
            # place in the "processed" table.
            idx_beg = 0
            grp_indices = {group_id: 0 for group_id in local_group_props.keys()}
            for edge_type_id_str, edge_type_grp in unprocessed_h5['unprocessed'][self._network_name].items():
                edge_type_id = int(edge_type_id_str)
                idx_end = idx_beg + edge_type_grp.attrs['size']
                proc_grp['source_node_id'][idx_beg:idx_end] = edge_type_grp['source_node_id']
                proc_grp['target_node_id'][idx_beg:idx_end] = edge_type_grp['target_node_id']
                proc_grp['edge_type_id'][idx_beg:idx_end] = edge_type_id

                group_props = self._model_groups_md[edge_type_id]
                group_id = group_props['edge_group_id']
                grp_idx_beg = grp_indices[group_id]
                grp_idx_end = grp_idx_beg + edge_type_grp.attrs['size']

                for pname in local_group_props[group_id]['pnames']:
                    proc_grp[str(group_id)][pname][grp_idx_beg:grp_idx_end] = edge_type_grp[pname]

                proc_grp['edge_group_id'][idx_beg:idx_end] = group_id

                group_index_offset = group_rank_offsets[group_id][mpi_rank]
                proc_grp['edge_group_index'][idx_beg:idx_end] = \
                    np.arange(grp_idx_beg, grp_idx_end, dtype=np.uint) + group_index_offset

                idx_beg = idx_end
                grp_indices[group_id] = grp_idx_end

        barrier()

    @property
    def group_ids(self):
        return list(self._group_ids)

    def sort(self, sort_by, sort_group_properties=True):
        logger.warning('Unable to sort edges.')

    def get_group_metadata(self, group_id):
        """for a given group_id return all the property dataset metadata; {name, type, size}, across all ranks."""
        ret_props = None
        prop_size = 0
        # There has to be a better way of doing this
        for _, edge_type_md in self._model_groups_md.items():
            if edge_type_md['edge_group_id'] != group_id:
                continue

            prop_size += edge_type_md['size']
            if ret_props is None:
                ret_props = edge_type_md['properties']

        if prop_size == 0:
            return []
        else:
            for p in ret_props:
                p['dim'] = (prop_size,)
            return ret_props

    def itr_chunks(self):
        idx_beg = 0
        self._group_offsets = {grp_id: 0 for grp_id in self.group_ids}

        for rank_id in range(mpi_size):
            idx_end = idx_beg + self._edges_by_rank[rank_id]
            yield rank_id, idx_beg, idx_end
            idx_beg = idx_end

    def _get_processed_h5(self, rank):
        h5_path = '.edge_types_table.processed.{}.h5'.format(rank)
        if h5_path in self._proc_fhandles:
            return self._proc_fhandles[h5_path]
        else:
            h5_handle = h5py.File(h5_path, 'r')
            self._proc_fhandles[h5_path] = h5_handle
            return h5_handle

    def get_source_node_ids(self, chunk_id):
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        return rank_h5['/processed/source_node_id'][()]

    def get_target_node_ids(self, chunk_id):
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        return rank_h5['/processed/target_node_id'][()]

    def get_edge_type_ids(self, chunk_id):
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        return rank_h5['/processed/edge_type_id'][()]

    def get_edge_group_ids(self, chunk_id):
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        return rank_h5['/processed/edge_group_id'][()]

    def get_edge_group_indices(self, chunk_id):
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        return rank_h5['/processed/edge_group_index'][()]

    def get_group_data(self, chunk_id):
        ret_data = []
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        processed = rank_h5['/processed']
        ranked_grp_ids = [grp_id_str for grp_id_str, h5_obj in processed.items() if isinstance(h5_obj, h5py.Group)]
        for grp_id in ranked_grp_ids:
            grp_size = 0
            grp_idx_beg = self._group_offsets[int(grp_id)]
            for pname, pdata in processed[grp_id].items():
                grp_size = len(pdata)
                ret_data.append((grp_id, pname, grp_idx_beg, grp_idx_beg + grp_size))
            self._group_offsets[int(grp_id)] += grp_size

        return ret_data

    def get_group_property(self, prop_name, group_id, chunk_id):
        rank_h5 = self._get_processed_h5(rank=chunk_id)
        return rank_h5['processed'][str(group_id)][prop_name][()]

    def __del__(self):
        # clean up .h5 file that is saved to disk
        tmp_h5_path = '.edge_types_table.processed.{}.h5'.format(mpi_rank)
        try:
            if os.path.exists(tmp_h5_path):
                os.remove(tmp_h5_path)
        except (FileNotFoundError, IOError, Exception) as e:
            logger.warning('Unable to delete temp edges file {}.'.format(tmp_h5_path))


class EdgesCollator(object):
    def __new__(cls, *args, **kwargs):
        if mpi_size > 1:
            return EdgesCollatorMPI(*args, **kwargs)
        else:
            return EdgesCollatorSingular(*args, **kwargs)
