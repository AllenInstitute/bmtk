import os
import h5py
import pandas as pd
import numpy as np
import json
import shutil
import logging
import glob

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


logger = logging.getLogger(__name__)


class ProgressFile(object):
    """Class for keeping track of the progress of the sorting of the hdf5 edges files. Will write progress to disk, so
    that if sorting fails we can still continue from where it left off."""

    def __init__(self, cache_dir, n_edges, n_chunks, sort_key, root_name=None):
        self._conf_parser = {}
        self.root_name = root_name
        self.cache_dir = cache_dir

        self.progress_file = os.path.join(self.cache_dir, 'progress.json')
        if os.path.exists(self.progress_file):
            self._conf_parser = json.load(open(self.progress_file, 'r'))
            self.initialized = self._conf_parser[self.root_name]['initialized']
            self.n_edges = self._conf_parser[self.root_name]['n_edges']
            self.n_chunks = self._conf_parser[self.root_name]['n_chunks']
            self.iteration = self._conf_parser[self.root_name]['iteration']
            self.sort_key = self._conf_parser[self.root_name]['sort_key']
            self.chunk_files = self._conf_parser[self.root_name]['chunk_files']
            self.chunk_indices = self._conf_parser[self.root_name]['chunk_indices']
            self.write_index = self._conf_parser[self.root_name]['write_index']
            self.completed = self._conf_parser[self.root_name].get('completed', False)

            # Make sure original edges file or the chunking size hasn't changed since the progress was last saved, if
            # so it can cause unforseen issues.
            if self.completed:
                raise ValueError('Already marked as complete.')

            if self.n_edges != n_edges:
                raise ValueError('n_edges={} has changed to {}. Unable to continue sorting from last saved point.'.format(self.n_edges, n_edges))

            if self.n_chunks != n_chunks:
                raise ValueError('n_chunks={} has changed to {}. Unable to continue sorting from last saved point.'.format(self.n_chunks, n_chunks))

            if self.sort_key != sort_key:
                raise ValueError('sort_key={} has changed to {}. Unable to continue from last saved point.'.format(
                    self.sort_key, sort_key
                ))

        else:
            self._conf_parser[self.root_name] = {}
            self.initialized = False
            self.n_edges = n_edges
            self.n_chunks = n_chunks
            self.iteration = 0
            self.sort_key = sort_key
            self.chunk_files = []
            self.chunk_indices = []
            self.write_index = 0
            self.completed = False
            self.update()

    def _can_serialize(self):
        json.dumps(self._conf_parser)
        return True

    def update(self, initialized=None, n_edges=None, n_chunks=None, iteration=None, chunk_files=None,
               chunk_indices=None, write_index=None, completed=None):
        if initialized is not None:
            self.initialized = initialized
        if n_edges is not None:
            self.n_edges = n_edges
        if n_chunks is not None:
            self.n_chunks = n_chunks
        if iteration is not None:
            self.iteration = iteration
        if chunk_files is not None:
            self.chunk_files = chunk_files
        if chunk_indices is not None:
            self.chunk_indices = chunk_indices
        if write_index is not None:
            self.write_index = int(write_index)
        if completed is not None:
            self.completed = completed

        self._conf_parser[self.root_name]['initialized'] = self.initialized
        self._conf_parser[self.root_name]['n_edges'] = self.n_edges
        self._conf_parser[self.root_name]['n_chunks'] = self.n_chunks
        self._conf_parser[self.root_name]['iteration'] = self.iteration
        self._conf_parser[self.root_name]['sort_key'] = self.sort_key
        self._conf_parser[self.root_name]['chunk_files'] = self.chunk_files
        self._conf_parser[self.root_name]['chunk_indices'] = self.chunk_indices
        self._conf_parser[self.root_name]['write_index'] = self.write_index
        self._conf_parser[self.root_name]['completed'] = self.completed

        if self._can_serialize():
            json.dump(self._conf_parser, open(self.progress_file, 'w'), indent=4)
        else:
            raise ValueError('Unable to serialize to json')


def _get_chunk_fn(cache_dir, itr_num, chunk_num):
    return os.path.join(cache_dir, 'itr{}_chunk{}.h5'.format(itr_num, chunk_num))


def _copy_attributes(in_grp, out_grp):
    """Recursively copy hdf5 Group/Dataset attributes from in_grp to out_grp

    :param in_grp: hdf5 Group object whose attributes will be copied from.
    :param out_grp: hdf5 Group object that will have it's attributes updated/copied to.
    """
    if in_grp.attrs:
        out_grp.attrs.update(in_grp.attrs)

    for in_name, in_h5_obj in in_grp.items():
        if in_name not in out_grp:
            # make sure matching subgroup/dataset exists in the output group
            continue

        elif isinstance(in_h5_obj, h5py.Dataset):
            out_grp[in_name].attrs.update(in_h5_obj.attrs)

        elif isinstance(in_h5_obj, h5py.Group):
            _copy_attributes(in_h5_obj, out_grp[in_name])


def _create_output_h5(input_file, output_file, edges_root, n_edges):
    mode = 'r+' if os.path.exists(output_file) else 'w'
    out_h5 = h5py.File(output_file, mode=mode)
    root_grp = out_h5.create_group(edges_root) if edges_root not in out_h5 else out_h5[edges_root]

    if 'source_node_id' not in root_grp:
        root_grp.create_dataset('source_node_id', (n_edges, ), dtype=np.uint32)

    if 'target_node_id' not in root_grp:
        root_grp.create_dataset('target_node_id', (n_edges, ), dtype=np.uint32)

    if 'edge_type_id' not in root_grp:
        root_grp.create_dataset('edge_type_id', (n_edges, ), dtype=np.uint32)

    if 'edge_group_id' not in root_grp:
        root_grp.create_dataset('edge_group_id', (n_edges, ), dtype=np.uint32)

    if 'edge_group_index' not in root_grp:
        root_grp.create_dataset('edge_group_index', (n_edges, ), dtype=np.uint32)

    # with h5py.File(input_file, 'r') as in_h5:
    #     for h5obj in in_h5[edges_root].values():
    #         if isinstance(h5obj, h5py.Group) and h5obj.name not in root_grp:
    #             root_grp.copy(h5obj, h5obj.name)

    _copy_attributes(h5py.File(input_file, 'r'), out_h5)

    return root_grp


def _order_model_groups(input_edges_path, output_edges_path, edges_population, chunk_size):
    """

    :param input_edges_path:
    :param output_edges_path:
    :param edges_population:
    :param chunk_size:
    :return:
    """
    # The output hdf5 should already have edges_group_id, edges_group_index columns which we can use to find order. This
    # hdf5 file handle should only be for reading

    # Get the model parameters from the input (original) edges file.
    in_edges_h5 = h5py.File(input_edges_path, 'r')
    in_edges_grp = in_edges_h5[edges_population]

    with h5py.File(output_edges_path, 'r+') as out_h5:
        out_root_grp = out_h5[edges_population]

        # Get a list of all model group columns that need to be resorted, create the dataset in the output hdf5 edges
        # file if it doesn't exist
        model_grp_tracker = {}
        for h5name, h5obj in in_edges_grp.items():
            if isinstance(h5obj, h5py.Group) and h5obj.name not in ['indices', 'indicies']:
                model_grp_tracker[h5name] = {'c_indx': 0, 'cols': []}
                for col_name, col_ds in h5obj.items():
                    model_grp_tracker[h5name]['cols'].append(col_name)
                    if col_ds.name not in out_h5:
                        out_h5.create_dataset(col_ds.name, shape=col_ds.shape, dtype=col_ds.dtype)

        # use edge_group_id and edge_group_index to sort the model group columns
        edge_group_ids_ds = out_root_grp['edge_group_id']
        edge_group_indices_ds = out_root_grp['edge_group_index']
        n_edges = len(edge_group_indices_ds)
        if 'edge_group_index_sorted' not in out_root_grp:
            # we will need to also need to keep track of the reordering and replace edge_group_index
            out_root_grp.create_dataset(
                'edge_group_index_sorted',
                shape=edge_group_indices_ds.shape,
                dtype=edge_group_indices_ds.dtype
            )

        chunk_idx_beg = 0
        while chunk_idx_beg < n_edges:
            # assuming we can't load and sort the entire edge_group_id/edge_group_index tables into memory, we need to
            # reorder into handable chunks

            chunk_idx_end = np.min((chunk_idx_beg + chunk_size, n_edges)).astype(np.uint)
            if chunk_idx_beg == chunk_idx_end:
                continue

            chunk_grp_ids = edge_group_ids_ds[chunk_idx_beg:chunk_idx_end][()]
            chunk_grp_idxs = edge_group_indices_ds[chunk_idx_beg:chunk_idx_end][()]
            updated_order = np.zeros(int(chunk_idx_end-chunk_idx_beg), dtype=np.uint32)

            # Go through all the groups and all their columns and reorder by group_id only
            for group_id in np.unique(chunk_grp_ids):
                # the order they appear in group_index is the order the columns will be resorted
                group_id_mask = np.argwhere(chunk_grp_ids == group_id).flatten()
                new_index_order = chunk_grp_idxs[group_id_mask]

                # If new_index_order is not sorted h5py will complain when we try to fetch those values in the index.
                # Use nio_incremental to fetch from the datasets and nio_argwhere to put back into the right order
                nio_incremental = np.sort(new_index_order)
                nio_argswhere = np.argsort(new_index_order)

                # for each column take values from in_edges and write them to out_edges in the correct order
                group_indx_beg = model_grp_tracker[str(group_id)]['c_indx']
                group_indx_end = group_indx_beg + len(new_index_order)
                for col in model_grp_tracker[str(group_id)]['cols']:
                    col_vals_tmp = in_edges_grp[str(group_id)][col][nio_incremental]
                    col_vals = col_vals_tmp[nio_argswhere]
                    out_root_grp[str(group_id)][col][group_indx_beg:group_indx_end] = col_vals

                model_grp_tracker[str(group_id)]['c_indx'] = group_indx_end
                updated_order[group_id_mask] = np.arange(group_indx_beg, group_indx_beg + len(new_index_order))

            out_root_grp['edge_group_index_sorted'][chunk_idx_beg:chunk_idx_end] = updated_order
            chunk_idx_beg = chunk_idx_end

        del out_root_grp['edge_group_index']
        out_root_grp['edge_group_index'] = out_root_grp['edge_group_index_sorted']
        del out_root_grp['edge_group_index_sorted']

        _copy_attributes(h5py.File(input_edges_path, 'r'), out_h5)


def _clean(progress):
    for c_fn in glob.glob(os.path.join(progress.cache_dir, 'itr*_chunk*.h5')):
        try:
            os.remove(c_fn)
        except Exception as e:
            pass

    try:
        os.remove(progress.progress_file)
    except Exception as e:
        pass

    try:
        shutil.rmtree(progress.cache_dir)
    except Exception as e:
        pass


def _initialize_chunks(edges_file_path, progress):
    """ Splits the original hdf5 edges file into n_chunks individual files"""
    edges_h5 = h5py.File(edges_file_path, 'r')
    edges_root_grp = edges_h5[progress.root_name]

    sort_key = progress.sort_key
    n_edges = progress.n_edges
    n_chunks = progress.n_chunks
    cache_dir = progress.cache_dir

    logger.debug('Splitting {} into {} sorted chunks'.format(edges_file_path, n_chunks))
    chunk_size = np.ceil(n_edges / n_chunks).astype(np.uint)

    # Split the edges table into N chunks, each chunk will be sorted (by the specified column) and the sorted table
    # will be saved it's own tmp hdf5 file.
    # Note: we don't have to keep track of subgroups since the edge_group_id w/
    # edge_group_index column will still hold the valid line references
    chunk_idx_beg = 0
    chunk_idx_end = np.min((chunk_size, n_edges)).astype(np.uint)
    iteration = 0
    chunk_num = 0
    chunk_files = []
    chunk_indices = []
    while chunk_idx_beg < n_edges:
        logger.debug('  Chunk {} of {} [{:,} - {:,})'.format(chunk_num+1, n_chunks, chunk_idx_beg, chunk_idx_end))
        # TODO: should be more efficencent if we just sort the specified column and keep track of the sort_order
        #   index for later.
        chunk_data_df = pd.DataFrame({
            'source_node_id': edges_root_grp['source_node_id'][chunk_idx_beg:chunk_idx_end],
            'target_node_id': edges_root_grp['target_node_id'][chunk_idx_beg:chunk_idx_end],
            'edge_type_id': edges_root_grp['edge_type_id'][chunk_idx_beg:chunk_idx_end],
            'edge_group_id': edges_root_grp['edge_group_id'][chunk_idx_beg:chunk_idx_end],
            'edge_group_index': edges_root_grp['edge_group_index'][chunk_idx_beg:chunk_idx_end]
        }).sort_values(sort_key)

        chunk_file = os.path.join(cache_dir, 'itr{}_chunk{}.h5'.format(iteration, chunk_num))
        chunk_h5 = h5py.File(chunk_file, 'w')
        chunk_h5.create_dataset('source_node_id', data=chunk_data_df['source_node_id'])
        chunk_h5.create_dataset('target_node_id', data=chunk_data_df['target_node_id'])
        chunk_h5.create_dataset('edge_type_id', data=chunk_data_df['edge_type_id'])
        chunk_h5.create_dataset('edge_group_id', data=chunk_data_df['edge_group_id'])
        chunk_h5.create_dataset('edge_group_index', data=chunk_data_df['edge_group_index'])

        # store the min/max sort-column value found in current chunk
        chunk_h5.attrs['min_id'] = chunk_data_df[sort_key].min()
        max_id = chunk_data_df[sort_key].max()
        chunk_h5.attrs['max_id'] = max_id

        # For each unique id that is being sorted on, keep a list of how many times the id shows up in the current
        # tmp chunk file
        id_counts = chunk_data_df[sort_key].value_counts().sort_index()
        id_counts = id_counts.reindex(pd.RangeIndex(max_id+1), fill_value=0)
        chunk_h5.create_dataset('id_counts', data=id_counts, dtype=np.uint32)

        chunk_files.append(chunk_file)
        chunk_indices.append(0) # chunk index is 0 b/c none of the data has been merged into the final file

        chunk_idx_beg = chunk_idx_end
        chunk_idx_end += np.min((chunk_size, n_edges)).astype(np.uint)
        chunk_num += 1

    progress.update(initialized=True, chunk_files=chunk_files, chunk_indices=chunk_indices)


def _sort_chunks(progress):
    """Takes the M (usually N or N-1) sorted chunk files, splits and merges them into N new chunk files"""
    logger.debug('Resorting Chunks')

    cache_dir = progress.cache_dir
    itr_num = progress.iteration
    n_chunks = progress.n_chunks

    # during the previous merge() process some of the rows of last iteration's sorted chunk file may have been merged
    # into the final result. Use the chunk_indices variable to keep track of what row to use for creating the next
    # sorted chunk files.
    chunk_files = progress.chunk_files
    chunk_h5s = [h5py.File(cf, 'r') for cf in chunk_files]
    chunk_start_indices = progress.chunk_indices
    chunk_sizes = [h5['source_node_id'].shape[0] for h5 in chunk_h5s]
    chunk_indices = {fn: np.linspace(c_start, c_size, num=n_chunks+1, endpoint=True, dtype=np.uint)
                     for fn, c_start, c_size in zip(chunk_files, chunk_start_indices, chunk_sizes)}

    for l in chunk_indices.values():
        assert(len(l) == n_chunks+1)

    new_chunk_files = []
    new_chunk_indices = np.zeros(n_chunks, dtype=np.uint).tolist()

    # loop through all the sorted chunk files created during the last iteration and divide them up into further chunks.
    # Take and merge together the corresponding chunks of data into a new table, sort and save for the next round of
    # merging,
    for chunk_num in range(n_chunks):
        logger.debug('  Chunk {} of {}'.format(chunk_num+1, n_chunks))
        combined_df = None
        for i, chunk_fn in enumerate(chunk_files):
            chunk_h5 = chunk_h5s[i]
            chunk_idx_beg = chunk_indices[chunk_fn][chunk_num]
            chunk_idx_end = chunk_indices[chunk_fn][chunk_num+1]
            if chunk_idx_end == chunk_idx_beg:
                continue

            data_df = pd.DataFrame({
                'source_node_id': chunk_h5['source_node_id'][chunk_idx_beg:chunk_idx_end],
                'target_node_id': chunk_h5['target_node_id'][chunk_idx_beg:chunk_idx_end],
                'edge_type_id': chunk_h5['edge_type_id'][chunk_idx_beg:chunk_idx_end],
                'edge_group_id': chunk_h5['edge_group_id'][chunk_idx_beg:chunk_idx_end],
                'edge_group_index': chunk_h5['edge_group_index'][chunk_idx_beg:chunk_idx_end]
            })
            combined_df = data_df if combined_df is None else pd.concat((combined_df, data_df))

        if combined_df is None:
            continue

        combined_df = combined_df.sort_values(progress.sort_key)

        new_chunk_file = os.path.join(cache_dir, 'itr{}_chunk{}.h5'.format(itr_num+1, chunk_num))
        new_chunk_h5 = h5py.File(new_chunk_file, 'w')
        new_chunk_h5.create_dataset('source_node_id', data=combined_df['source_node_id'])
        new_chunk_h5.create_dataset('target_node_id', data=combined_df['target_node_id'])
        new_chunk_h5.create_dataset('edge_type_id', data=combined_df['edge_type_id'])
        new_chunk_h5.create_dataset('edge_group_id', data=combined_df['edge_group_id'])
        new_chunk_h5.create_dataset('edge_group_index', data=combined_df['edge_group_index'])
        new_chunk_h5.attrs['min_id'] = combined_df[progress.sort_key].min()

        max_id = combined_df[progress.sort_key].max()
        new_chunk_h5.attrs['max_id'] = max_id

        # For each unique id that is being sorted on, keep a list of how many times the id shows up in the current
        # tmp chunk file
        id_counts = combined_df['edge_type_id'].value_counts().sort_index()
        id_counts = id_counts.reindex(pd.RangeIndex(max_id+1), fill_value=0)
        new_chunk_h5.create_dataset('id_counts', data=id_counts, dtype=np.uint32)

        new_chunk_files.append(new_chunk_file)

    progress.update(iteration=itr_num+1, chunk_files=new_chunk_files, chunk_indices=new_chunk_indices)

    logger.debug('Deleteing old chunks')
    for c_fn in chunk_files:
        try:
            os.remove(c_fn)
        except Exception as e:
            logger.warning('Could not delete file {}'.format(c_fn))
            print(e)


def _merge(progress, edges_grp):
    """Will merge data from  one or more sorted chunk files into the output file"""

    # Create a table of min and max ids in each chunk file and sort. Using this table we can find overlaps, and when
    # the max/max ids of one file doesn't overlap with another then it's safe to push that table to the output h5
    max_size = np.ceil(progress.n_edges / progress.n_chunks).astype(np.uint)
    chunk_files = progress.chunk_files.copy()
    chunk_indices = progress.chunk_indices.copy()
    chunk_nums = []
    chunk_h5s = []
    chunk_ids_min = []
    chunk_ids_max = []
    min_id = np.inf
    max_id = 0
    for chunk_num, chunk_file in enumerate(chunk_files):
        h5 = h5py.File(chunk_file, 'r+')
        chunk_nums.append(chunk_num)
        chunk_h5s.append(h5)
        chunk_ids_min.append(h5.attrs['min_id'])
        chunk_ids_max.append(h5.attrs['max_id'])
        max_id = int(np.max((max_id, h5.attrs['max_id'])))
        min_id = int(np.min((min_id, h5.attrs['min_id'])))

    logger.debug('Current status:')
    logger.debug(pd.DataFrame({
        'chunk file': [os.path.basename(c) for c in chunk_files],
        'min_id': chunk_ids_min,
        'max_id': chunk_ids_max,
    }))

    offsets = np.zeros(max_id+1, dtype=np.uint32)
    for h5 in chunk_h5s:
        id_counts = np.array(h5['id_counts'])
        offsets[:len(id_counts)] += id_counts

    offsets = np.cumsum(offsets)
    id_end = np.max((np.searchsorted(offsets, max_size, side='right'), min_id))

    collected_df = None
    for chunk_num, chunk_h5 in zip(chunk_nums, chunk_h5s):
        read_beg_idx = chunk_indices[chunk_num]
        sort_col = np.array(chunk_h5[progress.sort_key][read_beg_idx:])
        id_indices = np.argwhere(sort_col <= id_end).flatten()
        if len(id_indices) == 0:
            continue

        read_end_idx = id_indices[-1] + 1
        chunk_size = chunk_h5[progress.sort_key].shape[0]

        data_df = pd.DataFrame({
            'source_node_id': chunk_h5['source_node_id'][read_beg_idx:read_end_idx],
            'target_node_id': chunk_h5['target_node_id'][read_beg_idx:read_end_idx],
            'edge_type_id': chunk_h5['edge_type_id'][read_beg_idx:read_end_idx],
            'edge_group_id': chunk_h5['edge_group_id'][read_beg_idx:read_end_idx],
            'edge_group_index': chunk_h5['edge_group_index'][read_beg_idx:read_end_idx]
        })
        collected_df = data_df if collected_df is None else pd.concat((collected_df, data_df))

        if read_end_idx == chunk_size:
            chunk_indices[chunk_num] = -1
            chunk_files[chunk_num] = None

        else:
            chunk_indices[chunk_num] = int(read_end_idx)

    if collected_df is None or len(collected_df) == 0:
        logger.debug('No ids collected.')
        return

    collected_df = collected_df.sort_values(progress.sort_key)
    n_collected = len(collected_df)

    write_beg_idx = progress.write_index
    write_end_idx = write_beg_idx + n_collected

    logger.debug('Writing ids [{:,} - {:,}] to output indices [{:,} - {:,})'.format(
        collected_df[progress.sort_key].iloc[0],
        collected_df[progress.sort_key].iloc[-1],
        write_beg_idx,
        write_end_idx
    ))

    edges_grp['source_node_id'][write_beg_idx:write_end_idx] = collected_df['source_node_id'].values
    edges_grp['target_node_id'][write_beg_idx:write_end_idx] = collected_df['target_node_id'].values
    edges_grp['edge_type_id'][write_beg_idx:write_end_idx] = collected_df['edge_type_id'].values
    edges_grp['edge_group_id'][write_beg_idx:write_end_idx] = collected_df['edge_group_id'].values
    edges_grp['edge_group_index'][write_beg_idx:write_end_idx] = collected_df['edge_group_index'].values

    chunk_files = [c for c in chunk_files if c is not None]
    chunk_indices = [c for c in chunk_indices if c != -1]

    progress.update(chunk_files=chunk_files, chunk_indices=chunk_indices, write_index=write_end_idx,
                    completed=len(chunk_files) == 0)


def external_merge_sort(input_edges_path, output_edges_path, edges_population, sort_by, sort_model_properties=True,
                        n_chunks=12, max_itrs=10, cache_dir='.sort_cache', **kwargs):
    """Does an external merge sort on an input edges hdf5 file, saves value in new file. Usefull for large network
    files where we are not able to load into memory.

    Will split the original hdf5 into <n_chunks> chunks on the disk in cache_dir, Will sort each individual chunk
    of data in memory, then perform a merge on all the chunks. For speed considers may try to do the chunking and
    merging in multiple iterations.

    Itermediate sorting results are saved in cache_dir (eg .sort_cache) and if the sorting fails or doesn't finish in
    max_itrs, running this function again will continue where it last left off.

    :param input_edges_path: path to original edges file
    :param output_edges_path: path name of new file that will be created
    :param edges_population:
    :param sort_by: 'edge_type_id', 'source_node_id', etc.
    :param sort_model_properties: resort the model group so edges_group_id+edge_group_index is in order
    :param n_chunks: Number of chunks, eg the fraction of the edges.h5 file that will be loaded into memory at a given
        time. (default: 12)
    :param cache_dir: A temporary directory where itermeidate results will be stored. (default: './cache_dir/)
    :param max_itrs: The maximum number of iterations to run the merge sort.
    """
    with h5py.File(input_edges_path, 'r') as input_h5:
        n_edges = input_h5[edges_population]['source_node_id'].shape[0]

    cache_dir = os.path.join(os.path.dirname(output_edges_path), cache_dir,
                             os.path.splitext(os.path.basename(output_edges_path))[0])
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    n_chunks = np.min((n_chunks, n_edges))
    progress = ProgressFile(cache_dir=cache_dir, n_edges=n_edges, n_chunks=int(n_chunks), root_name=edges_population,
                            sort_key=sort_by)

    output_root_grp = _create_output_h5(input_file=input_edges_path, output_file=output_edges_path,
                                        edges_root=edges_population, n_edges=n_edges)

    # Split the edges files into N chunks.
    if not progress.initialized or not os.path.exists(cache_dir):
        _initialize_chunks(edges_file_path=input_edges_path, progress=progress)
    else:
        logger.debug('Edges files has already been initialized. Continuing from last saved point')

    # Try to merge chunk data to output, if there's still data left split and resort into another N chunks. Repeat.
    _merge(progress, edges_grp=output_root_grp)
    for i in range(max_itrs):
        if progress.completed:
            break
        _sort_chunks(progress)
        _merge(progress, edges_grp=output_root_grp)

    output_root_grp.file.flush()
    output_root_grp.file.close()

    if progress.completed and sort_model_properties:
        # copy over model group, and reorder so edge_group_ids/edge_group_index is ordered
        logger.debug('Sorting model group columns')
        chunk_size = np.max((np.ceil(n_edges / n_chunks), 2)).astype(np.uint)
        _order_model_groups(
            input_edges_path=input_edges_path,
            output_edges_path=output_edges_path,
            edges_population=progress.root_name,
            chunk_size=chunk_size
        )
    else:
        # Copy over model group columns without sorting
        out_h5 = h5py.File(output_edges_path, 'r+')
        root_grp = out_h5[progress.root_name]

        with h5py.File(input_edges_path, 'r') as in_h5:
            for h5obj in in_h5[progress.root_name].values():
                if isinstance(h5obj, h5py.Group) and h5obj.name not in root_grp and h5obj.name not in ['indices', 'indicies']:
                    root_grp.copy(h5obj, h5obj.name)

    logger.debug('Cleaning up cache directory')
    _clean(progress)

    logger.debug('Done sorting.')


if __name__ == '__main__':
    external_merge_sort(
        input_edges_path='network/right_local_edges.h5',
        output_edges_path='network/right_local_edges_by_edgetype_merge_sort_v1.h5',
        sort_by='edge_type_id',
        edges_population='/edges/right_local'
    )
