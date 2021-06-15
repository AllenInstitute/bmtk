import os
import h5py
import pandas as pd
import numpy as np
import json

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


class ProgressFile(object):
    """Class for keeping track of the progress of the sorting of the hdf5 edges files. Will write progress to disk, so
    that if sorting fails we can still continue from where it left off."""

    def __init__(self, cache_dir, n_edges, n_chunks, sort_key, root_name=None):
        self._conf_parser = {}
        self.root_name = root_name
        self.cache_dir = cache_dir

        self._progress_file = os.path.join(self.cache_dir, 'progress.json')
        if os.path.exists(self._progress_file):
            # self._conf_parser.read(self._progress_file)
            self._conf_parser = json.load(open(self._progress_file, 'r'))
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
            json.dump(self._conf_parser, open(self._progress_file, 'w'), indent=4)
        else:
            raise ValueError('Unable to serialize to json')


def _get_chunk_fn(cache_dir, itr_num, chunk_num):
    return os.path.join(cache_dir, 'itr{}_chunk{}.h5'.format(itr_num, chunk_num))


def _create_output_h5(input_file, output_file, edges_root, n_edges):
    mode = 'r+' if os.path.exists(output_file) else 'w'
    out_h5 = h5py.File(output_file, mode=mode)
    add_hdf5_version(out_h5)
    add_hdf5_magic(out_h5)
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

    with h5py.File(input_file, 'r') as in_h5:
        for h5obj in in_h5[edges_root].values():
            if isinstance(h5obj, h5py.Group):
                root_grp.copy(h5obj, h5obj.name)

    return root_grp


def initialize_chunks(edges_file_path, progress):
    """ Splits the original hdf5 edges file into n_chunks individual files"""
    edges_h5 = h5py.File(edges_file_path, 'r')
    edges_root_grp = edges_h5[progress.root_name]

    sort_key = progress.sort_key
    n_edges = progress.n_edges
    n_chunks = progress.n_chunks
    cache_dir = progress.cache_dir

    print('Splitting {} into {} sorted chunks'.format(edges_file_path, n_chunks))
    chunk_size = np.ceil(n_edges / n_chunks).astype(np.uint)
    chunk_idx_beg = 0
    chunk_idx_end = np.min((chunk_size, n_edges)).astype(np.uint)
    iteration = 0
    chunk_num = 0
    chunk_files = []
    chunk_indices = []
    while chunk_idx_beg < n_edges:
        print('  Chunk {} of {} [{:,} - {:,})'.format(chunk_num+1, n_chunks, chunk_idx_beg, chunk_idx_end))

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
        # Note we don't have to keep track of subgroups since the edge_group_id + edge_group_index column will still
        #  hold the valid line references

        chunk_h5.attrs['min_id'] = chunk_data_df[sort_key].min()
        max_id = chunk_data_df[sort_key].max()
        chunk_h5.attrs['max_id'] = max_id

        id_counts = chunk_data_df[sort_key].value_counts().sort_index()
        id_counts = id_counts.reindex(pd.RangeIndex(max_id+1), fill_value=0)
        chunk_h5.create_dataset('id_counts', data=id_counts, dtype=np.uint32)

        chunk_files.append(chunk_file)
        chunk_indices.append(0)

        chunk_idx_beg = chunk_idx_end
        chunk_idx_end += np.min((chunk_size, n_edges)).astype(np.uint)
        chunk_num += 1

    progress.update(initialized=True, chunk_files=chunk_files, chunk_indices=chunk_indices)


def sort_chunks(progress):
    """Takes the M (usually N or N-1) sorted chunk files, splits and merges them into N new chunk files"""
    print('Resorting Chunks')

    cache_dir = progress.cache_dir
    itr_num = progress.iteration
    n_chunks = progress.n_chunks  # len(alive_chunks)

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
    for chunk_num in range(n_chunks):
        print('  Chunk {} of {}'.format(chunk_num+1, n_chunks))
        combined_df = None
        for i, chunk_fn in enumerate(chunk_files):
            chunk_h5 = chunk_h5s[i]
            chunk_idx_beg = chunk_indices[chunk_fn][chunk_num]
            chunk_idx_end = chunk_indices[chunk_fn][chunk_num+1]
            data_df = pd.DataFrame({
                'source_node_id': chunk_h5['source_node_id'][chunk_idx_beg:chunk_idx_end],
                'target_node_id': chunk_h5['target_node_id'][chunk_idx_beg:chunk_idx_end],
                'edge_type_id': chunk_h5['edge_type_id'][chunk_idx_beg:chunk_idx_end],
                'edge_group_id': chunk_h5['edge_group_id'][chunk_idx_beg:chunk_idx_end],
                'edge_group_index': chunk_h5['edge_group_index'][chunk_idx_beg:chunk_idx_end]
            })
            combined_df = data_df if combined_df is None else pd.concat((combined_df, data_df))

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

        id_counts = combined_df['edge_type_id'].value_counts().sort_index()
        id_counts = id_counts.reindex(pd.RangeIndex(max_id+1), fill_value=0)
        new_chunk_h5.create_dataset('id_counts', data=id_counts, dtype=np.uint32)

        new_chunk_files.append(new_chunk_file)

    progress.update(iteration=itr_num+1, chunk_files=new_chunk_files, chunk_indices=new_chunk_indices)

    print('Deleteing old chunks')
    for c_fn in chunk_files:
        try:
            os.remove(c_fn)
        except Exception as e:
            print('Could not delete file {}'.format(c_fn))
            print(e)


# def _merge_chunk_file(edges_grp, chunk_h5, progress, read_end_idx=None):
#     merge_type = 'Completely' if read_end_idx is None else 'Partially'
#     read_end_idx = read_end_idx if read_end_idx is not None else chunk_h5['source_node_id'].shape[0]
#     write_beg_idx = progress.write_index
#     write_end_idx = write_beg_idx + read_end_idx
#
#     id_beg, id_end = chunk_h5[progress.sort_key][0], chunk_h5[progress.sort_key][read_end_idx-1]
#     print('{} merging file "{}" with ids ({:,} - {:,}) into master at indices [{:,} - {:,})'.format(
#         merge_type,
#         chunk_h5.filename,
#         id_beg,
#         id_end,
#         write_beg_idx,
#         write_end_idx
#     ))
#
#     edges_grp['source_node_id'][write_beg_idx:write_end_idx] = chunk_h5['source_node_id'][:read_end_idx]
#     edges_grp['target_node_id'][write_beg_idx:write_end_idx] = chunk_h5['target_node_id'][:read_end_idx]
#     edges_grp['edge_type_id'][write_beg_idx:write_end_idx] = chunk_h5['edge_type_id'][:read_end_idx]
#
#     return write_end_idx


# def merge(progress, edges_grp):
#     """Will merge data from  one or more sorted chunk files into the output file"""
#     cache_dir = progress.cache_dir
#     itr_num = progress.iteration
#
#     # Create a table of min and max ids in each chunk file and sort. Using this table we can find overlaps, and when
#     # the max/max ids of one file doesn't overlap with another then it's safe to push that table to the output h5
#     chunk_files = progress.chunk_files.copy()
#     chunk_indices = progress.chunk_indices.copy()
#     chunk_nums = []
#     chunks_h5 = []
#     chunks_id_min = []
#     chunks_id_max = []
#     for chunk_num, chunk_file in enumerate(chunk_files):
#         chunk_file = os.path.join(cache_dir, 'itr{}_chunk{}.h5'.format(itr_num, chunk_num))
#         h5 = h5py.File(chunk_file, 'r+')
#         chunk_nums.append(chunk_num)
#         chunks_h5.append(h5)
#         chunks_id_min.append(h5.attrs['min_id'])
#         chunks_id_max.append(h5.attrs['max_id'])
#
#     sorted_order = np.argsort(chunks_id_min)
#     print('Current status:')
#     print(pd.DataFrame({
#         'chunk file': [os.path.basename(c) for c in chunk_files],
#         'min_id': chunks_id_min,
#         'max_id': chunks_id_max
#     }))
#
#     while True:
#         if len(sorted_order) == 0:
#             progress.update(completed=True)
#             return
#
#         if len(sorted_order) == 1:
#             # Case: only one sorted chunk file so push it into the output h5
#             h5 = chunks_h5[lead_chunk]
#             # read_end_idx = h5['source_node_id'].shape[0]
#             write_index = _merge_chunk_file(edges_grp, h5, progress)
#
#             # write_beg_idx = progress.write_index
#             # write_end_idx = write_beg_idx + read_end_idx
#             #
#             # print('  Merging file {} (ids {:,} - {:,}) into master and indices [{:,} - {:,}'.format(
#             #     chunk_files[lead_chunk],
#             #     chunks_id_min[lead_chunk],
#             #     chunks_id_max[lead_chunk],
#             #     write_beg_idx,
#             #     write_end_idx
#             # ))
#             #
#             # edges_grp['source_node_id'][write_beg_idx:write_end_idx] = h5['source_node_id'][:read_end_idx]
#             # edges_grp['target_node_id'][write_beg_idx:write_end_idx] = h5['target_node_id'][:read_end_idx]
#             # edges_grp['edge_type_id'][write_beg_idx:write_end_idx] = h5['edge_type_id'][:read_end_idx]
#
#             del chunk_files[lead_chunk]
#             del chunk_indices[lead_chunk]
#             del chunk_nums[lead_chunk]
#             del chunks_h5[lead_chunk]
#             del chunks_id_min[lead_chunk]
#             del chunks_id_max[lead_chunk]
#             progress.update(write_index=write_index, chunk_files=chunk_files, chunk_indices=chunk_indices)
#
#             sorted_order = np.argsort(chunks_id_min)
#
#         lead_chunk = sorted_order[0]
#         second_place = sorted_order[1]
#         if chunks_id_max[lead_chunk] <= chunks_id_min[second_place]:
#             # Case: the first chunk has no id overlap with the next one in the table. Merge chunk into output, update
#             # progress.chunk_files so it's ignored on next iteration, the continue with the next chunk
#             h5 = chunks_h5[lead_chunk]
#             # read_end_idx = h5['source_node_id'].shape[0]
#             write_index = _merge_chunk_file(edges_grp, h5, progress)
#             # write_beg_idx = progress.write_index
#             # write_end_idx = write_beg_idx + read_end_idx
#             #
#             # print('  Merging file {} (ids {:,} - {:,}) into master and indices [{:,} - {:,}'.format(
#             #     chunk_files[lead_chunk],
#             #     chunks_id_min[lead_chunk],
#             #     chunks_id_max[lead_chunk],
#             #     write_beg_idx,
#             #     write_end_idx
#             # ))
#             #
#             # edges_grp['source_node_id'][write_beg_idx:write_end_idx] = h5['source_node_id'][:read_end_idx]
#             # edges_grp['target_node_id'][write_beg_idx:write_end_idx] = h5['target_node_id'][:read_end_idx]
#             # edges_grp['edge_type_id'][write_beg_idx:write_end_idx] = h5['edge_type_id'][:read_end_idx]
#
#             del chunk_files[lead_chunk]
#             del chunk_indices[lead_chunk]
#             del chunk_nums[lead_chunk]
#             del chunks_h5[lead_chunk]
#             del chunks_id_min[lead_chunk]
#             del chunks_id_max[lead_chunk]
#             progress.update(write_index=write_index, chunk_files=chunk_files, chunk_indices=chunk_indices)
#
#             sorted_order = np.argsort(chunks_id_min)
#
#         elif chunks_id_min[lead_chunk] <= chunks_id_min[second_place]:
#             # Case: first sorted chunk file has subset of ids not in the other chunks, push this subset into the
#             # output and update chunk_indices so on next iteration only the remaining part of the first chunk is used.
#             h5 = chunks_h5[lead_chunk]
#             sort_col = np.array(h5[progress.sort_key])
#             read_end_idx = np.argwhere(sort_col <= chunks_id_min[second_place]).flatten()[-1] + 1
#             write_index = _merge_chunk_file(edges_grp, h5, progress, read_end_idx=read_end_idx)
#             # write_beg_idx = progress.write_index
#             # write_end_idx = write_beg_idx + read_end_idx
#             #
#             # print(' Merging ids [{:,} - {:,}] from {} into master (indices [{:,} - {:,}))'.format(
#             #     chunks_id_min[lead_chunk],
#             #     chunks_id_min[second_place],
#             #     chunk_files[lead_chunk],
#             #     write_beg_idx,
#             #     write_end_idx
#             # ))
#             #
#             # edges_grp['source_node_id'][write_beg_idx:write_end_idx] = h5['source_node_id'][:read_end_idx]
#             # edges_grp['target_node_id'][write_beg_idx:write_end_idx] = h5['target_node_id'][:read_end_idx]
#             # edges_grp['edge_type_id'][write_beg_idx:write_end_idx] = h5['edge_type_id'][:read_end_idx]
#             chunk_indices[lead_chunk] = int(read_end_idx)
#             progress.update(write_index=write_index, chunk_indices=chunk_indices)
#             break
#
#             # print(sort_col[end_idx-3:end_idx+4])
#         else:
#             print('Unable to merge data into output, continuing iterating over chunks.')
#             break


def merge(progress, edges_grp):
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
        # chunk_file = os.path.join(cache_dir, 'itr{}_chunk{}.h5'.format(itr_num, chunk_num))
        h5 = h5py.File(chunk_file, 'r+')
        chunk_nums.append(chunk_num)
        chunk_h5s.append(h5)
        chunk_ids_min.append(h5.attrs['min_id'])
        chunk_ids_max.append(h5.attrs['max_id'])
        max_id = int(np.max((max_id, h5.attrs['max_id'])))
        min_id = int(np.min((min_id, h5.attrs['min_id'])))

    #  sorted_order = np.argsort(chunk_ids_max)
    print('Current status:')
    print(pd.DataFrame({
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
        print(chunk_num)
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
        print('No ids collected.')
        return

    collected_df = collected_df.sort_values(progress.sort_key)
    n_collected = len(collected_df)

    write_beg_idx = progress.write_index
    write_end_idx = write_beg_idx + n_collected

    print('Writing ids [{:,} - {:,}] to output indices [{:,} - {:,})'.format(
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


def external_merge_sort(input_file, output_file, sort_key, edges_root, n_chunks=12, max_itrs=10):
    """Does an external merge sort on an input edges hdf5 file, saves value in new file. Usefull for large network
    files where we are not able to load into memory.

    Will split the original hdf5 into <n_chunks> chunks on the disk in .sort_cache/, Will sort each individual chunk
    of data in memory, then perform a merge on all the chunks. For speed considers may try to do the chunking and
    merging in multiple iterations.

    :param input_file: path to original edges file
    :param output_file: path name of new file that will be created
    :param sort_key: 'edge_type_id', 'source_node_id', '
    :param edges_root: hdf5 path to edges population group (ex /edges/left_ispi)
    :param n_chunks: Number of chunks
    :param max_itrs: maximimum number of times will do a spliting + merging, mostly so it doesn't fall in an infite loop
    """
    with h5py.File(input_file, 'r') as input_h5:
        n_edges = input_h5[edges_root]['source_node_id'].shape[0]

    cache_dir = os.path.join(os.path.dirname(output_file), '.sort_cache',
                             os.path.splitext(os.path.basename(output_file))[0])
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    progress = ProgressFile(cache_dir=cache_dir, n_edges=n_edges, n_chunks=n_chunks, root_name=edges_root,
                            sort_key=sort_key)

    output_root_grp = _create_output_h5(input_file=input_file, output_file=output_file, edges_root=edges_root,
                                        n_edges=n_edges)

    # Split the edges files into N chunks.
    if not progress.initialized or not os.path.exists(cache_dir):
        initialize_chunks(edges_file_path=input_file, progress=progress)
    else:
        print('edges files has already been initialized. Continuing from last saved point')

    # Try to merge chunk data to output, if there's still data left split and resort into another N chunks. Repeat.
    merge(progress, edges_grp=output_root_grp)
    for i in range(max_itrs):
        if progress.completed:
            break
        sort_chunks(progress)
        merge(progress, edges_grp=output_root_grp)

    print('Done.')


if __name__ == '__main__':
    external_merge_sort(
        input_file='network/right_local_edges.h5',
        output_file='network/right_local_edges_by_edgetype_merge_sort_v1.h5',
        sort_key='edge_type_id',
        edges_root='/edges/right_local'
    )
