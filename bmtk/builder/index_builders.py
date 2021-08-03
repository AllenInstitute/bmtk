import os
import numpy as np
import pandas as pd
import h5py
import logging


logger = logging.getLogger(__name__)

# MAX_EDGE_READS = 500000000
# MAX_EDGE_READS = 200000000


def _get_names(index_type):
    if index_type.lower() in ['target', 'target_id', 'target_node_id', 'target_node_ids']:
        col_to_index = 'target_node_id'
        index_grp_name = 'indices/target_to_source'
    elif index_type in ['source', 'source_id', 'source_node_id', 'source_node_ids']:
        col_to_index = 'source_node_id'
        index_grp_name = 'indices/source_to_target'
    elif index_type in ['edge_type', 'edge_type_id', 'edge_type_ids']:
        col_to_index = 'edge_type_id'
        index_grp_name = 'indices/edge_type_to_index'
    else:
        raise ValueError('Unknown edges parameter {}'.format(index_type))

    return col_to_index, index_grp_name


def remove_index(edges_file, edges_population):
    with h5py.File(edges_file, mode='r+') as edges_h5:
        edges_pop_grp = edges_h5[edges_population]
        del edges_pop_grp['indices']


def create_index_in_memory(edges_file, edges_population, index_type, force_rebuild=True, **kwargs):
    col_to_index, index_grp_name = _get_names(index_type)

    with h5py.File(edges_file, mode='r+') as edges_h5:
        edges_pop_grp = edges_h5[edges_population]

        if index_grp_name in edges_pop_grp:
            # Remove existing index if it exists
            if not force_rebuild:
                logger.debug('create_index_in_memory> Edges index {} already exists, skipping.'.format(index_grp_name))
            else:
                logger.debug('create_index_in_memory> Removing existing index {}.'.format(index_grp_name))
                del edges_pop_grp[index_grp_name]

        index_grp = edges_pop_grp.create_group(index_grp_name)
        ids_array = np.array(edges_pop_grp[col_to_index])  # ids to be indexed
        total_edges = len(edges_pop_grp[col_to_index])

        if total_edges == 0:
            logger.warning('edges file {} does not contain any edges.'.format(edges_file))
            return

        # group together and save ids so contigous duplicates are represented using ranges, creating a range -> edge
        # index table.
        #  eg. [10 10 10 10 10 32 32 32 10 10 ...] ==> ... 10: [(0, 5), (8, 10)], 32: [(5, 8)], ...
        logger.debug('create_index_in_memory> Creating range_to_edge_id table')
        ids_diffs = np.diff(ids_array)
        ids_diffs_idx = ids_diffs.nonzero()[0]

        ranges_beg = np.concatenate(([0], ids_diffs_idx + 1))
        ranges_end = np.concatenate((ids_diffs_idx + 1, [total_edges]))
        id_idxs = np.concatenate(([0], ids_diffs_idx + 1))

        r2e_table_df = pd.DataFrame({
            'lu_ids': ids_array[id_idxs],
            'range_beg': ranges_beg,
            'range_end': ranges_end
        }).sort_values('lu_ids')

        index_grp.create_dataset('range_to_edge_id', data=r2e_table_df[['range_beg', 'range_end']].values,
                                 dtype='uint64') # np.uint64)

        # create a map to the range_to_edge_id dataset from id --> blocks ranges. The id value is implicitly equal to
        # the index
        # TODO: See if del r2e_table_df will significantly improve memory footprint?
        logger.debug('create_index_in_memory> Creating node_id_to_range table')
        ordered_ids = np.array(r2e_table_df['lu_ids'])
        ordered_ids_diffs = np.diff(ordered_ids).nonzero()[0]
        ordered_ids_beg = np.concatenate(([0], ordered_ids_diffs+1))
        ordered_ids_end = np.concatenate((ordered_ids_diffs+1, [len(r2e_table_df)]))
        id_idxs = np.concatenate(([0], ordered_ids_diffs+1))
        i2r_table_df = pd.DataFrame({
            'lu_ids': ordered_ids[id_idxs],
            'range_beg': ordered_ids_beg,
            'range_end': ordered_ids_end
        }).set_index('lu_ids')

        # There may be missing values in the id_cols table so fil them in
        i2r_table_df = i2r_table_df.reindex(pd.RangeIndex(i2r_table_df.index.max()+1), fill_value=0)

        index_grp.create_dataset('node_id_to_range', data=i2r_table_df[['range_beg', 'range_end']].values,
                                 dtype=np.uint64)


def create_index_on_disk(edges_file, edges_population, index_type, force_rebuild=False, cache_file=None,
                         max_edge_reads=200000000, **kwargs):
    col_to_index, index_grp_name = _get_names(index_type)
    cache_file = cache_file if cache_file is not None else edges_file

    mode = 'r+' if os.path.exists(cache_file) else 'w'
    with h5py.File(cache_file, mode=mode) as cache_h5:
        # Step 1: Separate the column being indexed into N partitons each of less than max_edge_reads length. Build the
        # index for each partition and save to disk. Make sure we don't have to have the read the entire col_to_index
        # in memory at one time

        edges_h5 = cache_h5 if edges_file == cache_file else h5py.File(edges_file, 'r')
        edges_root_grp = edges_h5[edges_population]

        if edges_population not in cache_h5:
            caches_root_grp = cache_h5.create_group(edges_population)
        else:
            caches_root_grp = cache_h5[edges_population]

        total_edges = edges_root_grp[col_to_index].shape[0]
        partition_size = np.min((max_edge_reads, total_edges))
        n_partitions = np.ceil(total_edges / max_edge_reads).astype(np.uint)
        block_begin_idx = 0  # initial index of current partition being created
        partition_index = 0
        max_id = 0
        total_blocks = 0

        if force_rebuild:
            del caches_root_grp[index_grp_name]

        if index_grp_name not in caches_root_grp or 'cache' not in caches_root_grp[index_grp_name]:
            index_grp = caches_root_grp.create_group(index_grp_name) \
                if index_grp_name not in caches_root_grp else caches_root_grp[index_grp_name]
            cache_grp = index_grp.create_group('cache')
            cache_grp.attrs['max_edge_reads'] = max_edge_reads
            cache_grp.attrs['cache_partition_size'] = partition_size

        else:
            cache_grp = caches_root_grp[index_grp_name]['cache']
            max_id = cache_grp.attrs.get('max_id', max_id)

            if cache_grp.attrs['max_edge_reads'] != max_edge_reads:
                raise Exception(
                    'Cache already exists but has a conflicting max_edge_reads' 
                    '({} vs {}). Use force-rebuild'.format(cache_grp.attrs['max_edge_reads'], max_edge_reads)
                )

            if cache_grp.attrs['cache_partition_size'] != partition_size:
                raise Exception(
                    'Cache already exists but has a conflicting partition_size' 
                    '({} vs {}). Use force-rebuild'.format(cache_grp.attrs['cache_partition_size'], partition_size)
                )

        logger.debug('Number of edges, {}, exceeds maximum ({}).'.format(total_edges, max_edge_reads))
        logger.debug('Separating into {} partitions'.format(n_partitions))
        while block_begin_idx < total_edges:
            # cache_grp_name = index_grp_name + '/cache/edges_partition_{}'.format(partition_index)
            partition_grp_name = 'edges_partition_{}'.format(partition_index)
            if partition_grp_name in cache_grp:
                logger.debug('Cache {}/cache/{} already exists, skipping'.format(index_grp_name, partition_grp_name))
                block_begin_idx += partition_size
                partition_index += 1
                total_blocks += cache_grp[partition_grp_name]['partitioned_table'].shape[0]
                continue

            logger.debug('Creating cache {} of {} to {}/{}'.format(partition_index+1, n_partitions, index_grp_name,
                                                                   partition_grp_name))

            block_end_idx = block_begin_idx + partition_size
            edge_ids = edges_root_grp[col_to_index][block_begin_idx:block_end_idx]

            # Group together contigious duplicates to create a list of [id [edge_index_beg, edge_index_end)], sort by
            # id and save table to disk
            diffs = np.diff(edge_ids)
            diffs_idx = diffs.nonzero()[0]
            ranges_beg = np.concatenate(([0], diffs_idx + 1)) + block_begin_idx
            ranges_end = np.concatenate((diffs_idx + 1, [partition_size])) + block_begin_idx
            ranges_end[-1] = np.min((total_edges+1, ranges_end[-1]))
            id_idxs = np.concatenate(([0], diffs_idx + 1))

            r2e_table_df = pd.DataFrame({
                'lu_ids': edge_ids[id_idxs],
                'range_beg': ranges_beg,
                'range_end': ranges_end
            }).sort_values('lu_ids')

            # Creates a lookup table id --> ranges
            ordered_ids = r2e_table_df['lu_ids'].values
            # id_block_indxs = np.diff(ordered_ids).nonzero()[0] + 1
            # id_block_sizes = np.concatenate(([id_block_indxs[0]], np.diff(id_block_indxs)))
            # sums_vals = np.ones(len(ordered_ids), dtype=np.int32)
            # sums_vals[id_block_indxs] = -id_block_sizes + 1

            ordered_ids_diffs = np.diff(ordered_ids).nonzero()[0]
            ordered_ids_beg = np.concatenate(([0], ordered_ids_diffs + 1))
            ordered_ids_end = np.concatenate((ordered_ids_diffs + 1, [len(r2e_table_df)]))
            id_idxs = np.concatenate(([0], ordered_ids_diffs + 1))
            i2r_table_df = pd.DataFrame({
                'lu_ids': ordered_ids[id_idxs],
                'range_beg': ordered_ids_beg,
                'range_end': ordered_ids_end
            }).set_index('lu_ids')

            # fill in missing ids and foward fill Nans with the last previous index index
            i2r_table_df = i2r_table_df.reindex(pd.RangeIndex(i2r_table_df.index.max() + 1))
            i2r_table_df['range_end'] = i2r_table_df['range_end'].fillna(method='ffill')
            i2r_table_df['range_end'].fillna(0, inplace=True)

            nans_mask = i2r_table_df['range_beg'].isna()
            i2r_table_df['range_beg'][nans_mask] = i2r_table_df['range_end'][nans_mask]

            # Save partition to disk
            partition_grp = cache_grp.create_group(partition_grp_name)
            partition_grp.create_dataset('partitioned_table',
                                         data=r2e_table_df[['lu_ids', 'range_beg', 'range_end']].values,
                                         dtype='uint64')
            partition_grp.create_dataset('lookup_tables', data=i2r_table_df[['range_beg', 'range_end']].values,
                                         dtype='uint64')
            partition_grp.attrs['columns'] = 'id,range_beg,range_end'
            cache_h5.flush()

            max_id = np.max((max_id, int(np.max(ordered_ids))))
            total_blocks += len(r2e_table_df)
            block_begin_idx += partition_size
            partition_index += 1

            del r2e_table_df
            del i2r_table_df

        # TODO: Find max_id, total_blocks, and n_partitions and save as attribute in "cache" group
        cache_grp.attrs['max_id'] = max_id
        cache_grp.attrs['n_partitions'] = n_partitions
        cache_grp.attrs['total_blocks'] = total_blocks

    with h5py.File(cache_file, mode='r+') as cache_h5:
        # Step 2: Using the partitions we cached to disk in the previous step to create the actual index
        edges_pop_grp = cache_h5[edges_population]
        index_grp = edges_pop_grp[index_grp_name]
        if 'cache' not in index_grp:
            raise ValueError('Error, could not find hdf5 group {}/cache. Unable to proceed.'.format(index_grp_name))
        cache_grp = edges_pop_grp[index_grp_name]['cache']

        logger.debug('Building edges index {}'.format(index_grp_name))

        total_blocks = cache_grp.attrs['total_blocks']  # total number of 'partitioned_table' rows across all partitions
        max_id = cache_grp.attrs['max_id']  # maximum value in 'col_to_index' across all partions
        n_partitions = len(cache_grp.keys())

        # Go through each partition and build a master id lookup table. eg for each id should contains the number
        # of ranges and where their offsets would go when ordered.
        block_sizes = np.zeros(max_id + 1, dtype=np.uint32)
        for grp_name, grp in edges_pop_grp[index_grp_name]['cache'].items():
            # block_sizes += np.diff(grp['lookup_tables']).flatten()
            partiton_max_id = grp['lookup_tables'].shape[0]  # some partions may not contain all the ids
            block_sizes[:partiton_max_id] += np.diff(grp['lookup_tables']).flatten()

        # global_offsets = np.concatenate(([0], np.cumsum(block_sizes)[:-1])).astype(np.uint)  # + 1
        global_offsets = np.concatenate(([0], np.cumsum(block_sizes))).astype(np.uint)  # + 1
        ranges = np.vstack((global_offsets, np.concatenate((global_offsets[1:], [total_blocks])))).T

        # edges_pop_grp.create_dataset('{}/node_id_to_range'.format(index_grp), data=ranges, dtype=np.uint64)
        if not cache_grp.attrs.get('building_index', False):
            if 'node_id_to_range' in index_grp:
                del index_grp['node_id_to_range']
            index_grp.create_dataset('node_id_to_range'.format(index_grp), data=ranges[:-1], dtype=np.uint64)

            if 'range_to_edge_id' in index_grp:
                del index_grp['range_to_edge_id']
            index_grp.create_dataset('range_to_edge_id', (total_blocks, 2), dtype=np.uint64)

        # Using the ids offsets we can calculate the number N such that the associated "partitioned_table" for those N
        # ids have less than MAX_EDGE_READS rows (across all partitions). Procedure is to find N_0, iterate through all
        # the partitions to find ranges for ids [0, N_0], merge and sort and save to dataset. Then do the same thing
        # for ids [N_0, N_1]
        # TODO: Save index_beg, index_end, and block number as attributes in "cache"
        caches = {grp_name: grp for grp_name, grp in edges_pop_grp[index_grp_name]['cache'].items()}
        read_block_size = max_edge_reads / (n_partitions*2+1)

        if cache_grp.attrs.get('building_index', False):
            print('Index has already been partially built, resumming from last point')
            block_num = cache_grp.attrs['block_num']
            id_beg = cache_grp.attrs['id_beg']
            id_end = cache_grp.attrs['id_end']

        else:
            # separating global offsets into N blocks. Need to track which block were on, the begging index/row of the
            # block and the end index/row of block
            cache_grp.attrs['building_index'] = True
            cache_grp.attrs['block_num'] = block_num = 1
            cache_grp.attrs['id_beg'] = id_beg = 0
            id_end = np.max((np.searchsorted(global_offsets, read_block_size * block_num, side='right'), 1))
            id_end = np.min((max_id, id_end))
            cache_grp.attrs['id_end'] = id_end

        while id_beg <= max_id:
            logger.debug('Building "range_to_edge_id" for nodes ({}, {})'.format(id_beg, id_end))
            indx_beg = global_offsets[id_beg]
            indx_end = global_offsets[id_end+1]
            block_num += 1

            to_remove = []
            master_df = None
            for grp_name, grp in caches.items():
                # Use "lookup_tables" to find the which rows in "partioned_table" corresponds to ids [id_beg, id_end].
                # The partitioned tables are already sorted.
                if id_beg > grp['lookup_tables'].shape[0]:
                    to_remove.append(grp_name)
                    continue

                if id_end >= grp['lookup_tables'].shape[0]:
                    part_beg, part_end = grp['lookup_tables'][id_beg][0], grp['lookup_tables'][-1][1]
                    to_remove.append(grp_name)
                    # del caches[grp_name]
                else:
                    part_beg, part_end = grp['lookup_tables'][id_beg][0], grp['lookup_tables'][id_end][1]

                if part_end - part_beg <= 0:
                    continue  # possible partion does contain any edges for the subset of ids we are looking for

                tmp_df = pd.DataFrame({
                    'id': grp['partitioned_table'][part_beg:part_end, 0],
                    'range_beg': grp['partitioned_table'][part_beg:part_end, 1],
                    'range_end': grp['partitioned_table'][part_beg:part_end, 2]
                })
                master_df = tmp_df if master_df is None else pd.concat((master_df, tmp_df))

            if master_df is not None:
                master_df = master_df.sort_values(['id', 'range_beg'])
                index_grp['range_to_edge_id'][indx_beg:indx_end, :] = master_df[['range_beg', 'range_end']]

                for name in to_remove:
                    del caches[name]

                del master_df

            id_beg = id_end + 1
            id_end = np.searchsorted(global_offsets, read_block_size*block_num, side='right')
            id_end = np.min((max_id, id_end))

            # Save state of loop in case of failure and we need to progress later
            cache_grp.attrs['block_num'] = block_num
            cache_grp.attrs['id_beg'] = indx_beg
            cache_grp.attrs['id_end'] = indx_end

        # Remove cache
        if 'cache' in index_grp:
            del index_grp['cache']
