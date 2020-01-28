import os
from functools import partial
import numpy as np
import pandas as pd
import h5py

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


column_renames = {
    'params_file': 'dynamics_params',
    'level_of_detail': 'model_type',
    'morphology': 'morphology',
    'x_soma': 'x',
    'y_soma': 'y',
    'z_soma': 'z',
    'weight_max': 'syn_weight',
    'set_params_function': 'model_template'
}


def convert_edges(edges_file, edge_types_file, **params):
    is_flat_h5 = False
    is_new_h5 = False
    try:
        h5file = h5py.File(edges_file, 'r')
        if 'edges' in h5file:
            is_new_h5 = True
        elif 'num_syns' in h5file:
            is_flat_h5 = True
    except Exception as e:
        pass

    if is_flat_h5:
        update_aibs_edges(edges_file, edge_types_file, **params)
        return
    elif is_new_h5:
        update_h5_edges(edges_file, edge_types_file, **params)
        return

    try:
        edges_csv2h5(edges_file, **params)
        return
    except Exception as exc:
        raise exc

    raise Exception('Could not parse edges file')


def update_edge_types_file(edge_types_file, src_network=None, trg_network=None, output_dir='network'):
    edge_types_csv = pd.read_csv(edge_types_file, sep=' ')

    # rename required columns
    edge_types_csv = edge_types_csv.rename(index=str, columns=column_renames)

    edge_types_output_fn = os.path.join(output_dir, '{}_{}_edge_types.csv'.format(src_network, trg_network))
    edge_types_csv.to_csv(edge_types_output_fn, sep=' ', index=False, na_rep='NONE')


def update_h5_edges(edges_file, edge_types_file, src_network=None, population_name=None, trg_network=None,
                    output_dir='network'):
    population_name = population_name if population_name is not None else '{}_to_{}'.format(src_network, trg_network)
    input_h5 = h5py.File(edges_file, 'r')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    edges_output_fn = os.path.join(output_dir, '{}_{}_edges.h5'.format(src_network, trg_network))
    with h5py.File(edges_output_fn, 'w') as h5:
        edges_path = '/edges/{}'.format(population_name)
        h5.copy(input_h5['/edges'], edges_path)
        grp = h5[edges_path]
        grp.move('source_gid', 'source_node_id')
        grp.move('target_gid', 'target_node_id')
        grp.move('edge_group', 'edge_group_id')

        if 'network' in grp['source_node_id'].attrs:
            del grp['source_node_id'].attrs['network']
        grp['source_node_id'].attrs['node_population'] = src_network

        if 'network' in grp['target_node_id'].attrs:
            del grp['target_node_id'].attrs['network']
        grp['target_node_id'].attrs['node_population'] = trg_network

        create_index(input_h5['edges/target_gid'], grp, index_type=INDEX_TARGET)
        create_index(input_h5['edges/source_gid'], grp, index_type=INDEX_SOURCE)

    update_edge_types_file(edge_types_file, src_network, trg_network, output_dir)


def update_aibs_edges(edges_file, edge_types_file, trg_network, src_network, population_name=None, output_dir='output'):
    population_name = population_name if population_name is not None else '{}_to_{}'.format(src_network, trg_network)

    edges_h5 = h5py.File(edges_file, 'r')
    src_gids = edges_h5['/src_gids']
    n_edges = len(src_gids)
    trg_gids = np.zeros(n_edges, dtype=np.uint64)
    start = edges_h5['/edge_ptr'][0]
    for trg_gid, end in enumerate(edges_h5['/edge_ptr'][1:]):
        trg_gids[start:end] = [trg_gid]*(end-start)
        start = end

    edges_output_fn = os.path.join(output_dir, '{}_{}_edges.h5'.format(src_network, trg_network))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with h5py.File(edges_output_fn, 'w') as hf:
        add_hdf5_magic(hf)
        add_hdf5_version(hf)
        grp = hf.create_group('/edges/{}'.format(population_name))

        grp.create_dataset('target_node_id', data=trg_gids, dtype='uint64')
        grp['target_node_id'].attrs['node_population'] = trg_network
        grp.create_dataset('source_node_id', data=edges_h5['src_gids'], dtype='uint64')
        grp['source_node_id'].attrs['node_population'] = src_network
        grp.create_dataset('edge_group_id', data=np.zeros(n_edges), dtype='uint32')
        grp.create_dataset('edge_group_index', data=np.arange(0, n_edges))
        grp.create_dataset('edge_type_id', data=edges_h5['edge_types'])
        grp.create_dataset('0/nsyns', data=edges_h5['num_syns'], dtype='uint32')
        grp.create_group('0/dynamics_params')

        create_index(trg_gids, grp, index_type=INDEX_TARGET)
        create_index(src_gids, grp, index_type=INDEX_SOURCE)

    update_edge_types_file(edge_types_file, src_network, trg_network, output_dir)


def edges_csv2h5(edges_file, edge_types_file, src_network, src_nodes, src_node_types, trg_network, trg_nodes,
                  trg_node_types, output_dir='network', src_label='location', trg_label='pop_name'):
    """Used to convert oldest (isee engine) edges files

    :param edges_file:
    :param edge_types_file:
    :param src_network:
    :param src_nodes:
    :param src_node_types:
    :param trg_network:
    :param trg_nodes:
    :param trg_node_types:
    :param output_dir:
    :param src_label:
    :param trg_label:
    """
    column_renames = {
        'target_model_id': 'node_type_id',
        'weight': 'weight_max',
        'weight_function': 'weight_func',
    }

    columns_order = ['edge_type_id', 'target_query', 'source_query']

    edges_h5 = h5py.File(edges_file, 'r')
    edge_types_df = pd.read_csv(edge_types_file, sep=' ')
    n_edges = len(edges_h5['src_gids'])
    n_targets = len(edges_h5['indptr']) - 1

    # rename specified columns in edge-types
    edge_types_df = edge_types_df.rename(columns=column_renames)

    # Add a "target_query" and "source_query" columns from target_label and source_label
    def query_col(row, labels, search_col):
        return '&'.join("{}=='{}'".format(l, row[search_col]) for l in labels)
    trg_query_fnc = partial(query_col, labels=['node_type_id', trg_label], search_col='target_label')
    src_query_fnc = partial(query_col, labels=[src_label], search_col='source_label')

    edge_types_df['target_query'] = edge_types_df.apply(trg_query_fnc, axis=1)
    edge_types_df['source_query'] = edge_types_df.apply(src_query_fnc, axis=1)

    # Add an edge_type_id column
    edge_types_df['edge_type_id'] = np.arange(100, 100 + len(edge_types_df.index), dtype='uint32')

    nodes_tmp = pd.read_csv(src_nodes, sep=' ', index_col=['id'])
    node_types_tmp = pd.read_csv(src_node_types, sep=' ')
    src_nodes_df = pd.merge(nodes_tmp, node_types_tmp, on='model_id')

    nodes_tmp = pd.read_csv(trg_nodes, sep=' ', index_col=['id'])
    node_types_tmp = pd.read_csv(trg_node_types, sep=' ')
    trg_nodes_df = pd.merge(nodes_tmp, node_types_tmp, on='model_id')

    # For assigning edge types to each edge. For a given src --> trg pair we need to lookup source_label and
    # target_label values of the nodes, then use it to find the corresponding edge_types row.
    print('Processing edge_type_id dataset')
    edge_types_ids = np.zeros(n_edges, dtype='uint32')
    edge_types_df = edge_types_df.set_index(['node_type_id', 'target_label', 'source_label'])
    ten_percent = int(n_targets*.1)  # for keepting track of progress
    index = 0  # keeping track of row index
    for trg_gid in range(n_targets):
        # for the target find value node_type_id and target_label
        nodes_row = trg_nodes_df.loc[trg_gid]
        model_id = nodes_row['model_id']
        trg_label_val = nodes_row[trg_label]

        # iterate through all the sources
        idx_begin = edges_h5['indptr'][trg_gid]
        idx_end = edges_h5['indptr'][trg_gid+1]
        for src_gid in edges_h5['src_gids'][idx_begin:idx_end]:
            # find each source_label, use value to find edge_type_id
            # TODO: may be faster to filter by model_id, trg_label_val before iterating through the sources
            src_label_val = src_nodes_df.loc[src_gid][src_label]
            edge_type_id = edge_types_df.loc[model_id, trg_label_val, src_label_val]['edge_type_id']
            edge_types_ids[index] = edge_type_id
            index += 1

        if trg_gid % ten_percent == 0 and trg_gid != 0:
            print('   processed {} out of {} targets'.format(trg_gid, n_targets))

    # Create the target_gid table
    print('Creating target_gid dataset')
    trg_gids = np.zeros(n_edges)
    for trg_gid in range(n_targets):
        idx_begin = edges_h5['indptr'][trg_gid]
        idx_end = edges_h5['indptr'][trg_gid+1]
        trg_gids[idx_begin:idx_end] = [trg_gid]*(idx_end - idx_begin)

    # Save edges.h5
    edges_output_fn = '{}/{}_{}_edges.h5'.format(output_dir, src_network, trg_network)
    print('Saving edges to {}.'.format(edges_output_fn))
    with h5py.File(edges_output_fn, 'w') as hf:
        hf.create_dataset('edges/target_gid', data=trg_gids, dtype='uint64')
        hf['edges/target_gid'].attrs['node_population'] = trg_network
        hf.create_dataset('edges/source_gid', data=edges_h5['src_gids'], dtype='uint64')
        hf['edges/source_gid'].attrs['node_population'] = trg_network
        hf.create_dataset('edges/index_pointer', data=edges_h5['indptr'])
        hf.create_dataset('edges/edge_group', data=np.zeros(n_edges), dtype='uint32')
        hf.create_dataset('edges/edge_group_index', data=np.arange(0, n_edges))
        hf.create_dataset('edges/edge_type_id', data=edge_types_ids)

        hf.create_dataset('edges/0/nsyns', data=edges_h5['nsyns'], dtype='uint32')

    # Save edge_types.csv
    update_edge_types_file(edge_types_file, src_network, trg_network, output_dir)
    '''
    edges_types_output_fn = '{}/{}_{}_edge_types.csv'.format(output_dir, src_network, trg_network)
    print('Saving edge-types to {}'.format(edges_types_output_fn))
    edge_types_df = edge_types_df[edge_types_df['edge_type_id'].isin(np.unique(edge_types_ids))]
    # reorder columns
    reorderd_cols = columns_order + [cn for cn in edge_types_df.columns.tolist() if cn not in columns_order]
    edge_types_df = edge_types_df[reorderd_cols]
    edge_types_df.to_csv(edges_types_output_fn, sep=' ', index=False, na_rep='NONE')
    '''


INDEX_TARGET = 0
INDEX_SOURCE = 1


def create_index(node_ids_ds, output_grp, index_type=INDEX_TARGET):
    if index_type == INDEX_TARGET:
        edge_nodes = np.array(node_ids_ds, dtype=np.int64)
        output_grp = output_grp.create_group('indicies/target_to_source')
    elif index_type == INDEX_SOURCE:
        edge_nodes = np.array(node_ids_ds, dtype=np.int64)
        output_grp = output_grp.create_group('indicies/source_to_target')

    edge_nodes = np.append(edge_nodes, [-1])
    n_targets = np.max(edge_nodes)
    ranges_list = [[] for _ in range(n_targets + 1)]

    n_ranges = 0
    begin_index = 0
    cur_trg = edge_nodes[begin_index]
    for end_index, trg_gid in enumerate(edge_nodes):
        if cur_trg != trg_gid:
            ranges_list[cur_trg].append((begin_index, end_index))
            cur_trg = int(trg_gid)
            begin_index = end_index
            n_ranges += 1

    node_id_to_range = np.zeros((n_targets+1, 2))
    range_to_edge_id = np.zeros((n_ranges, 2))
    range_index = 0
    for node_index, trg_ranges in enumerate(ranges_list):
        if len(trg_ranges) > 0:
            node_id_to_range[node_index, 0] = range_index
            for r in trg_ranges:
                range_to_edge_id[range_index, :] = r
                range_index += 1
            node_id_to_range[node_index, 1] = range_index

    output_grp.create_dataset('range_to_edge_id', data=range_to_edge_id, dtype='uint64')
    output_grp.create_dataset('node_id_to_range', data=node_id_to_range, dtype='uint64')
