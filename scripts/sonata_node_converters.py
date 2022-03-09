import os

import h5py
import pandas as pd
import numpy as np


def convert_nodes(nodes_file, node_types_file, **params):
    is_h5 = False
    try:
        h5file = h5py.File(nodes_file, 'r')
        is_h5 = True
    except Exception as e:
        pass

    if is_h5:
        update_h5_nodes(nodes_file, node_types_file, **params)
        return

    update_csv_nodes(nodes_file, node_types_file, **params)


# columns which need to be renamed, key is original name and value is the updated name
column_renames = {
    'id': 'node_id',
    'model_id': 'node_type_id',
    'electrophysiology': 'dynamics_params',
    'level_of_detail': 'model_type',
    'morphology': 'morphology',
    'params_file': 'dynamics_params',
    'x_soma': 'x',
    'y_soma': 'y',
    'z_soma': 'z'
}


def update_h5_nodes(nodes_file, node_types_file, network_name, output_dir='output',
                    column_order=('node_type_id', 'model_type', 'model_template', 'model_processing', 'dynamics_params',
                                  'morphology')):
    # open nodes and node-types into a single table
    input_h5 = h5py.File(nodes_file, 'r')

    output_name = '{}_nodes.h5'.format(network_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    nodes_output_fn = os.path.join(output_dir, output_name)

    # save nodes hdf5
    with h5py.File(nodes_output_fn, 'w') as h5:
        #h5.copy()
        #grp = h5.create_group('/nodes/{}'.format(network_name))
        #input_grp = input_h5['/nodes/']
        nodes_path = '/nodes/{}'.format(network_name)
        h5.copy(input_h5['/nodes/'], nodes_path)
        grp = h5[nodes_path]
        grp.move('node_gid', 'node_id')
        grp.move('node_group', 'node_group_id')

    node_types_csv = pd.read_csv(node_types_file, sep=' ')

    node_types_csv = node_types_csv.rename(index=str, columns=column_renames)

    # Change values for model type
    model_type_map = {
        'biophysical': 'biophysical',
        'point_IntFire1': 'point_process',
        'intfire': 'point_process',
        'virtual': 'virtual',
        'iaf_psc_alpha': 'nest:iaf_psc_alpha',
        'filter': 'virtual'
    }
    node_types_csv['model_type'] = node_types_csv.apply(lambda row: model_type_map[row['model_type']], axis=1)

    # Add model_template column
    def model_template(row):
        model_type = row['model_type']
        if model_type == 'biophysical':
            return 'ctdb:Biophys1.hoc'
        elif model_type == 'point_process':
            return 'nrn:IntFire1'
        else:
            return 'NONE'
    node_types_csv['model_template'] = node_types_csv.apply(model_template, axis=1)

    # Add model_processing column
    def model_processing(row):
        model_type = row['model_type']
        if model_type == 'biophysical':
            return 'aibs_perisomatic'
        else:
            return 'NONE'
    node_types_csv['model_processing'] = node_types_csv.apply(model_processing, axis=1)

    # Reorder columns
    orig_columns = node_types_csv.columns
    col_order = [cn for cn in column_order if cn in orig_columns]
    col_order += [cn for cn in node_types_csv.columns if cn not in column_order]
    node_types_csv = node_types_csv[col_order]

    # Save node-types csv
    node_types_output_fn = os.path.join(output_dir, '{}_node_types.csv'.format(network_name))
    node_types_csv.to_csv(node_types_output_fn, sep=' ', index=False, na_rep='NONE')
    # open nodes and node-types into a single table

    '''
    print('loading csv files')
    nodes_tmp = pd.read_csv(nodes_file, sep=' ')
    node_types_tmp = pd.read_csv(node_types_file, sep=' ')
    nodes_df = pd.merge(nodes_tmp, node_types_tmp, on='node_type_id')
    n_nodes = len(nodes_df.index)

    # rename required columns
    nodes_df = nodes_df.rename(index=str, columns=column_renames)

    # Old versions of node_type_id may be set to strings/floats, convert to integers
    dtype_ntid = nodes_df['node_type_id'].dtype
    if dtype_ntid == 'object':
        # if string, move model_id to pop_name and create an integer node_type_id column
        if 'pop_name' in nodes_df.columns:
            nodes_df = nodes_df.drop('pop_name', axis=1)
        nodes_df = nodes_df.rename(index=str, columns={'node_type_id': 'pop_name'})
        ntid_map = {pop_name: indx for indx, pop_name in enumerate(nodes_df['pop_name'].unique())}
        nodes_df['node_type_id'] = nodes_df.apply(lambda row: ntid_map[row['pop_name']], axis=1)

    elif dtype_ntid == 'float64':
        nodes_df['node_type_id'] = nodes_df['node_type_id'].astype('uint64')

    # divide columns up into nodes and node-types columns, and for nodes determine which columns are valid for every
    # node-type. The rules are
    #  1. If all values are the same for a node-type-id, column belongs in node_types csv. If there's any intra
    #     node-type heterogenity then the column belongs in the nodes h5.
    #  2. For nodes h5 columns, a column belongs to a node-type-id if it contains at least one non-null value
    print('parsing input')
    opt_columns = [n for n in nodes_df.columns if n not in ['node_id', 'node_type_id']]
    heterogeneous_cols = {cn: False for cn in opt_columns}
    nonnull_cols = {}  # for each node-type, a list of columns that contains at least one non-null value
    for node_type_id, nt_group in nodes_df.groupby(['node_type_id']):
        nonnull_cols[node_type_id] = set(nt_group.columns[nt_group.isnull().any() == False].tolist())
        for col_name in opt_columns:
            heterogeneous_cols[col_name] |= len(nt_group[col_name].unique()) > 1

    nodes_columns = set(cn for cn, val in heterogeneous_cols.items() if val)
    nodes_types_columns = [cn for cn, val in heterogeneous_cols.items() if not val]

    # Check for nodes columns that has non-numeric values, these will require some special processing to save to hdf5
    string_nodes_columns = set()
    for col_name in nodes_columns:
        if nodes_df[col_name].dtype == 'object':
            string_nodes_columns.add(col_name)
    if len(string_nodes_columns) > 0:
        print('Warning: column(s) {} have non-numeric values that vary within a node-type and will be stored in h5 format'.format(list(string_nodes_columns)))

    # Divide the nodes columns into groups and create neccessary lookup tables. If two node-types share the same
    # non-null columns then they belong to the same group
    grp_idx2cols = {}  # group-id --> group-columns
    grp_cols2idx = {}  # group-columns --> group-id
    grp_id2idx = {}  # node-type-id --> group-id
    group_index = -1
    for nt_id, cols in nonnull_cols.items():
        group_columns = sorted(list(nodes_columns & cols))
        col_key = tuple(group_columns)
        if col_key in grp_cols2idx:
            grp_id2idx[nt_id] = grp_cols2idx[col_key]
        else:
            group_index += 1
            grp_cols2idx[col_key] = group_index
            grp_idx2cols[group_index] = group_columns
            grp_id2idx[nt_id] = group_index

    # merge x,y,z columns, if they exists, into 'positions' dataset
    grp_pos_cols = {}
    for grp_idx, cols in grp_idx2cols.items():
        pos_list = []
        for coord in ['x', 'y', 'z']:
            if coord in cols:
                pos_list += coord
                grp_idx2cols[grp_idx].remove(coord)
        if len(pos_list) > 0:
            grp_pos_cols[grp_idx] = pos_list

    # Create the node_group and node_group_index columns
    nodes_df['__bmtk_node_group'] = nodes_df.apply(lambda row: grp_id2idx[row['node_type_id']], axis=1)
    nodes_df['__bmtk_node_group_index'] = [0]*n_nodes
    for grpid in grp_idx2cols.keys():
        group_size = len(nodes_df[nodes_df['__bmtk_node_group'] == grpid])
        nodes_df.loc[nodes_df['__bmtk_node_group'] == grpid, '__bmtk_node_group_index'] = range(group_size)

    # Save nodes.h5 file
    nodes_output_fn = os.path.join(output_dir, '{}_nodes.h5'.format(network_name))
    node_types_output_fn = os.path.join(output_dir, '{}_node_types.csv'.format(network_name))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Creating {}'.format(nodes_output_fn))
    with h5py.File(nodes_output_fn, 'w') as hf:
        hf.create_dataset('nodes/node_gid', data=nodes_df['node_id'], dtype='uint64')
        hf['nodes/node_gid'].attrs['network'] = network_name
        hf.create_dataset('nodes/node_type_id', data=nodes_df['node_type_id'], dtype='uint64')
        hf.create_dataset('nodes/node_group', data=nodes_df['__bmtk_node_group'], dtype='uint32')
        hf.create_dataset('nodes/node_group_index', data=nodes_df['__bmtk_node_group_index'], dtype='uint64')

        for grpid, cols in grp_idx2cols.items():
            group_slice = nodes_df[nodes_df['__bmtk_node_group'] == grpid]
            for col_name in cols:
                dataset_name = 'nodes/{}/{}'.format(grpid, col_name)
                if col_name in string_nodes_columns:
                    # for columns with non-numeric values
                    dt = h5py.special_dtype(vlen=bytes)
                    hf.create_dataset(dataset_name, data=group_slice[col_name], dtype=dt)
                else:
                    hf.create_dataset(dataset_name, data=group_slice[col_name])

            # special case for positions
            if grpid in grp_pos_cols:
                hf.create_dataset('nodes/{}/positions'.format(grpid),
                                  data=group_slice.as_matrix(columns=grp_pos_cols[grpid]))

    # Save the node_types.csv file
    print('Creating {}'.format(node_types_output_fn))
    node_types_table = nodes_df[['node_type_id'] + nodes_types_columns]
    node_types_table = node_types_table.drop_duplicates()
    if len(sort_order) > 0:
        node_types_table = node_types_table.sort_values(by=sort_order)

    node_types_table.to_csv(node_types_output_fn, sep=' ', index=False) # , na_rep='NONE')
    '''


def update_csv_nodes(nodes_file, node_types_file, network_name, output_dir='network',
                     column_order=('node_type_id', 'model_type', 'model_template', 'model_processing',
                                   'dynamics_params', 'morphology')):
    # open nodes and node-types into a single table
    print('loading csv files')
    nodes_tmp = pd.read_csv(nodes_file, sep=' ')
    node_types_tmp = pd.read_csv(node_types_file, sep=' ')
    if 'model_id' in nodes_tmp:
        nodes_df = pd.merge(nodes_tmp, node_types_tmp, on='model_id')
    elif 'node_type_id' in nodes_tmp:
        nodes_df = pd.merge(nodes_tmp, node_types_tmp, on='node_type_id')
    else:
        raise Exception('Could not find column to merge nodes and node_types')

    n_nodes = len(nodes_df.index)

    # rename required columns
    nodes_df = nodes_df.rename(index=str, columns=column_renames)

    # Old versions of node_type_id may be set to strings/floats, convert to integers
    dtype_ntid = nodes_df['node_type_id'].dtype
    if dtype_ntid == 'object':
        # if string, move model_id to pop_name and create an integer node_type_id column
        if 'pop_name' in nodes_df:
            nodes_df = nodes_df.drop('pop_name', axis=1)

        nodes_df = nodes_df.rename(index=str, columns={'node_type_id': 'pop_name'})

        ntid_map = {pop_name: indx for indx, pop_name in enumerate(nodes_df['pop_name'].unique())}
        nodes_df['node_type_id'] = nodes_df.apply(lambda row: ntid_map[row['pop_name']], axis=1)

    elif dtype_ntid == 'float64':
        nodes_df['node_type_id'] = nodes_df['node_type_id'].astype('uint64')

    # divide columns up into nodes and node-types columns, and for nodes determine which columns are valid for every
    # node-type. The rules are
    #  1. If all values are the same for a node-type-id, column belongs in node_types csv. If there's any intra
    #     node-type heterogenity then the column belongs in the nodes h5.
    #  2. For nodes h5 columns, a column belongs to a node-type-id if it contains at least one non-null value
    print('parsing input')
    opt_columns = [n for n in nodes_df.columns if n not in ['node_id', 'node_type_id']]
    heterogeneous_cols = {cn: False for cn in opt_columns}
    nonnull_cols = {}  # for each node-type, a list of columns that contains at least one non-null value
    for node_type_id, nt_group in nodes_df.groupby(['node_type_id']):
        nonnull_cols[node_type_id] = set(nt_group.columns[nt_group.isnull().any() == False].tolist())
        for col_name in opt_columns:
            heterogeneous_cols[col_name] |= len(nt_group[col_name].unique()) > 1

    nodes_columns = set(cn for cn, val in heterogeneous_cols.items() if val)
    nodes_types_columns = [cn for cn, val in heterogeneous_cols.items() if not val]

    # Check for nodes columns that has non-numeric values, these will require some special processing to save to hdf5
    string_nodes_columns = set()
    for col_name in nodes_columns:
        if nodes_df[col_name].dtype == 'object':
            string_nodes_columns.add(col_name)
    if len(string_nodes_columns) > 0:
        print('Warning: column(s) {} have non-numeric values that vary within a node-type and will be stored in h5 format'.format(list(string_nodes_columns)))

    # Divide the nodes columns into groups and create neccessary lookup tables. If two node-types share the same
    # non-null columns then they belong to the same group
    grp_idx2cols = {}  # group-id --> group-columns
    grp_cols2idx = {}  # group-columns --> group-id
    grp_id2idx = {}  # node-type-id --> group-id
    group_index = -1
    for nt_id, cols in nonnull_cols.items():
        group_columns = sorted(list(nodes_columns & cols))
        col_key = tuple(group_columns)
        if col_key in grp_cols2idx:
            grp_id2idx[nt_id] = grp_cols2idx[col_key]
        else:
            group_index += 1
            grp_cols2idx[col_key] = group_index
            grp_idx2cols[group_index] = group_columns
            grp_id2idx[nt_id] = group_index

    # merge x,y,z columns, if they exists, into 'positions' dataset
    grp_pos_cols = {}
    for grp_idx, cols in grp_idx2cols.items():
        pos_list = []
        for coord in ['x', 'y', 'z']:
            if coord in cols:
                pos_list += coord
                grp_idx2cols[grp_idx].remove(coord)
        if len(pos_list) > 0:
            grp_pos_cols[grp_idx] = pos_list

    # Create the node_group and node_group_index columns
    nodes_df['__bmtk_node_group'] = nodes_df.apply(lambda row: grp_id2idx[row['node_type_id']], axis=1)
    nodes_df['__bmtk_node_group_index'] = [0]*n_nodes
    for grpid in grp_idx2cols.keys():
        group_size = len(nodes_df[nodes_df['__bmtk_node_group'] == grpid])
        nodes_df.loc[nodes_df['__bmtk_node_group'] == grpid, '__bmtk_node_group_index'] = range(group_size)

    # Save nodes.h5 file
    nodes_output_fn = os.path.join(output_dir, '{}_nodes.h5'.format(network_name))
    node_types_output_fn = os.path.join(output_dir, '{}_node_types.csv'.format(network_name))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Creating {}'.format(nodes_output_fn))
    with h5py.File(nodes_output_fn, 'w') as hf:
        grp = hf.create_group('/nodes/{}'.format(network_name))
        grp.create_dataset('node_id', data=nodes_df['node_id'], dtype='uint64')
        grp.create_dataset('node_type_id', data=nodes_df['node_type_id'], dtype='uint64')
        grp.create_dataset('node_group_id', data=nodes_df['__bmtk_node_group'], dtype='uint32')
        grp.create_dataset('node_group_index', data=nodes_df['__bmtk_node_group_index'], dtype='uint64')

        for grpid, cols in grp_idx2cols.items():
            group_slice = nodes_df[nodes_df['__bmtk_node_group'] == grpid]
            for col_name in cols:
                dataset_name = '{}/{}'.format(grpid, col_name)
                if col_name in string_nodes_columns:
                    # for columns with non-numeric values
                    dt = h5py.special_dtype(vlen=bytes)
                    grp.create_dataset(dataset_name, data=group_slice[col_name], dtype=dt)
                else:
                    grp.create_dataset(dataset_name, data=group_slice[col_name])

            # special case for positions
            if grpid in grp_pos_cols:
                grp.create_dataset('{}/positions'.format(grpid),
                                  data=group_slice.as_matrix(columns=grp_pos_cols[grpid]))

            # Create empty dynamics_params
            grp.create_group('{}/dynamics_params'.format(grpid))

    # Save the node_types.csv file
    print('Creating {}'.format(node_types_output_fn))
    node_types_table = nodes_df[['node_type_id'] + nodes_types_columns]
    node_types_table = node_types_table.drop_duplicates()

    # Change values for model type
    model_type_map = {
        'biophysical': 'biophysical',
        'point_IntFire1': 'point_process',
        'virtual': 'virtual',
        'intfire': 'point_process',
        'filter': 'virtual'
    }
    node_types_table['model_type'] = node_types_table.apply(lambda row: model_type_map[row['model_type']], axis=1)
    if 'set_params_function' in node_types_table:
        node_types_table = node_types_table.drop('set_params_function', axis=1)

    # Add model_template column
    def model_template(row):
        model_type = row['model_type']
        if model_type == 'biophysical':
            return 'ctdb:Biophys1.hoc'
        elif model_type == 'point_process':
            return 'nrn:IntFire1'
        else:
            return 'NONE'
    node_types_table['model_template'] = node_types_table.apply(model_template, axis=1)

    # Add model_processing column
    def model_processing(row):
        model_type = row['model_type']
        if model_type == 'biophysical':
            return 'aibs_perisomatic'
        else:
            return 'NONE'
    node_types_table['model_processing'] = node_types_table.apply(model_processing, axis=1)

    # Reorder columns
    orig_columns = node_types_table.columns
    col_order = [cn for cn in column_order if cn in orig_columns]
    col_order += [cn for cn in node_types_table.columns if cn not in column_order]
    node_types_table = node_types_table[col_order]

    node_types_table.to_csv(node_types_output_fn, sep=' ', index=False, na_rep='NONE')
