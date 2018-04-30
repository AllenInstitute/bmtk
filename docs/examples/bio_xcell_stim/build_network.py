import h5py
import pandas as pd
import numpy as np


def build_node_types():
    cell_types_df = pd.read_csv('example/inputs/cell_models_perisomatic_axonfix.csv', sep=' ')

    def get_pop_name(row):
        lst = row['morphology'].split('-')
        if len(lst) == 1:
            return 'Human'
        else:
            return lst[0]

    def get_potential(row):
        if row['pop_name'] == 'Pvalb':
            return 'i'
        else:
            return 'e'

    # Add new columns
    cell_types_df['pop_name'] = cell_types_df.apply(get_pop_name, axis=1)
    cell_types_df['model_template'] = cell_types_df.apply(lambda _: 'ctdb:Biophys1.hoc', axis=1)
    cell_types_df['location'] = cell_types_df.apply(lambda _: 'VisL4', axis=1)
    cell_types_df['ei'] = cell_types_df.apply(get_potential, axis=1)
    cell_types_df['fixaxon'] = cell_types_df.apply(lambda _: 'aibs_perisomatic_directed', axis=1)

    column_renames = {
        'model_id': 'node_type_id',
        'level_of_detail': 'model_type',
        'fixaxon': 'model_processing',
        'morphology': 'morphology_file',
        'electrophysiology': 'dynamics_params'
    }

    # Change column names
    cell_types_df = cell_types_df.rename(index=str, columns=column_renames)

    # reorder columns
    column_order = ['node_type_id', 'model_type', 'model_template', 'model_processing', 'dynamics_params']
    col_order = column_order + [cn for cn in cell_types_df.columns if cn not in column_order]
    cell_types_df = cell_types_df[col_order]

    # save
    cell_types_df.to_csv('network/v1_node_types.csv', sep=' ', index=False, na_rep='NONE')


def build_nodes():
    cells_df = pd.read_csv('example/inputs/485184849_cell.csv', sep=' ')
    node_ids = np.array(cells_df['id'], dtype=np.uint64)
    node_group_ids = np.zeros(len(node_ids), dtype=np.uint32)
    node_group_indices = range(0, len(node_ids))
    node_type_ids = np.array(cells_df['model_id'], dtype=np.uint64)

    positions = np.array([cells_df['x_soma'], cells_df['y_soma'], cells_df['z_soma']], dtype=np.float).T
    rotation_angle_yaxis = np.array(cells_df['rotation_angle_yaxis'], dtype=np.float)
    tuning_angle = np.array(cells_df['tuning_angle'], dtype=np.float)

    with h5py.File('network/v1_nodes.h5', 'w') as h5:
        nodes_grp = h5.create_group('/nodes/v1')
        nodes_grp.create_dataset('node_id', data=node_ids)
        nodes_grp.create_dataset('node_group_id', data=node_group_ids)
        nodes_grp.create_dataset('node_group_index', data=node_group_indices)
        nodes_grp.create_dataset('node_type_id', data=node_type_ids)

        prop_grp = nodes_grp.create_group('0')
        prop_grp.create_dataset('positions', data=positions)
        prop_grp.create_dataset('rotation_angle_yaxis', data=rotation_angle_yaxis)
        prop_grp.create_dataset('tuning_angle', data=tuning_angle)


if __name__ == '__main__':
    build_node_types()
    build_nodes()
