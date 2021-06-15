import os
import h5py
import logging
import numpy as np
import pandas as pd

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


def _create_output_h5(input_file, output_file, edges_root, n_edges):
    mode = 'r+' if os.path.exists(output_file) else 'w'
    out_h5 = h5py.File(output_file, mode=mode)
    add_hdf5_version(out_h5)
    add_hdf5_magic(out_h5)
    root_grp = out_h5.create_group(edges_root) if edges_root not in out_h5 else out_h5[edges_root]

    if 'source_node_id' not in root_grp:
        root_grp.create_dataset('source_node_id', (n_edges, ), dtype=np.uint64)

    if 'target_node_id' not in root_grp:
        root_grp.create_dataset('target_node_id', (n_edges, ), dtype=np.uint64)

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


# def _get_sort(index_type):
#     if index_type.lower() in ['target', 'target_id', 'target_node_id', 'target_node_ids']:
#         col_to_index = 'target_node_id'
#         index_grp_name = 'indices/target_to_source'
#     elif index_type in ['source', 'source_id', 'source_node_id', 'source_node_ids']:
#         col_to_index = 'source_node_id'
#         index_grp_name = 'indices/source_to_target'
#     elif index_type == ['edge_type', 'edge_type_id', 'edge_type_ids']:
#         col_to_index = 'edge_type_id'
#         index_grp_name = 'indices/edge_type_to_index'
#     else:
#         raise ValueError('Unknown edges parameter {}'.format(index_type))
#
#     return col_to_index, index_grp_name


def resort_edges(input_edges_path, output_edges_path, edges_population, sort_by, sort_model_properties=True):
    assert(os.path.exists(input_edges_path))

    output_h5 = h5py.File(output_edges_path, 'w')

    with h5py.File(input_edges_path, 'r') as input_h5:
        in_pop_grp = input_h5[edges_population]

        out_pop_grp = output_h5.create_group(edges_population)

        sort_vals = in_pop_grp[sort_by][()]
        sort_order = np.argsort(sort_vals)
        # TODO: Check if already sorted

        for col_name in ['source_node_id', 'target_node_id', 'edge_type_id', 'edge_group_id']:  # , 'edge_group_index']:
            col_type = in_pop_grp[col_name].dtype
            col_vals = in_pop_grp[col_name][()]
            sorted_col_vals = col_vals[sort_order]

            out_pop_grp.create_dataset(col_name, data=sorted_col_vals, dtype=col_type)

        sorted_group_indx = in_pop_grp['edge_group_index'][()][sort_order]
        group_index_dtype = in_pop_grp['edge_group_index'].dtype
        group_ids = np.unique(in_pop_grp['edge_group_id'][()])
        for group_id in group_ids:
            out_model_grp = out_pop_grp.create_group(str(group_id))
            model_cols = [n for n, h5_obj in in_pop_grp[str(group_id)].items() if isinstance(h5_obj, h5py.Dataset)]
            group_id_mask = np.argwhere(out_pop_grp['edge_group_id'][()] == group_id).flatten()

            new_index_order = sorted_group_indx[group_id_mask]

            for col_name in model_cols:
                prop_data = in_pop_grp[str(group_id)][col_name][()][new_index_order]
                out_model_grp.create_dataset(col_name, data=prop_data)

            sorted_group_indx[group_id_mask] = np.arange(0, len(group_id_mask), dtype=group_index_dtype)

        out_pop_grp.create_dataset('edge_group_index', data=sorted_group_indx)
