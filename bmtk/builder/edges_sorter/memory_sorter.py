import os
import h5py
import logging
import numpy as np

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


def copy_attributes(in_grp, out_grp):
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
            copy_attributes(in_h5_obj, out_grp[in_name])


def quicksort_edges(input_edges_path, output_edges_path, edges_population, sort_by, sort_model_properties=True,
                    **kwargs):
    assert(os.path.exists(input_edges_path))

    output_h5 = h5py.File(output_edges_path, 'w')

    with h5py.File(input_edges_path, 'r') as input_h5:
        in_pop_grp = input_h5[edges_population]

        out_pop_grp = output_h5.create_group(edges_population)

        sort_vals = in_pop_grp[sort_by][()]
        sort_order = np.argsort(sort_vals)
        # TODO: Check if already sorted

        for col_name in ['source_node_id', 'target_node_id', 'edge_type_id', 'edge_group_id']:
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
        copy_attributes(input_h5['/'], output_h5['/'])
