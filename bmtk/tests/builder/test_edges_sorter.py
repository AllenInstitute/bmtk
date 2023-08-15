import pytest
import tempfile
import h5py
import numpy as np
import logging

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version, check_magic, get_version
from bmtk.builder.edges_sorter import quicksort_edges, external_merge_sort


def _check_edges(h5, n_edges):
    assert(check_magic(h5))
    assert(get_version(h5))

    assert(len(h5['/edges/a_to_b/source_node_id']) == n_edges)
    assert(h5['/edges/a_to_b/source_node_id'].attrs['node_population'] == 'a')
    assert(len(h5['/edges/a_to_b/target_node_id']) == n_edges)
    assert(h5['/edges/a_to_b/target_node_id'].attrs['node_population'] == 'b')
    assert(len(h5['/edges/a_to_b/edge_type_id']) == n_edges)
    assert(len(h5['/edges/a_to_b/edge_group_id']) == n_edges)
    assert(len(h5['/edges/a_to_b/edge_group_index']) == n_edges)
    for i in range(n_edges):
        grp_id = h5['/edges/a_to_b/edge_group_id'][i]
        grp_indx = h5['/edges/a_to_b/edge_group_index'][i]
        assert (h5['/edges/a_to_b/source_node_id'][i] == h5['/edges/a_to_b'][str(grp_id)]['src_id'][grp_indx])
        assert (h5['/edges/a_to_b/target_node_id'][i] == h5['/edges/a_to_b'][str(grp_id)]['trg_id'][grp_indx])
        assert (h5['/edges/a_to_b/edge_type_id'][i] == h5['/edges/a_to_b'][str(grp_id)]['et_id'][grp_indx])


@pytest.mark.parametrize('sort_func,sort_params', [
    (quicksort_edges, {}),
    (external_merge_sort, {'sort_model_properties': False, 'n_chunks': 5}),
    (external_merge_sort, {'sort_model_properties': True, 'n_chunks': 5}),
])
def test_sort(sort_func, sort_params):
    tmp_edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    source_node_ids = np.tile([0, 1], 5)
    target_node_ids = np.arange(20, 0, -2, dtype=int)
    edge_type_ids = np.repeat([103, 100, 104, 101, 102], 2)
    edge_group_ids = np.repeat([1, 0], 5)
    edge_group_indices = np.tile(range(5), 2)
    n_edges = 10
    with h5py.File(tmp_edges_h5.name, 'w') as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)

        h5.create_dataset('/edges/a_to_b/source_node_id', data=source_node_ids)
        h5['/edges/a_to_b/source_node_id'].attrs['node_population'] = 'a'
        h5.create_dataset('/edges/a_to_b/target_node_id', data=target_node_ids)
        h5['/edges/a_to_b/target_node_id'].attrs['node_population'] = 'b'
        h5.create_dataset('/edges/a_to_b/edge_group_id', data=edge_group_ids)
        h5.create_dataset('/edges/a_to_b/edge_group_index', data=edge_group_indices)
        h5.create_dataset('/edges/a_to_b/edge_type_id', data=edge_type_ids)

        for grp_id in np.unique(h5['/edges/a_to_b/source_node_id'][()]):
            model_grp = h5.create_group('/edges/a_to_b/{}'.format(grp_id))
            grp_mask = edge_group_ids == grp_id
            model_grp.create_dataset('src_id', data=source_node_ids[grp_mask])
            model_grp.create_dataset('trg_id', data=target_node_ids[grp_mask])
            model_grp.create_dataset('et_id', data=edge_type_ids[grp_mask])

    # Sort by source_node_id
    sorted_tmp_edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    sort_func(
        input_edges_path=tmp_edges_h5.name,
        output_edges_path=sorted_tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        sort_by='source_node_id',
        **sort_params
    )
    with h5py.File(sorted_tmp_edges_h5.name, 'r') as h5:
        assert(np.all(np.diff(h5['/edges/a_to_b/source_node_id'][()]) >= 0))
        _check_edges(h5, n_edges=n_edges)

    # Sort by target_node_id
    sorted_tmp_edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    sort_func(
        input_edges_path=tmp_edges_h5.name,
        output_edges_path=sorted_tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        sort_by='target_node_id',
        **sort_params
    )
    with h5py.File(sorted_tmp_edges_h5.name, 'r') as h5:
        assert(np.all(np.diff(h5['/edges/a_to_b/target_node_id'][()]) >= 0))
        _check_edges(h5, n_edges=n_edges)

    # Sort by edge_type_id
    sorted_tmp_edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    sort_func(
        input_edges_path=tmp_edges_h5.name,
        output_edges_path=sorted_tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        sort_by='edge_type_id',
        **sort_params
    )
    with h5py.File(sorted_tmp_edges_h5.name, 'r') as h5:
        assert(np.all(np.diff(h5['/edges/a_to_b/edge_type_id'][()]) >= 0))
        _check_edges(h5, n_edges=n_edges)

    # Sort by edge_group_id
    sorted_tmp_edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    sort_func(
        input_edges_path=tmp_edges_h5.name,
        output_edges_path=sorted_tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        sort_by='edge_group_id',
        **sort_params
    )
    with h5py.File(sorted_tmp_edges_h5.name, 'r') as h5:
        assert(np.all(np.diff(h5['/edges/a_to_b/edge_group_id'][()]) >= 0))
        _check_edges(h5, n_edges=n_edges)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # test_sort(quicksort_edges, {})
    test_sort(external_merge_sort, {'sort_model_properties': False, 'n_chunks': 5})
    # test_sort(external_merge_sort, {'sort_model_properties': True, 'n_chunks': 5})
