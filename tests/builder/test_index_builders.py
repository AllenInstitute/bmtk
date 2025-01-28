import pytest
import tempfile
import h5py
import numpy as np

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version, check_magic, get_version
from bmtk.builder.index_builders import create_index_on_disk, create_index_in_memory


def _check_index(h5, index_col, id_to_range_col, range_to_edge_col, n_edges):
    assert(index_col in h5)
    assert(id_to_range_col in h5)
    assert(range_to_edge_col in h5)
    id_lu = h5[id_to_range_col][()]
    edge_idx_lu = h5[range_to_edge_col][()]
    edge_count = 0
    for i, id_range in enumerate(id_lu):
        if id_range[1] - id_range[0] > 0:
            for range_idx in range(id_range[0], id_range[1]):
                for edge_idx in range(edge_idx_lu[range_idx][0], edge_idx_lu[range_idx][1]):
                    assert(h5[index_col][edge_idx] == i)
                    edge_count += 1

    assert(edge_count == n_edges)


@pytest.mark.parametrize('indexer_func,indexer_args', [
    (create_index_in_memory, {}),
    (create_index_on_disk, {'max_edge_reads': 10})
])
def test_create_index(indexer_func, indexer_args):
    tmp_edges_h5 = tempfile.NamedTemporaryFile(suffix='.h5')
    n_edges = 20
    source_node_ids = np.tile([0, 1, 3, 4], 5)
    target_node_ids = np.repeat([73, 72, 52, 4], 5)
    edge_type_ids = np.random.choice([100, 102, 103], size=n_edges, replace=True)
    edge_group_ids = np.full(n_edges, fill_value=0)
    edge_group_indices = np.arange(0, n_edges, dtype=int)

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
        h5.create_group('/edges/a_to_b/0')

    # Add target_node_id index
    indexer_func(
        edges_file=tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        index_type='target_node_id',
        **indexer_args
    )
    with h5py.File(tmp_edges_h5.name, 'r') as h5:
        _check_index(
            h5=h5,
            index_col='/edges/a_to_b/target_node_id',
            id_to_range_col='/edges/a_to_b/indices/target_to_source/node_id_to_range',
            range_to_edge_col='/edges/a_to_b/indices/target_to_source/range_to_edge_id',
            n_edges=n_edges
        )

    # Add source_node_id index
    indexer_func(
        edges_file=tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        index_type='source_node_id',
        **indexer_args
    )
    with h5py.File(tmp_edges_h5.name, 'r') as h5:
        _check_index(
            h5=h5,
            index_col='/edges/a_to_b/source_node_id',
            id_to_range_col='/edges/a_to_b/indices/source_to_target/node_id_to_range',
            range_to_edge_col='/edges/a_to_b/indices/source_to_target/range_to_edge_id',
            n_edges=n_edges
        )

    # Add edge_type_id index
    indexer_func(
        edges_file=tmp_edges_h5.name,
        edges_population='/edges/a_to_b',
        index_type='edge_type_id',
        **indexer_args
    )
    with h5py.File(tmp_edges_h5.name, 'r') as h5:
        _check_index(
            h5=h5,
            index_col='/edges/a_to_b/edge_type_id',
            id_to_range_col='/edges/a_to_b/indices/edge_type_to_index/node_id_to_range',
            range_to_edge_col='/edges/a_to_b/indices/edge_type_to_index/range_to_edge_id',
            n_edges=n_edges
        )


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.DEBUG)

    test_create_index(create_index_in_memory, {})
    test_create_index(create_index_on_disk, {'max_edge_reads': 10})
