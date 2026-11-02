"""Low-memory host CSR construction equals the straightforward int64 algorithm.

``build_csr_connectivity`` keeps narrow integer inputs narrow, works on edge
chunks and marks compact (target, type) pairs in a dense table. The oracle
below is the original whole-array int64 construction (lexicographic edge
order, ``np.unique`` pairs); every metadata byte, the host edge order and the
derived flags must match it for any input dtype, chunk size and pair path.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops

NAMES = (
    "dpointnet_csr_spike_forward_csr_master", "dpointnet_csr_weight_grad",
    "dpointnet_csr_spike_grad_csr_master",
    "dpointnet_csr_spike_grad_accumulate_csr_master",
    "dpointnet_combined_sweep_csr_master",
)


@pytest.fixture
def host_capability(monkeypatch):
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(**{name: (lambda *, uint8_types=False: uint8_types)
                                                        for name in NAMES}))


def oracle(indices, types, n_sources, n_targets, n_types, *, pairs, sort_by_target, host_lifecycle,
           uint8_types, scalar_types, projection_types=None):
    indices = np.asarray(indices).astype(np.int64)
    types = np.asarray(types).astype(np.int64)
    dtype = ops._csr_index_dtype(len(indices), n_sources, n_targets, n_types).as_numpy_dtype
    pre, post = indices[:, 1], indices[:, 0]
    order = np.lexsort((types, post, pre)) if sort_by_target else np.argsort(pre, kind="stable")
    row_splits = np.concatenate(([0], np.cumsum(np.bincount(pre, minlength=n_sources)))).astype(dtype)
    sorted_pre, sorted_post = pre[order], post[order]
    repeated = bool(sort_by_target and len(pre) > 1 and np.any(
        (sorted_pre[1:] == sorted_pre[:-1]) & (sorted_post[1:] == sorted_post[:-1])))
    post_ids, sorted_types = sorted_post.astype(dtype), types[order].astype(dtype)
    parts = [post_ids, sorted_types, row_splits] + ([] if host_lifecycle else [order.astype(dtype)])
    n_pairs = 0
    if pairs and len(pre):
        pair_types = sorted_types if projection_types is None else projection_types[sorted_types]
        keys = post_ids.astype(np.uint64) * np.uint64(n_types) + pair_types.astype(np.uint64)
        unique, inverse = np.unique(keys, return_inverse=True)
        parts += [inverse.reshape(-1).astype(dtype), (unique // np.uint64(n_types)).astype(dtype),
                  (unique % np.uint64(n_types)).astype(dtype)]
        n_pairs = len(unique)
    if uint8_types:
        wide = np.concatenate([parts[0], *parts[2:]])
        padding = (-len(sorted_types)) % np.dtype(dtype).itemsize
        metadata = np.concatenate((wide.view(np.uint8), sorted_types.astype(np.uint8),
                                   np.zeros(padding, np.uint8))).view(dtype)
    else:
        metadata = np.concatenate(parts)
    scalar = None
    if scalar_types and len(post):
        scalar = np.zeros(n_targets, np.int64)
        np.maximum.at(scalar, post, types)
        if np.any(scalar[post] != types):
            scalar = None
    return dict(metadata=metadata, order=order, repeated=repeated, n_pairs=n_pairs, scalar=scalar)


def network(edges, n_sources, n_targets, n_types, seed, duplicates):
    rng = np.random.default_rng(seed)
    indices = np.column_stack((rng.integers(0, n_targets, edges), rng.integers(0, n_sources, edges)))
    if duplicates and edges > 3:
        indices[edges // 2] = indices[edges // 2 - 1]
    return indices, rng.integers(0, n_types, edges)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64, np.uint32, np.float64])
@pytest.mark.parametrize("type_dtype", [np.uint8, np.int32, np.int64])
@pytest.mark.parametrize("chunk", [7, 1 << 24])
@pytest.mark.parametrize("dense_pairs", [True, False])
@pytest.mark.parametrize("edges,duplicates", [(0, False), (1, False), (41, True), (1000, False), (1000, True)])
def test_build_matches_whole_array_int64_oracle(host_capability, monkeypatch, index_dtype, type_dtype, chunk,
                                                dense_pairs, edges, duplicates):
    monkeypatch.setattr(ops, "_HOST_CHUNK_EDGES", chunk)
    monkeypatch.setattr(ops, "_DENSE_PAIR_TABLE_MIN", 1 << 20 if dense_pairs else 0)
    n_sources, n_targets, n_types = 53, 900, 4
    indices, types = network(edges, n_sources, n_targets, n_types, edges, duplicates)
    indices, types = indices.astype(index_dtype), types.astype(type_dtype)
    for sort_by_target, host_lifecycle, uint8_types, pairs, scalar in (
        (True, True, True, True, True), (True, False, False, True, False),
        (False, True, False, False, True), (True, True, False, False, False),
    ):
        expected = oracle(indices, types, n_sources, n_targets, n_types, pairs=pairs,
                          sort_by_target=sort_by_target, host_lifecycle=host_lifecycle,
                          uint8_types=uint8_types, scalar_types=scalar)
        conn = ops.build_csr_connectivity(
            indices, types, n_sources, n_targets, n_types, build_compact_pairs=pairs,
            sort_by_target=sort_by_target, host_lifecycle=host_lifecycle, uint8_types=uint8_types,
            build_scalar_target_types=scalar,
        )
        try:
            metadata = tf.raw_ops.ReadVariableOp(
                resource=conn["metadata_handle"], dtype=tf.as_dtype(conn["index_dtype"])).numpy()
            assert metadata.dtype == expected["metadata"].dtype
            np.testing.assert_array_equal(metadata, expected["metadata"])
            assert conn["has_repeated_targets"] == expected["repeated"]
            assert conn["n_pairs"] == expected["n_pairs"]
            if host_lifecycle:
                np.testing.assert_array_equal(conn["host_edge_order"], expected["order"])
            if scalar:
                if expected["scalar"] is None:
                    assert conn["scalar_target_types"] is None
                else:
                    np.testing.assert_array_equal(conn["scalar_target_types"], expected["scalar"])
        finally:
            conn.close()


def test_projection_representatives_share_pairs(host_capability):
    indices, types = network(200, 11, 30, 4, 3, True)
    basis = np.array([[1.0, 0.5], [1.0, 0.5], [0.2, 0.1], [0.2, 0.1]], np.float32)
    expected = oracle(indices, types, 11, 30, 4, pairs=True, sort_by_target=True, host_lifecycle=False,
                      uint8_types=False, scalar_types=False, projection_types=np.array([0, 0, 2, 2]))
    conn = ops.build_csr_connectivity(indices.astype(np.int32), types.astype(np.int32), 11, 30, 4,
                                      build_compact_pairs=True, sort_by_target=True, projection_basis=basis)
    try:
        metadata = tf.raw_ops.ReadVariableOp(resource=conn["metadata_handle"], dtype=tf.uint32).numpy()
        np.testing.assert_array_equal(metadata, expected["metadata"])
        assert conn["n_pairs"] == expected["n_pairs"]
    finally:
        conn.close()


def test_integer_inputs_are_not_widened(monkeypatch):
    seen = []
    original = ops._composite_keys

    def record(major, minor, stride):
        seen.append((major.dtype, minor.dtype))
        return original(major, minor, stride)

    monkeypatch.setattr(ops, "_composite_keys", record)
    indices, types = network(100, 13, 17, 4, 5, False)
    conn = ops.build_csr_connectivity(indices.astype(np.int32), types.astype(np.int32), 13, 17, 4,
                                      build_compact_pairs=True, sort_by_target=True)
    conn.close()
    assert seen and all(np.dtype(major).itemsize <= 4 for major, _ in seen)
    with pytest.raises(ValueError, match="int64 range"):
        ops.build_csr_connectivity(np.array([[0, 2**63]], np.uint64), np.array([0]), 2, 2, 1)
    with pytest.raises(ValueError, match="finite integer"):
        ops.build_csr_connectivity(np.array([[0, 0.5]]), np.array([0]), 2, 2, 1)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 1 << 24])
def test_repeated_targets_detected_across_chunk_boundaries(host_capability, monkeypatch, chunk):
    monkeypatch.setattr(ops, "_HOST_CHUNK_EDGES", chunk)
    # CSR order (source 0 first, then targets): targets 0, 1, 2, 3, 3; the repeated
    # (source, target) pair occupies positions 3 and 4.
    repeated = np.array([[3, 0], [0, 0], [3, 0], [2, 0], [1, 0], [4, 1]], np.int32)
    distinct = repeated.copy()
    distinct[2] = [4, 0]
    for indices, expected in ((repeated, True), (distinct, False)):
        conn = ops.build_csr_connectivity(indices, np.zeros(len(indices), np.int32), 2, 5, 1, sort_by_target=True)
        try:
            assert conn["has_repeated_targets"] is expected
        finally:
            conn.close()


def test_input_index_dtype_keeps_small_signed_indices_narrow():
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import _input_index_dtype

    assert _input_index_dtype(np.zeros((3, 2), np.int32), 2**31 - 1) is np.int32
    assert _input_index_dtype(np.zeros((3, 2), np.uint16), 100) is np.int32
    assert _input_index_dtype(np.zeros((3, 2), np.int32), 2**31) is np.int64
    assert _input_index_dtype(np.zeros((3, 2), np.uint32), 100) is np.int64
    assert _input_index_dtype(np.zeros((3, 2), np.int64), 100) is np.int64
    assert _input_index_dtype([[0, 1]], 100) is np.int64
