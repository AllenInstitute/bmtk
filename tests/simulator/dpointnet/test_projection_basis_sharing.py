"""Exact coefficient sharing without changing edge identity or mutable-basis VJPs."""

from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops


def metadata(connectivity):
    return tf.raw_ops.ReadVariableOp(
        resource=connectivity["metadata_handle"],
        dtype=tf.as_dtype(connectivity["index_dtype"]),
    ).numpy()


def test_representatives_preserve_signed_zero_and_nan_payloads():
    first = np.array([0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7FC00001, 0])
    basis = np.column_stack((first.astype(np.uint32).view(np.float32), np.ones(6)))
    basis = basis.astype(np.float32)
    np.testing.assert_array_equal(
        ops._projection_type_representatives(basis, 6), [0, 1, 2, 3, 2, 0]
    )


def test_representatives_do_not_limit_synapse_types_to_one_byte():
    basis = np.arange(300, dtype=np.float32)[:, None]
    basis[-1] = basis[270]
    representatives = ops._projection_type_representatives(basis, 300)
    np.testing.assert_array_equal(representatives[:299], np.arange(299))
    assert representatives[-1] == 270


@pytest.mark.parametrize(
    "basis", [np.ones((2, 0)), np.ones((3, 4)), np.ones(2), np.ones((2, 4), np.int32)]
)
def test_representatives_reject_incompatible_basis(basis):
    with pytest.raises(ValueError, match="floating matrix"):
        ops._projection_type_representatives(basis, 2)


@pytest.mark.parametrize("sort_by_target", [False, True])
def test_sharing_changes_only_compact_projection_metadata(sort_by_target):
    indices = np.column_stack((np.zeros(6, dtype=np.int64), [2, 0, 1, 0, 2, 1]))
    types = np.arange(6)
    basis = np.array([[1, 2], [3, 4], [1, 2], [5, 6], [3, 4], [1, 2]], np.float32)
    original = ops.build_csr_connectivity(
        indices, types, 3, 1, 6, build_compact_pairs=True, sort_by_target=sort_by_target
    )
    shared = ops.build_csr_connectivity(
        indices, types, 3, 1, 6, build_compact_pairs=True,
        sort_by_target=sort_by_target, projection_basis=basis,
    )
    try:
        assert original["n_pairs"] == 6
        assert shared["n_pairs"] == 3
        original_values, shared_values = metadata(original), metadata(shared)
        prefix = 3 * len(indices) + 4
        np.testing.assert_array_equal(original_values[:prefix], shared_values[:prefix])
        pair_ids = shared_values[prefix:prefix + len(indices)]
        pair_posts = shared_values[prefix + len(indices):prefix + len(indices) + 3]
        pair_types = shared_values[-3:]
        np.testing.assert_array_equal(pair_posts, 0)
        csr_types = original_values[len(indices):2 * len(indices)]
        np.testing.assert_array_equal(basis[pair_types[pair_ids]], basis[csr_types])
    finally:
        original.close()
        shared.close()


def test_sharing_requires_compact_pairs():
    with pytest.raises(ValueError, match="build_compact_pairs"):
        ops.build_csr_connectivity(
            [[0, 0]], [0], 1, 1, 1, projection_basis=np.ones((1, 4), np.float32)
        )


@pytest.mark.parametrize("alternate_first", [False, True])
@pytest.mark.parametrize("replacement_rows", [2, 3])
@pytest.mark.parametrize("combined_sweep", [False, True])
def test_recurrent_routing_falls_back_for_replaced_and_alternate_basis(
    monkeypatch, alternate_first, replacement_rows, combined_sweep
):
    basis = tf.constant([[1.0], [1.0]])
    shared = {"projection_basis_shared": True, "n_synapse_types": 2}
    canonical = {"projection_basis_shared": False}
    cell = SimpleNamespace(
        recurrent_fused_connectivity=shared,
        _recurrent_projection_basis_reference=basis,
        synaptic_basis_weights=basis,
        _unshared_recurrent_fused_connectivity=None,
        recurrent_indices=tf.constant([[0, 0], [0, 0]]),
        syn_ids=tf.constant([0, 1]),
        recurrent_dense_shape=(1, 1),
        _n_neurons=1,
        _use_pair_projection=True,
        _use_forward_run_aggregation=False,
        _use_combined_current_sweep=combined_sweep,
    )
    calls = []

    def build(*args, **kwargs):
        assert "projection_basis" not in kwargs
        calls.append((args, kwargs))
        return canonical

    monkeypatch.setattr(
        "bmtk.simulator.dpointnet.cell_models.glif3_cell.build_csr_connectivity", build
    )
    select = GLIF3Cell._recurrent_projection_connectivity
    assert select(cell) is shared
    assert not calls
    if alternate_first:
        assert select(cell, alternate_basis=True) is canonical
    cell.synaptic_basis_weights = tf.Variable(
        np.arange(replacement_rows, dtype=np.float32)[:, None], trainable=False
    )
    assert select(cell) is canonical
    cell.synaptic_basis_weights.assign(cell.synaptic_basis_weights * 2)
    assert select(cell) is canonical
    cell.synaptic_basis_weights = basis
    assert select(cell) is shared
    assert select(cell, alternate_basis=True) is canonical
    assert len(calls) == 1
    assert calls[0][0][4] == 2
    assert calls[0][1]["sort_by_target"] is combined_sweep


@pytest.mark.skipif(
    not ops.fused_recurrent_accumulation_available(), reason="SM86 accumulation op required"
)
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [3, 16, 32])
def test_native_shared_projection_value_and_vjp_are_bitwise_identical(dtype, batch):
    from test_recurrent_accumulation import raw_gradient

    rng = np.random.default_rng(9241)
    indices = np.column_stack((rng.integers(7, size=513), rng.integers(5, size=513)))
    types = rng.integers(6, size=len(indices))
    basis = rng.integers(-1, 2, size=(2, 4)).astype(dtype)[[0, 1, 0, 1, 0, 1]]
    spikes = rng.poisson(0.4, size=(batch, 5)).astype(dtype)
    spikes[:, 0] = -1
    spikes[:, 2] = 0
    # Exact integer currents isolate metadata sharing from baseline atomic ordering.
    weights = rng.integers(-1, 2, size=len(indices)).astype(dtype)
    assert np.bincount(indices[:, 0]).max() * np.abs(spikes).max() < 2048
    upstream = rng.normal(size=(batch * 7, 4)).astype(dtype)
    seed = tf.constant(rng.normal(size=len(indices)).astype(np.float32))
    results = []
    for coefficients in (None, basis):
        connectivity = ops.build_csr_connectivity(
            indices, types, 5, 7, 6, build_compact_pairs=True,
            projection_basis=coefficients,
        )
        try:
            z, w, b, dy = map(tf.constant, (spikes, weights, basis, upstream))
            csr = ops.reorder_csr_values(w, connectivity)
            y = ops.fused_spike_currents(
                z, tf.cast(w, tf.float32), csr, connectivity, b, 7,
                compute_spike_gradient=True,
                use_packed_sm120_backward=False,
            )
            dz, dw = raw_gradient(
                connectivity, z, csr, b, dy, seed, scaled=dtype == np.float16
            )
            results.append((y.numpy(), dz.numpy(), dw.numpy(), connectivity["n_pairs"]))
        finally:
            connectivity.close()
    for actual, expected in zip(results[1][:3], results[0][:3]):
        np.testing.assert_array_equal(actual, expected)
    assert results[1][3] < results[0][3]
