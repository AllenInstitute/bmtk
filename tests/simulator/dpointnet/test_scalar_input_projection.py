"""Scalar named-input projection with unchanged source and weight adjoints."""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from test_adaptive_routing import (
    N_TYPES, build, check_against_oracle, fp64_oracle, make_source, native,
    run_sweep, skip_unlaunchable, sweep_nodes,
)


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("post,types,expected", [
    ([0, 0, 2], [1, 1, 3], [1, 0, 3, 0]),
    ([0, 0, 2], [0, 1, 3], None),
    ([], [], None),
])
def test_host_target_types(post, types, expected):
    result = ops._scalar_target_type_map(np.array(post, np.int64), np.array(types, np.int64), 4)
    if expected is None:
        assert result is None
    else:
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.int64 and not result.flags.writeable


@pytest.mark.parametrize("value", [None, "true", 0, 1])
def test_cell_requires_explicit_boolean_or_auto(value):
    from test_precision_credit import make_cell
    with pytest.raises(ValueError, match="use_scalar_input_projection must be true or false"):
        make_cell("nest", use_scalar_input_projection=value)


def test_cell_auto_keeps_ordinary_projection_without_adaptive_sweep():
    """``"auto"`` never raises: without the adaptive joint sweep it resolves to False."""
    from test_precision_credit import make_cell
    cell = make_cell("nest", use_scalar_input_projection="auto")
    try:
        assert cell.native_execution_options["use_scalar_input_projection"] is False
        assert cell._scalar_input_source is None
    finally:
        cell.close_fused_cuda()


def test_cell_requires_adaptive_combined_sweep():
    from test_precision_credit import make_cell
    with pytest.raises(ValueError, match="requires the adaptive combined"):
        make_cell("nest", use_scalar_input_projection=True)


@pytest.mark.parametrize("source", [True, 0, -1, 2, "1"])
def test_invalid_source_index(source):
    conns = [{"n_sources": 2, "n_edges": 4, "n_post": 4},
             {"n_sources": 2, "n_edges": 4, "n_post": 4,
              "scalar_target_types": np.zeros(4, np.int64)}]
    with pytest.raises((TypeError, ValueError), match="scalar_source|named input"):
        ops.adaptive_routing_plan(conns, scalar_source=source)


def test_old_library_rejects_scalar_capability(monkeypatch):
    monkeypatch.setattr(ops, "adaptive_routing_available", lambda: True)
    monkeypatch.setattr(ops, "_operator_accepts", lambda _name, _attributes: False)
    assert not ops.scalar_input_projection_available()


def scalar_sources(batch, dtype, n_post):
    rng = np.random.default_rng(4187)
    sources = [make_source(kind, rng, n_post, batch, dtype)
               for kind in ("recurrent", "lgn", "bkg")]
    indices, _, weights, spikes = sources[1]
    sources[1] = indices, indices[:, 0] % N_TYPES, weights, spikes
    basis = rng.uniform(-0.9, 1.3, (N_TYPES, 4)).astype(dtype)
    dy = rng.uniform(-0.5, 0.5, (batch, n_post, 4)).astype(np.float32)
    return sources, basis, dy


def connect(sources, n_post):
    conns = []
    for index, (indices, types, _, spikes) in enumerate(sources):
        if index == 1:
            conn = ops.build_csr_connectivity(
                indices, types, spikes.shape[1], n_post, N_TYPES,
                build_compact_pairs=True, sort_by_target=True, host_lifecycle=True,
                uint8_types=True, build_scalar_target_types=True,
            )
        else:
            conn = build(indices, types, spikes.shape[1], n_post)
        conns.append(conn)
    return conns


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("index", [tf.uint32, tf.int64])
@pytest.mark.parametrize("tile", [256, 2048, 5120])
def test_gpu_scalar_value_and_all_source_vjps(monkeypatch, dtype, batch, index, tile):
    native()
    assert ops.scalar_input_projection_available()
    skip_unlaunchable(tile, index)
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *_args: index)
    n_post = 10243 if tile == 5120 else 4099
    sources, basis, dy = scalar_sources(batch, dtype, n_post)
    conns = connect(sources, n_post)
    try:
        reference = fp64_oracle(sources, basis, dy)
        for scalar in (None, 1):
            out, gradients, layouts, run = run_sweep(
                sources, conns, basis, dy, adaptive=True, tile=tile, gather=True,
                n_post=n_post, scalar_source=scalar,
            )
            check_against_oracle(out, gradients, layouts, reference, dtype)
            assert np.any(gradients[1][:, -1] != 0), "Silent input rows must retain source credit."
            node, = sweep_nodes(run)
            assert node.attr["scalar_source"].i == (-1 if scalar is None else 1)
            if scalar is not None:
                assert node.attr["adaptive_traversal"].list.i[1] == 2
                assert len(node.attr["scalar_target_types"].list.i) == n_post
    finally:
        for conn in conns:
            conn.close()


@pytest.mark.parametrize("fault", ["mismatch", "length", "range", "negative"])
def test_gpu_invalid_live_map_rejects(fault):
    native()
    sources, basis, dy = scalar_sources(1, np.float32, 67)
    conns = connect(sources, 67)
    try:
        target_types = conns[1]["scalar_target_types"].copy()
        if fault == "mismatch":
            post = sources[1][0][0, 0]
            target_types[post] = (target_types[post] + 1) % N_TYPES
        elif fault == "length":
            target_types = target_types[:-1]
        elif fault == "range":
            target_types[0] = N_TYPES
        else:
            target_types[0] = -1
        invalid_conns = [conns[0], {**dict(conns[1]), "scalar_target_types": target_types}, conns[2]]
        with pytest.raises(tf.errors.InvalidArgumentError, match="scalar_target_types"):
            run_sweep(
                sources, invalid_conns, basis, dy, adaptive=True, tile=256,
                gather=True, n_post=67, scalar_source=1,
            )
    finally:
        for conn in conns:
            conn.close()
