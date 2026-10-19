"""Independent values/VJPs for cooperative short rows, fanout walks and target gathers."""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_cooperative_selector_requires_boolean(value):
    with pytest.raises(ValueError, match="true or false"):
        ops.fused_combined_currents([], [], [], None, [], cooperative_rows=value)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_fanout_selector_requires_boolean(value):
    with pytest.raises(ValueError, match="fanout_rows must be true or false"):
        ops.fused_combined_currents([], [], [], None, [], fanout_rows=value)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_gather_selector_requires_boolean(value):
    with pytest.raises(ValueError, match="gather_sources must be true or false"):
        ops.fused_combined_currents([], [], [], None, [], gather_sources=value)


def test_gather_eligibility_is_mean_in_degree():
    assert ops.combined_gather_eligible({"n_edges": 4 * 100, "n_post": 100})
    assert ops.combined_gather_eligible({"n_edges": 16 * 100, "n_post": 100})
    assert not ops.combined_gather_eligible({"n_edges": 16 * 100 + 1, "n_post": 100})
    assert not ops.combined_gather_eligible({"n_edges": 0, "n_post": 100})


def _fanout_cell(**options):
    from test_precision_credit import make_cell

    policy = tf.keras.mixed_precision.global_policy()
    try:
        return make_cell(mode="nest", selective=False, noise=True, **options)
    finally:
        tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_cell_fanout_selector_requires_boolean(value):
    with pytest.raises(ValueError, match="use_fanout_row_projection"):
        _fanout_cell(use_fanout_row_projection=value)


def test_cell_fanout_requires_combined_route():
    with pytest.raises(ValueError, match="requires the combined"):
        _fanout_cell(use_fanout_row_projection=True, use_combined_current_sweep=False)


def test_cell_fanout_requires_native_capability(monkeypatch):
    monkeypatch.setattr(ops, "fanout_combined_sweep_available", lambda: False)
    with pytest.raises(RuntimeError, match="Rebuild both"):
        _fanout_cell(use_fanout_row_projection=True, use_combined_current_sweep=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_cell_gather_selector_requires_boolean(value):
    with pytest.raises(ValueError, match="use_target_gather_projection"):
        _fanout_cell(use_target_gather_projection=value)


def test_cell_gather_requires_combined_route():
    with pytest.raises(ValueError, match="requires the combined"):
        _fanout_cell(use_target_gather_projection=True, use_combined_current_sweep=False)


def test_cell_gather_requires_native_capability(monkeypatch):
    monkeypatch.setattr(ops, "gather_combined_sweep_available", lambda: False)
    with pytest.raises(RuntimeError, match="Rebuild both"):
        _fanout_cell(use_target_gather_projection=True, use_combined_current_sweep=True)


def test_old_operator_abi_default_is_compatible_explicit_opt_in_errors(monkeypatch):
    from types import SimpleNamespace

    called = []
    def old_operator(spikes, weights, metadata, basis, *, Tindex, n_post, n_edges, n_pairs, target_tile_size):
        called.append(target_tile_size)
        return tf.zeros((1, 4), tf.float32)
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(dpointnet_combined_sweep=old_operator))
    monkeypatch.setattr(ops, "fused_cuda_available", lambda: True)
    assert ops.combined_sweep_available()
    assert not ops.cooperative_combined_sweep_available()
    conn = {
        "n_post": 1, "n_sources": 1, "n_synapse_types": 1, "n_edges": 1,
        "n_pairs": 0, "sorted_by_target": True, "index_dtype": "uint32", "metadata_handle": None,
    }
    arguments = ([tf.ones((1, 1))] * 2, [tf.ones((1,))] * 2, [conn] * 2,
                 tf.ones((1, 4)), [tf.zeros((1, 4))] * 2)
    np.testing.assert_array_equal(ops.fused_combined_currents(*arguments), np.zeros((1, 4)))
    assert called == [2048]
    with pytest.raises(RuntimeError, match="Rebuild both"):
        ops.fused_combined_currents(*arguments, cooperative_rows=True)
    assert not ops.fanout_combined_sweep_available()
    with pytest.raises(RuntimeError, match="Rebuild both"):
        ops.fused_combined_currents(*arguments, fanout_rows=True)
    assert not ops.gather_combined_sweep_available()
    with pytest.raises(RuntimeError, match="Rebuild both"):
        ops.fused_combined_currents(*arguments, gather_sources=True)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
def test_short_rows_tail_empty_segments_and_canonical_vjps(
    dtype, batch, index_dtype, monkeypatch, high_fanout=False, fanout_rows=False,
    target_tile_size=2048, gather_sources=False, background=False,
):
    if not ops.cooperative_combined_sweep_available():
        pytest.skip("Rebuilt cooperative sweep required")
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.as_dtype(index_dtype))
    rng = np.random.default_rng(719)
    targets = 4109
    basis = rng.integers(-3, 4, (2, 4)).astype(dtype) / 4
    sources, expected = [], np.zeros((batch, targets, 4), np.float64)
    references_z, references_w = [], []
    upstream = rng.integers(-3, 4, (batch, targets, 4)).astype(np.float32) / 4
    layouts = [(47, False, False), (777 if high_fanout else 3, True, False)]
    if background:
        # BKG-like: 100 sources, every target receives exactly four edges, one
        # target repeats a source, Poisson-count activity.
        layouts.append((100, True, True))
    for width, repeated, bkg_like in layouts:
        rows = []
        if bkg_like:
            for post in range(targets):
                picks = rng.choice(width, size=4, replace=False)
                rows.extend((post, int(row)) for row in picks)
            rows.append((5, rows[5 * 4][1]))
        else:
            for row in range(width):
                count = 400 if high_fanout and repeated else 17
                posts = rng.choice(targets, size=count if row % 3 else 0, replace=False)
                rows.extend((post, row) for post in posts)
        indices = np.asarray(rows, dtype=np.int64).reshape(-1, 2)
        if repeated and not bkg_like:
            indices = np.concatenate((indices, indices))
        types = rng.integers(2, size=len(indices))
        weights = rng.integers(-4, 5, size=len(indices)).astype(dtype) / (64 if high_fanout else 4)
        activity = rng.integers(0, 4, (batch, width)).astype(dtype)
        activity[:, ::4] = 0
        activity[:, 1] = 0
        if bkg_like:
            activity = rng.poisson(.25, (batch, width)).astype(dtype)
        elif high_fanout and repeated:
            activity.fill(1)
        connection = ops.build_csr_connectivity(
            indices, types, width, targets, 2, build_compact_pairs=True, sort_by_target=True,
        )
        assert connection["has_repeated_targets"] is repeated
        if bkg_like:
            assert ops.combined_gather_eligible(connection)
        if high_fanout and repeated and not bkg_like:
            assert connection["n_edges"] // connection["n_sources"] >= 512
            assert np.all(np.count_nonzero(activity, axis=1) > 512)
        products = weights[:, None].astype(np.float64) * basis[types].astype(np.float64)
        projected = np.sum(upstream[:, indices[:, 0]].astype(np.float64) * basis[types], axis=-1)
        dz = np.zeros((batch, width), np.float64)
        for sample in range(batch):
            np.add.at(
                expected[sample], indices[:, 0],
                np.maximum(activity[sample, indices[:, 1]], 0)[:, None] * products,
            )
            np.add.at(dz[sample], indices[:, 1], projected[sample] * weights)
        references_z.append(dz)
        references_w.append(np.sum(activity[:, indices[:, 1]] * projected, axis=0))
        sources.append((activity, weights, connection))
    try:
        with tf.device("/GPU:0"):
            spikes = [tf.Variable(value) for value, _, _ in sources]
            masters = [tf.Variable(value.astype(np.float32)) for _, value, _ in sources]
            weights = [
                ops.reorder_csr_values(tf.cast(master, tf.as_dtype(dtype)), source[2])
                for master, source in zip(masters, sources)
            ]
            coefficients = tf.constant(basis)

            @tf.function
            def evaluate():
                with tf.GradientTape() as tape:
                    proxies = [
                        ops.fused_spike_currents(
                            z, master, weight, source[2], coefficients, targets, True,
                            use_packed_sm120_backward=False, vjp_only=True,
                        ) for z, master, weight, source in zip(spikes, masters, weights, sources)
                    ]
                    value = ops.fused_combined_currents(
                        spikes, weights, [source[2] for source in sources], coefficients, proxies,
                        cooperative_rows=True,
                        fanout_rows=fanout_rows,
                        gather_sources=gather_sources,
                        target_tile_size=target_tile_size,
                    )
                    objective = tf.reduce_sum(tf.cast(value, tf.float32) * upstream.reshape(-1, 4))
                return value, tape.gradient(objective, spikes + masters)
            value, gradients = evaluate()
        np.testing.assert_allclose(value.numpy().reshape(expected.shape), expected, rtol=1e-3, atol=1e-3)
        for actual, reference in zip(gradients, references_z + references_w):
            np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
        assert np.any(gradients[0].numpy()[:, 4]), "Silent-source credit remains live"
        nodes = evaluate.get_concrete_function().graph.as_graph_def().node
        sweep = [node for node in nodes if node.op == "DpointnetCombinedSweep"]
        assert len(sweep) == 1
        assert list(sweep[0].attr["cooperative_rows"].list.b) == [True, False] + ([False] if background else [])
        if fanout_rows:
            assert sweep[0].attr["fanout_rows"].b
            assert sweep[0].attr["target_tile_size"].i == target_tile_size
        gathered = list(sweep[0].attr["gather_sources"].list.b) if "gather_sources" in sweep[0].attr else []
        if gather_sources:
            expected_gather = [ops.combined_gather_eligible(source[2]) for source in sources]
            assert any(expected_gather) and gathered == expected_gather
        else:
            assert gathered == []
    finally:
        for _, _, connection in sources:
            connection.close()


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
def test_high_fanout_multiple_rounds_and_cross_row_warp_boundaries(dtype, index_dtype, monkeypatch):
    test_short_rows_tail_empty_segments_and_canonical_vjps(
        dtype, 3, index_dtype, monkeypatch, high_fanout=True,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("target_tile_size", [256, 2048, 5120])
def test_fanout_descriptors_preserve_repeated_targets_and_vjps(
    dtype, index_dtype, batch, target_tile_size, monkeypatch,
):
    if not ops.fanout_combined_sweep_available():
        pytest.skip("Rebuilt fanout sweep required")
    test_short_rows_tail_empty_segments_and_canonical_vjps(
        dtype, batch, index_dtype, monkeypatch, high_fanout=True, fanout_rows=True,
        target_tile_size=target_tile_size,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("target_tile_size", [256, 2048, 5120])
def test_target_gather_preserves_values_repeats_and_vjps(
    dtype, index_dtype, batch, target_tile_size, monkeypatch,
):
    if not ops.gather_combined_sweep_available():
        pytest.skip("Rebuilt gather sweep required")
    # Low in-degree sources (including a BKG-like one with a repeated source
    # on one target) are gathered; the high-fanout source keeps its walk.
    test_short_rows_tail_empty_segments_and_canonical_vjps(
        dtype, batch, index_dtype, monkeypatch, high_fanout=True, fanout_rows=True,
        target_tile_size=target_tile_size, gather_sources=True, background=True,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_background_source_without_gather_matches_oracle(dtype, monkeypatch):
    if not ops.fanout_combined_sweep_available():
        pytest.skip("Rebuilt fanout sweep required")
    test_short_rows_tail_empty_segments_and_canonical_vjps(
        dtype, 4, "uint32", monkeypatch, fanout_rows=True, background=True,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("active_rows", [1024, 1100])
def test_fanout_wide_tile_round_with_every_row_nonempty(dtype, active_rows, monkeypatch):
    """At tile5120 the fanout walk runs1,024-thread rounds. When every row of a
    round has a segment in the tile, the packed scan's count field must hold
    1,024 (11 bits); a10-bit field wraps to zero and corrupts the offsets."""
    if not ops.fanout_combined_sweep_available():
        pytest.skip("Rebuilt fanout sweep required")
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.uint32)
    rng = np.random.default_rng(1024)
    targets = 5120  # one tile: every edge of every row lands in it
    basis = rng.integers(-3, 4, (2, 4)).astype(dtype) / 4
    width = active_rows
    # Each row has >=600 edges in the tile so the source is high-fanout and
    # every row's segment is nonempty; a few repeated targets per row.
    rows = []
    for row in range(width):
        posts = rng.choice(targets, size=600, replace=False)
        rows.extend((post, row) for post in posts)
        rows.extend((int(posts[0]), row) for _ in range(3))
    indices = np.asarray(rows, dtype=np.int64)
    types = rng.integers(2, size=len(indices))
    weights = rng.integers(-4, 5, size=len(indices)).astype(dtype) / 64
    activity = np.ones((1, width), dtype)
    connection = ops.build_csr_connectivity(
        indices, types, width, targets, 2, build_compact_pairs=True, sort_by_target=True,
    )
    assert connection["n_edges"] // connection["n_sources"] >= 512
    # A second tiny, inactive source keeps the two-source contract.
    other = ops.build_csr_connectivity(
        np.array([[0, 0]], np.int64), np.array([0]), 1, targets, 2,
        build_compact_pairs=True, sort_by_target=True,
    )
    products = weights[:, None].astype(np.float64) * basis[types].astype(np.float64)
    expected = np.zeros((targets, 4), np.float64)
    np.add.at(expected, indices[:, 0], products)
    try:
        with tf.device("/GPU:0"):
            spikes = [tf.constant(activity), tf.zeros((1, 1), dtype)]
            values = [
                ops.reorder_csr_values(tf.constant(weights), connection),
                tf.zeros((1,), dtype),
            ]
            coefficients = tf.constant(basis)
            proxies = [tf.zeros((targets, 4), dtype)] * 2
            result = ops.fused_combined_currents(
                spikes, values, [connection, other], coefficients, proxies,
                cooperative_rows=True, fanout_rows=True, target_tile_size=5120,
            )
        np.testing.assert_allclose(
            result.numpy().reshape(targets, 4), expected,
            rtol=2e-3 if dtype == np.float16 else 1e-5, atol=2e-3 if dtype == np.float16 else 1e-5,
        )
    finally:
        connection.close()
        other.close()
