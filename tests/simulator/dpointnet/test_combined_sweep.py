"""Independent FP64 current and canonical/carrier VJP oracles for joint rounding."""

from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def fixture(dtype, batch, n_post=521, high_fanout=False, wide_rows=False):
    rng = np.random.default_rng(43)
    basis = rng.uniform(-0.75, 1.25, (3, 4)).astype(dtype)
    if high_fanout or wide_rows:
        basis = (rng.integers(-3, 6, (3, 4)) / 4).astype(dtype)
    sources = []
    first = (777, 400001) if wide_rows else (19, 17001 if high_fanout else 1701)
    for width, edges in (first, (11, 701), (3, 151)):
        indices = np.column_stack((rng.integers(n_post, size=edges), rng.integers(width, size=edges)))
        types = rng.integers(len(basis), size=edges)
        weights = rng.uniform(-1, 1, edges).astype(dtype)
        if high_fanout or wide_rows:
            weights = (rng.integers(-4, 5, edges) / 4).astype(dtype)
        spikes = rng.integers(0, 4, (batch, width)).astype(dtype)
        spikes[:, 0] = 0
        if wide_rows and width == 777:
            assert np.all(np.count_nonzero(spikes, axis=1) > 512)
        if width == 3:
            spikes.fill(0)
        conn = ops.build_csr_connectivity(
            indices, types, width, n_post, len(basis),
            build_compact_pairs=True, sort_by_target=True,
        )
        sources.append((indices, types, weights, spikes, conn))
    return basis, sources


@pytest.mark.parametrize("dtype,batch", [
    (np.float16, 1), (np.float16, 3), (np.float16, 32),
    (np.float32, 1), (np.float32, 3), (np.float32, 32),
])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
@pytest.mark.parametrize("n_post", [521, 4099])
def test_combined_accuracy_and_all_canonical_vjps(
    dtype, batch, compiled, index_dtype, n_post, monkeypatch, target_tile_size=2048,
    high_fanout=False, wide_rows=False,
):
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.as_dtype(index_dtype))
    basis, sources = fixture(dtype, batch, n_post, high_fanout, wide_rows)
    rng = np.random.default_rng(17)
    upstream = rng.uniform(-1, 1, (batch, n_post, 4)).astype(dtype)
    if high_fanout or wide_rows:
        # Binary-exact operands isolate traversal from long-sum VJP rounding.
        upstream = (rng.integers(-4, 5, (batch, n_post, 4)) / 4).astype(dtype)
    reference = np.zeros((batch, n_post, 4), np.float64)
    derivative_z, derivative_w = [], []
    for indices, types, weights, spikes, _ in sources:
        products = weights.astype(np.float64)[:, None] * basis[types].astype(np.float64)
        active = np.maximum(spikes.astype(np.float64), 0)
        for b in range(batch):
            np.add.at(reference[b], indices[:, 0], active[b, indices[:, 1], None] * products)
        projected = np.sum(upstream.astype(np.float64)[:, indices[:, 0]] * basis[types], axis=-1)
        dw = np.sum(spikes.astype(np.float64)[:, indices[:, 1]] * projected, axis=0)
        dz = np.zeros_like(spikes, dtype=np.float64)
        for b in range(batch):
            np.add.at(dz[b], indices[:, 1], projected[b] * weights)
        derivative_z.append(dz)
        derivative_w.append(dw)
    try:
        with tf.device("/GPU:0"):
            zs = [tf.Variable(source[3]) for source in sources]
            masters = [tf.Variable(source[2].astype(np.float32)) for source in sources]
            csr = [ops.reorder_csr_values(tf.cast(w, zs[0].dtype), source[4])
                   for w, source in zip(masters, sources)]
            coefficients = tf.constant(basis)

            def calculate():
                with tf.GradientTape() as tape:
                    proxies = [
                        ops.fused_spike_currents(
                            z, w, weight, source[4], coefficients, n_post, True,
                            use_packed_sm120_backward=False, vjp_only=True,
                        ) for z, w, weight, source in zip(zs, masters, csr, sources)
                    ]
                    combined = ops.fused_combined_currents(
                        zs, csr, [source[4] for source in sources], coefficients, proxies,
                        target_tile_size=target_tile_size,
                    )
                    loss = tf.reduce_sum(tf.cast(combined, tf.float32) * upstream.reshape(-1, 4))
                gradients = tape.gradient(loss, zs + masters)
                return combined, gradients

            actual, gradients = tf.function(calculate)() if compiled else calculate()
        tolerance = 1e-3 if dtype == np.float16 else 3e-6
        np.testing.assert_allclose(actual.numpy().reshape(reference.shape), reference,
                                   rtol=tolerance, atol=tolerance)
        for actual_grad, expected in zip(gradients, derivative_z + derivative_w):
            np.testing.assert_allclose(actual_grad, expected, rtol=0.015 if dtype == np.float16 else 1e-5,
                                       atol=0.012 if dtype == np.float16 else 5e-6)
        assert np.any(gradients[0].numpy()[:, 0]), "Silent-source recurrent credit must remain live"
    finally:
        for source in sources:
            source[4].close()


@pytest.mark.parametrize("tile", [3072, 4096, 5120])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
def test_wide_tiles_values_and_all_canonical_vjps(tile, dtype, batch, compiled, index_dtype, monkeypatch):
    from bmtk.simulator.dpointnet import acceleration

    architecture = ops._gpu_compute_architecture()
    geometry = acceleration.combined_sweep_launch_geometry(
        architecture, tile, index_bytes=4 if index_dtype == "uint32" else 8,
    )
    if architecture is not None and not geometry["valid"]:
        pytest.skip(f"tile {tile} is not launchable on SM{architecture}: {geometry['reasons']}")
    test_combined_accuracy_and_all_canonical_vjps(
        dtype, batch, compiled, index_dtype, 10243, monkeypatch, target_tile_size=tile,
        high_fanout=True,
    )


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("batch", [1, 3])
def test_balanced_multiple_rounds_values_and_all_canonical_vjps(dtype, batch, monkeypatch):
    test_combined_accuracy_and_all_canonical_vjps(
        dtype, batch, True, "uint32", 10243, monkeypatch, wide_rows=True,
    )


def test_balanced_aggregation_keeps_disjoint_row_runs_separate():
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    indices = np.column_stack((
        np.r_[np.zeros(500), np.ones(11), 0, np.ones(512)],
        np.r_[np.zeros(511), np.ones(513)],
    )).astype(np.int64)
    conn = ops.build_csr_connectivity(
        indices, np.zeros(1024), 2, 2, 1, sort_by_target=True,
    )
    try:
        with tf.device("/GPU:0"):
            active = tf.constant([[1., 3.]])
            weights = tf.ones(1024)
            result = ops.fused_combined_currents(
                [active, tf.zeros_like(active)], [weights, weights], [conn, conn],
                tf.ones((1, 4)), [tf.zeros((2, 4))] * 2,
            )
        np.testing.assert_array_equal(result, np.repeat([[503.], [1547.]], 4, axis=1))
    finally:
        conn.close()


def test_joint_rounding_fp16_high_fan_in_improves_fp64_error():
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    # Each source's small increments disappear after repeated half additions to one.
    indices = np.column_stack((np.zeros(2049, np.int64), np.arange(2049)))
    conn = ops.build_csr_connectivity(indices, np.zeros(2049), 2049, 1, 1, sort_by_target=True)
    try:
        with tf.device("/GPU:0"):
            z = tf.ones((1, 2049), tf.float16)
            w = tf.constant(np.r_[1, np.full(2048, 2**-12)], tf.float16)
            basis = tf.ones((1, 4), tf.float16)
            old = ops.fused_spike_currents(z, tf.cast(w, tf.float32), w, conn, basis, 1, True)
            old = ops.fused_spike_currents(z, tf.cast(w, tf.float32), w, conn, basis, 1, True,
                                          initial_currents=old)
            actual = ops.fused_combined_currents([z, z], [w, w], [conn, conn], basis, [old, old])
        np.testing.assert_array_equal(actual, 3.0)
        assert np.max(np.abs(old.numpy() - 3)) > 0.1
    finally:
        conn.close()


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("index_dtype", ["uint32", "int64"])
def test_cached_routing_refreshes_topology_and_reads_live_weights(dtype, index_dtype, monkeypatch):
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    monkeypatch.setattr(ops, "_csr_index_dtype", lambda *args: tf.as_dtype(index_dtype))
    first = ops.build_csr_connectivity(
        np.array([[2047, 0], [4098, 1]]), np.zeros(2), 2, 4099, 1,
        sort_by_target=True,
    )
    second = ops.build_csr_connectivity(
        np.array([[0, 0]]), np.zeros(1), 1, 4099, 1, sort_by_target=True,
    )
    try:
        with tf.device("/GPU:0"):
            weights = tf.Variable([1., 2.], dtype=dtype)

            @tf.function
            def calculate():
                return ops.fused_combined_currents(
                    [tf.ones((1, 2), dtype), tf.ones((1, 1), dtype)],
                    [weights, tf.ones(1, dtype)], [first, second], tf.ones((1, 4), dtype),
                    [tf.zeros((4099, 4), dtype)] * 2,
                )

            original = calculate().numpy()
            metadata = tf.raw_ops.ReadVariableOp(
                resource=first["metadata_handle"], dtype=tf.as_dtype(index_dtype),
            )
            changed = tf.tensor_scatter_nd_update(
                metadata, [[0]], tf.constant([2048], tf.as_dtype(index_dtype)),
            )
            tf.raw_ops.AssignVariableOp(resource=first["metadata_handle"], value=changed)
            updated = calculate().numpy()
            expected = original.copy()
            expected[2048] = expected[2047]
            expected[2047] = 0
            np.testing.assert_array_equal(updated, expected)
            weights.assign([2., 4.])
            expected[[2048, 4098]] *= 2
            np.testing.assert_array_equal(calculate().numpy(), expected)
    finally:
        first.close()
        second.close()


def test_negative_and_silent_activity_do_not_enter_joint_forward():
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    conn = ops.build_csr_connectivity(
        np.array([[0, 0], [0, 1], [0, 2]]), np.zeros(3), 3, 1, 1, sort_by_target=True,
    )
    try:
        with tf.device("/GPU:0"):
            result = ops.fused_combined_currents(
                [tf.constant([[-2., 0., 3.]]), tf.zeros((1, 3))],
                [tf.ones(3), tf.ones(3)], [conn, conn], tf.ones((1, 4)),
                [tf.zeros((1, 4)), tf.zeros((1, 4))],
            )
        np.testing.assert_array_equal(result, [[3.] * 4])
    finally:
        conn.close()


def test_substitution_retains_each_proxy_gradient_not_old_forward(monkeypatch):
    monkeypatch.setattr(ops, "combined_sweep_available", lambda: True)
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(
        dpointnet_combined_sweep=lambda *args, **kwargs: tf.constant([[7.] * 4])
    ))
    conn = dict(index_dtype="uint32", sorted_by_target=True, n_post=1,
                n_sources=1, n_edges=1, n_synapse_types=1, n_pairs=0, metadata_handle=None)
    first, second = tf.Variable(2.), tf.Variable(3.)
    with tf.GradientTape() as tape:
        actual = ops.fused_combined_currents(
            [tf.ones((1, 1))] * 2, [tf.ones((1,))] * 2, [conn] * 2,
            tf.ones((1, 4)), [tf.ones((1, 4)) * first, tf.ones((1, 4)) * second],
        )
        loss = tf.reduce_sum(actual)
    np.testing.assert_array_equal(actual, [[7.] * 4])
    np.testing.assert_array_equal(tape.gradient(loss, [first, second]), [4., 4.])


@pytest.mark.parametrize("value", [None, "auto", 1])
def test_combined_option_requires_boolean(value):
    from test_precision_credit import make_cell

    with pytest.raises(ValueError, match="use_combined_current_sweep"):
        make_cell("nest", noise=True, use_combined_current_sweep=value)


def test_combined_option_is_false_by_default_and_rejects_legacy_or_adjoint():
    from test_precision_credit import make_cell

    cell = make_cell("nest", noise=True)
    assert cell._use_combined_current_sweep is False
    with pytest.raises(ValueError, match="NEST and compute temporal gradients"):
        make_cell("legacy", noise=True, use_combined_current_sweep=True)
    with pytest.raises(ValueError, match="NEST and compute temporal gradients"):
        make_cell("nest", noise=True, use_combined_current_sweep=True, temporal_gradient_precision="float32")


@pytest.mark.parametrize("batch", [3, 32])
def test_combined_carrier_retains_reverse_seed_and_silent_spike_credit(batch):
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    conn = ops.build_csr_connectivity(
        [[0, 0], [1, 1], [0, 1]], [0, 0, 0], 2, 2, 1,
        build_compact_pairs=True, sort_by_target=True,
    )
    try:
        z = tf.Variable(np.tile([[0, 2]], (batch, 1)).astype(np.float16))
        carrier = tf.Variable([0.5, 0.25, -0.5], dtype=tf.float32)
        weights = ops.reorder_csr_values(tf.cast(carrier, tf.float16), conn)
        basis = tf.constant([[1, 0.5, 0.25, 0.125]], tf.float16)
        seed = tf.constant([3., -2., 1.])

        @tf.function
        def run(combined):
            with tf.GradientTape() as tape:
                proxy, last = ops.fused_recurrent_weight_carry(
                    z, carrier, weights, conn, basis, 2, 0.5, vjp_only=True,
                    use_javier_batch32_backward=True,
                )
                if combined:
                    primal = ops.fused_combined_currents(
                        [z, z], [weights, weights], [conn, conn], basis,
                        [proxy, tf.zeros_like(proxy)],
                    )
                else:
                    primal = proxy
                loss = tf.reduce_sum(tf.cast(primal, tf.float32)) + tf.reduce_sum(last * seed)
            return tape.gradient(loss, (z, carrier))

        actual, expected = run(True), run(False)
        for result, reference in zip(actual, expected):
            np.testing.assert_array_equal(result, reference)
        assert np.any(actual[0].numpy()[:, 0])
        assert np.any(actual[1].numpy() != seed.numpy())
    finally:
        conn.close()


@pytest.mark.parametrize("carrier", [False, True])
def test_combined_named_inputs_full_and_checkpointed_updates_and_restore(carrier, tmp_path):
    if not ops.combined_sweep_available():
        pytest.skip("Rebuilt combined-sweep CUDA required")
    from test_precision_credit import make_cell, build_core
    from bmtk.simulator.dpointnet.segmented_recompute import (
        FullBPTTGradientRunner, SegmentedRecomputeRunner,
    )

    old_policy = tf.keras.mixed_precision.global_policy()
    cell = make_cell(
        "nest", noise=True, two_inputs=True, batch_size=32,
        fused=True, use_fused_cuda=True, use_pair_projection=True,
        use_direct_csr_recurrent_gradient=True, _canonical_gradient_boundary=True,
        use_direct_state_rnn_loop=True, use_fused_recurrent_accumulation=carrier,
        use_javier_recurrent_vjp=carrier, use_combined_current_sweep=True,
        use_packed_sm120_backward=True, use_packed_sm120_external_backward=True,
    )
    try:
        core, state, sequence = build_core(cell, length=13, batch=32)
        targets = [cell.recurrent_weight_values] + [
            net["input_weight_values"] for net in cell.inputs.values()
        ]
        runners = [
            FullBPTTGradientRunner(core, cell.restore_segmented_variable_gradients),
            SegmentedRecomputeRunner(
                core, 13, 5, 2, pack_spike_checkpoints=True,
                variable_gradient_transform=cell.restore_segmented_variable_gradients,
            ),
        ]
        results = []
        for runner in runners:
            @tf.function
            def evaluate():
                with tf.GradientTape() as tape:
                    out = runner(sequence, state)
                    loss = tf.reduce_mean(tf.cast(out[0], tf.float32)) + tf.reduce_mean(out[1])
                return out, tape.gradient(loss, targets)
            results.append(evaluate())
        for actual, expected in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1])):
            assert actual is not None and expected is not None
            np.testing.assert_allclose(actual, expected, rtol=4e-3, atol=5e-4)
        assert all(np.any(gradient.numpy()) for gradient in results[0][1])
        optimizer = tf.keras.optimizers.SGD(0.01)
        optimizer.apply_gradients(zip(results[0][1], targets))
        checkpoint = tf.train.Checkpoint(
            optimizer=optimizer, **{f"master_{i}": target for i, target in enumerate(targets)}
        )
        snapshots = [target.numpy().copy() for target in targets]
        path = checkpoint.save(str(tmp_path / "model"))
        for target in targets:
            target.assign(tf.zeros_like(target))
        checkpoint.restore(path).assert_consumed()
        for target, expected in zip(targets, snapshots):
            np.testing.assert_array_equal(target, expected)
        cell.refresh_recurrent_weight_shadow()
        for master, connectivity, shadow in [
            (cell.recurrent_weight_values, cell.recurrent_fused_connectivity,
             cell.recurrent_csr_weight_values_compute),
            *[(net["input_weight_values"], net["fused_connectivity"], net["csr_weight_values_compute"])
              for net in cell.inputs.values()],
        ]:
            np.testing.assert_array_equal(master, master.constraint(master))
            np.testing.assert_array_equal(shadow, ops.reorder_csr_values(tf.cast(master, tf.float16), connectivity))
        cell.synaptic_basis_weights = tf.identity(cell.synaptic_basis_weights)
        current = cell._project_step_currents(sequence[:, 0], state)[0]
        assert bool(tf.reduce_all(tf.math.is_finite(current)))
        assert cell._recurrent_projection_connectivity()["sorted_by_target"] is True
    finally:
        cell.close_fused_cuda()
        tf.keras.mixed_precision.set_global_policy(old_policy)
