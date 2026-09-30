"""Producer fusion, temporal carrier, ordering and TensorFlow buffer ownership."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity,
    reorder_csr_values,
)
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner
from test_precision_credit import make_cell

gpu = pytest.mark.skipif(
    not ops.fused_recurrent_accumulation_available(),
    reason="SM86 accumulation op unavailable",
)


@pytest.fixture(autouse=True)
def policy():
    before = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(before)


def raw_gradient(conn, spikes, weights, basis, upstream, accumulator=None):
    fn = (
        ops._OPS.dpointnet_csr_spike_grad
        if accumulator is None
        else ops._OPS.dpointnet_csr_spike_grad_accumulate
    )
    args = (
        spikes,
        upstream,
        conn["metadata_handle"],
        weights,
        basis,
        tf.constant(0.7, dtype=spikes.dtype),
    )
    if accumulator is not None:
        args += (accumulator,)
    return fn(
        *args,
        Tindex=tf.uint32,
        n_post=conn["n_post"],
        n_edges=conn["n_edges"],
        n_pairs=conn["n_pairs"],
        write_csr_weight_gradient=True,
    )


@gpu
@pytest.mark.parametrize("fanout", [1, 31, 32, 33, 129])
@pytest.mark.parametrize("silent", [False, True])
def test_producer_nonzero_accumulator_alias_and_dense_oracle(fanout, silent):
    rng = np.random.default_rng(513)
    pre = np.repeat([0, 3, 12], fanout)
    indices = np.column_stack((rng.integers(7, size=len(pre)), pre))
    order = rng.permutation(len(pre))
    indices = indices[order]
    types = rng.integers(3, size=len(pre))
    conn = build_csr_connectivity(indices, types, 13, 7, 3, build_compact_pairs=True)
    spikes = rng.poisson(1.3, (32, 13)).astype(np.float32) * 5
    spikes[:, 3] = -2
    if silent:
        spikes.fill(0)
    basis = rng.normal(size=(3, 4)).astype(np.float32)
    upstream = rng.normal(size=(32, 7, 4)).astype(np.float32)
    weights = rng.normal(size=len(pre)).astype(np.float32)
    accumulator = rng.normal(size=len(pre)).astype(np.float32)
    try:
        csr = reorder_csr_values(tf.constant(weights), conn)
        z, b, dy, seed = map(
            tf.constant, (spikes, basis, upstream.reshape(-1, 4), accumulator)
        )
        dz, dw = raw_gradient(conn, z, csr, b, dy)

        @tf.function
        def run(old):
            first = raw_gradient(conn, z, csr, b, dy, old)
            second = raw_gradient(conn, z, csr, b, dy, old)
            chained = raw_gradient(conn, z, csr, b, dy, first[1])
            return old, first, second, chained

        retained, first, second, chained = run(seed)
        np.testing.assert_array_equal(retained, accumulator)
        np.testing.assert_array_equal(seed, accumulator)
        np.testing.assert_array_equal(first[0], dz)
        np.testing.assert_array_equal(first[1], seed + dw)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(chained[1], (seed + dw) + dw)
        projection = np.einsum(
            "ber,er->be",
            upstream[:, indices[:, 0]].astype(np.float64),
            basis[types].astype(np.float64),
        )
        ref = np.sum(projection * np.maximum(spikes[:, indices[:, 1]], 0), axis=0)
        ref = ref[np.argsort(indices[:, 1], kind="stable")]
        np.testing.assert_allclose(dw, ref, rtol=2e-5, atol=3e-5)
        if np.linalg.norm(ref):
            a = np.asarray(dw, np.float64)
            assert (
                np.dot(a, ref) / (np.linalg.norm(a) * np.linalg.norm(ref)) > 1 - 1e-10
            )
        np.testing.assert_array_equal(run(seed)[1][1], first[1])
    finally:
        conn.close()


@gpu
def test_weight_carrier_threads_nonzero_reverse_seed():
    conn = build_csr_connectivity(
        np.array([[1, 0], [0, 1], [0, 0]]),
        np.array([0, 0, 0]),
        2,
        2,
        1,
        build_compact_pairs=True,
    )
    carrier = tf.Variable([0.1, -0.3, 0.5])
    weights = reorder_csr_values(carrier, conn)
    z = tf.reshape(tf.cast(tf.range(64) % 4, tf.float32), (32, 2))
    basis = tf.constant([[1.0, 0.3, -0.1, 0.05]])
    upstream = tf.reshape(tf.cast(tf.range(256) % 11, tf.float32), (64, 4))
    seed = tf.constant([3.0, -2.0, 1.0])
    try:

        @tf.function
        def run():
            with tf.GradientTape() as tape:
                tape.watch(z)
                a, c = ops.fused_recurrent_weight_carry(
                    z, carrier, weights, conn, basis, 2, 0.7
                )
                b, c = ops.fused_recurrent_weight_carry(
                    z, c, weights, conn, basis, 2, 0.7
                )
                loss = tf.reduce_sum(a * upstream) + tf.reduce_sum(b * (upstream * 2))
                loss += tf.reduce_sum(c * seed)
            return tape.gradient(loss, (z, carrier))

        actual_z, actual_w = run()
        dz1, dw1 = raw_gradient(conn, z, weights, basis, upstream)
        dz2, dw2 = raw_gradient(conn, z, weights, basis, upstream * 2)
        np.testing.assert_array_equal(actual_w, (seed + dw2) + dw1)
        np.testing.assert_allclose(actual_z, dz1 + dz2, rtol=1e-6, atol=1e-5)
    finally:
        conn.close()


def make_fused_cell(dynamics, replay_mode, fused_state=True, **options):
    return make_cell(
        dynamics,
        noise=True,
        two_inputs=True,
        fused=fused_state,
        use_fused_cuda=True,
        batch_size=32,
        temporal_gradient_precision="float32",
        current_replay_mode=replay_mode,
        use_direct_csr_recurrent_gradient=True,
        use_packed_sm120_backward=False,
        use_packed_sm120_external_backward=False,
        use_fused_recurrent_accumulation=True,
        **options,
    )


@gpu
@pytest.mark.parametrize("dynamics", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
@pytest.mark.parametrize("graph", [False, True])
def test_identical_cached_forward_gradients_and_lifetimes(dynamics, replay_mode, graph):
    cell = make_fused_cell(dynamics, replay_mode)
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    initial = list(cell.zero_state(32, cell.compute_dtype))
    initial[0] = tf.ones_like(initial[0])
    initial = tuple(initial)
    x = tf.zeros((32, 11, 0), tf.float32)
    masters = tuple(
        v.value if not callable(getattr(v, "value", None)) else v
        for v in cell.trainable_variables
    )

    def run():
        cache = runner._forward(x, initial, probe_steps=(2, 8))
        final_gradients = [None] * len(initial)
        final_gradients[1] = 2 * cache[1][1]

        def reverse(fused):
            cell.use_fused_recurrent_accumulation = fused
            return runner._backward(
                x, initial, cache, (None, None), final_gradients, masters, capture=True
            )

        expected = reverse(False)
        actual = reverse(True)
        for v in cell.trainable_variables:
            v.assign(v * 1.1)
        cell.refresh_recurrent_weight_shadow()
        cell.advance_noise_seed()
        later = runner._forward(x, initial)
        return expected, actual, reverse(True), tf.reduce_sum(later[1][1])

    try:
        expected, actual, repeated, _ = (tf.function(run) if graph else run)()
        for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
            np.testing.assert_allclose(a, b, rtol=3e-6, atol=1e-7)
        assert all(v.dtype == tf.float32 for v in actual[2])
        assert np.any(actual[2][0].numpy() != 0)
        for a, b in zip(tf.nest.flatten(repeated), tf.nest.flatten(actual)):
            # Named-input atomic reductions are not bitwise repeatable on GPU.
            np.testing.assert_allclose(a, b, rtol=3e-6, atol=1e-7)
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("value", ["auto", 1, None])
def test_accumulation_option_rejects_unknown(value):
    with pytest.raises(ValueError, match="use_fused_recurrent_accumulation"):
        make_cell(use_fused_recurrent_accumulation=value)


def test_accumulation_requires_supported_route():
    assert make_cell().use_fused_recurrent_accumulation is False
    with pytest.raises(ValueError, match="Fused recurrent accumulation requires"):
        make_cell(use_fused_recurrent_accumulation=True)


@gpu
def test_accumulation_accepts_ordinary_direct_loop_route(monkeypatch):
    import bmtk.simulator.dpointnet.cell_models.glif3_cell as glif3_cell

    monkeypatch.setattr(
        glif3_cell, "fused_recurrent_accumulation_available", lambda: True
    )
    cell = make_cell(
        batch_size=32,
        use_fused_cuda=True,
        use_direct_csr_recurrent_gradient=True,
        use_direct_state_rnn_loop=True,
        temporal_gradient_precision="compute",
        use_fused_recurrent_accumulation=True,
    )
    try:
        assert cell.use_fused_recurrent_accumulation is True
    finally:
        cell.close_fused_cuda()


@gpu
@pytest.mark.parametrize("dynamics", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_regularizer_persistent_backward_and_optimizer_shadows(dynamics, replay_mode):
    results = []
    for enabled in (False, True):
        # Baseline legacy fused-state cannot trace a persistent tape twice.
        # Keep CUDA currents enabled; only this lifetime test uses TF neuron state.
        cell = make_fused_cell(dynamics, replay_mode, fused_state=dynamics != "legacy")
        cell.use_fused_recurrent_accumulation = enabled
        runner = TemporalAdjointRunner(cell, chunk_size=3)
        initial = list(cell.zero_state(32, cell.compute_dtype))
        initial[0] = tf.ones_like(initial[0])
        x = tf.zeros((32, 11, 0), tf.float32)
        try:

            @tf.function
            def gradients():
                with tf.GradientTape(persistent=True) as tape:
                    outputs, final = runner(x, initial)
                    loss = tf.reduce_mean(outputs[1]) + tf.reduce_mean(final[1] ** 2)
                    loss += 0.003 * tf.reduce_sum(cell.recurrent_weight_values**2)
                first = tape.gradient(loss, cell.trainable_variables)
                second = tape.gradient(loss, cell.trainable_variables)
                return loss, first, second

            loss, grads, repeated = gradients()
            for a, b in zip(grads, repeated):
                np.testing.assert_allclose(a, b, rtol=3e-6, atol=1e-7)
            optimizer = tf.keras.optimizers.SGD(0.001)
            optimizer.apply_gradients(zip(grads, cell.trainable_variables))
            cell.refresh_recurrent_weight_shadow()
            for variable in cell.trainable_variables:
                if variable.constraint is not None:
                    np.testing.assert_array_equal(
                        variable, variable.constraint(variable)
                    )
            np.testing.assert_array_equal(
                cell.recurrent_csr_weight_values_compute,
                reorder_csr_values(
                    tf.cast(cell.recurrent_weight_values, tf.float16),
                    cell.recurrent_fused_connectivity,
                ),
            )
            for net in cell.inputs.values():
                np.testing.assert_array_equal(
                    net["input_weight_values_compute"],
                    tf.cast(net["input_weight_values"], tf.float16),
                )
                np.testing.assert_array_equal(
                    net["csr_weight_values_compute"],
                    reorder_csr_values(
                        tf.cast(net["input_weight_values"], tf.float16),
                        net["fused_connectivity"],
                    ),
                )
            results.append((loss, grads, [v.numpy() for v in cell.trainable_variables]))
        finally:
            cell.close_fused_cuda()
    for a, b in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1])):
        np.testing.assert_allclose(a, b, rtol=3e-6, atol=1e-7)


def test_carrier_graph_compatibility_with_tensorflow_test_double(monkeypatch):
    from types import SimpleNamespace

    metadata = tf.Variable(0, trainable=False, dtype=tf.int32)
    conn = dict(
        metadata_handle=metadata.handle,
        index_dtype="uint32",
        n_edges=1,
        n_sources=1,
        n_post=1,
        n_synapse_types=1,
        n_pairs=1,
        fixed4_incoming=False,
        incoming_pre_ids=[],
        incoming_edge_ids=[],
        incoming_types=[],
    )

    def forward(z, master, handle, weights, basis, scale, *args, **kwargs):
        return tf.raw_ops.IdentityN(
            input=[tf.zeros((tf.shape(z)[0], 4), z.dtype), master, handle]
        )[0]

    def backward(z, dy, handle, weights, basis, scale, acc, **kwargs):
        projection = tf.reduce_sum(dy * basis, axis=1, keepdims=True)
        return (
            projection * weights[0] * scale,
            acc + tf.cast(tf.reduce_sum(z * projection)[None], acc.dtype),
        )

    monkeypatch.setattr(
        ops,
        "_OPS",
        SimpleNamespace(
            dpointnet_csr_spike_forward=forward,
            dpointnet_csr_spike_grad_accumulate=backward,
        ),
    )
    master = tf.Variable([2.0])

    @tf.function
    def run():
        with tf.GradientTape() as tape:

            def step(index, carrier, loss):
                z = tf.ones((32, 1), dtype=carrier.dtype)
                currents, carrier = ops.fused_recurrent_weight_carry(
                    z,
                    carrier,
                    tf.ones((1,), dtype=carrier.dtype),
                    conn,
                    tf.ones((1, 4), dtype=carrier.dtype),
                    1,
                    1.0,
                    vjp_only=False,
                )
                return (
                    index + 1,
                    carrier,
                    loss + tf.cast(
                        tf.reduce_sum(currents) * tf.cast(index + 1, currents.dtype),
                        tf.float32,
                    ),
                )

            _, carrier, loss = tf.while_loop(
                lambda index, *_: index < 3,
                step,
                (0, tf.identity(master), tf.constant(0.0)),
            )
            loss += tf.reduce_sum(carrier)
        return tape.gradient(loss, master)

    np.testing.assert_array_equal(run(), [769.0])

    master.assign([2.0])

    @tf.function
    def run_half():
        with tf.GradientTape() as tape:
            carrier0 = tf.cast(master, tf.float32)

            def step(index, carrier, loss):
                currents, carrier = ops.fused_recurrent_weight_carry(
                    tf.ones((32, 1), dtype=tf.float16),
                    carrier,
                    tf.ones((1,), dtype=tf.float16),
                    conn,
                    tf.ones((1, 4), dtype=tf.float16),
                    1,
                    tf.constant(1.0, dtype=tf.float16),
                    vjp_only=False,
                )
                return (
                    index + 1,
                    carrier,
                    loss + tf.cast(tf.reduce_sum(currents), tf.float32),
                )

            _, carrier, loss = tf.while_loop(
                lambda index, *_: index < 2,
                step,
                (0, carrier0, tf.constant(0.0)),
            )
            loss += tf.reduce_sum(carrier)
        return tape.gradient(loss, master)

    np.testing.assert_array_equal(run_half(), [257.0])


@gpu
def test_half_producer_nonzero_accumulator_alias_and_dense_oracle():
    conn = build_csr_connectivity(
        np.array([[0, 0], [1, 0], [1, 1], [0, 2]], np.uint32),
        np.array([0, 1, 0, 1], np.uint32),
        3,
        2,
        2,
        build_compact_pairs=True,
    )
    try:
        spikes = (tf.reshape(tf.cast(tf.range(96) % 5, tf.float16), (32, 3)) / 4)
        upstream = (
            tf.reshape(tf.cast(tf.range(256) % 7, tf.float16), (64, 4)) / 8
        )
        basis = tf.constant([[1.0, -0.25, 0.5, 0.125], [0.75, 0.2, -0.3, 0.4]], tf.float16)
        weights = reorder_csr_values(
            tf.constant([0.5, -0.75, 1.25, 0.875], tf.float16), conn
        )
        seed = tf.constant([0.1, -0.2, 0.3, -0.4], tf.float32)
        dz, dw = raw_gradient(conn, spikes, weights, basis, upstream)
        actual_z, actual_w = raw_gradient(conn, spikes, weights, basis, upstream, seed)
        np.testing.assert_array_equal(actual_z, dz)
        np.testing.assert_allclose(actual_w, seed + dw, rtol=1e-3, atol=2e-3)
        assert actual_w.dtype == tf.float32
    finally:
        conn.close()


@gpu
def test_subnormal_carry_and_accumulator_aliasing_weights():
    conn = build_csr_connectivity(
        np.array([[0, 0]]), np.array([0]), 1, 1, 1, build_compact_pairs=True
    )
    try:
        weights = tf.constant([0.125], tf.float32)
        basis = tf.ones((1, 4))
        tiny = tf.constant([2.0**-140], tf.float32)
        _, carried = raw_gradient(
            conn, tf.zeros((32, 1)), weights, basis, tf.zeros((32, 4)), tiny
        )
        np.testing.assert_array_equal(carried, tiny)
        assert carried.numpy()[0] > 0
        z, upstream = tf.ones((32, 1)), tf.ones((32, 4)) * 0.25
        dz, dw = raw_gradient(conn, z, weights, basis, upstream)
        actual_z, actual_w = raw_gradient(conn, z, weights, basis, upstream, weights)
        np.testing.assert_array_equal(weights, [0.125])
        np.testing.assert_array_equal(actual_z, dz)
        np.testing.assert_array_equal(actual_w, weights + dw)
    finally:
        conn.close()
