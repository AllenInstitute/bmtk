from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import build_csr_connectivity, reorder_csr_values
from bmtk.simulator.dpointnet.custom_ops import csr_spike_ops as ops


gpu = pytest.mark.skipif(
    not ops.fused_recurrent_accumulation_available() or not ops.weight_only_accumulation_available(),
    reason="Rebuilt SM86+ accumulator required",
)


@gpu
@pytest.mark.parametrize("batch", [1, 7, 32])
@pytest.mark.parametrize("silent", [False, True])
def test_weight_only_preserves_accumulator_and_default_live_spike_credit(batch, silent):
    rng = np.random.default_rng(61)
    indices = np.column_stack((rng.integers(11, size=135), rng.integers(8, size=135)))
    conn = build_csr_connectivity(indices, rng.integers(3, size=135), 8, 11, 3, build_compact_pairs=True)
    try:
        spikes = rng.poisson(.3, size=(batch, 8)).astype(np.float16)
        if silent:
            spikes.fill(0)
        weights = reorder_csr_values(tf.constant(rng.normal(size=135).astype(np.float16)), conn)
        basis = tf.constant(rng.normal(size=(3, 4)).astype(np.float16))
        upstream = tf.constant(rng.normal(size=(batch * 11, 4)).astype(np.float16))
        seed = tf.constant(rng.normal(size=135).astype(np.float32))
        z = tf.constant(spikes)

        @tf.function
        def run(initial):
            arguments = (z, upstream, conn["metadata_handle"], weights, basis,
                         tf.constant(.7, tf.float16), initial)
            options = dict(Tindex=tf.uint32, n_post=11, n_edges=135,
                           n_pairs=conn["n_pairs"], use_javier_batch32_backward=True)
            default = ops._OPS.dpointnet_csr_spike_grad_accumulate(*arguments, **options)
            weight_only = ops._OPS.dpointnet_csr_spike_grad_accumulate(
                *arguments, **options, compute_spike_gradient=False
            )
            return initial, default, weight_only

        retained, default, actual = run(seed)
        np.testing.assert_array_equal(retained, seed)
        np.testing.assert_array_equal(default[1], actual[1])
        np.testing.assert_array_equal(actual[0], np.zeros_like(spikes))
        assert np.any(default[0].numpy() != 0)
        if silent:
            np.testing.assert_array_equal(actual[1], seed)
    finally:
        conn.close()


@gpu
def test_wrapper_stops_only_requested_input_adjoint():
    conn = build_csr_connectivity(
        np.array([[0, 0], [1, 1], [1, 0]]), np.array([0, 0, 0]),
        2, 2, 1, build_compact_pairs=True,
    )
    try:
        z = tf.constant(np.ones((32, 2)), tf.float16)
        weights = tf.constant([.1, .2, .3], tf.float16)
        basis = tf.constant([[.5, .25, .1, .05]], tf.float16)
        seed = tf.Variable([.1, .2, .3], dtype=tf.float32)
        results = []
        for compute_spike_gradient in (True, False):
            with tf.GradientTape() as tape:
                tape.watch(z)
                output, carried = ops.fused_recurrent_weight_carry(
                    z, seed, weights, conn, basis, 2, .7, vjp_only=False,
                    use_javier_batch32_backward=True,
                    compute_spike_gradient=compute_spike_gradient,
                )
                loss = tf.reduce_sum(tf.cast(output, tf.float32)) + tf.reduce_sum(carried)
            results.append(tape.gradient(loss, [z, seed]))
        assert results[0][0] is not None
        assert results[1][0] is None
        np.testing.assert_array_equal(results[0][1], results[1][1])
    finally:
        conn.close()


def test_invalid_weight_only_configuration_fails_before_execution(monkeypatch):
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(dpointnet_csr_spike_grad_accumulate=lambda: None))
    with pytest.raises(ValueError, match="FP16 Javier"):
        ops.fused_recurrent_weight_carry(
            tf.zeros((32, 2)), tf.zeros((3,)), tf.zeros((3,)),
            {}, tf.zeros((1, 4)), 2, 0., compute_spike_gradient=False,
        )
    with pytest.raises(TypeError, match="explicit boolean"):
        ops.fused_recurrent_weight_carry(
            tf.zeros((32, 2)), tf.zeros((3,)), tf.zeros((3,)),
            {}, tf.zeros((1, 4)), 2, 0., compute_spike_gradient=0,
        )
    with pytest.raises(RuntimeError, match="Rebuild CUDA operators"):
        ops.fused_recurrent_weight_carry(
            tf.zeros((32, 2), tf.float16), tf.zeros((3,)), tf.zeros((3,), tf.float16),
            {}, tf.zeros((1, 4), tf.float16), 2, 0.,
            use_javier_batch32_backward=True, compute_spike_gradient=False,
        )


def test_default_wrapper_still_accepts_old_binary_signature(monkeypatch):
    calls = []

    def old_accumulator(z, dc, metadata, weights, basis, scale, carrier,
                        Tindex, n_post, n_edges, n_pairs, use_javier_batch32_backward):
        calls.append(use_javier_batch32_backward)
        return tf.ones_like(z), carrier + 1.

    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(dpointnet_csr_spike_grad_accumulate=old_accumulator))
    monkeypatch.setattr(ops, "fused_spike_currents", lambda z, *args, **kwargs: z)
    z = tf.ones((2, 2))
    carrier = tf.ones((3,))
    with tf.GradientTape() as tape:
        tape.watch((z, carrier))
        output, carried = ops.fused_recurrent_weight_carry(
            z, carrier, tf.ones((3,)),
            {"metadata_handle": None, "index_dtype": "uint32", "n_edges": 3, "n_pairs": 2},
            tf.ones((1, 4)), 2, 1.,
        )
        loss = tf.reduce_sum(output) + tf.reduce_sum(carried)
    dz, dw = tape.gradient(loss, (z, carrier))
    np.testing.assert_array_equal(dz, np.ones((2, 2)))
    np.testing.assert_array_equal(dw, np.full(3, 2.))
    assert calls == [False]
    assert not ops.weight_only_accumulation_available()
