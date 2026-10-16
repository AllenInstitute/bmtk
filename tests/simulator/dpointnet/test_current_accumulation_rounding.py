import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import (
    build_csr_connectivity,
    fused_cuda_available,
    fused_spike_currents,
    reorder_csr_values,
)


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("batch", [1, 4, 16, 32])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("route", ["ordinary", "fixed4", "active", "queue", "queue_runs"])
@pytest.mark.parametrize("compiled", [False, True])
def test_accumulation_rounds_projection_before_adding_initial(batch, dtype, route, compiled):
    indices = np.array([[post, pre] for pre in range(4) for post in range(3)])
    types = np.zeros(len(indices), np.int64)
    connectivity = build_csr_connectivity(
        indices, types, 4, 3, 1, build_fixed4_incoming=True, sort_by_target=True)
    master = tf.Variable(np.ones(len(indices)), dtype=tf.float32)
    weights = reorder_csr_values(tf.cast(master, dtype), connectivity)
    basis = tf.constant([[1., .5, .25, .125]], dtype)
    scale = 2048. if dtype == tf.float16 else 16777216.
    initial = tf.Variable(tf.fill((batch * 3, 4), tf.cast(scale, dtype)))
    activity = tf.Variable(
        np.repeat([[1., 1., 1., 1.]], batch, axis=0), dtype=dtype)
    options = dict(
        use_fixed4_forward=route == "fixed4",
        use_active_row_forward=route == "active",
        use_device_active_queue_forward=route in ("queue", "queue_runs"),
        use_forward_run_aggregation=route == "queue_runs",
    )

    def evaluate(spikes, baseline):
        with tf.GradientTape() as tape:
            tape.watch((spikes, baseline))
            projected = fused_spike_currents(
                spikes, master, weights, connectivity, basis, 3,
                compute_spike_gradient=True, **options)
            accumulated = fused_spike_currents(
                spikes, master, weights, connectivity, basis, 3,
                compute_spike_gradient=True, initial_currents=baseline, **options)
            loss = tf.reduce_sum(tf.cast(accumulated, tf.float32))
        gradients = tape.gradient(loss, (spikes, master, baseline))
        return accumulated, baseline + projected, gradients

    run = tf.function(evaluate) if compiled else evaluate
    for _ in range(3):
        accumulated, reference, gradients = run(activity, initial)
        np.testing.assert_array_equal(accumulated, reference)
        np.testing.assert_array_equal(initial, np.full((batch * 3, 4), scale))
        np.testing.assert_array_equal(gradients[0], np.full((batch, 4), 3 * 1.875))
        np.testing.assert_array_equal(gradients[1], np.full(len(indices), batch * 1.875))
        np.testing.assert_array_equal(gradients[2], np.ones((batch * 3, 4)))


@pytest.mark.skipif(not fused_cuda_available(), reason="Fused CUDA op is unavailable.")
@pytest.mark.parametrize("batch", [1, 4, 16, 32])
@pytest.mark.parametrize("route", ["ordinary", "fixed4", "queue"])
def test_distinct_counted_inputs_accumulation_and_checkpoint_vjp(batch, route):
    indices = np.array([[post, pre] for pre in range(4) for post in range(3)])
    connectivity = build_csr_connectivity(
        indices, np.zeros(len(indices), np.int64), 4, 3, 1,
        build_fixed4_incoming=True, sort_by_target=True)
    master = tf.Variable(np.arange(1, len(indices) + 1), dtype=tf.float32)
    weights = reorder_csr_values(master, connectivity)
    basis = tf.constant([[1., .5, .25, .125]], tf.float32)
    spikes = tf.constant(np.arange(batch * 4).reshape(batch, 4) % 5, tf.float32)
    initial = tf.constant(np.arange(batch * 3 * 4).reshape(batch * 3, 4), tf.float32)
    options = dict(use_fixed4_forward=route == "fixed4",
                   use_device_active_queue_forward=route == "queue")

    def operation(x, baseline):
        return fused_spike_currents(
            x, master, weights, connectivity, basis, 3,
            compute_spike_gradient=True, initial_currents=baseline, **options)

    @tf.function
    def evaluate(recompute):
        with tf.GradientTape() as tape:
            tape.watch((spikes, initial))
            result = tf.recompute_grad(operation)(spikes, initial) if recompute else operation(spikes, initial)
            loss = tf.reduce_sum(result)
        return result, tape.gradient(loss, (spikes, master, initial))

    expected = initial.numpy().reshape(batch, 3, 4).astype(np.float64)
    for edge, (post, pre) in enumerate(indices):
        expected[:, post] += spikes.numpy()[:, pre, None] * master.numpy()[edge] * basis.numpy()[0]
    ordinary = evaluate(False)
    replay = evaluate(True)
    np.testing.assert_array_equal(ordinary[0], expected.reshape(batch * 3, 4))
    for actual, reference in zip(tf.nest.flatten(replay), tf.nest.flatten(ordinary)):
        np.testing.assert_array_equal(actual, reference)
