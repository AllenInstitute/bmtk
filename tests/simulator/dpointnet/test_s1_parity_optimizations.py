import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import (
    fused_nest_state,
    fused_nest_state_available,
    pack_nest_state_coefficients,
)
from bmtk.simulator.dpointnet.loss_functions.weight_regularization import (
    EMDWeightRegularization,
)


def _emd_object(current, initial, groups, *, custom=False, dedup=False):
    group_ids = np.asarray(groups, dtype=np.int64)
    n_groups = int(group_ids.max() + 1) if group_ids.size else 0
    order = np.argsort(group_ids, kind="stable")
    counts = np.bincount(group_ids, minlength=n_groups)
    row_splits = np.empty(n_groups + 1, dtype=np.int64)
    row_splits[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_splits[1:])
    sorted_initial = np.empty_like(np.asarray(initial, dtype=np.float32))
    for group in range(n_groups):
        start, end = row_splits[group], row_splits[group + 1]
        sorted_initial[start:end] = np.sort(np.asarray(initial, dtype=np.float32)[order[start:end]])

    obj = object.__new__(EMDWeightRegularization)
    obj._weights = current
    obj._dtype = tf.float32
    obj._cost = tf.constant(1.75, tf.float32)
    obj._n_groups = n_groups
    obj.num_unique = tf.constant(n_groups, tf.int32)
    obj._group_order = tf.Variable(order.astype(np.int32), trainable=False)
    obj._row_splits = tf.Variable(row_splits, dtype=tf.int64, trainable=False)
    obj._sorted_initial_values = tf.Variable(sorted_initial, dtype=tf.float32, trainable=False)
    obj._group_slices = tuple(
        (int(row_splits[i]), int(row_splits[i + 1])) for i in range(n_groups)
    )
    obj._use_grouped_custom_gradient = custom
    obj._deduplicate_within_graph = dedup
    obj._graph_cache = {}
    return obj


def _numpy_reference(current, initial, groups, cost=1.75):
    values = []
    groups = np.asarray(groups)
    for group in sorted(np.unique(groups)):
        mask = groups == group
        values.append(np.mean(np.abs(np.sort(current[mask]) - np.sort(initial[mask]))))
    return cost * np.mean(values)


def test_grouped_emd_custom_gradient_matches_loop_and_reference():
    current = tf.Variable([0.2, -0.4, 0.2, 1.1, -0.7, 0.3, 0.8], dtype=tf.float32)
    initial = np.array([0.0, -0.2, 0.6, 0.9, -0.9, 0.1, 1.2], dtype=np.float32)
    groups = np.array([2, 1, 2, 0, 1, 0, 2])
    loop = _emd_object(current, initial, groups, custom=False)
    custom = _emd_object(current, initial, groups, custom=True)

    with tf.GradientTape() as tape:
        loop_value = loop._compute(current)
    loop_grad = tape.gradient(loop_value, current)
    with tf.GradientTape() as tape:
        custom_value = custom._compute(current)
    custom_grad = tape.gradient(custom_value, current)

    np.testing.assert_allclose(
        custom_value.numpy(), _numpy_reference(current.numpy(), initial, groups), rtol=1e-6
    )
    np.testing.assert_allclose(custom_value.numpy(), loop_value.numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        tf.convert_to_tensor(custom_grad).numpy(),
        tf.convert_to_tensor(loop_grad).numpy(),
        rtol=1e-6,
        atol=1e-6,
    )


def test_grouped_emd_custom_gradient_keeps_tie_values_exact():
    current = tf.Variable([0.5, 0.5, -0.25, -0.25, 1.0, 1.0], dtype=tf.float32)
    initial = np.array([0.1, 0.7, -0.5, -0.1, 1.2, 0.8], dtype=np.float32)
    groups = np.array([0, 0, 1, 1, 1, 1])
    custom = _emd_object(current, initial, groups, custom=True)
    value = custom._compute(current)
    np.testing.assert_allclose(
        value.numpy(), _numpy_reference(current.numpy(), initial, groups), rtol=1e-6
    )


def test_emd_graph_dedup_reuses_value_but_counts_gradient_twice():
    current = tf.Variable([0.2, -0.4, 1.1, -0.7], dtype=tf.float32)
    initial = np.array([0.0, -0.2, 0.9, -0.9], dtype=np.float32)
    groups = np.array([0, 0, 1, 1])
    dedup = _emd_object(current, initial, groups, custom=True, dedup=True)
    single = _emd_object(current, initial, groups, custom=True, dedup=False)

    @tf.function
    def two_terms():
        with tf.GradientTape() as tape:
            value = dedup() + dedup()
        return value, tape.gradient(value, current)

    with tf.GradientTape() as tape:
        single_value = single._compute(current)
    single_grad = tape.gradient(single_value, current)
    value, grad = two_terms()
    np.testing.assert_allclose(value.numpy(), 2.0 * single_value.numpy(), rtol=1e-6)
    np.testing.assert_allclose(grad.numpy(), 2.0 * single_grad.numpy(), rtol=1e-6)


def test_pack_nest_state_coefficients_reads_current_values_each_invocation():
    neurons = tf.constant(3, tf.int32)
    syn_decay = tf.Variable([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
    psc_initial = tf.Variable([[0.5, 0.6, 0.7, 0.8]], dtype=tf.float32)
    asc_decay = tf.Variable([[0.9, 1.0], [1.1, 1.2], [1.3, 1.4]], dtype=tf.float32)
    asc_amps = tf.Variable([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]], dtype=tf.float32)
    decay = tf.Variable([0.25, 0.5, 0.75], dtype=tf.float32)
    current_factor = tf.Variable([2.0, 3.0, 4.0], dtype=tf.float32)
    asc_mean = tf.Variable([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2]], dtype=tf.float32)
    asc_refractory_decay = tf.Variable(
        [[1.2, 1.0], [0.8, 0.6], [0.4, 0.2]], dtype=tf.float32
    )
    psc_voltage = tf.Variable(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
        dtype=tf.float32,
    )
    rise_voltage = tf.Variable(
        [[1.2, 1.1, 1.0, 0.9], [0.8, 0.7, 0.6, 0.5], [0.4, 0.3, 0.2, 0.1]],
        dtype=tf.float32,
    )
    v_reset = tf.Variable([0.0, -0.1, -0.2], dtype=tf.float32)
    damping = tf.Variable(0.25, dtype=tf.float32)

    @tf.function
    def packed():
        return pack_nest_state_coefficients(
            neurons,
            tf.float32,
            syn_decay=syn_decay,
            psc_initial=psc_initial,
            asc_decay=asc_decay,
            asc_amps=asc_amps,
            decay=decay,
            current_factor=current_factor,
            asc_mean=asc_mean,
            asc_refractory_decay=asc_refractory_decay,
            psc_voltage=psc_voltage,
            rise_voltage=rise_voltage,
            v_reset=v_reset,
            voltage_gradient_dampening=damping,
        )

    first = packed().numpy()
    damping.assign(0.75)
    v_reset.assign([0.3, 0.2, 0.1])
    second = packed().numpy()
    assert first.shape == (3, 28)
    assert second.shape == (3, 28)
    np.testing.assert_allclose(first[:, -1], 0.25)
    np.testing.assert_allclose(second[:, -2], [0.3, 0.2, 0.1])
    np.testing.assert_allclose(second[:, -1], 0.75)


def test_prepacked_fused_nest_coefficients_match_inline_for_two_forwards_before_backward():
    if not fused_nest_state_available():
        pytest.skip("Fused NEST state operator is unavailable")
    dtype = tf.float32
    neurons = 4
    batch = 2
    rng = np.random.default_rng(29)

    def constant(shape, low, high):
        return tf.constant(rng.uniform(low, high, shape), dtype)

    parameters = dict(
        syn_decay=constant((neurons, 4), 0.7, 0.9),
        psc_initial=constant((neurons, 4), 0.2, 0.4),
        asc_decay=constant((neurons, 2), 0.7, 0.9),
        asc_amps=constant((neurons, 2), -0.2, 0.0),
        decay=constant((neurons,), 0.8, 0.95),
        current_factor=constant((neurons,), 0.05, 0.1),
        asc_mean=constant((neurons, 2), 0.8, 0.95),
        asc_refractory_decay=constant((neurons, 2), 0.5, 0.8),
        psc_voltage=constant((neurons, 4), 0.02, 0.06),
        rise_voltage=constant((neurons, 4), 0.01, 0.03),
        t_ref_steps=tf.constant([2, 3, 4, 2], tf.int8),
        dt=tf.cast(0.5, dtype),
        v_reset=constant((neurons,), -0.2, 0.2),
        v_th=tf.cast(1.0, dtype),
        dampening=tf.cast(0.3, dtype),
        voltage_gradient_dampening=tf.cast(0.2, dtype),
        hard_reset=False,
        detach_reset=True,
        detach_asc_reset=True,
        pseudo_gauss=True,
        use_fused_event_vjp=True,
    )
    coefficients = pack_nest_state_coefficients(
        tf.constant(neurons, tf.int32),
        dtype,
        **{
            key: parameters[key]
            for key in (
                "syn_decay",
                "psc_initial",
                "asc_decay",
                "asc_amps",
                "decay",
                "current_factor",
                "asc_mean",
                "asc_refractory_decay",
                "psc_voltage",
                "rise_voltage",
                "v_reset",
                "voltage_gradient_dampening",
            )
        },
    )
    kernel_coefficients = tf.transpose(coefficients)
    refractory = tf.zeros((batch, neurons), tf.int8)
    values = [
        constant((batch, neurons), 0.4, 1.4),
        constant((batch, neurons * 2), -0.1, 0.1),
        constant((batch, neurons * 4), -0.2, 0.3),
        constant((batch, neurons * 4), -0.2, 0.3),
        constant((batch, neurons * 4), -0.2, 0.3),
        tf.cast(rng.integers(0, 2, (batch, neurons * 3)), dtype),
    ]

    def evaluate(prepacked):
        with tf.GradientTape() as tape:
            tape.watch(values)
            kwargs = dict(parameters)
            if prepacked:
                kwargs.update(
                    packed_coefficients=coefficients,
                    packed_kernel_coefficients=kernel_coefficients,
                )
            first = fused_nest_state(values[0], refractory, *values[1:], **kwargs)
            second_inputs = [
                first[1],
                first[3],
                first[4],
                first[5],
                values[4] * tf.cast(0.5, dtype),
                first[6],
            ]
            second = fused_nest_state(second_inputs[0], refractory, *second_inputs[1:], **kwargs)
            loss = tf.add_n(
                [
                    tf.reduce_sum(tf.cast(output, tf.float32) * (index + 1))
                    for index, output in enumerate(first + second)
                    if output.dtype.is_floating
                ]
            )
        return first + second, tape.gradient(loss, values)

    expected, expected_grad = tf.function(lambda: evaluate(False))()
    actual, actual_grad = tf.function(lambda: evaluate(True))()
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference, rtol=1e-6, atol=1e-6)
    for observed, reference in zip(actual_grad, expected_grad):
        np.testing.assert_allclose(observed, reference, rtol=1e-6, atol=1e-6)
