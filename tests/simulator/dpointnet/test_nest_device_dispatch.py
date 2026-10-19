"""Live device dispatch must retain the ordinary state operator's arithmetic."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("dtype,psc_dtype", [
    (tf.float32, tf.float32), (tf.float32, tf.float16), (tf.float16, tf.float16),
])
@pytest.mark.parametrize("refractory_dtype", [tf.int8, tf.int16])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize("kind", [
    "forward", "history_forward", "backward", "backward_events",
    "history_backward", "history_backward_events",
])
def test_live_dispatch_bitwise_mutation_and_reused_graph(
    dtype, psc_dtype, refractory_dtype, hard_reset, kind,
):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    if not glif_state_ops.fused_nest_device_dispatch_available():
        pytest.skip("Compatible rebuilt device-dispatch operators required")
    ops = glif_state_ops._OPS
    rng = np.random.default_rng(137)
    batch, neurons = 3, 6
    backward, history, events = "backward" in kind, "history" in kind, "events" in kind
    with tf.device("/GPU:0"):
        indices = tf.constant([0, 1, 0, 2, 1, 2], tf.int64)
        table = rng.uniform(.01, .2, (3, 28))
        table[:, :4] += .7
        table[:, 8:10] = .9
        table[:, 10:12] *= -1
        table[:, 12] = .95
        table[:, 26] = -.1
        table[:, 27] = .25
        full = tf.Variable(tf.gather(tf.constant(table, dtype), indices))
        original = full.numpy()

        def values(shape, target=dtype):
            return tf.constant(rng.uniform(-.3, .4, shape), target)

        voltage = values((batch, neurons))
        refractory = tf.constant(np.tile([0, 1, 0, 2, 0, 1], (batch, 1)), refractory_dtype)
        dt, threshold = tf.constant(.5, dtype), tf.constant(.1, dtype)
        if backward:
            prefix = [voltage, refractory]
            suffix = [
                dt, tf.constant(.75, dtype), values((batch, neurons)), values((batch, neurons)),
                values((batch, 2 * neurons)), values((batch, 4 * neurons), psc_dtype),
                values((batch, 4 * neurons), psc_dtype),
            ]
            options = dict(hard_reset=hard_reset)
            if events:
                suffix += [
                    values((batch, 2 * neurons)), threshold,
                    tf.constant(.3, dtype), tf.constant(.5, dtype),
                ]
                options.update(detach_reset=False, detach_asc_reset=False, pseudo_gauss=True)
            elif history:
                suffix += [tf.constant(.3, dtype), tf.constant(.5, dtype)]
                options["pseudo_gauss"] = True
            if history:
                suffix += [
                    values((batch, neurons), psc_dtype), values((batch, 3 * neurons), psc_dtype),
                ]
        else:
            prefix = [
                voltage, refractory, values((batch, 2 * neurons)),
                values((batch, 4 * neurons), psc_dtype), values((batch, 4 * neurons), psc_dtype),
                values((batch, 4 * neurons), psc_dtype),
            ]
            suffix = [tf.constant([2, 3, 2, 4, 3, 4], refractory_dtype), dt, threshold]
            if history:
                suffix += [values((batch, 3 * neurons), psc_dtype)]
            options = dict(hard_reset=hard_reset, emit_pre_reset_voltage=True)
        operation = getattr(ops, "dpointnet_nest_state_" + kind + "_device_dispatch")
        reference = getattr(ops, "dpointnet_nest_state_" + kind)

        @tf.function
        def evaluate():
            coefficients = full.read_value()
            live_table = tf.gather(coefficients, [0, 1, 3])
            identity = tf.reduce_all(tf.equal(coefficients, tf.gather(live_table, indices)))
            actual = operation(
                *prefix, live_table, indices, *suffix,
                fallback_coefficients=coefficients, type_identity=identity, **options,
            )
            expected = reference(*prefix, coefficients, *suffix, **options)
            return actual, expected, identity

        def compare(identity):
            actual, expected, resolved = evaluate()
            assert bool(resolved.numpy()) is identity
            for left, right in zip(actual, expected):
                np.testing.assert_array_equal(left.numpy(), right.numpy())
            return [value.numpy() for value in actual]

        initial = compare(True)
        traces = evaluate.experimental_get_tracing_count()
        changed = original.copy()
        changed[2, 12] *= .5
        full.assign(changed)
        compare(False)
        changed[0] = changed[2]
        full.assign(changed)
        compare(True)
        full.assign(original)
        restored = compare(True)
        for left, right in zip(initial, restored):
            np.testing.assert_array_equal(left, right)
        assert evaluate.experimental_get_tracing_count() == traces
        operations = [node.op for node in evaluate.get_concrete_function().graph.as_graph_def().node]
        assert sum(name.endswith("DeviceDispatch") for name in operations) == 1
        assert not any(name in ("If", "StatelessIf") for name in operations)


@pytest.mark.parametrize("bad_shape", [(4, 28), (5, 27), (28, 5), (140,)])
def test_device_dispatch_rejects_fallback_shape(bad_shape):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops
    from test_s8_type_indexed_nest_coefficients import _type_indexed_specs

    if glif_state_ops._OPS is None or not hasattr(
        glif_state_ops._OPS, "dpointnet_nest_state_forward_device_dispatch",
    ):
        pytest.skip("Rebuilt device-dispatch registration required")
    specs = _type_indexed_specs(False)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        tf.function(
            glif_state_ops._OPS.dpointnet_nest_state_forward_device_dispatch,
        ).get_concrete_function(
            *specs, tf.TensorSpec(bad_shape, tf.float32), tf.TensorSpec((), tf.bool),
        )
