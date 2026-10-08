"""Opt-in NEST event VJP: unchanged primal and independent event derivatives."""
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from bmtk.simulator.dpointnet.custom_ops import glif_state_ops as ops
import test_precision_credit as reference
import test_voltage_floor_temporal as floor_reference
from test_recurrent_accumulation import make_fused_cell
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


def require_gpu():
    if not ops.fused_nest_event_vjp_available():
        pytest.skip("Rebuilt NEST event VJP GPU operator unavailable")


def test_option_requires_boolean():
    with pytest.raises(ValueError, match="use_fused_nest_event_vjp must be"):
        reference.make_cell("nest", use_fused_nest_event_vjp="true")


@pytest.mark.parametrize("mode,fused", [("legacy", False), ("nest", False)])
def test_option_rejects_unsupported_model(mode, fused):
    with pytest.raises(ValueError, match="use_fused_nest_event_vjp=True requires"):
        reference.make_cell(mode, fused=fused, use_fused_nest_event_vjp=True)


def test_old_binary_does_not_satisfy_event_abi(monkeypatch):
    monkeypatch.setattr(ops, "fused_nest_state_available", lambda: True)
    monkeypatch.setattr(ops, "_OPS", object())
    assert not ops.fused_nest_event_vjp_available()


def test_event_abi_shape_inference():
    library = os.environ.get("NEST_EVENT_SHAPE_LIBRARY")
    if library:
        native = tf.load_op_library(library)
    elif ops.fused_nest_event_vjp_available():
        native = ops._OPS
    else:
        pytest.skip("Registration-only library or GPU build required")
    specifications = [
        tf.TensorSpec((2, 5), tf.float32), tf.TensorSpec((2, 5), tf.int16),
        tf.TensorSpec((28, 5), tf.float32), tf.TensorSpec((), tf.float32),
        tf.TensorSpec((), tf.float32), tf.TensorSpec((2, 5), tf.float32),
        tf.TensorSpec((2, 5), tf.float32), tf.TensorSpec((2, 10), tf.float32),
        tf.TensorSpec((2, 20), tf.float16), tf.TensorSpec((2, 20), tf.float16),
        tf.TensorSpec((2, 10), tf.float32), tf.TensorSpec((), tf.float32),
        tf.TensorSpec((), tf.float32), tf.TensorSpec((), tf.float32)]
    function = tf.function(lambda *values: native.dpointnet_nest_state_backward_events(
        *values, coefficients_layout="soa", detach_asc_reset=False, pseudo_gauss=True))
    concrete = function.get_concrete_function(*specifications)
    assert [tuple(value.shape) for value in concrete.outputs] == [
        (2, 5), (2, 10), (2, 20), (2, 20), (2, 20)]
    specifications[10] = tf.TensorSpec((2, 9), tf.float32)
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError)):
        function.get_concrete_function(*specifications)


@pytest.mark.parametrize("precision", ["compute", "selective"])
@pytest.mark.parametrize("gaussian,hard_reset,detach_reset,detach_asc",
                         itertools.product([False, True], repeat=4))
def test_event_fusion_against_independent_nest_oracle(
        monkeypatch, precision, gaussian, hard_reset, detach_reset, detach_asc):
    require_gpu()
    original = reference.make_cell

    def make_cell(*args, **kwargs):
        if kwargs.get("fused"):
            kwargs["use_fused_nest_event_vjp"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "make_cell", make_cell)
    reference.test_fused_cell_matches_fallback(
        "nest", precision, gaussian, hard_reset, detach_reset, detach_asc)


@pytest.mark.parametrize("selective,gaussian,detach,hard_reset",
                         itertools.product([False, True], repeat=4))
def test_native_voltage_floor_credit_remains_independent(
        monkeypatch, selective, gaussian, detach, hard_reset):
    require_gpu()
    original = floor_reference.make_cell

    def make_cell(*args, **kwargs):
        if kwargs.get("fused"):
            kwargs["use_fused_nest_event_vjp"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(floor_reference, "make_cell", make_cell)
    floor_reference.test_fused_native_pre_reset_credit_matches_tensorflow(
        selective, gaussian, detach, hard_reset)


@pytest.mark.parametrize("chunk", [1, 7, 25])
def test_ordinary_checkpoint_and_poisson(monkeypatch, chunk):
    require_gpu()
    original = reference.make_cell

    def make_cell(*args, **kwargs):
        kwargs["use_fused_nest_event_vjp"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "make_cell", make_cell)
    reference.test_selective_exact_replay_and_poisson("nest", chunk, True, False)


@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
@pytest.mark.parametrize("penalty_mode", ["range", "threshold"])
@pytest.mark.parametrize("with_floor", [False, True])
def test_fp32_same_cache_preserves_canonical_vjps(replay_mode, penalty_mode, with_floor):
    require_gpu()
    from types import SimpleNamespace
    from bmtk.simulator.dpointnet.loss_functions import VoltageRateFloor
    loss = VoltageRateFloor(SimpleNamespace(dt=1.0)) if with_floor else None
    cell = make_fused_cell(
        "nest", replay_mode, use_fused_nest_event_vjp=True,
        online_voltage_losses=[loss] if loss else [],
        return_voltage_sequences=False, voltage_penalty_mode=penalty_mode,
        pseudo_gauss=True, detach_reset=False, detach_asc_reset=False)
    if loss:
        loss.build(cell)
        loss.initialize_rates([0.0, 0.0])
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    state = list(cell.zero_state(32, cell.compute_dtype))
    state[0] = tf.ones_like(state[0])
    state[3] = tf.ones_like(state[3]) * .1
    state = tuple(state)
    x = tf.zeros((32, 11, 0), tf.float32)
    masters = tuple(value.value if not callable(getattr(value, "value", None)) else value
                    for value in cell.trainable_variables)
    original = [value.numpy().copy() for value in masters]

    @tf.function
    def evaluate():
        cache = runner._forward(x, state)
        final_gradients = [tf.ones_like(value, dtype=tf.float32) * .03
                           if value.dtype.is_floating else None for value in state]

        def reverse(fused):
            cell.use_fused_nest_event_vjp = fused
            return runner._backward(
                x, state, cache, (None, None), final_gradients, masters, capture=True)
        return reverse(False), reverse(True)

    try:
        expected, actual = evaluate()
        for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
            assert a.dtype == b.dtype
            np.testing.assert_allclose(a, b, rtol=4e-5, atol=1e-7)
        assert all(value.dtype == tf.float32 for value in actual[2])
        assert np.any(np.asarray(actual[2][0]) != 0)
        for value, saved in zip(masters, original):
            np.testing.assert_array_equal(value, saved)
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("dtype,psc_dtype", [
    (tf.float16, tf.float16), (tf.float32, tf.float16), (tf.float32, tf.float32)])
@pytest.mark.parametrize("gaussian,hard_reset,detach_reset,detach_asc",
                         itertools.product([False, True], repeat=4))
@pytest.mark.parametrize("tiny", [False, True])
def test_original_fused_vjp_rounding_and_refractory(
        dtype, psc_dtype, gaussian, hard_reset, detach_reset, detach_asc, tiny):
    require_gpu()
    rng = np.random.default_rng(733)
    batch, neurons = 2, 5

    def tensor(shape, low, high, target=dtype):
        return tf.constant(rng.uniform(low, high, shape), target)

    parameters = dict(
        syn_decay=tensor((neurons, 4), .7, .9),
        psc_initial=tensor((neurons, 4), .2, .4),
        asc_decay=tensor((neurons, 2), .7, .9),
        asc_amps=tensor((neurons, 2), -.2, .2),
        decay=tensor((neurons,), .8, .95),
        current_factor=tensor((neurons,), .05, .1),
        asc_mean=tensor((neurons, 2), .8, .95),
        asc_refractory_decay=tensor((neurons, 2), .5, 1.3),
        psc_voltage=tensor((neurons, 4), .02, .06),
        rise_voltage=tensor((neurons, 4), .01, .03),
        t_ref_steps=tf.constant([2, 3, 300, 2, 3], tf.int16),
        dt=tf.cast(.5, dtype), v_reset=tensor((neurons,), -.2, .2),
        v_th=tf.cast(1., dtype), dampening=tf.cast(.05, dtype),
        voltage_gradient_dampening=tf.cast(0., dtype),
        hard_reset=hard_reset, detach_reset=detach_reset,
        detach_asc_reset=detach_asc, pseudo_gauss=gaussian, gauss_std=.28)
    refractory = tf.constant([[0, 300, 0, 2, 0]] * batch, tf.int16)
    values = [
        tf.constant([[.4, 1.4, 1.4, .8, 1.]] * batch, dtype),
        tensor((batch, neurons * 2), -.1, .1),
        tensor((batch, neurons * 4), -.2, .3, psc_dtype),
        tensor((batch, neurons * 4), -.2, .3, psc_dtype),
        tensor((batch, neurons * 4), -.2, .3, psc_dtype),
        tf.cast(rng.integers(0, 2, (batch, neurons * 3)), psc_dtype)]
    scale = (2.0 ** -22 if dtype == tf.float16 else 2.0 ** -130) if tiny else 1.

    def evaluate(fused):
        with tf.GradientTape() as tape:
            tape.watch(values)
            outputs = ops.fused_nest_state(
                values[0], refractory, *values[1:], use_fused_event_vjp=fused,
                **parameters)
        floating = [x for x in outputs if x.dtype.is_floating]
        gradients = tape.gradient(
            floating, values,
            output_gradients=[tf.ones_like(x) * tf.cast(scale, x.dtype) for x in floating],
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        return outputs, gradients

    with tf.device("/GPU:0"):
        expected, expected_gradients = tf.function(lambda: evaluate(False))()
        function = tf.function(lambda: evaluate(True)).get_concrete_function()
        actual, gradients = function()
    assert any(op.type == "DpointnetNestStateBackwardEvents"
               for op in function.graph.get_operations())
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
        assert a.dtype == b.dtype
    maximum = mismatch = 0
    for a, b in zip(gradients, expected_gradients):
        assert a.dtype == b.dtype
        aa, bb = a.numpy().astype(np.float64), b.numpy().astype(np.float64)
        maximum = max(maximum, float(np.max(np.abs(aa - bb))))
        mismatch += int(np.count_nonzero(aa != bb))
        if tiny:
            tolerance = 2 ** -24 if a.dtype == tf.float16 else 4 * 2 ** -149
            np.testing.assert_allclose(aa, bb, atol=tolerance, rtol=2e-6)
        else:
            np.testing.assert_allclose(aa, bb, atol=2e-6, rtol=3e-3 if dtype == tf.float16 else 2e-6)
    receipt = os.environ.get("NEST_EVENT_NUMERIC_RECEIPT")
    if receipt:
        with Path(receipt).open("a") as stream:
            stream.write(json.dumps(dict(dtype=dtype.name, psc_dtype=psc_dtype.name,
                gaussian=gaussian, hard_reset=hard_reset, detach_reset=detach_reset,
                detach_asc_reset=detach_asc, tiny=tiny, max_absolute_error=maximum,
                non_bitwise_elements=mismatch)) + "\n")
