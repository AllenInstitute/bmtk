"""Hard native values and independent soft-surrogate voltage credit."""

import itertools
import os

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    spike_function, spike_gauss, straight_through_dampen,
)
from bmtk.simulator.dpointnet.cell_models.nest_dynamics import active_update, spike_reset
from bmtk.simulator.dpointnet.custom_ops import glif_state_ops as ops
from test_s6_nest_state_history import StateHistoryOpsDouble, _fixture
import test_precision_credit as reference


class TypeIndexedOpsDouble(StateHistoryOpsDouble):
    def __getattr__(self, name):
        suffix = "_type_indexed"
        if not name.endswith(suffix):
            raise AttributeError(name)
        base = name[:-len(suffix)]
        operation = getattr(super(), base)
        coefficient_position = 2 if "backward" in base else 6

        def indexed(*args, **kwargs):
            position = coefficient_position
            coefficients = tf.gather(args[position], args[position + 1])
            return operation(
                *args[:position], coefficients, *args[position + 2:],
                coefficients_layout="aos", **kwargs,
            )

        return indexed


def oracle(values, refractory, params, mode="soft_surrogate"):
    voltage, adaptation, rise, psc, current, history = values
    batch, neurons = tf.shape(voltage)[0], tf.shape(voltage)[1]
    asc3 = tf.reshape(adaptation, (batch, neurons, 2))
    rise3 = tf.cast(tf.reshape(rise, (batch, neurons, 4)), voltage.dtype)
    psc3 = tf.cast(tf.reshape(psc, (batch, neurons, 4)), voltage.dtype)
    current3 = tf.cast(tf.reshape(current, (batch, neurons, 4)), voltage.dtype)
    before, new_r, asc3, active = active_update(
        straight_through_dampen(voltage, params["voltage_gradient_dampening"]),
        refractory, asc3, psc3, rise3,
        decay=params["decay"], current_factor=params["current_factor"],
        asc_decay=params["asc_decay"], asc_mean=params["asc_mean"],
        psc_voltage=params["psc_voltage"], rise_voltage=params["rise_voltage"],
        reset_voltage=params["v_reset"], hard_reset=True,
        hard_reset_gradient_mode=mode,
    )
    threshold = before - params["v_th"]
    spikes = (
        spike_gauss(threshold, params["gauss_std"], params["dampening"])
        if params["pseudo_gauss"]
        else spike_function(threshold, params["dampening"])
    )
    spikes = tf.where(active, spikes, tf.zeros_like(spikes))
    new_v, new_r, asc3 = spike_reset(
        before, new_r, asc3, spikes, reset_voltage=params["v_reset"],
        refractory_steps=params["t_ref_steps"], asc_amplitudes=params["asc_amps"],
        asc_refractory_decay=params["asc_refractory_decay"], hard_reset=True,
        detach_reset=params["detach_reset"], detach_asc_reset=params["detach_asc_reset"],
        hard_reset_gradient_mode=mode,
    )
    new_rise = rise3 * params["syn_decay"] + current3 * params["psc_initial"]
    new_psc = psc3 * params["syn_decay"] + (
        params["dt"] * params["syn_decay"]
    ) * rise3
    spikes = tf.cast(spikes, history.dtype)
    result = (
        spikes, new_v, new_r, tf.reshape(asc3, tf.shape(adaptation)),
        tf.cast(tf.reshape(new_rise, tf.shape(rise)), rise.dtype),
        tf.cast(tf.reshape(new_psc, tf.shape(psc)), psc.dtype),
        tf.concat([spikes, history[:, :-neurons]], axis=1),
    )
    if params["return_pre_reset_voltage"]:
        result += (before,)
    return result


def evaluate(values, function):
    with tf.GradientTape() as tape:
        tape.watch(values)
        outputs = function()
        loss = tf.add_n([
            tf.reduce_sum(tf.cast(output, tf.float32) ** 2) * (index + 1) / 17
            for index, output in enumerate(outputs) if output.dtype.is_floating
        ])
    return outputs, tape.gradient(
        loss, values, unconnected_gradients=tf.UnconnectedGradients.ZERO,
    )


@pytest.mark.parametrize("route", ["separate", "events", "history", "indexed", "static", "live", "fallback"])
@pytest.mark.parametrize("detach_reset,detach_asc", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("gaussian,pre_reset", [(False, False), (True, True)])
def test_native_wrapper_matches_independent_reference(
    monkeypatch, route, detach_reset, detach_asc, gaussian, pre_reset,
):
    monkeypatch.setattr(ops, "_OPS", TypeIndexedOpsDouble())
    monkeypatch.setattr(ops, "_glif_gpu_compatibility_error", lambda: None)
    check_route(route, detach_reset, detach_asc, gaussian, pre_reset)


def check_route(route, detach_reset, detach_asc, gaussian, pre_reset, syn_dtype=tf.float32):
    values, refractory, params = _fixture(syn_dtype=syn_dtype)
    params.update(
        hard_reset=True, detach_reset=detach_reset, detach_asc_reset=detach_asc,
        pseudo_gauss=gaussian, return_pre_reset_voltage=pre_reset,
    )
    coefficient_args = {}
    if route in ("indexed", "static", "live", "fallback"):
        coefficients = ops.pack_nest_state_coefficients(
            5, tf.float32, **{
                key: params[key] for key in (
                    "syn_decay", "psc_initial", "asc_decay", "asc_amps", "decay",
                    "current_factor", "asc_mean", "asc_refractory_decay", "psc_voltage",
                    "rise_voltage", "v_reset", "voltage_gradient_dampening",
                )
            },
        )
        identity = (
            tf.Variable(route != "fallback", trainable=False)
            if route in ("live", "fallback") else tf.constant(True)
        )
        coefficient_args = dict(
            packed_coefficients=coefficients,
            packed_kernel_coefficients=tf.transpose(coefficients),
            packed_type_coefficients=coefficients,
            type_indices=tf.range(5, dtype=tf.int64),
            type_indexed_identity=identity,
        )

    def native(mode):
        return ops.fused_nest_state(
            values[0], refractory, *values[1:], **params,
            hard_reset_gradient_mode=mode,
            use_fused_event_vjp=route != "separate",
            fuse_history=route in ("history", "static", "live", "fallback"),
            **coefficient_args,
        )

    expected = tf.function(lambda: evaluate(values, lambda: oracle(values, refractory, params)))()
    actual = tf.function(lambda: evaluate(values, lambda: native("soft_surrogate")))()
    exact = tf.function(lambda: native("exact"))()
    for index, (left, right) in enumerate(zip(actual[0], exact)):
        np.testing.assert_array_equal(left, right, err_msg=f"hard forward output {index}")
    for left, right in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
        assert left.dtype == right.dtype
        np.testing.assert_allclose(
            left, right, rtol=3e-3 if left.dtype == tf.float16 else 1e-6, atol=1e-6,
        )
    assert np.isfinite(actual[1][0]).all()
    assert np.any(np.asarray(actual[1][0])[np.asarray(refractory) > 0] != 0)


@pytest.mark.parametrize("route", ["separate", "events", "history", "indexed", "static", "live", "fallback"])
@pytest.mark.parametrize("detach_reset,detach_asc", itertools.product([False, True], repeat=2))
@pytest.mark.parametrize("gaussian,pre_reset", [(False, False), (True, True)])
@pytest.mark.parametrize("syn_dtype", [tf.float32, tf.float16])
def test_cuda_routes_match_independent_reference(
    route, detach_reset, detach_asc, gaussian, pre_reset, syn_dtype,
):
    if not ops.fused_nest_state_history_type_indexed_available():
        if os.environ.get("DPOINTNET_REQUIRE_GPU") == "1":
            pytest.fail("Required native NEST GPU capabilities unavailable")
        pytest.skip("Compatible native NEST GPU operators required")
    with tf.device("/GPU:0"):
        check_route(route, detach_reset, detach_asc, gaussian, pre_reset, syn_dtype)


@pytest.mark.parametrize("precision", ["compute", "selective"])
@pytest.mark.parametrize("gaussian,detach_reset,detach_asc", itertools.product([False, True], repeat=3))
def test_cuda_rollout_matches_tensorflow(
    monkeypatch, precision, gaussian, detach_reset, detach_asc,
):
    if not ops.fused_nest_state_available():
        if os.environ.get("DPOINTNET_REQUIRE_GPU") == "1":
            pytest.fail("Required native NEST GPU operator unavailable")
        pytest.skip("Compatible native NEST GPU operator required")
    original = reference.make_cell

    def make_cell(*args, **kwargs):
        kwargs["hard_reset_gradient_mode"] = "soft_surrogate"
        if kwargs.get("fused"):
            kwargs["use_fused_nest_event_vjp"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "make_cell", make_cell)
    reference.test_fused_cell_matches_fallback(
        "nest", precision, gaussian, True, detach_reset, detach_asc,
    )


@pytest.mark.parametrize("chunk", [3, 25])
@pytest.mark.parametrize("device_poisson", [False, True])
def test_cuda_checkpoint_replays_native_hard_surrogate(monkeypatch, chunk, device_poisson):
    if not ops.fused_nest_state_history_type_indexed_available():
        if os.environ.get("DPOINTNET_REQUIRE_GPU") == "1":
            pytest.fail("Required native NEST GPU capabilities unavailable")
        pytest.skip("Compatible native NEST GPU operators required")
    original = reference.make_cell

    def make_cell(*args, **kwargs):
        kwargs.update(
            hard_reset=True, hard_reset_gradient_mode="soft_surrogate",
            use_fused_nest_event_vjp=True, use_fused_state_history=True,
            use_prepacked_nest_coefficients=True, use_type_indexed_nest_coefficients=True,
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(reference, "make_cell", make_cell)
    reference.test_selective_exact_replay_and_poisson("nest", chunk, True, device_poisson)


@pytest.mark.parametrize("selective,gaussian,detach", itertools.product([False, True], repeat=3))
def test_cuda_native_pre_reset_credit(monkeypatch, selective, gaussian, detach):
    if not ops.fused_nest_event_vjp_available():
        if os.environ.get("DPOINTNET_REQUIRE_GPU") == "1":
            pytest.fail("Required native NEST GPU capabilities unavailable")
        pytest.skip("Compatible native NEST GPU operators required")
    import test_voltage_floor_temporal as floor_reference
    original = floor_reference.make_cell

    def make_cell(*args, **kwargs):
        kwargs["hard_reset_gradient_mode"] = "soft_surrogate"
        if kwargs.get("fused"):
            kwargs["use_fused_nest_event_vjp"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(floor_reference, "make_cell", make_cell)
    floor_reference.test_fused_native_pre_reset_credit_matches_tensorflow(
        selective, gaussian, detach, True,
    )


@pytest.mark.parametrize("replay,penalty,floor", itertools.product(
    ["record", "recompute"], ["range", "threshold"], [False, True],
))
def test_cuda_fp32_temporal_event_vjps(monkeypatch, replay, penalty, floor):
    if not ops.fused_nest_event_vjp_available():
        if os.environ.get("DPOINTNET_REQUIRE_GPU") == "1":
            pytest.fail("Required native NEST GPU capabilities unavailable")
        pytest.skip("Compatible native NEST GPU operators required")
    import test_nest_event_vjp as event_reference
    original = event_reference.make_fused_cell

    def make_cell(*args, **kwargs):
        kwargs.update(hard_reset=True, hard_reset_gradient_mode="soft_surrogate")
        return original(*args, **kwargs)

    monkeypatch.setattr(event_reference, "make_fused_cell", make_cell)
    event_reference.test_fp32_same_cache_preserves_canonical_vjps(replay, penalty, floor)
