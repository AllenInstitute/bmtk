"""S6 NEST state/history fused wrapper contracts."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.custom_ops import glif_state_ops as ops
from bmtk.simulator.dpointnet.cell_models import glif3_cell


class StateHistoryOpsDouble:
    def dpointnet_nest_state_forward(
        self, v, r, asc, rise, psc, current, coefficients, t_ref, dt, v_th,
        hard_reset=False, coefficients_layout="aos", emit_pre_reset_voltage=False,
    ):
        c = tf.transpose(coefficients) if coefficients_layout == "soa" else coefficients
        batch, neurons = tf.shape(v)[0], tf.shape(v)[1]
        asc3 = tf.reshape(asc, (batch, neurons, 2))
        rise3 = tf.cast(tf.reshape(rise, (batch, neurons, 4)), v.dtype)
        psc3 = tf.cast(tf.reshape(psc, (batch, neurons, 4)), v.dtype)
        current3 = tf.cast(tf.reshape(current, (batch, neurons, 4)), v.dtype)
        integrated = psc3 * c[:, 18:22] + rise3 * c[:, 22:26]
        integrated = (integrated[..., 0] + integrated[..., 1]) + (
            integrated[..., 2] + integrated[..., 3]
        )
        mean_asc = asc3[..., 0] * c[:, 14] + asc3[..., 1] * c[:, 15]
        voltage = c[:, 12] * (v * (1 - c[:, 27]) + v * c[:, 27]) + (
            c[:, 13] * mean_asc + integrated
        )
        active = r <= 0
        if hard_reset:
            voltage = tf.where(active, voltage, c[:, 26])
        fired = active & (voltage - v_th > 0)
        new_v = (
            tf.where(fired, c[:, 26], voltage) if hard_reset
            else voltage - tf.cast(fired, v.dtype) * (1 - c[:, 26])
        )
        asc3 = tf.where(active[..., None], asc3 * c[:, 8:10], asc3)
        asc3 = tf.where(fired[..., None], c[:, 10:12] + asc3 * c[:, 16:18], asc3)
        new_r = tf.where(fired, t_ref, tf.maximum(r - 1, 0))
        return (
            voltage if emit_pre_reset_voltage else voltage - v_th,
            new_v,
            new_r,
            tf.reshape(asc3, tf.shape(asc)),
            tf.cast(tf.reshape(rise3 * c[:, :4] + current3 * c[:, 4:8], tf.shape(rise)), rise.dtype),
            tf.cast(tf.reshape(psc3 * c[:, :4] + (dt * c[:, :4]) * rise3, tf.shape(psc)), psc.dtype),
        )

    def dpointnet_nest_state_history_forward(self, *args, **kwargs):
        outputs = self.dpointnet_nest_state_forward(*args[:10], **kwargs)
        threshold = outputs[0] - args[9] if kwargs.get("emit_pre_reset_voltage", False) else outputs[0]
        return outputs + self.dpointnet_spike_shift(threshold, args[1] > 0, args[10])

    def dpointnet_spike_shift(self, threshold, refractory, history):
        spikes = tf.cast((threshold > 0) & ~refractory, history.dtype)
        neurons = tf.shape(threshold)[1]
        return spikes, tf.concat([spikes, history[:, :-neurons]], axis=1)

    def dpointnet_spike_shift_backward_v2(
        self, threshold, refractory, dz, dh, gain, width, pseudo_gauss=False,
    ):
        neurons = tf.shape(threshold)[1]
        derivative = ops._surrogate_derivative(threshold, gain, pseudo_gauss, width)
        dv = tf.cast(dz, threshold.dtype) + tf.cast(dh[:, :neurons], threshold.dtype)
        dv *= tf.where(refractory, tf.zeros_like(derivative), derivative)
        return dv, tf.concat([dh[:, neurons:], tf.zeros_like(dh[:, :neurons])], axis=1)

    def _event_grad(
        self, threshold, r, c, previous_asc, v_th, dampening, gauss_std,
        grad_v, grad_asc, hard_reset, detach_reset, detach_asc_reset, pseudo_gauss,
    ):
        active = r <= 0
        event_grad = tf.zeros_like(threshold)
        if not detach_reset:
            one = tf.cast(1.0, threshold.dtype)
            sensitivity = (
                c[:, 26] - (threshold + v_th)
                if hard_reset else -(one - c[:, 26])
            )
            event_grad += grad_v * sensitivity
        if not detach_asc_reset:
            batch, neurons = tf.shape(threshold)[0], tf.shape(threshold)[1]
            adaptation = tf.reshape(previous_asc, (batch, neurons, 2)) * c[:, 8:10]
            sensitivity = c[:, 10:12] + (c[:, 16:18] - 1) * adaptation
            event_grad += tf.reduce_sum(tf.reshape(grad_asc, (batch, neurons, 2)) * sensitivity, axis=-1)
        derivative = ops._surrogate_derivative(threshold, dampening, pseudo_gauss, gauss_std)
        return tf.where(active, event_grad * derivative, tf.zeros_like(event_grad))

    def dpointnet_nest_state_backward(
        self, threshold, r, coefficients, dt, retention, gt, gv, ga, gr, gp,
        hard_reset=False, coefficients_layout="aos",
    ):
        c = tf.transpose(coefficients) if coefficients_layout == "soa" else coefficients
        batch, neurons = tf.shape(threshold)[0], tf.shape(threshold)[1]
        active = r <= 0
        fired = active & (threshold > 0)
        candidate = gt + (tf.where(fired, tf.zeros_like(gv), gv) if hard_reset else gv)
        if hard_reset:
            candidate = tf.where(active, candidate, tf.zeros_like(candidate))
        ga3 = tf.reshape(ga, (batch, neurons, 2))
        ga3 = tf.where(fired[..., None], ga3 * c[:, 16:18], ga3)
        ga3 = tf.where(active[..., None], ga3 * c[:, 8:10], ga3)
        ga3 += (candidate * c[:, 13])[..., None] * c[:, 14:16]
        gr3 = tf.cast(tf.reshape(gr, (batch, neurons, 4)), threshold.dtype)
        gp3 = tf.cast(tf.reshape(gp, (batch, neurons, 4)), threshold.dtype)
        return (
            (candidate * c[:, 12]) * retention,
            tf.reshape(ga3, tf.shape(ga)),
            tf.cast(tf.reshape(gr3 * c[:, :4] + gp3 * (dt * c[:, :4]) + candidate[..., None] * c[:, 22:26], tf.shape(gr)), gr.dtype),
            tf.cast(tf.reshape(gp3 * c[:, :4] + candidate[..., None] * c[:, 18:22], tf.shape(gp)), gp.dtype),
            tf.cast(tf.reshape(gr3 * c[:, 4:8], tf.shape(gr)), gr.dtype),
        )

    def dpointnet_nest_state_backward_events(
        self, threshold, r, coefficients, dt, retention, gt, gv, ga, gr, gp,
        previous_asc, v_th, dampening, gauss_std, hard_reset=False,
        coefficients_layout="aos", detach_reset=True, detach_asc_reset=True,
        pseudo_gauss=False,
    ):
        c = tf.transpose(coefficients) if coefficients_layout == "soa" else coefficients
        gt = gt + self._event_grad(
            threshold, r, c, previous_asc, v_th, dampening, gauss_std,
            gv, ga, hard_reset, detach_reset, detach_asc_reset, pseudo_gauss,
        )
        return self.dpointnet_nest_state_backward(
            threshold, r, coefficients, dt, retention, gt, gv, ga, gr, gp,
            hard_reset=hard_reset, coefficients_layout=coefficients_layout,
        )

    def dpointnet_nest_state_history_backward(self, *args, **kwargs):
        dampening, gauss_std, spike_grad, history_grad = args[10], args[11], args[12], args[13]
        dv, old_history = self.dpointnet_spike_shift_backward_v2(
            args[0], args[1] > 0, spike_grad, history_grad, dampening, gauss_std,
            pseudo_gauss=kwargs.get("pseudo_gauss", False),
        )
        state = self.dpointnet_nest_state_backward(
            *args[:5], args[5] + dv, *args[6:10],
            hard_reset=kwargs.get("hard_reset", False),
            coefficients_layout=kwargs.get("coefficients_layout", "aos"),
        )
        return state + (old_history,)

    def dpointnet_nest_state_history_backward_events(
        self, threshold, r, coefficients, dt, retention, gt, gv, ga, gr, gp,
        previous_asc, v_th, dampening, gauss_std, spike_grad, history_grad,
        hard_reset=False, coefficients_layout="aos", detach_reset=True,
        detach_asc_reset=True, pseudo_gauss=False,
    ):
        dv, old_history = self.dpointnet_spike_shift_backward_v2(
            threshold, r > 0, spike_grad, history_grad, dampening, gauss_std,
            pseudo_gauss=pseudo_gauss,
        )
        state = self.dpointnet_nest_state_backward_events(
            threshold, r, coefficients, dt, retention, gt + dv, gv, ga, gr, gp,
            previous_asc, v_th, dampening, gauss_std, hard_reset=hard_reset,
            coefficients_layout=coefficients_layout, detach_reset=detach_reset,
            detach_asc_reset=detach_asc_reset, pseudo_gauss=pseudo_gauss,
        )
        return state + (old_history,)


@pytest.fixture
def cpu_ops(monkeypatch):
    monkeypatch.setattr(ops, "_OPS", StateHistoryOpsDouble())
    monkeypatch.setattr(ops, "_glif_gpu_compatibility_error", lambda: None)


def _fixture(dtype=tf.float32, syn_dtype=tf.float32, batch=3, neurons=5, delays=3):
    rng = np.random.default_rng(11)
    v = tf.constant(rng.normal(0.1, 0.4, (batch, neurons)), dtype)
    r = tf.constant(rng.integers(0, 3, (batch, neurons)), tf.int8)
    asc = tf.constant(rng.normal(0, 0.2, (batch, neurons * 2)), dtype)
    rise = tf.constant(rng.normal(0, 0.2, (batch, neurons * 4)), syn_dtype)
    psc = tf.constant(rng.normal(0, 0.2, (batch, neurons * 4)), syn_dtype)
    current = tf.constant(rng.normal(0, 0.1, (batch, neurons * 4)), syn_dtype)
    history = tf.constant(rng.integers(0, 2, (batch, neurons * delays)), syn_dtype)
    params = dict(
        syn_decay=tf.constant(rng.uniform(0.65, 0.95, (neurons, 4)), dtype),
        psc_initial=tf.constant(rng.uniform(0.01, 0.05, (neurons, 4)), dtype),
        asc_decay=tf.constant(rng.uniform(0.6, 0.95, (neurons, 2)), dtype),
        asc_amps=tf.constant(rng.uniform(0.01, 0.08, (neurons, 2)), dtype),
        decay=tf.constant(rng.uniform(0.7, 0.98, neurons), dtype),
        current_factor=tf.constant(rng.uniform(0.2, 0.5, neurons), dtype),
        asc_mean=tf.constant(rng.uniform(0.1, 0.4, (neurons, 2)), dtype),
        asc_refractory_decay=tf.constant(rng.uniform(0.2, 0.8, (neurons, 2)), dtype),
        psc_voltage=tf.constant(rng.uniform(0.03, 0.1, (neurons, 4)), dtype),
        rise_voltage=tf.constant(rng.uniform(0.02, 0.09, (neurons, 4)), dtype),
        t_ref_steps=tf.constant(rng.integers(1, 5, neurons), tf.int8),
        dt=tf.constant(1.0, dtype),
        v_reset=tf.constant(rng.uniform(-0.3, 0.1, neurons), dtype),
        v_th=tf.constant(0.2, dtype),
        dampening=tf.constant(0.35, dtype),
        voltage_gradient_dampening=tf.constant(0.0, dtype),
        gauss_std=tf.constant(0.4, dtype),
    )
    return [v, asc, rise, psc, current, history], r, params


@pytest.mark.parametrize("option", ["auto", None, 1, "true"])
def test_state_history_option_requires_boolean(option):
    with pytest.raises(ValueError, match="boolean"):
        glif3_cell._resolve_fused_state_history(option, "nest", True)


@pytest.mark.parametrize("mode,state,available", itertools.product(
    ["legacy", "nest"], [False, True], [False, True]
))
def test_state_history_eligibility(monkeypatch, mode, state, available):
    monkeypatch.setattr(glif3_cell, "fused_nest_state_history_available", lambda: available)
    assert not glif3_cell._resolve_fused_state_history(False, mode, state)
    if mode == "nest" and state and available:
        assert glif3_cell._resolve_fused_state_history(True, mode, state)
    else:
        with pytest.raises(ValueError, match="requires NEST"):
            glif3_cell._resolve_fused_state_history(True, mode, state)


def test_state_history_stale_abi(monkeypatch):
    monkeypatch.setattr(ops, "_OPS", SimpleNamespace(
        dpointnet_nest_state_forward=True,
        dpointnet_nest_state_backward=True,
        dpointnet_spike_shift_backward_v2=True,
    ))
    monkeypatch.setattr(ops, "_glif_gpu_compatibility_error", lambda: None)
    assert ops.fused_nest_state_available()
    assert not ops.fused_nest_state_history_available()


def test_separate_state_preserves_history_dtype(cpu_ops):
    values, refractory, params = _fixture(syn_dtype=tf.float16)
    values[-1] = tf.cast(values[-1], tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(values[-1])
        outputs = ops.fused_nest_state(
            values[0], refractory, *values[1:], **params, fuse_history=False
        )
        loss = tf.reduce_sum(outputs[6])
    assert outputs[0].dtype == tf.float32
    assert outputs[6].dtype == tf.float32
    gradient = tape.gradient(loss, values[-1])
    assert gradient.dtype == tf.float32
    np.testing.assert_array_equal(
        gradient,
        tf.concat([
            tf.ones_like(values[-1][:, :-tf.shape(values[0])[1]]),
            tf.zeros_like(values[0]),
        ], axis=1),
    )


@pytest.mark.parametrize("event_vjp", [False, True])
@pytest.mark.parametrize("pre_reset", [False, True])
@pytest.mark.parametrize("gaussian", [False, True])
def test_cpu_state_history_matches_separate_path(cpu_ops, event_vjp, pre_reset, gaussian):
    values, refractory, params = _fixture()
    params.update(
        detach_reset=False,
        detach_asc_reset=False,
        pseudo_gauss=gaussian,
        return_pre_reset_voltage=pre_reset,
        use_fused_event_vjp=event_vjp,
    )

    def evaluate(fusion):
        with tf.GradientTape() as tape:
            tape.watch(values)
            outputs = ops.fused_nest_state(
                values[0], refractory, *values[1:], fuse_history=fusion, **params
            )
            loss = tf.add_n([
                tf.reduce_sum(tf.cast(output, tf.float32) ** 2) * (index + 1) / 17
                for index, output in enumerate(outputs) if output.dtype.is_floating
            ])
        return outputs, tape.gradient(loss, values)

    expected = tf.function(lambda: evaluate(False))()
    actual = tf.function(lambda: evaluate(True))()
    for left, right in zip(actual[0], expected[0]):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    for left, right in zip(actual[1], expected[1]):
        np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-6, atol=1e-6)
