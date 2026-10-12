import numpy as np
from scipy.special import exprel
import tensorflow as tf
from .._options import UNSET, validate_bool_option


def resolve_hard_reset_options(
    hard_reset=UNSET, gradient_mode=UNSET, *, dynamics_mode="nest", training=False
):
    omitted_reset = hard_reset is UNSET
    if omitted_reset:
        hard_reset = dynamics_mode == "nest"
    elif hard_reset is None:
        hard_reset = dynamics_mode == "nest" and not training
    validate_bool_option(hard_reset, "hard_reset")
    if gradient_mode is UNSET:
        gradient_mode = "soft_surrogate" if omitted_reset and dynamics_mode == "nest" else "exact"
    validate_hard_reset_gradient_mode(
        gradient_mode, hard_reset=hard_reset, dynamics_mode=dynamics_mode
    )
    return hard_reset, gradient_mode


def validate_hard_reset_gradient_mode(mode, *, hard_reset, dynamics_mode="nest"):
    if mode not in ("exact", "soft_surrogate"):
        raise ValueError("hard_reset_gradient_mode must be 'exact' or 'soft_surrogate'.")
    if mode == "soft_surrogate" and (hard_reset is not True or dynamics_mode != "nest"):
        raise ValueError(
            "hard_reset_gradient_mode='soft_surrogate' requires explicit "
            "hard_reset=True and dynamics_mode='nest'."
        )


@tf.custom_gradient
def _hard_forward_surrogate_backward(hard_value, surrogate_value):
    def grad(dy):
        return None, dy

    return tf.identity(hard_value), grad


def time_steps(milliseconds, dt):
    step_ticks = int(round(dt * 1000))
    if step_ticks < 1 or not np.isclose(step_ticks, dt * 1000, rtol=0, atol=1e-6):
        raise ValueError("NEST-compatible dt must be a multiple of 0.001 ms")
    values = np.asarray(milliseconds, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("NEST times must be finite and nonnegative")
    ticks = np.rint(values * 1000).astype(np.int64)
    return (ticks + step_ticks - 1) // step_ticks


def integration_coefficients(dt, capacitance, conductance, tau_basis):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    capacitance = np.asarray(capacitance, dtype=np.float64)
    conductance = np.asarray(conductance, dtype=np.float64)
    tau_basis = np.asarray(tau_basis, dtype=np.float64)
    if any(
        not np.isfinite(values).all() or (values <= 0).any()
        for values in (capacitance, conductance, tau_basis)
    ):
        raise ValueError(
            "Capacitance, conductance and synaptic time constants must be positive"
        )
    membrane_rate = conductance / capacitance
    decay = np.exp(-dt * membrane_rate)
    difference = dt * (membrane_rate[:, None] - 1 / tau_basis[None, :])
    current = decay[:, None] * dt * exprel(difference) / capacitance[:, None]
    integral = np.empty_like(difference)
    small = np.abs(difference) < 1e-3
    values = difference[small]
    integral[small] = (
        0.5 + values / 3 + values**2 / 8 + values**3 / 30 + values**4 / 144
    )
    values = difference[~small]
    integral[~small] = ((values - 1) * np.expm1(values) + values) / values**2
    rise = decay[:, None] * dt**2 * integral / capacitance[:, None]
    return decay, -np.expm1(-dt * membrane_rate) / conductance, current, rise


def active_update(
    voltage,
    refractory,
    adaptation,
    psc,
    psc_rise,
    *,
    decay,
    current_factor,
    asc_decay,
    asc_mean,
    psc_voltage,
    rise_voltage,
    reset_voltage,
    hard_reset,
    direct_current=0.0,
    hard_reset_gradient_mode="exact",
):
    active = refractory <= 0
    mean_adaptation = tf.reduce_sum(adaptation * asc_mean, axis=-1)
    candidate = (
        decay * voltage
        + current_factor * (direct_current + mean_adaptation)
        + tf.reduce_sum(psc * psc_voltage + psc_rise * rise_voltage, axis=-1)
    )
    voltage = tf.where(active, candidate, reset_voltage) if hard_reset else candidate
    if hard_reset_gradient_mode == "soft_surrogate":
        voltage = _hard_forward_surrogate_backward(voltage, candidate)
    adaptation = tf.where(active[..., None], adaptation * asc_decay, adaptation)
    remaining = tf.maximum(refractory - tf.cast(1, refractory.dtype), 0)
    return voltage, remaining, adaptation, active


@tf.custom_gradient
def _event_select(before, after, event, sensitivity):
    """Keep the Boolean forward map, with its explicitly chosen event Jacobian."""
    fired = event > 0
    result = tf.where(fired, after, before)

    def grad(dy):
        return (
            tf.where(fired, tf.zeros_like(dy), dy),
            tf.where(fired, dy, tf.zeros_like(dy)),
            dy * sensitivity,
            None,
        )

    return result, grad


def spike_reset(
    voltage,
    refractory,
    adaptation,
    spikes,
    *,
    reset_voltage,
    refractory_steps,
    asc_amplitudes,
    asc_refractory_decay,
    hard_reset,
    detach_reset=True,
    detach_asc_reset=True,
    hard_reset_gradient_mode="exact",
):
    fired = tf.stop_gradient(spikes) > 0
    reset_event = tf.stop_gradient(spikes) if detach_reset else spikes
    voltage_before_reset = voltage
    voltage = (
        _event_select(
            voltage,
            tf.broadcast_to(reset_voltage, tf.shape(voltage)),
            reset_event,
            reset_voltage - voltage,
        )
        if hard_reset
        else voltage - reset_event * (1 - reset_voltage)
    )
    if hard_reset_gradient_mode == "soft_surrogate":
        soft_voltage = voltage_before_reset - reset_event * (1 - reset_voltage)
        voltage = _hard_forward_surrogate_backward(voltage, soft_voltage)
    refractory = tf.where(
        fired, tf.cast(refractory_steps, refractory.dtype), refractory
    )
    asc_event = tf.stop_gradient(spikes) if detach_asc_reset else spikes
    adaptation = _event_select(
        adaptation,
        asc_amplitudes + adaptation * asc_refractory_decay,
        tf.broadcast_to(asc_event[..., None], tf.shape(adaptation)),
        asc_amplitudes + (asc_refractory_decay - 1) * adaptation,
    )
    return voltage, refractory, adaptation
