import numpy as np
from scipy.special import exprel
import tensorflow as tf


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
    direct_current=0.0
):
    active = refractory <= 0
    mean_adaptation = tf.reduce_sum(adaptation * asc_mean, axis=-1)
    candidate = (
        decay * voltage
        + current_factor * (direct_current + mean_adaptation)
        + tf.reduce_sum(psc * psc_voltage + psc_rise * rise_voltage, axis=-1)
    )
    voltage = tf.where(active, candidate, reset_voltage) if hard_reset else candidate
    adaptation = tf.where(active[..., None], adaptation * asc_decay, adaptation)
    remaining = tf.maximum(refractory - tf.cast(1, refractory.dtype), 0)
    return voltage, remaining, adaptation, active


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
    hard_reset
):
    fired = tf.stop_gradient(spikes) > 0
    voltage = (
        tf.where(fired, reset_voltage, voltage)
        if hard_reset
        else voltage - tf.stop_gradient(spikes) * (1 - reset_voltage)
    )
    refractory = tf.where(
        fired, tf.cast(refractory_steps, refractory.dtype), refractory
    )
    adaptation = tf.where(
        fired[..., None], asc_amplitudes + adaptation * asc_refractory_decay, adaptation
    )
    return voltage, refractory, adaptation
