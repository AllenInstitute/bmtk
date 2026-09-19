import os
from pathlib import Path

import tensorflow as tf

from .csr_spike_ops import _gpu_compatibility_error, _read_built_architectures

_LIBRARY_PATH = Path(__file__).with_name("_glif_state_ops.so")
_ARCHITECTURE_PATH = Path(__file__).with_name("_glif_state_ops.archs")
_SM_ARCHITECTURES, _PTX_ARCHITECTURE = _read_built_architectures(_ARCHITECTURE_PATH)
_OPS = None
_LOAD_ERROR = None


def _environment_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _glif_gpu_compatibility_error():
    return _gpu_compatibility_error(
        _SM_ARCHITECTURES,
        _PTX_ARCHITECTURE,
        _ARCHITECTURE_PATH,
    )


if not _environment_flag("BMTK_DPOINTNET_DISABLE_FUSED_CUDA"):
    if _LIBRARY_PATH.exists():
        try:
            _OPS = tf.load_op_library(str(_LIBRARY_PATH))
        except (tf.errors.NotFoundError, OSError) as exc:
            _LOAD_ERROR = exc
    else:
        _LOAD_ERROR = FileNotFoundError(
            f"Fused DPointNet GLIF state library does not exist at {_LIBRARY_PATH}."
        )


def glif_state_op_status():
    if _environment_flag("BMTK_DPOINTNET_DISABLE_FUSED_CUDA"):
        return "disabled by BMTK_DPOINTNET_DISABLE_FUSED_CUDA"
    if _OPS is not None:
        compatibility_error = _glif_gpu_compatibility_error()
        if compatibility_error is not None:
            return f"loaded, but {compatibility_error}"
        return f"loaded from {_LIBRARY_PATH}"
    return str(_LOAD_ERROR)


def fused_glif_state_available():
    return _OPS is not None and _glif_gpu_compatibility_error() is None


def fused_nest_state_available():
    return fused_glif_state_available() and all(
        hasattr(_OPS, name)
        for name in ("dpointnet_nest_state_forward", "dpointnet_nest_state_backward")
    )


def fused_nest_state(
    voltage,
    refractory,
    asc,
    psc_rise,
    psc,
    currents,
    history,
    *,
    syn_decay,
    psc_initial,
    asc_decay,
    asc_amps,
    decay,
    current_factor,
    asc_mean,
    asc_refractory_decay,
    psc_voltage,
    rise_voltage,
    t_ref_steps,
    dt,
    v_reset,
    v_th,
    dampening,
    voltage_gradient_dampening,
    hard_reset=False,
):
    """Four-basis NEST transition with frozen coefficients and triangular surrogate."""
    if not fused_nest_state_available():
        raise RuntimeError(
            "Fused NEST state requires rebuilt CUDA operators: "
            + glif_state_op_status()
        )
    dtype = voltage.dtype
    neurons = tf.shape(voltage)[1]
    voltage_gradient_dampening = tf.clip_by_value(
        tf.cast(voltage_gradient_dampening, dtype),
        tf.cast(0.0, dtype),
        tf.cast(1.0, dtype),
    )
    coefficients = tf.concat(
        [
            tf.broadcast_to(
                tf.reshape(tf.cast(value, dtype), (-1, width)), (neurons, width)
            )
            for value, width in (
                (syn_decay, 4),
                (psc_initial, 4),
                (asc_decay, 2),
                (asc_amps, 2),
                (decay, 1),
                (current_factor, 1),
                (asc_mean, 2),
                (asc_refractory_decay, 2),
                (psc_voltage, 4),
                (rise_voltage, 4),
                (v_reset, 1),
                (voltage_gradient_dampening, 1),
            )
        ],
        axis=1,
    )
    retention = tf.cast(1.0, dtype) - tf.cast(voltage_gradient_dampening, dtype)

    @tf.custom_gradient
    def transition(*arguments):
        outputs = _OPS.dpointnet_nest_state_forward(*arguments, hard_reset=hard_reset)
        backward_refractory = tf.cast(arguments[1] > 0, dtype)

        def grad(grad_threshold, grad_v, _grad_r, grad_asc, grad_rise, grad_psc):
            gradients = _OPS.dpointnet_nest_state_backward(
                outputs[0],
                tf.cast(backward_refractory, refractory.dtype),
                arguments[6],
                arguments[8],
                retention,
                *[
                    _gradient_like(gradient, output)
                    for gradient, output in zip(
                        (grad_threshold, grad_v, grad_asc, grad_rise, grad_psc),
                        (outputs[0], outputs[1], outputs[3], outputs[4], outputs[5]),
                    )
                ],
                hard_reset=hard_reset,
            )
            return (gradients[0], None, *gradients[1:], None, None, None, None)

        return outputs, grad

    threshold, new_v, new_r, new_asc, new_rise, new_psc = transition(
        voltage,
        refractory,
        asc,
        psc_rise,
        psc,
        currents,
        coefficients,
        tf.cast(t_ref_steps, refractory.dtype),
        tf.cast(dt, dtype),
        tf.cast(v_th, dtype),
    )
    spikes, new_history = fused_spike_shift(
        threshold, refractory > 0, history, dampening
    )
    return spikes, new_v, new_r, new_asc, new_rise, new_psc, new_history


def _gradient_like(gradient, output):
    if gradient is None:
        return tf.zeros_like(output)
    gradient = tf.cast(gradient, output.dtype)
    if gradient.shape.rank == output.shape.rank and gradient.shape.is_compatible_with(
        output.shape
    ):
        return gradient
    gradient = tf.broadcast_to(gradient, tf.shape(output))
    return tf.ensure_shape(gradient, output.shape)


def fused_dense_state(
    prev_z,
    voltage,
    refractory,
    asc,
    psc_rise,
    psc,
    rec_inputs,
    *,
    syn_decay,
    psc_initial,
    asc_decay,
    asc_amps,
    decay,
    current_factor,
    t_ref_steps,
    dt,
    v_reset,
    voltage_gradient_dampening,
    hard_reset=False,
):
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet GLIF state operator is unavailable: {glif_state_op_status()}"
        )

    @tf.custom_gradient
    def transition(
        z,
        v,
        r,
        adaptation,
        rise,
        postsynaptic,
        inputs,
        synaptic_decay,
        initial,
        adaptation_decay,
        adaptation_amplitudes,
        membrane_decay,
        membrane_current_factor,
        refractory_steps,
        timestep,
        reset_voltage,
        gradient_retention,
    ):
        outputs = _OPS.dpointnet_glif_state_forward(
            z,
            v,
            r,
            adaptation,
            rise,
            postsynaptic,
            inputs,
            synaptic_decay,
            initial,
            adaptation_decay,
            adaptation_amplitudes,
            membrane_decay,
            membrane_current_factor,
            refractory_steps,
            timestep,
            reset_voltage,
            hard_reset=hard_reset,
        )

        def grad(grad_v, _grad_r, grad_asc, grad_rise, grad_psc):
            grad_v = _gradient_like(grad_v, outputs[0])
            grad_asc = _gradient_like(grad_asc, outputs[2])
            grad_rise = _gradient_like(grad_rise, outputs[3])
            grad_psc = _gradient_like(grad_psc, outputs[4])
            backward_refractory = r if hard_reset else tf.zeros_like(r)
            gradients = _OPS.dpointnet_glif_state_backward(
                z,
                backward_refractory,
                synaptic_decay,
                initial,
                adaptation_decay,
                adaptation_amplitudes,
                membrane_decay,
                membrane_current_factor,
                refractory_steps,
                timestep,
                gradient_retention,
                grad_v,
                grad_asc,
                grad_rise,
                grad_psc,
                hard_reset=hard_reset,
            )
            return gradients[:2] + (None,) + gradients[2:] + (None,) * 10

        return outputs, grad

    dtype = voltage.dtype
    gradient_retention = tf.cast(1.0, dtype) - tf.cast(
        voltage_gradient_dampening, dtype
    )
    outputs = transition(
        prev_z,
        voltage,
        tf.cast(refractory, t_ref_steps.dtype),
        asc,
        psc_rise,
        psc,
        rec_inputs,
        tf.cast(syn_decay, dtype),
        tf.cast(psc_initial, dtype),
        tf.cast(asc_decay, dtype),
        tf.cast(asc_amps, dtype),
        tf.cast(decay, dtype),
        tf.cast(current_factor, dtype),
        t_ref_steps,
        tf.cast(dt, dtype),
        tf.cast(v_reset, dtype),
        gradient_retention,
    )
    return (outputs[0], tf.cast(outputs[1], refractory.dtype)) + outputs[2:]


def fused_spike_shift(voltage, refractory, history, dampening):
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet GLIF state operator is unavailable: {glif_state_op_status()}"
        )
    refractory = tf.cast(refractory, tf.bool)

    @tf.custom_gradient
    def transition(voltage_value, refractory_value, history_value, scale):
        spikes, new_history = _OPS.dpointnet_spike_shift(
            voltage_value, refractory_value, history_value
        )

        def grad(spike_gradient, history_gradient):
            spike_gradient = _gradient_like(spike_gradient, spikes)
            history_gradient = _gradient_like(history_gradient, new_history)
            voltage_gradient, old_history_gradient = (
                _OPS.dpointnet_spike_shift_backward(
                    voltage_value,
                    refractory_value,
                    spike_gradient,
                    history_gradient,
                    scale,
                )
            )
            return voltage_gradient, None, old_history_gradient, None

        return (spikes, new_history), grad

    return transition(
        voltage,
        refractory,
        history,
        tf.cast(dampening, voltage.dtype),
    )
