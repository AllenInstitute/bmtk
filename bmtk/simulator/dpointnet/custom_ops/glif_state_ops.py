import os
from pathlib import Path

import tensorflow as tf

from .csr_spike_ops import _gpu_compatibility_error

_LIBRARY_PATH = Path(__file__).with_name("_glif_state_ops.so")
_OPS = None
_LOAD_ERROR = None


def _environment_flag(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


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
        compatibility_error = _gpu_compatibility_error()
        if compatibility_error is not None:
            return f"loaded, but {compatibility_error}"
        return f"loaded from {_LIBRARY_PATH}"
    return str(_LOAD_ERROR)


def fused_glif_state_available():
    return _OPS is not None and _gpu_compatibility_error() is None


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
