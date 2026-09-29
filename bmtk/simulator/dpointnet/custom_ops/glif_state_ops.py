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
    return (
        _OPS is not None
        and hasattr(_OPS, "dpointnet_spike_shift_backward_v2")
        and _glif_gpu_compatibility_error() is None
    )


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
    detach_reset=True,
    detach_asc_reset=True,
    pseudo_gauss=False,
    gauss_std=0.5,
    return_pre_reset_voltage=False,
):
    """Four-basis NEST transition with live frozen coefficients and event VJPs."""
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
    forward_options = {}
    if return_pre_reset_voltage:
        import inspect
        if "emit_pre_reset_voltage" not in inspect.signature(
            _OPS.dpointnet_nest_state_forward
        ).parameters:
            raise RuntimeError("VoltageRateFloor requires rebuilt fused NEST state operators.")
        forward_options["emit_pre_reset_voltage"] = True

    @tf.custom_gradient
    def transition(*arguments):
        layout = "soa" if dtype == tf.float32 else "aos"
        kernel_coefficients = tf.transpose(arguments[6]) if layout == "soa" else arguments[6]
        outputs = _OPS.dpointnet_nest_state_forward(
            *arguments[:6], kernel_coefficients, *arguments[7:10],
            hard_reset=hard_reset, coefficients_layout=layout, **forward_options
        )
        # The enclosing custom gradient owns the VJP. Nested rollout tapes must
        # not attempt to differentiate the opaque forward CUDA op itself.
        outputs = tuple(tf.stop_gradient(value) for value in outputs)
        backward_refractory = tf.cast(arguments[1] > 0, dtype)
        threshold_for_backward = (
            outputs[0] - arguments[9] if return_pre_reset_voltage else outputs[0]
        )

        def grad(grad_threshold, grad_v, _grad_r, grad_asc, grad_rise, grad_psc):
            grad_threshold = _gradient_like(grad_threshold, outputs[0])
            grad_v = _gradient_like(grad_v, outputs[1])
            grad_asc = _gradient_like(grad_asc, outputs[3])
            if not detach_reset or not detach_asc_reset:
                event_grad = tf.zeros_like(outputs[0])
                params = arguments[6]
                if not detach_reset:
                    reset_sensitivity = (
                        params[:, 26] - (threshold_for_backward + arguments[9])
                        if hard_reset
                        else -(1 - params[:, 26])
                    )
                    event_grad += grad_v * reset_sensitivity
                if not detach_asc_reset:
                    adaptation = tf.reshape(arguments[2], (-1, neurons, 2))
                    adaptation = tf.where(
                        backward_refractory[..., None] > 0,
                        adaptation,
                        adaptation * params[:, 8:10],
                    )
                    sensitivity = params[:, 10:12] + (params[:, 16:18] - 1) * adaptation
                    event_grad += tf.reduce_sum(
                        tf.reshape(grad_asc, (-1, neurons, 2)) * sensitivity, axis=-1
                    )
                derivative = _surrogate_derivative(
                    threshold_for_backward, arguments[10], pseudo_gauss, arguments[11]
                )
                grad_threshold += tf.where(
                    backward_refractory > 0,
                    tf.zeros_like(event_grad),
                    event_grad * derivative,
                )
            gradients = _OPS.dpointnet_nest_state_backward(
                threshold_for_backward,
                tf.cast(backward_refractory, refractory.dtype),
                kernel_coefficients,
                arguments[8],
                1 - arguments[6][0, 27],
                *[
                    _gradient_like(gradient, output)
                    for gradient, output in zip(
                        (grad_threshold, grad_v, grad_asc, grad_rise, grad_psc),
                        (outputs[0], outputs[1], outputs[3], outputs[4], outputs[5]),
                    )
                ],
                hard_reset=hard_reset,
                coefficients_layout=layout,
            )
            return (
                gradients[0],
                None,
                *gradients[1:],
                None,
                None,
                None,
                None,
                None,
                None,
            )

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
        tf.cast(dampening, dtype),
        tf.cast(gauss_std, dtype),
    )
    spikes, new_history = fused_spike_shift(
        threshold - tf.cast(v_th, dtype) if return_pre_reset_voltage else threshold,
        refractory > 0,
        history,
        dampening,
        pseudo_gauss=pseudo_gauss,
        gauss_std=gauss_std,
    )
    result = (spikes, new_v, new_r, new_asc, new_rise, new_psc, new_history)
    if return_pre_reset_voltage:
        result += (threshold,)
    return result


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
    detach_reset=True,
    detach_asc_reset=True,
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
                detach_reset=detach_reset,
                detach_asc_reset=detach_asc_reset,
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


def _surrogate_derivative(voltage, dampening, pseudo_gauss, gauss_std):
    scale = tf.cast(dampening, voltage.dtype)
    if pseudo_gauss:
        return scale * tf.exp(
            -tf.square(voltage) / tf.square(tf.cast(gauss_std, voltage.dtype))
        )
    return scale * tf.maximum(1 - tf.abs(voltage), 0)


def fused_spike_shift(
    voltage, refractory, history, dampening, *, pseudo_gauss=False, gauss_std=0.5
):
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet GLIF state operator is unavailable: {glif_state_op_status()}"
        )
    refractory = tf.cast(refractory, tf.bool)

    @tf.custom_gradient
    def transition(voltage_value, refractory_value, history_value, scale, width):
        spikes, new_history = _OPS.dpointnet_spike_shift(
            voltage_value, refractory_value, history_value
        )

        def grad(spike_gradient, history_gradient):
            spike_gradient = _gradient_like(spike_gradient, spikes)
            history_gradient = _gradient_like(history_gradient, new_history)
            voltage_gradient, old_history_gradient = (
                _OPS.dpointnet_spike_shift_backward_v2(
                    voltage_value,
                    refractory_value,
                    spike_gradient,
                    history_gradient,
                    scale,
                    width,
                    pseudo_gauss=pseudo_gauss,
                )
            )
            return voltage_gradient, None, old_history_gradient, None, None

        return (spikes, new_history), grad

    return transition(
        voltage,
        refractory,
        history,
        tf.cast(dampening, voltage.dtype),
        tf.cast(gauss_std, voltage.dtype),
    )
