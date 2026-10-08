import os
from pathlib import Path

import tensorflow as tf

from .._options import validate_bool_option
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


def fused_nest_event_vjp_available():
    return fused_nest_state_available() and hasattr(
        _OPS, "dpointnet_nest_state_backward_events"
    )


def fused_voltage_penalty_available():
    return fused_glif_state_available() and all(
        hasattr(_OPS, name)
        for name in (
            "dpointnet_voltage_penalty_forward",
            "dpointnet_voltage_penalty_backward",
        )
    )


def fused_voltage_penalty_mean_step(voltage, penalty_mode="range"):
    """Native per-sample mean voltage penalty with a custom VJP."""
    if penalty_mode not in ("range", "threshold"):
        raise ValueError("penalty_mode must be 'range' or 'threshold'.")
    if not fused_voltage_penalty_available():
        raise RuntimeError(
            "Fused voltage penalty requires rebuilt CUDA operators: "
            + glif_state_op_status()
        )

    @tf.custom_gradient
    def transition(voltage_value):
        penalty = _OPS.dpointnet_voltage_penalty_forward(
            voltage_value, mode=penalty_mode
        )

        def grad(penalty_grad):
            penalty_grad = _gradient_like(penalty_grad, penalty)
            return _OPS.dpointnet_voltage_penalty_backward(
                voltage_value, penalty_grad, mode=penalty_mode
            )

        return penalty, grad

    return transition(voltage)


def fused_nest_type_indexed_coefficients_available():
    return fused_nest_state_available() and all(
        hasattr(_OPS, name)
        for name in (
            "dpointnet_nest_state_forward_type_indexed",
            "dpointnet_nest_state_backward_type_indexed",
            "dpointnet_nest_state_backward_events_type_indexed",
        )
    )


def fused_nest_state_history_available():
    return fused_nest_state_available() and all(
        hasattr(_OPS, name)
        for name in (
            "dpointnet_nest_state_history_forward",
            "dpointnet_nest_state_history_backward",
            "dpointnet_nest_state_history_backward_events",
        )
    )


def fused_nest_state_history_type_indexed_available():
    return fused_nest_state_history_available() and all(
        hasattr(_OPS, name)
        for name in (
            "dpointnet_nest_state_history_forward_type_indexed",
            "dpointnet_nest_state_history_backward_type_indexed",
            "dpointnet_nest_state_history_backward_events_type_indexed",
        )
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
    use_fused_event_vjp=False,
    packed_coefficients=None,
    packed_kernel_coefficients=None,
    packed_type_coefficients=None,
    type_indices=None,
    type_indexed_identity=None,
    require_type_indexed_coefficients=False,
    fuse_history=False,
):
    """Four-basis NEST transition with live frozen coefficients and event VJPs."""
    if not fused_nest_state_available():
        raise RuntimeError(
            "Fused NEST state requires rebuilt CUDA operators: "
            + glif_state_op_status()
        )
    dtype = voltage.dtype
    validate_bool_option(use_fused_event_vjp, "use_fused_event_vjp")
    if use_fused_event_vjp and not fused_nest_event_vjp_available():
        raise RuntimeError("NEST event VJP requires rebuilt CUDA operators.")
    if fuse_history and not fused_nest_state_history_available():
        raise RuntimeError("NEST state/history fusion requires rebuilt CUDA operators.")
    if fuse_history and history.dtype != psc.dtype:
        raise ValueError("NEST state/history fusion requires history and PSC dtypes to match.")
    use_type_indexed = packed_type_coefficients is not None
    if use_type_indexed:
        if fuse_history:
            available = fused_nest_state_history_type_indexed_available()
            requirement = "NEST type-indexed state/history fusion"
        else:
            available = fused_nest_type_indexed_coefficients_available()
            requirement = "NEST type-indexed coefficients"
        if not available:
            raise RuntimeError(requirement + " requires rebuilt CUDA operators.")
    if (packed_type_coefficients is None) != (type_indices is None):
        raise ValueError(
            "packed_type_coefficients and type_indices must be provided together."
        )
    if use_type_indexed and type_indexed_identity is None:
        raise ValueError("type_indexed_identity is required with type-indexed coefficients.")
    validate_bool_option(require_type_indexed_coefficients, "require_type_indexed_coefficients")
    type_indexed_identity_static = None
    if use_type_indexed:
        type_indexed_identity_static = tf.get_static_value(type_indexed_identity)
        if type_indexed_identity_static is not None:
            type_indexed_identity_static = bool(type_indexed_identity_static)
    neurons = tf.shape(voltage)[1]
    voltage_gradient_dampening = tf.clip_by_value(
        tf.cast(voltage_gradient_dampening, dtype),
        tf.cast(0.0, dtype),
        tf.cast(1.0, dtype),
    )
    if (packed_coefficients is None) != (packed_kernel_coefficients is None):
        raise ValueError(
            "packed_coefficients and packed_kernel_coefficients must be provided together."
        )
    if packed_coefficients is None:
        coefficients = pack_nest_state_coefficients(
            neurons,
            dtype,
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
            voltage_gradient_dampening=voltage_gradient_dampening,
        )
        kernel_coefficients_prepared = (
            tf.transpose(coefficients) if dtype == tf.float32 else coefficients
        )
    else:
        coefficients = tf.cast(packed_coefficients, dtype)
        kernel_coefficients_prepared = tf.cast(packed_kernel_coefficients, dtype)
    if use_type_indexed:
        type_coefficients = tf.cast(packed_type_coefficients, dtype)
        type_indices = tf.cast(type_indices, tf.int64)
        type_indexed_identity = tf.cast(type_indexed_identity, tf.bool)
        if require_type_indexed_coefficients:
            with tf.control_dependencies(
                [
                    tf.debugging.assert_equal(
                        type_indexed_identity,
                        True,
                        message=(
                            "NEST type-indexed coefficients were required"
                            + (" for state/history fusion" if fuse_history else "")
                            + ", but live per-neuron coefficients are not type-identical."
                        ),
                    )
                ]
            ):
                type_indexed_identity = tf.identity(type_indexed_identity)
    else:
        type_coefficients = None

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
        kernel_coefficients = arguments[7]
        forward = (
            _OPS.dpointnet_nest_state_history_forward
            if fuse_history else _OPS.dpointnet_nest_state_forward
        )
        history_arguments = (arguments[13],) if fuse_history else ()
        if use_type_indexed:
            type_coefficients_arg = arguments[14]
            type_indices_arg = arguments[15]
            type_identity_arg = arguments[16]

            def type_indexed_forward():
                if fuse_history:
                    return tuple(_OPS.dpointnet_nest_state_history_forward_type_indexed(
                        *arguments[:6], type_coefficients_arg, type_indices_arg,
                        *arguments[8:11], arguments[13],
                        hard_reset=hard_reset, **forward_options
                    ))
                return tuple(_OPS.dpointnet_nest_state_forward_type_indexed(
                    *arguments[:6], type_coefficients_arg, type_indices_arg,
                    *arguments[8:11], hard_reset=hard_reset, **forward_options
                ))

            def full_forward():
                return tuple(forward(
                    *arguments[:6], kernel_coefficients, *arguments[8:11],
                    *history_arguments,
                    hard_reset=hard_reset, coefficients_layout=layout, **forward_options
                ))

            if type_indexed_identity_static is True:
                outputs = type_indexed_forward()
            elif type_indexed_identity_static is False:
                outputs = full_forward()
            else:
                outputs = tf.cond(type_identity_arg, type_indexed_forward, full_forward)
        else:
            outputs = forward(
                *arguments[:6], kernel_coefficients, *arguments[8:11],
                *history_arguments,
                hard_reset=hard_reset, coefficients_layout=layout, **forward_options
            )
        # The enclosing custom gradient owns the VJP. Nested rollout tapes must
        # not attempt to differentiate the opaque forward CUDA op itself.
        outputs = tuple(tf.stop_gradient(value) for value in outputs)
        backward_refractory = tf.cast(arguments[1] > 0, dtype)
        threshold_for_backward = (
            outputs[0] - arguments[10] if return_pre_reset_voltage else outputs[0]
        )

        def grad(grad_threshold, grad_v, _grad_r, grad_asc, grad_rise, grad_psc,
                 *history_gradients):
            grad_threshold = _gradient_like(grad_threshold, outputs[0])
            grad_v = _gradient_like(grad_v, outputs[1])
            grad_asc = _gradient_like(grad_asc, outputs[3])
            old_history_gradient = None
            if not use_fused_event_vjp and (not detach_reset or not detach_asc_reset):
                event_grad = tf.zeros_like(outputs[0])
                params = arguments[6]
                if not detach_reset:
                    reset_sensitivity = (
                        params[:, 26] - (threshold_for_backward + arguments[10])
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
                    threshold_for_backward, arguments[11], pseudo_gauss, arguments[12]
                )
                grad_threshold += tf.where(
                    backward_refractory > 0,
                    tf.zeros_like(event_grad),
                    event_grad * derivative,
                )
            backward = _OPS.dpointnet_nest_state_backward
            event_inputs, event_options = (), {}
            if fuse_history:
                spike_gradient, history_gradient = history_gradients
                backward = (
                    _OPS.dpointnet_nest_state_history_backward_events
                    if use_fused_event_vjp
                    else _OPS.dpointnet_nest_state_history_backward
                )
                event_inputs = (
                    arguments[2], arguments[10], arguments[11], arguments[12]
                ) if use_fused_event_vjp else ()
                if not use_fused_event_vjp:
                    event_inputs = (arguments[11], arguments[12])
                event_options = dict(
                    detach_reset=detach_reset,
                    detach_asc_reset=detach_asc_reset,
                    pseudo_gauss=pseudo_gauss,
                ) if use_fused_event_vjp else dict(pseudo_gauss=pseudo_gauss)
                extra_inputs = (
                    _gradient_like(spike_gradient, outputs[6]),
                    _gradient_like(history_gradient, outputs[7]),
                )
            elif use_fused_event_vjp:
                backward = _OPS.dpointnet_nest_state_backward_events
                event_inputs = (arguments[2], arguments[10], arguments[11], arguments[12])
                event_options = dict(
                    detach_reset=detach_reset, detach_asc_reset=detach_asc_reset,
                    pseudo_gauss=pseudo_gauss,
                )
                extra_inputs = ()
            else:
                extra_inputs = ()
            gradient_inputs = [
                _gradient_like(gradient, output)
                for gradient, output in zip(
                    (grad_threshold, grad_v, grad_asc, grad_rise, grad_psc),
                    (outputs[0], outputs[1], outputs[3], outputs[4], outputs[5]),
                )
            ]

            def full_backward():
                return tuple(backward(
                    threshold_for_backward,
                    tf.cast(backward_refractory, refractory.dtype),
                    kernel_coefficients,
                    arguments[9],
                    1 - arguments[6][0, 27],
                    *gradient_inputs,
                    *event_inputs,
                    *extra_inputs,
                    hard_reset=hard_reset,
                    coefficients_layout=layout,
                    **event_options,
                ))

            if use_type_indexed:
                if fuse_history:
                    type_backward = (
                        _OPS.dpointnet_nest_state_history_backward_events_type_indexed
                        if use_fused_event_vjp
                        else _OPS.dpointnet_nest_state_history_backward_type_indexed
                    )
                    type_event_inputs = event_inputs
                    type_extra_inputs = extra_inputs
                else:
                    type_backward = (
                        _OPS.dpointnet_nest_state_backward_events_type_indexed
                        if use_fused_event_vjp
                        else _OPS.dpointnet_nest_state_backward_type_indexed
                    )
                    type_event_inputs = (
                        (arguments[2], arguments[10], arguments[11], arguments[12])
                        if use_fused_event_vjp
                        else ()
                    )
                    type_extra_inputs = ()

                def type_indexed_backward():
                    return tuple(type_backward(
                        threshold_for_backward,
                        tf.cast(backward_refractory, refractory.dtype),
                        arguments[14],
                        arguments[15],
                        arguments[9],
                        1 - arguments[6][0, 27],
                        *gradient_inputs,
                        *type_event_inputs,
                        *type_extra_inputs,
                        hard_reset=hard_reset,
                        **event_options,
                    ))

                if type_indexed_identity_static is True:
                    gradients = type_indexed_backward()
                elif type_indexed_identity_static is False:
                    gradients = full_backward()
                else:
                    gradients = tf.cond(arguments[16], type_indexed_backward, full_backward)
            else:
                gradients = full_backward()
            if fuse_history:
                old_history_gradient = gradients[5]
            extra_nones = (None, None, None) if use_type_indexed else ()
            return (
                gradients[0],
                None,
                gradients[1],
                gradients[2],
                gradients[3],
                gradients[4],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                old_history_gradient,
                *extra_nones,
            )

        return outputs, grad

    @tf.custom_gradient
    def transition_static_type_indexed(
        voltage_arg,
        refractory_arg,
        asc_arg,
        psc_rise_arg,
        psc_arg,
        currents_arg,
        t_ref_arg,
        dt_arg,
        v_th_arg,
        dampening_arg,
        gauss_std_arg,
        history_arg,
        type_coefficients_arg,
        type_indices_arg,
    ):
        outputs = tuple(_OPS.dpointnet_nest_state_history_forward_type_indexed(
            voltage_arg,
            refractory_arg,
            asc_arg,
            psc_rise_arg,
            psc_arg,
            currents_arg,
            type_coefficients_arg,
            type_indices_arg,
            t_ref_arg,
            dt_arg,
            v_th_arg,
            history_arg,
            hard_reset=hard_reset,
            **forward_options,
        ))
        outputs = tuple(tf.stop_gradient(value) for value in outputs)
        backward_refractory = tf.cast(refractory_arg > 0, dtype)
        threshold_for_backward = (
            outputs[0] - v_th_arg if return_pre_reset_voltage else outputs[0]
        )

        def grad(grad_threshold, grad_v, _grad_r, grad_asc, grad_rise, grad_psc,
                 spike_gradient, history_gradient):
            grad_threshold = _gradient_like(grad_threshold, outputs[0])
            grad_v = _gradient_like(grad_v, outputs[1])
            grad_asc = _gradient_like(grad_asc, outputs[3])
            if not use_fused_event_vjp and (not detach_reset or not detach_asc_reset):
                params = tf.gather(type_coefficients_arg, type_indices_arg)
                event_grad = tf.zeros_like(outputs[0])
                if not detach_reset:
                    reset_sensitivity = (
                        params[:, 26] - (threshold_for_backward + v_th_arg)
                        if hard_reset
                        else -(1 - params[:, 26])
                    )
                    event_grad += grad_v * reset_sensitivity
                if not detach_asc_reset:
                    adaptation = tf.reshape(asc_arg, (-1, neurons, 2))
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
                    threshold_for_backward, dampening_arg, pseudo_gauss, gauss_std_arg
                )
                grad_threshold += tf.where(
                    backward_refractory > 0,
                    tf.zeros_like(event_grad),
                    event_grad * derivative,
                )
            gradient_inputs = [
                _gradient_like(gradient, output)
                for gradient, output in zip(
                    (grad_threshold, grad_v, grad_asc, grad_rise, grad_psc),
                    (outputs[0], outputs[1], outputs[3], outputs[4], outputs[5]),
                )
            ]
            type_backward = (
                _OPS.dpointnet_nest_state_history_backward_events_type_indexed
                if use_fused_event_vjp
                else _OPS.dpointnet_nest_state_history_backward_type_indexed
            )
            type_event_inputs = (
                (asc_arg, v_th_arg, dampening_arg, gauss_std_arg)
                if use_fused_event_vjp
                else (dampening_arg, gauss_std_arg)
            )
            event_options = dict(
                detach_reset=detach_reset,
                detach_asc_reset=detach_asc_reset,
                pseudo_gauss=pseudo_gauss,
            ) if use_fused_event_vjp else dict(pseudo_gauss=pseudo_gauss)
            gradients = tuple(type_backward(
                threshold_for_backward,
                tf.cast(backward_refractory, refractory_arg.dtype),
                type_coefficients_arg,
                type_indices_arg,
                dt_arg,
                1 - type_coefficients_arg[0, 27],
                *gradient_inputs,
                *type_event_inputs,
                _gradient_like(spike_gradient, outputs[6]),
                _gradient_like(history_gradient, outputs[7]),
                hard_reset=hard_reset,
                **event_options,
            ))
            return (
                gradients[0],
                None,
                gradients[1],
                gradients[2],
                gradients[3],
                gradients[4],
                None,
                None,
                None,
                None,
                None,
                gradients[5],
                None,
                None,
            )

        return outputs, grad

    if use_type_indexed and type_indexed_identity_static is True and fuse_history:
        transition_outputs = transition_static_type_indexed(
            voltage,
            refractory,
            asc,
            psc_rise,
            psc,
            currents,
            tf.cast(t_ref_steps, refractory.dtype),
            tf.cast(dt, dtype),
            tf.cast(v_th, dtype),
            tf.cast(dampening, dtype),
            tf.cast(gauss_std, dtype),
            tf.cast(history, psc.dtype) if fuse_history else history,
            type_coefficients,
            type_indices,
        )
    else:
        transition_args = (
            voltage,
            refractory,
            asc,
            psc_rise,
            psc,
            currents,
            coefficients,
            kernel_coefficients_prepared,
            tf.cast(t_ref_steps, refractory.dtype),
            tf.cast(dt, dtype),
            tf.cast(v_th, dtype),
            tf.cast(dampening, dtype),
            tf.cast(gauss_std, dtype),
            tf.cast(history, psc.dtype),
        )
        if use_type_indexed:
            transition_args += (type_coefficients, type_indices, type_indexed_identity)
        transition_outputs = transition(*transition_args)
    threshold, new_v, new_r, new_asc, new_rise, new_psc = transition_outputs[:6]
    if fuse_history:
        spikes, new_history = transition_outputs[6], transition_outputs[7]
    else:
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



def pack_nest_state_coefficients(
    neurons,
    dtype,
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
    v_reset,
    voltage_gradient_dampening,
):
    voltage_gradient_dampening = tf.clip_by_value(
        tf.cast(voltage_gradient_dampening, dtype),
        tf.cast(0.0, dtype),
        tf.cast(1.0, dtype),
    )
    return tf.concat(
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


def pack_type_indexed_nest_state_coefficients(coefficients, type_indices, first_indices):
    """Compact live per-neuron NEST coefficients by type and verify exact identity.

    Returns ``(type_coefficients, type_indices, identity)`` where ``identity`` is a
    scalar boolean Tensor indicating whether gathering the compact table exactly
    reconstructs the supplied per-neuron coefficient matrix.
    """
    coefficients = tf.convert_to_tensor(coefficients)
    type_indices = tf.cast(type_indices, tf.int64)
    first_indices = tf.cast(first_indices, tf.int32)
    type_coefficients = tf.gather(coefficients, first_indices)
    reconstructed = tf.gather(type_coefficients, type_indices)
    identity = tf.reduce_all(tf.equal(reconstructed, coefficients))
    return type_coefficients, type_indices, identity


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
