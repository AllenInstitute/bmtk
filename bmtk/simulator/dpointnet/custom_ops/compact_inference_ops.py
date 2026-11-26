"""Optional native frozen projection; missing binaries never trigger a build."""

from pathlib import Path

import tensorflow as tf

from . import csr_spike_ops

_PATH = Path(__file__).with_name("_compact_inference_ops.so")
_OPS = None
_ERROR = "compact inference operator is not built"
if _PATH.exists():
    try:
        _OPS = tf.load_op_library(str(_PATH))
    except (tf.errors.OpError, OSError) as error:
        _ERROR = f"compact inference operator could not be loaded: {error}"


def _compatibility_error():
    if tf.config.get_visible_devices("GPU"):
        if csr_spike_ops._environment_flag("BMTK_DPOINTNET_DISABLE_FUSED_CUDA"):
            return "compact CUDA disabled by BMTK_DPOINTNET_DISABLE_FUSED_CUDA"
        architecture_path = _PATH.with_suffix(".archs")
        targets, ptx = csr_spike_ops._read_built_architectures(architecture_path)
        return csr_spike_ops._gpu_compatibility_error(
            sm_architectures=targets, ptx_architecture=ptx,
            architecture_path=architecture_path)
    return None


def status():
    if _OPS is None:
        return _ERROR
    error = _compatibility_error()
    if error:
        return error
    return f"compact inference operator loaded from {_PATH}"


def available():
    return _OPS is not None and _compatibility_error() is None


def currents(spikes, row_ptr, targets, weights, basis, neurons):
    if _OPS is None:
        raise RuntimeError(status())

    @tf.custom_gradient
    def project(values, coefficients):
        result = _OPS.compact_inference_currents(
            values, row_ptr, targets, coefficients, basis, n_neurons=neurons)

        def gradient(_):
            raise ValueError("Compact inference is frozen and has no adjoint; use standard connectivity for training.")

        return result, gradient

    return project(spikes, weights)
