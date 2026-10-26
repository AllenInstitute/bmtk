"""Native drifting-grating LGN: exact factorization of the movie -> filters -> rate pipeline.

Adapted from Javier Galvan's ``v1_model_utils/cuda_lgn/wrapper.py`` (commit
d8fb466b; credit: Javier). See ``lgn_grating_ops.cc`` for the math: a grating
frame is ``Im(e^{i(k.r + w f + phi)})`` and every stage up to the rectification
is linear, so each unit's filtered response is ``Im(A_n R_n(t))`` with a
per-sample complex spatial coefficient ``A`` (computed by the op in float64)
and a per-unit complex temporal response ``R`` precomputed here once in
float64 from the float32 LGN constants. Rates and spike probabilities equal
the TensorFlow path's to float32 rounding; spikes use the same stateless
float32 uniforms. The operators contain no architecture-specific intrinsics
and are built for every target of the fat binary.
"""

from pathlib import Path
import warnings

import numpy as np
import tensorflow as tf

from .csr_spike_ops import _gpu_compatibility_error, _read_built_architectures

_LIBRARY_PATH = Path(__file__).with_name("_lgn_grating_ops.so")
_ARCHITECTURE_PATH = Path(__file__).with_name("_lgn_grating_ops.archs")
_SM_ARCHITECTURES, _PTX_ARCHITECTURE = _read_built_architectures(_ARCHITECTURE_PATH)
_OPS = None
_LOAD_ERROR = None
MAX_CHUNK = 16


def _load():
    global _OPS, _LOAD_ERROR
    if _OPS is None and _LOAD_ERROR is None:
        if not _LIBRARY_PATH.exists():
            _LOAD_ERROR = f"Native LGN grating library does not exist at {_LIBRARY_PATH}."
            return None
        try:
            _OPS = tf.load_op_library(str(_LIBRARY_PATH))
        except (tf.errors.NotFoundError, OSError) as exc:
            _LOAD_ERROR = f"Native LGN grating library failed to load: {exc}"
    return _OPS


def lgn_grating_op_status():
    """Human-readable load status of the native grating library."""
    _load()
    if _OPS is not None:
        return f"loaded from {_LIBRARY_PATH}"
    return _LOAD_ERROR or "not loaded"


def native_grating_lgn_available():
    """Whether the native grating operators can run: a GPU is visible, the fat
    binary targets it and the library loads."""
    if not tf.config.list_physical_devices("GPU"):
        return False
    if _load() is None:
        return False
    error = _gpu_compatibility_error(_SM_ARCHITECTURES, _PTX_ARCHITECTURE, _ARCHITECTURE_PATH)
    if error:
        warnings.warn(str(error))
        return False
    return True


REQUIRED_CONSTANTS = (
    "is_composite", "gaussian_filters", "spatial_range_indices", "amplitude", "non_dom_amplitude",
    "x", "y", "non_dominant_x", "non_dominant_y", "dom_temporal_kernels", "non_dom_temporal_kernels",
    "spontaneous_firing_rates",
)


def supports_lgn(lgn):
    """Whether ``lgn`` exposes the ``lgn_tf.lgn.LGN`` constants the factorization needs."""
    return all(hasattr(lgn, name) for name in REQUIRED_CONSTANTS)


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def _separable_taps(filters):
    """Rank-one float64 factors of each 2-D Gaussian, centered in a common odd width."""
    taps = max(max(f.shape) for f in filters)
    vertical, horizontal = np.zeros((len(filters), taps)), np.zeros((len(filters), taps))
    for k, kernel in enumerate(filters):
        u, singular, vh = np.linalg.svd(kernel, full_matrices=False)
        if singular[1:].sum() > singular[0] * 1e-5 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("Expected odd, rank-one Gaussian spatial kernels")
        column, row = u[:, 0] * np.sqrt(singular[0]), vh[0] * np.sqrt(singular[0])
        if column.sum() < 0:
            column, row = -column, -row
        before_v, before_h = (taps - column.size) // 2, (taps - row.size) // 2
        vertical[k, before_v:before_v + column.size] = column
        horizontal[k, before_h:before_h + row.size] = row
    return vertical, horizontal


def _bilinear(x, y, reciprocal, scale):
    """Corner coordinates [x0, x1, y0, y1] and weights of (y0,x0), (y1,x0), (y0,x1), (y1,x1)."""
    x0, x1, y0, y1 = np.floor(x), np.ceil(x), np.floor(y), np.ceil(y)
    fx, fy = x - x0, y - y0
    corners = np.stack((x0, x1, y0, y1), axis=1).astype(np.int64)
    weights = np.stack(((1 - fx) * (1 - fy), (1 - fx) * fy, fx * (1 - fy), fx * fy), axis=1)
    rows = corners[:, [2, 3, 2, 3]]
    cols = corners[:, [0, 0, 1, 1]]
    return corners, weights * reciprocal[rows, cols] * scale[:, None]


def _same_correlation_normalizer(rows, cols, kernel):
    """The 'SAME' zero-padded correlation of an all-ones frame with ``kernel``
    (what the TensorFlow path divides by under ``bmtk_compat``), computed on the
    host in float64: the frame is all ones, so each entry is the sum of the taps
    that fall inside the frame."""
    height, width = kernel.shape
    rows_sum = np.zeros((rows, height)); cols_sum = np.zeros((cols, width))
    # TensorFlow 'SAME' pads (k - 1) // 2 before and the remainder after.
    pad_top, pad_left = (height - 1) // 2, (width - 1) // 2
    for r in range(rows):
        for i in range(height):
            rows_sum[r, i] = 0 <= r + i - pad_top < rows
    for c in range(cols):
        for j in range(width):
            cols_sum[c, j] = 0 <= c + j - pad_left < cols
    return rows_sum @ kernel @ cols_sum.T


def _temporal_response(kernels, omega, seq_len, pre_delay, duration, chunk=4096):
    """R[t, n] = sum over lags m with the frame t - m - pre_delay in the stimulus of
    K[L - 1 - m, n] e^{i omega (t - m - pre_delay)}: the causal depthwise
    convolution of e^{i omega f}, zero-padded before the kernel and outside the window."""
    length, units = kernels.shape
    frame = np.arange(seq_len) - pre_delay
    high = np.clip(frame + 1, 0, length)
    low = np.clip(frame - duration + 1, 0, length)
    rotation = np.exp(1j * omega * frame)[:, None]
    shift = np.exp(-1j * omega * np.arange(length))[:, None]
    response = np.empty((seq_len, units, 2), np.float32)
    for start in range(0, units, chunk):
        lagged = kernels[::-1, start:start + chunk].astype(np.float64) * shift
        prefix = np.concatenate((np.zeros((1, lagged.shape[1])), np.cumsum(lagged, axis=0)))
        value = rotation * (prefix[high] - prefix[low])
        response[:, start:start + chunk, 0] = value.real
        response[:, start:start + chunk, 1] = value.imag
    return response


class GratingLGNKernel:
    """The native grating operators and their constants for one ``DriftingGratingLGN``.

    ``lgn`` is a built ``lgn_tf.lgn.LGN``. ``chunk`` samples are drawn per spike
    op call (at most 16): a chunk reuses the temporal responses across its
    samples; at 17,400 units and 500 steps a chunk of 16 float32 uniform
    tensors occupies about 531 MiB while one sample's occupies 33 MiB.
    Constants are ``tf.constant`` on the current device; under a distribution
    strategy construct one kernel per replica.
    """

    def __init__(self, lgn, *, seq_len, pre_delay, post_delay, temporal_f, cpd, contrast,
                 rows, cols, theta_sign, theta_offset, bmtk_compat=True, chunk=16):
        if not 1 <= chunk <= MAX_CHUNK:
            raise ValueError(f"chunk must be in [1, {MAX_CHUNK}], got {chunk}")
        ops = _load()
        if ops is None:
            raise RuntimeError(lgn_grating_op_status())
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("Native LGN grating operators require a visible GPU.")
        error = _gpu_compatibility_error(_SM_ARCHITECTURES, _PTX_ARCHITECTURE, _ARCHITECTURE_PATH)
        if error:
            raise RuntimeError(str(error))
        duration = seq_len - pre_delay - post_delay
        if duration < 0:
            raise ValueError("Grating delays must not exceed seq_len.")
        composite = _numpy(lgn.is_composite).astype(np.float32) > 0.5
        units = composite.size
        f64 = lambda value: _numpy(value).astype(np.float32).astype(np.float64)  # noqa: E731
        # Our LGN keeps conv2d kernels [h, w, 1, 1]; the factorization wants 2-D Gaussians.
        filters = [f64(kernel).reshape(f64(kernel).shape[:2]) for kernel in lgn.gaussian_filters]
        vertical, horizontal = _separable_taps(filters)
        bins = np.full(units, -1, np.int64)
        for k, indices in enumerate(lgn.spatial_range_indices):
            bins[np.asarray(indices)] = k
        if (bins < 0).any():
            raise ValueError("Every LGN unit needs a spatial filter bin")
        corners = np.zeros((units, 8), np.int64)
        weights = np.zeros((units, 8))
        amplitude = f64(lgn.amplitude) * contrast
        non_dom_amplitude = f64(lgn.non_dom_amplitude) * contrast
        x, y = f64(lgn.x), f64(lgn.y)
        non_dom_x, non_dom_y = f64(lgn.non_dominant_x), f64(lgn.non_dominant_y)
        for k, kernel in enumerate(filters):
            reciprocal = (
                1.0 / _same_correlation_normalizer(rows, cols, kernel) if bmtk_compat
                else np.ones((rows, cols))
            )
            units_k = np.flatnonzero(bins == k)
            corners[units_k, :4], weights[units_k, :4] = _bilinear(
                x[units_k], y[units_k], reciprocal, amplitude[units_k])
            units_k = units_k[composite[units_k]]
            corners[units_k, 4:], weights[units_k, 4:] = _bilinear(
                non_dom_x[units_k], non_dom_y[units_k], reciprocal, non_dom_amplitude[units_k])
        omega = 2 * np.pi * temporal_f / 1000.0
        composite_ids = np.flatnonzero(composite)
        slot = np.full(units, -1, np.int64)
        slot[composite_ids] = np.arange(composite_ids.size)
        response = _temporal_response(_numpy(lgn.dom_temporal_kernels), omega, seq_len, pre_delay, duration)
        composite_response = _temporal_response(
            _numpy(lgn.non_dom_temporal_kernels)[:, composite_ids], omega, seq_len, pre_delay, duration)
        self.seq_len, self.units, self.chunk = seq_len, units, chunk
        self._ops = ops
        self._attrs = dict(theta_sign=float(theta_sign), theta_offset=float(theta_offset), rows=rows, cols=cols)
        self._spatial = [tf.constant(value, dtype) for value, dtype in (
            (2 * np.pi * cpd, tf.float64), (vertical, tf.float64), (horizontal, tf.float64),
            (bins, tf.int64), (corners, tf.int64), (weights, tf.float64))]
        self._temporal = [tf.constant(value, dtype) for value, dtype in (
            (response, tf.float32), (composite_response, tf.float32), (slot, tf.int64),
            (_numpy(lgn.spontaneous_firing_rates).astype(np.float32), tf.float32))]

    def coefficients(self, theta, phase):
        """Complex spatial coefficients [batch, 2, units, 2] of the batch's gratings."""
        return self._ops.dpointnet_lgn_grating_coefficients(theta, phase, *self._spatial, **self._attrs)

    def probabilities(self, theta, phase, rates=False):
        """Float32 spike probabilities (or rates in Hz with ``rates``), [batch, seq_len, units]."""
        return self._ops.dpointnet_lgn_grating_probabilities(
            self.coefficients(theta, phase), *self._temporal, rates=rates)

    def spikes(self, theta, phase, spike_seeds, dtype=tf.float32):
        """Spikes [batch, seq_len, units]: stateless uniforms in ``dtype`` (use float32;
        float16 uniforms have 1024 levels and bias the rate upwards) below the
        probabilities, drawn ``chunk`` samples at a time into one buffer."""
        coefficients = self.coefficients(theta, phase)
        shape = (self.seq_len, self.units)

        def uniform(seed):
            return tf.random.stateless_uniform(shape, seed=seed, dtype=dtype)

        def sample(spikes, uniforms, offset):
            return self._ops.dpointnet_lgn_grating_spikes(
                spikes, coefficients, *self._temporal, uniforms, offset=offset)

        batch = theta.shape[0]
        if batch is None:
            uniforms = tf.map_fn(uniform, spike_seeds, fn_output_signature=tf.TensorSpec(shape, dtype))
            return sample(tf.zeros((0,), tf.bool), [uniforms], 0)
        if batch == 0:
            return tf.zeros((0,) + shape, tf.bool)
        spikes = tf.zeros((0,), tf.bool)
        for begin in range(0, batch, self.chunk):
            with tf.control_dependencies([spikes] if begin else []):
                uniforms = [uniform(spike_seeds[b]) for b in range(begin, min(begin + self.chunk, batch))]
            spikes = sample(spikes, uniforms, begin)
        return spikes
