"""Native LGN response to arbitrary movies: spatial filter, temporal filter, rates.

Adapted from Javier Galvan's ``v1_model_utils/cuda_lgn/movie.py`` (commit
d8fb466b; credit: Javier). See ``lgn_movie_ops.cc`` for the math: the same
pipeline as ``lgn_tf.lgn.LGN.spatial_response`` + ``firing_rates_from_spatial``
in float32. The constants are derived here once, in float64 from the float32
LGN constants, and stored as float32: the separable filter taps, each subunit's
four bilinear corners with corner weight, edge normalization and amplitude
folded in, and the temporal kernels by lag in blocks of four
(``[lags / 4, columns, 4]``) with the range of nonzero lags of every 32-column
tile. The operators use only plain float loads and are built for every target
of the fat binary.
"""

from pathlib import Path
import warnings

import numpy as np
import tensorflow as tf

from .csr_spike_ops import _gpu_compatibility_error, _read_built_architectures
from .lgn_grating_ops import (
    MAX_CHUNK, REQUIRED_CONSTANTS, _bilinear, _numpy, _same_correlation_normalizer,
    _separable_taps, supports_lgn,
)

_LIBRARY_PATH = Path(__file__).with_name("_lgn_movie_ops.so")
_ARCHITECTURE_PATH = Path(__file__).with_name("_lgn_movie_ops.archs")
_SM_ARCHITECTURES, _PTX_ARCHITECTURE = _read_built_architectures(_ARCHITECTURE_PATH)
_OPS = None
_LOAD_ERROR = None
TILE = 32  # subunit columns per lag range (one warp of the temporal kernel)
LAG_PADDING = 32  # trailing zero lags, so the op's lag blocks never check bounds
# Limits of the spatial kernel (lgn_movie_ops.cu.cc): one block of SPATIAL_THREADS
# threads filters a frame, each thread HORIZONTAL_COLUMNS columns of one row, and
# the separable taps sit in registers up to MAX_TAPS wide.
SPATIAL_THREADS = 1024
HORIZONTAL_COLUMNS = 11
MAX_TAPS = 27

__all__ = [
    "MovieLGNKernel", "native_movie_lgn_available", "lgn_movie_op_status", "supports_lgn",
    "movie_support_error", "REQUIRED_CONSTANTS",
]


def _load():
    global _OPS, _LOAD_ERROR
    if _OPS is None and _LOAD_ERROR is None:
        if not _LIBRARY_PATH.exists():
            _LOAD_ERROR = f"Native LGN movie library does not exist at {_LIBRARY_PATH}."
            return None
        try:
            _OPS = tf.load_op_library(str(_LIBRARY_PATH))
        except (tf.errors.NotFoundError, OSError) as exc:
            _LOAD_ERROR = f"Native LGN movie library failed to load: {exc}"
    return _OPS


def lgn_movie_op_status():
    """Human-readable load status of the native movie library."""
    _load()
    if _OPS is not None:
        return f"loaded from {_LIBRARY_PATH}"
    return _LOAD_ERROR or "not loaded"


def native_movie_lgn_available():
    """Whether the native movie operators can run: a GPU is visible, the fat
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


def movie_support_error(lgn, rows, cols):
    """Why the native movie operators cannot serve ``lgn`` for ``rows`` x ``cols``
    frames, or None when they can: the LGN must expose the required constants,
    a frame must fit one block's horizontal work items and the spatial filters
    must be odd, rank-one Gaussians of at most MAX_TAPS taps."""
    if not supports_lgn(lgn):
        return "the LGN lacks the constants the native movie operators need"
    if rows * -(-cols // HORIZONTAL_COLUMNS) > SPATIAL_THREADS:
        return (f"frames of {rows} x {cols} exceed the native spatial pass "
                f"({SPATIAL_THREADS} rows x {HORIZONTAL_COLUMNS}-column groups)")
    try:
        filters = [_numpy(kernel).reshape(_numpy(kernel).shape[:2]) for kernel in lgn.gaussian_filters]
        _separable_taps(filters)
    except ValueError as exc:
        return str(exc)
    if max(max(kernel.shape) for kernel in filters) > MAX_TAPS:
        return f"spatial filters wider than {MAX_TAPS} taps exceed the native spatial pass"
    return None


def _lag_major(kernels):
    """[columns, lags]: lag m weighs the frame m steps back (lag 0 the current frame),
    zero-padded to a multiple of LAG_PADDING lags with at least LAG_PADDING zeros."""
    lagged = np.asarray(kernels, np.float32)[::-1]
    lags = (lagged.shape[0] // LAG_PADDING + 2) * LAG_PADDING
    return np.ascontiguousarray(np.pad(lagged, ((0, lags - lagged.shape[0]), (0, 0))).T)


def _time_blocked(kernels):
    """[lags / 4, columns, 4] of [columns, lags]: a warp's float4 loads are contiguous."""
    columns, lags = kernels.shape
    return np.ascontiguousarray(kernels.reshape(columns, lags // 4, 4).transpose(1, 0, 2))


def _lag_ranges(kernels):
    """[tiles, 2] first and last nonzero lag of each TILE-column tile of [columns, lags]
    kernels ([LAG_PADDING, 0], empty, when all are zero)."""
    columns, lags = kernels.shape
    tiles = -(-columns // TILE)
    nonzero = np.zeros((tiles * TILE, lags), bool)
    nonzero[:columns] = kernels != 0
    nonzero = nonzero.reshape(tiles, TILE, lags).any(axis=1)
    first, last = nonzero.argmax(axis=1), lags - 1 - nonzero[:, ::-1].argmax(axis=1)
    return np.where(nonzero.any(axis=1)[:, None], np.stack((first, last), axis=1), [LAG_PADDING, 0])


class MovieLGNKernel:
    """The native movie operators and their constants for one built ``lgn_tf.lgn.LGN``
    and edge-normalization choice.

    ``chunk`` samples are sampled per spike op call (1-16): 1 keeps one sample's
    movie and uniforms alive at a time. Constants are ``tf.constant`` on the
    current device; under a distribution strategy construct one kernel per replica.
    """

    def __init__(self, lgn, *, rows, cols, bmtk_compat=True, chunk=1):
        if not 1 <= chunk <= MAX_CHUNK:
            raise ValueError(f"chunk must be in [1, {MAX_CHUNK}], got {chunk}")
        ops = _load()
        if ops is None:
            raise RuntimeError(lgn_movie_op_status())
        if not tf.config.list_physical_devices("GPU"):
            raise RuntimeError("Native LGN movie operators require a visible GPU.")
        error = _gpu_compatibility_error(_SM_ARCHITECTURES, _PTX_ARCHITECTURE, _ARCHITECTURE_PATH)
        if error:
            raise RuntimeError(str(error))
        error = movie_support_error(lgn, rows, cols)
        if error:
            raise ValueError(f"Native LGN movie operators cannot serve this LGN: {error}")
        f64 = lambda value: _numpy(value).astype(np.float32).astype(np.float64)  # noqa: E731
        filters = [f64(kernel).reshape(f64(kernel).shape[:2]) for kernel in lgn.gaussian_filters]
        vertical, horizontal = _separable_taps(filters)
        composite = _numpy(lgn.is_composite).astype(np.float32) > 0.5
        units = composite.size
        composite_ids = np.flatnonzero(composite)
        slot = np.full(units, -1, np.int64)
        slot[composite_ids] = np.arange(composite_ids.size)
        amplitude, non_dom_amplitude = f64(lgn.amplitude), f64(lgn.non_dom_amplitude)
        x, y = f64(lgn.x), f64(lgn.y)
        non_dom_x, non_dom_y = f64(lgn.non_dominant_x), f64(lgn.non_dominant_y)
        samples, weights, offsets = [], [], [0]
        for kernel, indices in zip(filters, lgn.spatial_range_indices):
            reciprocal = (
                1.0 / _same_correlation_normalizer(rows, cols, kernel) if bmtk_compat
                else np.ones((rows, cols))
            )
            indices = np.asarray(indices, np.int64)
            selected = indices[composite[indices]]
            for column, px, py, scale in (
                    (indices, x[indices], y[indices], amplitude[indices]),
                    (units + slot[selected], non_dom_x[selected], non_dom_y[selected],
                     non_dom_amplitude[selected])):
                corners, weight = _bilinear(px, py, reciprocal, scale)
                x0, x1, y0, y1 = corners.T
                samples.append((y0 * cols + x0) | (x1 - x0) << 30 | (y1 - y0) << 31 | column << 32)
                weights.append(weight)
            offsets.append(offsets[-1] + indices.size + selected.size)
        if offsets[-1] != units + composite_ids.size:
            raise ValueError("Every LGN unit needs exactly one spatial filter bin")
        kernels = _lag_major(_numpy(lgn.dom_temporal_kernels))
        composite_kernels = _lag_major(_numpy(lgn.non_dom_temporal_kernels)[:, composite_ids])
        spontaneous = _numpy(lgn.spontaneous_firing_rates).astype(np.float32)
        self.units, self.chunk = units, chunk
        self._ops = ops
        self._attrs = dict(rows=rows, cols=cols)
        self._constants = [tf.constant(value, dtype) for value, dtype in (
            (np.stack((vertical, horizontal)), tf.float32),
            ([(f.shape[0] - 1) // 2 for f in filters], tf.int64),
            (offsets, tf.int64),
            (np.concatenate(samples), tf.int64),
            (np.concatenate(weights), tf.float32),
            (_time_blocked(kernels), tf.float32), (_lag_ranges(kernels), tf.int64),
            (spontaneous, tf.float32), (slot, tf.int64),
            (_time_blocked(composite_kernels), tf.float32),
            (_lag_ranges(composite_kernels), tf.int64),
            (spontaneous[composite_ids], tf.float32))]

    @staticmethod
    def _movie(movie):
        """The movie [batch, time, rows, cols] in float16 or float32 (others become float32)."""
        movie = tf.convert_to_tensor(movie)
        return movie if movie.dtype in (tf.float16, tf.float32) else tf.cast(movie, tf.float32)

    def response(self, movie, output="rates"):
        """Float32 rates in Hz (``output="rates"``) or 1 ms spike probabilities
        (``"probabilities"``), [batch, time, units], of a movie [batch, time, rows, cols]."""
        return self._ops.dpointnet_lgn_movie_response(
            self._movie(movie), *self._constants, output=output, **self._attrs)

    def spikes(self, movie, spike_seeds, dtype=tf.float32, batch=None):
        """Spikes [batch, time, units]: stateless uniforms of each sample's seed
        [batch, 2] in ``dtype`` (use float32) below the probabilities, ``chunk``
        samples at a time. ``movie`` is the batch's movie [batch, time, rows,
        cols] or a function ``movie(begin, end)`` returning the movie of samples
        [begin, end) of ``batch`` samples, built only once the previous chunk is
        sampled so at most one chunk of movie is alive."""
        if callable(movie):
            if batch is None:
                raise ValueError("a movie function needs the batch size")
            chunk_movie = lambda begin, end: self._movie(movie(begin, end))  # noqa: E731
            steps = None
        else:
            movie = self._movie(movie)
            batch, steps = movie.shape[0], movie.shape[1]
            chunk_movie = None

        def uniform(seed, steps):
            return tf.random.stateless_uniform((steps, self.units), seed=seed, dtype=dtype)

        def sample(spikes, movie, uniforms, offset, whole_batch=0):
            return self._ops.dpointnet_lgn_movie_spikes(
                spikes, movie, *self._constants, uniforms, offset=offset, batch=whole_batch,
                **self._attrs)

        if chunk_movie is not None:
            spikes = tf.zeros((0,), tf.bool)
            for begin in range(0, batch, self.chunk):
                end = min(begin + self.chunk, batch)
                with tf.control_dependencies([spikes] if begin else []):
                    part = chunk_movie(begin, end)
                    uniforms = [uniform(spike_seeds[b], part.shape[1]) for b in range(begin, end)]
                spikes = sample(spikes, part, uniforms, begin, batch)
                # Eagerly, drop this chunk's movie and uniforms before the next
                # chunk is built so only one chunk is alive at a time.
                del part, uniforms
            return spikes
        if batch is None or steps is None:
            # Symbolic shape: one sample at a time so memory stays bounded.
            def one(sample):
                frames, seed = sample
                return sample_one(frames, seed)

            def sample_one(frames, seed):
                return self._ops.dpointnet_lgn_movie_spikes(
                    tf.zeros((0,), tf.bool), frames[None], *self._constants,
                    [uniform(seed, tf.shape(frames)[0])], offset=0, batch=0, **self._attrs)[0]

            return tf.map_fn(
                one, (movie, spike_seeds),
                fn_output_signature=tf.TensorSpec((None, self.units), tf.bool), parallel_iterations=1,
            )
        shape = (steps, self.units)
        if batch == 0:
            return tf.zeros((0,) + shape, tf.bool)
        spikes = tf.zeros((0,), tf.bool)
        for begin in range(0, batch, self.chunk):
            with tf.control_dependencies([spikes] if begin else []):
                uniforms = [uniform(spike_seeds[b], steps) for b in range(begin, min(begin + self.chunk, batch))]
            spikes = sample(spikes, movie, uniforms, begin)
            del uniforms
        return spikes
