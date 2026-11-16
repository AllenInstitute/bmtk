"""Pinned-host record store for the direct recorded adjoint.

Exact, asynchronous offload of per-step records to pinned host memory during
the forward loop and prefetched return during the reverse loop. The store moves
bytes only: values, dtypes and the arithmetic of the recorded adjoint are
unchanged, and the four fat operator libraries are not touched.

Design adapted from Javier Galvan's ``v1_model_utils/cuda_frame_store`` raw
mode (commit 5d58bc3; credit: Javier), without its codec.
"""

import threading
import uuid
import weakref
from pathlib import Path

import tensorflow as tf

_LIBRARY_PATH = Path(__file__).with_name("_record_store_ops.so")
_OPS = None
_LOAD_ERROR = None
_LOCK = threading.Lock()

__all__ = [
    "HostRecordStore", "host_record_store_available", "record_store_op_status",
    "DEFAULT_PREFETCH", "DEFAULT_STAGING_SLOTS",
]

# Frames kept resident on the device ahead of the reverse loop's reads, and
# staging slots the forward writes before the oldest must have left the device.
# Both bound device memory to a few frames; the link keeps up with the forward
# at a small fraction of the per-step time on PCIe 3/4 for V1-scale records.
DEFAULT_PREFETCH = 8
DEFAULT_STAGING_SLOTS = 8


def _load():
    global _OPS, _LOAD_ERROR
    with _LOCK:
        if _OPS is None and _LOAD_ERROR is None:
            if not _LIBRARY_PATH.exists():
                _LOAD_ERROR = f"Native record store library does not exist at {_LIBRARY_PATH}."
                return None
            try:
                _OPS = tf.load_op_library(str(_LIBRARY_PATH))
            except (tf.errors.NotFoundError, OSError) as exc:
                _LOAD_ERROR = f"Native record store library failed to load: {exc}"
    return _OPS


def record_store_op_status():
    """Human-readable load status of the record store library."""
    _load()
    if _OPS is not None:
        return f"loaded from {_LIBRARY_PATH}"
    return _LOAD_ERROR or "not loaded"


def host_record_store_available():
    """Whether host record offload can run: a GPU is visible and the library loads."""
    if not tf.config.list_physical_devices("GPU"):
        return False
    return _load() is not None


def _destroy_store(name, frames, frame_shape, dtype, prefetch):
    """Delete the named store resource (pinned host memory, cached device frames)."""
    try:
        with tf.init_scope():
            handle = _OPS.dpointnet_host_record_store(
                frames=frames, frame_shape=frame_shape, dtype=dtype, prefetch=prefetch, shared_name=name,
            )
            tf.raw_ops.DestroyResourceOp(resource=handle, ignore_lookup_error=True)
    except (tf.errors.OpError, RuntimeError, AttributeError, TypeError):
        pass


class HostRecordStore:
    """``frames`` equally shaped records in pinned host memory, one store per rollout role.

    Each graph (the forward function and the custom gradient's reverse function)
    creates its own handle to the same named resource; the resource itself is
    created on first use. The Python object owns it: ``close()`` (or garbage
    collection) destroys the resource, releasing the pinned allocation and any
    cached device frames, as ``CsrConnectivity`` does for its metadata.
    """

    def __init__(self, frames, frame_shape, dtype, *, prefetch=DEFAULT_PREFETCH,
                 staging_slots=DEFAULT_STAGING_SLOTS, name=None):
        if not host_record_store_available():
            raise RuntimeError("Host record offload requires the record store library: " + record_store_op_status())
        self.frames = int(frames)
        self.frame_shape = tuple(int(size) for size in frame_shape)
        self.dtype = tf.as_dtype(dtype)
        self.prefetch_depth = int(prefetch)
        self.staging_slots = int(staging_slots)
        if self.frames < 1 or self.prefetch_depth < 1 or self.staging_slots < 1:
            raise ValueError("frames, prefetch and staging_slots must be positive.")
        # int32 variables are host-resident in TensorFlow, so integer records are
        # staged as float32 bit views (tf.bitcast) rather than as int32 frames.
        if self.dtype not in (tf.float16, tf.float32):
            raise ValueError(f"Host record stores hold float16 or float32 frames, not {self.dtype.name}.")
        self.name = name or f"dpointnet_host_record_store_{uuid.uuid4().hex}"
        self._handles = {}
        self._finalizer = weakref.finalize(
            self, _destroy_store, self.name, self.frames, self.frame_shape, self.dtype,
            self.prefetch_depth,
        )

    def close(self):
        """Destroy the store resource now (idempotent)."""
        self._finalizer()
        self._handles = {}

    @property
    def frame_bytes(self):
        bytes_per_element = self.dtype.size
        count = 1
        for size in self.frame_shape:
            count *= size
        return count * bytes_per_element

    @property
    def host_bytes(self):
        return self.frames * self.frame_bytes

    def handle(self):
        """The resource handle in the current graph (one op per graph; fresh in eager mode)."""
        if tf.executing_eagerly():
            return _OPS.dpointnet_host_record_store(
                frames=self.frames, frame_shape=self.frame_shape, dtype=self.dtype,
                prefetch=self.prefetch_depth, shared_name=self.name,
            )
        graph = tf.compat.v1.get_default_graph()
        key = id(graph)
        handle = self._handles.get(key)
        if handle is None or handle.graph is not graph:
            handle = _OPS.dpointnet_host_record_store(
                frames=self.frames, frame_shape=self.frame_shape, dtype=self.dtype,
                prefetch=self.prefetch_depth, shared_name=self.name,
            )
            self._handles[key] = handle
        return handle

    def offload(self, staging, slot, index, after):
        """Copy frame ``slot`` of the ``[staging_slots, *frame_shape]`` variable ``staging``
        to host frame ``index`` once the compute stream has produced ``after``;
        returns an int32 token to order later work against."""
        handle = staging.handle if isinstance(staging, tf.Variable) else staging
        return _OPS.dpointnet_host_record_offload(
            self.handle(), handle, tf.cast(slot, tf.int32), tf.cast(index, tf.int32), after,
            T=self.dtype, staging_slots=self.staging_slots,
        )

    def take(self, index, prefetch=-1):
        """Host frame ``index`` as a device tensor; starts the upload of frame
        ``prefetch`` (ignored when negative or beyond the store)."""
        frame = _OPS.dpointnet_host_record_take(
            self.handle(), tf.cast(index, tf.int32), tf.cast(prefetch, tf.int32), T=self.dtype,
        )
        return tf.ensure_shape(frame, self.frame_shape)

    def prefetch(self, index, after):
        """Start the upload of frame ``index`` after ``after``; out-of-range indices are ignored."""
        return _OPS.dpointnet_host_record_prefetch(self.handle(), tf.cast(index, tf.int32), after)

    def stats(self):
        """``[frames, frame_bytes, host_bytes, offloaded, resident, takes]`` as int64."""
        return _OPS.dpointnet_host_record_stats(self.handle())
