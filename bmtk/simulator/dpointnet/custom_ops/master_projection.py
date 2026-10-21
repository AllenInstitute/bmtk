"""Private direct-canonical FP32-weight experiment; not a production option."""

from pathlib import Path

import numpy as np
import tensorflow as tf

_LIBRARY = None


def library():
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = tf.load_op_library(str(Path(__file__).with_name("_csr_spike_ops.so")))
    return _LIBRARY


def project(spikes, master, metadata, basis, n_post, n_edges, spike_scale=1.0):
    """Return FP32 currents with canonical FP32 weight and live source VJPs."""
    spikes = tf.convert_to_tensor(spikes)
    master = tf.convert_to_tensor(master)
    basis = tf.convert_to_tensor(basis)
    if master.dtype != tf.float32 or spikes.dtype not in (tf.float16, tf.float32):
        raise ValueError("Direct masters require FP32 weights and FP16/FP32 spikes.")
    if basis.dtype != spikes.dtype:
        raise ValueError("Direct-master basis and spike dtypes must match.")
    native = library()

    @tf.custom_gradient
    def explicit_operands(z, w, b, routing, scale):
        current = native.dpointnet_master_projection(
            z, w, routing, b, n_post=n_post, n_edges=n_edges
        )

        def grad(dy):
            dz, dw = native.dpointnet_master_projection_grad(
                z, w, routing, b, tf.cast(dy, tf.float32),
                n_post=n_post, n_edges=n_edges,
            )
            return tf.cast(dz * scale, z.dtype), dw, None, None, None

        return current, grad

    return explicit_operands(spikes, master, basis, metadata, tf.cast(spike_scale, tf.float32))


def connected_project(spikes, master, connectivity, basis, spike_scale=1.0):
    metadata = connectivity.get("master_metadata")
    if metadata is None:
        metadata = tf.raw_ops.ReadVariableOp(
            resource=connectivity["metadata_handle"],
            dtype=tf.as_dtype(connectivity["index_dtype"]),
        )
    return project(
        spikes, master, metadata, basis, connectivity["n_post"],
        connectivity["n_edges"], spike_scale,
    )


def cpu_connectivity(indices, types, width, posts, *, host_lifecycle=False):
    """Same CSR field schema, without requiring a visible GPU for CPU tests."""
    indices = np.asarray(indices, dtype=np.int64)
    types = np.asarray(types, dtype=np.int64)
    if (
        indices.ndim != 2 or indices.shape[1] != 2
        or types.shape != (len(indices),)
        or np.any(indices < 0) or np.any(types < 0)
        or np.any(indices[:, 0] >= posts) or np.any(indices[:, 1] >= width)
    ):
        raise ValueError("Invalid direct-master connectivity.")
    order = np.argsort(indices[:, 1], kind="stable")
    splits = np.r_[0, np.cumsum(np.bincount(indices[:, 1], minlength=width))]
    metadata = np.concatenate((indices[order, 0], types[order], splits, order))
    result = {
        "master_metadata": tf.constant(metadata, tf.int64),
        "n_post": posts, "n_edges": len(indices), "n_sources": width,
    }
    if host_lifecycle:
        result["host_edge_order"] = order.astype(
            np.uint32 if len(indices) <= np.iinfo(np.uint32).max else np.uint64
        )
    return result
