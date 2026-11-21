"""Locality row schedules for the recurrent spike-gradient backward.

The FP16 Javier-derived backward computes every source row's spike cotangent in
one warp and gathers one projected ``(target, type)`` line per edge, so on GPUs
whose L2 cache is much smaller than the projected operand it is bound by DRAM
gathers. Rows that run concurrently share cached lines when they belong to the
same presynaptic class (the same set of outgoing projection types) and lie close
together. A schedule orders rows by class, then along a Hilbert curve over the
neurons' positions or, without positions, over a two-dimensional spectral
embedding of the connectivity. It only reassigns rows to warps: every row's sum,
and therefore the result, is unchanged.
"""

import time

import numpy as np


def hilbert_index(a, b, bits=12):
    """Index along a ``2**bits`` Hilbert curve of the points ``(a, b)`` scaled to a square."""
    def grid(values):
        values = np.asarray(values, dtype=np.float64)
        span = np.ptp(values)
        scaled = (values - values.min()) / span if span > 0 else np.zeros_like(values)
        return np.minimum((scaled * (2**bits)).astype(np.int64), 2**bits - 1)

    x, y = grid(a), grid(b)
    index = np.zeros(x.size, dtype=np.int64)
    step = 2 ** (bits - 1)
    while step > 0:
        rx = ((x & step) > 0).astype(np.int64)
        ry = ((y & step) > 0).astype(np.int64)
        index += step * step * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        x = np.where(flip, step - 1 - x, x)
        y = np.where(flip, step - 1 - y, y)
        swap = ry == 0
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        step //= 2
    return index


def spectral_coordinates(pre, post, n_neurons, seed=0):
    """Two coordinates per neuron from the second and third left singular vectors of the
    degree-normalized binary adjacency; for distance-dependent connectivity they recover
    the layout up to rotation."""
    import scipy.sparse as sparse
    from scipy.sparse.linalg import svds

    adjacency = sparse.csr_matrix(
        (np.ones(pre.size, dtype=np.float64), (pre, post)), shape=(n_neurons, n_neurons)
    )
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    rows = np.asarray(adjacency.sum(axis=1)).ravel()
    columns = np.asarray(adjacency.sum(axis=0)).ravel()
    normalized = (
        sparse.diags(1.0 / np.sqrt(np.maximum(rows, 1.0)))
        @ adjacency
        @ sparse.diags(1.0 / np.sqrt(np.maximum(columns, 1.0)))
    )
    if n_neurons < 3:
        return np.zeros((n_neurons, 2))
    if n_neurons <= 4096:
        # Small networks: dense SVD (svds needs k < n and is not faster here).
        vectors, values, _ = np.linalg.svd(normalized.toarray())
        return vectors[:, 1:3]
    vectors, values, _ = svds(normalized, k=3, random_state=seed)
    order = np.argsort(-values)
    return vectors[:, order[1:3]]


def presynaptic_classes(pre, types, n_neurons):
    """Class rank per neuron: identical sets of outgoing types share a class, ranked by first neuron."""
    n_types = int(types.max()) + 1 if types.size else 1
    present = np.zeros((n_neurons, n_types), dtype=bool)
    present[pre, types] = True
    _, classes = np.unique(np.packbits(present, axis=1), axis=0, return_inverse=True)
    classes = classes.reshape(-1)
    first = np.full(classes.max() + 1, n_neurons, dtype=np.int64)
    np.minimum.at(first, classes, np.arange(n_neurons))
    return np.argsort(np.argsort(first))[classes]


def gpu_l2_cache_bytes():
    """L2 cache bytes of the selected GPU from the CUDA runtime, or ``None``."""
    import ctypes
    import ctypes.util

    import tensorflow as tf

    if not tf.config.list_physical_devices("GPU"):
        return None
    for name in ("libcudart.so.12", "libcudart.so", ctypes.util.find_library("cudart")):
        if not name:
            continue
        try:
            runtime = ctypes.CDLL(name)
            device = int(tf.config.list_logical_devices("GPU")[0].name.rsplit(":", 1)[1])
            value = ctypes.c_int()
            # cudaDevAttrL2CacheSize
            if runtime.cudaDeviceGetAttribute(ctypes.byref(value), 38, device) == 0 and value.value > 0:
                return int(value.value)
        except (OSError, AttributeError, IndexError, ValueError):
            continue
    return None


def locality_row_schedule(indices, types, n_neurons, n_sources, positions=None):
    """``(schedule, report)``: a uint32 permutation of the ``n_sources`` CSR source rows.

    ``indices`` are ``[target, source]`` recurrent edges whose source is
    ``neuron + n_neurons * slot`` (``slot`` the delay slab) and ``types`` their
    projection types. ``positions`` (``[n_neurons, 2]``) are optional plane
    coordinates; without them a spectral embedding of the connectivity is used.
    """
    start = time.perf_counter()
    if n_neurons <= 0 or n_sources % n_neurons:
        raise ValueError("Recurrent CSR sources must be whole delay slabs of the neurons.")
    sources = np.asarray(indices[:, 1], dtype=np.int64)
    neurons = sources % n_neurons
    classes = presynaptic_classes(neurons, np.asarray(types, dtype=np.int64), n_neurons)
    if positions is not None:
        coordinates = np.asarray(positions, dtype=np.float64)
        if coordinates.shape != (n_neurons, 2) or not np.all(np.isfinite(coordinates)):
            raise ValueError("positions must be finite [n_neurons, 2] plane coordinates.")
        basis = "positions"
    else:
        coordinates = spectral_coordinates(neurons, np.asarray(indices[:, 0], dtype=np.int64), n_neurons)
        basis = "spectral embedding of the connectivity"
    curve = hilbert_index(coordinates[:, 0], coordinates[:, 1])
    order = np.lexsort((curve, classes)).astype(np.int64)
    slabs = n_sources // n_neurons
    schedule = (np.arange(slabs, dtype=np.int64)[:, None] * n_neurons + order[None, :]).reshape(-1)
    if np.any(np.bincount(schedule, minlength=n_sources) != 1):
        raise AssertionError("The row schedule is not a permutation.")
    report = (
        f"locality (presynaptic class then Hilbert order over {basis}; "
        f"{int(classes.max()) + 1} classes, {slabs} delay slabs, "
        f"{time.perf_counter() - start:.1f} s)"
    )
    return schedule.astype(np.uint32), report
