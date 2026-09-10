import os
import itertools
import weakref
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops


_LIBRARY_PATH = Path(__file__).with_name('_csr_spike_ops.so')
_ARCHITECTURE_PATH = Path(__file__).with_name('_csr_spike_ops.archs')
_OPS = None
_LOAD_ERROR = None
_RESOURCE_COUNTER = itertools.count()
_RESOURCE_CONTAINER = 'bmtk_dpointnet_csr'


def _read_built_architectures():
    if not _ARCHITECTURE_PATH.exists():
        return (), None
    values = {}
    for line in _ARCHITECTURE_PATH.read_text().splitlines():
        key, separator, value = line.partition('=')
        if separator:
            values[key] = value
    sm_architectures = tuple(
        int(value) for value in values.get('sm', '').split()
    )
    ptx_architecture = values.get('ptx')
    return sm_architectures, (
        int(ptx_architecture) if ptx_architecture else None
    )


_SM_ARCHITECTURES, _PTX_ARCHITECTURE = _read_built_architectures()


def _gpu_compatibility_error():
    visible_gpus = tf.config.get_visible_devices('GPU')
    if len(visible_gpus) != 1:
        return f'fused CUDA requires exactly one visible GPU; found {len(visible_gpus)}'
    if not _SM_ARCHITECTURES and _PTX_ARCHITECTURE is None:
        return f'build architecture metadata is missing at {_ARCHITECTURE_PATH}'
    details = tf.config.experimental.get_device_details(visible_gpus[0])
    capability = details.get('compute_capability')
    if capability is None:
        return f'compute capability is unavailable for {visible_gpus[0].name}'
    architecture = int(capability[0]) * 10 + int(capability[1])
    if architecture in _SM_ARCHITECTURES:
        return None
    if _PTX_ARCHITECTURE is not None and architecture >= _PTX_ARCHITECTURE:
        return None
    return (
        f'GPU compute capability sm_{architecture} is incompatible with '
        f'sm targets {_SM_ARCHITECTURES} and compute_{_PTX_ARCHITECTURE} PTX'
    )


def _destroy_metadata_resource(handle):
    try:
        tf.raw_ops.DestroyResourceOp(
            resource=handle, ignore_lookup_error=True
        )
    except (tf.errors.OpError, RuntimeError):
        pass


class CsrConnectivity(Mapping):
    def __init__(self, values):
        self._values = values
        self._finalizer = weakref.finalize(
            self, _destroy_metadata_resource, values['metadata_handle']
        )

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def close(self):
        self._finalizer()


def _environment_flag(name):
    value = os.environ.get(name, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _validate_packed_sm120_option(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is True or value is False:
        return value
    if isinstance(value, (bytes, np.bytes_)):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            pass
    if isinstance(value, (str, np.str_)) and value == "auto":
        return "auto"
    raise ValueError('use_packed_sm120_backward must be true, false, or "auto".')


def _gpu_compute_architecture():
    visible_gpus = tf.config.get_visible_devices("GPU")
    if len(visible_gpus) != 1:
        return None
    capability = tf.config.experimental.get_device_details(visible_gpus[0]).get(
        "compute_capability"
    )
    if capability is None:
        return None
    return int(capability[0]) * 10 + int(capability[1])


def _resolve_packed_sm120_backward(option, spikes, connectivity, basis):
    option = _validate_packed_sm120_option(option)
    incompatibilities = []
    if spikes.dtype != tf.float16:
        incompatibilities.append(f"compute dtype is {spikes.dtype.name}, not float16")
    if connectivity["index_dtype"] != tf.uint32.name:
        incompatibilities.append(
            f'CSR metadata dtype is {connectivity["index_dtype"]}, not uint32'
        )
    if spikes.shape[0] != 32:
        incompatibilities.append(f"batch size is {spikes.shape[0]}, not 32")
    if basis.shape[1] != 4:
        incompatibilities.append(f"basis width is {basis.shape[1]}, not 4")
    if connectivity["n_pairs"] <= 0:
        incompatibilities.append("compact pair metadata is unavailable")
    architecture = _gpu_compute_architecture()
    if architecture is None or architecture < 86:
        description = "unavailable" if architecture is None else f"SM{architecture}"
        incompatibilities.append(
            f"GPU compute capability is {description}, not SM86 or newer"
        )
    if option is True and incompatibilities:
        raise ValueError(
            "use_packed_sm120_backward=True is incompatible: "
            + "; ".join(incompatibilities)
        )
    return option is not False and not incompatibilities


def _resolve_packed_sm120_model_option(
    option, fused_cuda, compute_dtype, batch_size, basis_width
):
    option = _validate_packed_sm120_option(option)
    incompatibilities = []
    if not fused_cuda:
        incompatibilities.append("fused CUDA currents are disabled or unavailable")
    if tf.as_dtype(compute_dtype) != tf.float16:
        incompatibilities.append(
            f"compute dtype is {tf.as_dtype(compute_dtype).name}, not float16"
        )
    if batch_size != 32:
        incompatibilities.append(f"batch size is {batch_size}, not 32")
    if basis_width != 4:
        incompatibilities.append(f"basis width is {basis_width}, not 4")
    architecture = _gpu_compute_architecture()
    if architecture is None or architecture < 86:
        description = "unavailable" if architecture is None else f"SM{architecture}"
        incompatibilities.append(
            f"GPU compute capability is {description}, not SM86 or newer"
        )
    if option is True and incompatibilities:
        raise ValueError(
            "use_packed_sm120_external_backward=True is incompatible: "
            + "; ".join(incompatibilities)
        )
    if incompatibilities or option is False:
        return False
    return option


if not _environment_flag("BMTK_DPOINTNET_DISABLE_FUSED_CUDA"):
    if _LIBRARY_PATH.exists():
        try:
            _OPS = tf.load_op_library(str(_LIBRARY_PATH))
        except (tf.errors.NotFoundError, OSError) as exc:
            _LOAD_ERROR = exc
    else:
        _LOAD_ERROR = FileNotFoundError(
            f"Fused DPointNet CUDA library does not exist at {_LIBRARY_PATH}."
        )


def cuda_op_status():
    if _environment_flag("BMTK_DPOINTNET_DISABLE_FUSED_CUDA"):
        return "disabled by BMTK_DPOINTNET_DISABLE_FUSED_CUDA"
    if _OPS is not None:
        compatibility_error = _gpu_compatibility_error()
        if compatibility_error is not None:
            return f"loaded, but {compatibility_error}"
        return f"loaded from {_LIBRARY_PATH}"
    return str(_LOAD_ERROR)


def fused_cuda_available():
    return _OPS is not None and _gpu_compatibility_error() is None


def _csr_index_dtype(
        n_edges, n_source_neurons, n_target_neurons, n_synapse_types):
    maximum_value = max(
        int(n_edges),
        int(n_source_neurons) - 1,
        int(n_target_neurons) - 1,
        int(n_synapse_types) - 1,
    )
    return tf.uint32 if maximum_value <= np.iinfo(np.uint32).max else tf.int64


def _create_metadata_resource(metadata, index_dtype):
    resource_name = f'csr_{os.getpid()}_{next(_RESOURCE_COUNTER)}'
    metadata_handle = tf.raw_ops.VarHandleOp(
        dtype=index_dtype,
        shape=metadata.shape,
        container=_RESOURCE_CONTAINER,
        shared_name=resource_name,
    )
    tf.raw_ops.AssignVariableOp(
        resource=metadata_handle,
        value=tf.constant(metadata, dtype=index_dtype),
    )
    return metadata_handle


def build_csr_connectivity(
    indices,
    synapse_types,
    n_source_neurons,
    n_target_neurons,
    n_synapse_types,
    build_compact_pairs=False,
    build_fixed4_incoming=False,
):
    indices = np.asarray(indices)
    synapse_types = np.asarray(synapse_types)
    for dimension, name in (
        (n_source_neurons, "n_source_neurons"),
        (n_target_neurons, "n_target_neurons"),
        (n_synapse_types, "n_synapse_types"),
    ):
        if not isinstance(dimension, (int, np.integer)) or dimension <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"indices must have shape [n_edges, 2], got {indices.shape}.")
    if synapse_types.shape != (indices.shape[0],):
        raise ValueError(
            "synapse_types must contain one value per edge, got "
            f"{synapse_types.shape} for {indices.shape[0]} edges."
        )
    for values, name in ((indices, "indices"), (synapse_types, "synapse_types")):
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{name} must contain numeric integer values.")
        if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
            raise ValueError(f"{name} must contain finite integer values.")
        int64_info = np.iinfo(np.int64)
        if np.any(values < int64_info.min) or np.any(values > int64_info.max):
            raise ValueError(f"{name} values must be within the int64 range.")

    indices = indices.astype(np.int64, copy=False)
    synapse_types = synapse_types.astype(np.int64, copy=False)
    pre_ids = indices[:, 1]
    if np.any(pre_ids < 0) or np.any(pre_ids >= n_source_neurons):
        raise ValueError(
            "Presynaptic indices are outside the declared source dimension."
        )
    post_ids = indices[:, 0]
    if np.any(post_ids < 0) or np.any(post_ids >= n_target_neurons):
        raise ValueError(
            "Postsynaptic indices are outside the declared target dimension."
        )
    if np.any(synapse_types < 0) or np.any(synapse_types >= n_synapse_types):
        raise ValueError("Synapse type indices are outside the basis table.")
    index_dtype = _csr_index_dtype(
        indices.shape[0],
        n_source_neurons,
        n_target_neurons,
        n_synapse_types,
    )
    numpy_index_dtype = index_dtype.as_numpy_dtype
    edge_ids = np.argsort(pre_ids, kind="stable").astype(numpy_index_dtype, copy=False)
    sorted_pre_ids = pre_ids[edge_ids]
    counts = np.bincount(sorted_pre_ids, minlength=n_source_neurons)
    row_splits = np.empty(n_source_neurons + 1, dtype=numpy_index_dtype)
    row_splits[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_splits[1:])

    device = "/GPU:0" if tf.config.get_visible_devices("GPU") else "/CPU:0"
    with tf.device(device):
        post_ids = indices[edge_ids, 0].astype(numpy_index_dtype, copy=False)
        sorted_synapse_types = synapse_types[edge_ids].astype(
            numpy_index_dtype, copy=False
        )
        incoming_pre_ids = np.empty(0, dtype=numpy_index_dtype)
        incoming_edge_ids = np.empty(0, dtype=numpy_index_dtype)
        incoming_types = np.empty(0, dtype=numpy_index_dtype)
        if build_fixed4_incoming and post_ids.size == 4 * n_target_neurons:
            post_counts = np.bincount(post_ids, minlength=n_target_neurons)
            if np.all(post_counts == 4):
                incoming_order = np.argsort(post_ids, kind="stable").astype(
                    numpy_index_dtype, copy=False
                )
                source_ids = np.repeat(
                    np.arange(n_source_neurons, dtype=numpy_index_dtype),
                    counts,
                )
                incoming_pre_ids = source_ids[incoming_order]
                incoming_edge_ids = incoming_order
                incoming_types = sorted_synapse_types[incoming_order]
        metadata_parts = [
            post_ids,
            sorted_synapse_types,
            row_splits,
            edge_ids,
        ]
        n_pairs = 0
        if build_compact_pairs and post_ids.size:
            pairs, pair_ids = np.unique(
                np.column_stack((post_ids, sorted_synapse_types)),
                axis=0,
                return_inverse=True,
            )
            metadata_parts.extend(
                (
                    pair_ids.astype(numpy_index_dtype, copy=False),
                    pairs[:, 0].astype(numpy_index_dtype, copy=False),
                    pairs[:, 1].astype(numpy_index_dtype, copy=False),
                )
            )
            n_pairs = int(pairs.shape[0])
        metadata = np.concatenate(metadata_parts)
        metadata_handle = _create_metadata_resource(metadata, index_dtype)
        return CsrConnectivity(
            {
                "metadata_handle": metadata_handle,
                "index_dtype": index_dtype.name,
                "n_edges": int(indices.shape[0]),
                "n_sources": int(n_source_neurons),
                "n_post": int(n_target_neurons),
                "n_synapse_types": int(n_synapse_types),
                "n_pairs": n_pairs,
                "fixed4_incoming": bool(incoming_pre_ids.size),
                "incoming_pre_ids": tf.constant(incoming_pre_ids, dtype=index_dtype),
                "incoming_edge_ids": tf.constant(incoming_edge_ids, dtype=index_dtype),
                "incoming_types": tf.constant(incoming_types, dtype=index_dtype),
            }
        )


def reorder_csr_values(values, connectivity):
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet CUDA operator is unavailable: {cuda_op_status()}"
        )
    return _OPS.dpointnet_csr_reorder(
        values,
        connectivity["metadata_handle"],
        Tindex=tf.dtypes.as_dtype(connectivity["index_dtype"]),
        n_edges=connectivity["n_edges"],
        n_sources=connectivity["n_sources"],
        n_pairs=connectivity["n_pairs"],
    )


@ops.RegisterGradient("DpointnetCsrSpikeForward")
def _fused_spike_currents_gradient(op, current_grad):
    compute_spike_gradient = op.get_attr("compute_spike_gradient")
    compute_weight_gradient = op.get_attr("compute_weight_gradient")
    n_post = op.get_attr("n_post")
    index_dtype = op.get_attr("Tindex")
    if compute_spike_gradient:
        spike_grad, weight_grad = _OPS.dpointnet_csr_spike_grad(
            op.inputs[0],
            current_grad,
            op.inputs[2],
            op.inputs[3],
            op.inputs[4],
            op.inputs[5],
            Tindex=index_dtype,
            n_post=n_post,
            n_edges=op.get_attr("n_edges"),
            n_pairs=op.get_attr("n_pairs"),
            use_packed_sm120_backward=op.get_attr("use_packed_sm120_backward"),
        )
        if not compute_weight_gradient:
            weight_grad = None
    elif compute_weight_gradient:
        spike_grad = None
        weight_grad = _OPS.dpointnet_csr_weight_grad(
            op.inputs[0],
            current_grad,
            op.inputs[2],
            op.inputs[4],
            Tindex=index_dtype,
            n_post=n_post,
            n_edges=op.get_attr("n_edges"),
            n_pairs=op.get_attr("n_pairs"),
            use_packed_sm120_backward=op.get_attr("use_packed_sm120_backward"),
        )
    else:
        spike_grad = None
        weight_grad = None
    initial_grad = current_grad if op.inputs[10].shape.rank == 2 else None
    return (
        spike_grad,
        (tf.cast(weight_grad, op.inputs[1].dtype) if weight_grad is not None else None),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        initial_grad,
    )


def fused_spike_currents(
    spikes,
    master_weights,
    csr_weights,
    connectivity,
    basis,
    n_post,
    compute_spike_gradient,
    compute_weight_gradient=True,
    spike_gradient_scale=1.0,
    use_packed_sm120_backward="auto",
    use_fixed4_forward=False,
    initial_currents=None,
):
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet CUDA operator is unavailable: {cuda_op_status()}"
        )
    if spikes.dtype not in (tf.float16, tf.float32):
        raise TypeError(
            f"Fused DPointNet CUDA operator requires float16 or float32 spikes, got {spikes.dtype}."
        )
    if csr_weights.dtype != spikes.dtype or basis.dtype != spikes.dtype:
        raise TypeError("spikes, csr_weights, and basis must have the same dtype.")
    if master_weights.shape.rank != 1 or csr_weights.shape.rank != 1:
        raise ValueError("master_weights and csr_weights must be rank 1.")
    spike_gradient_scale = tf.convert_to_tensor(
        spike_gradient_scale, dtype=spikes.dtype
    )
    if spike_gradient_scale.shape.rank != 0:
        raise ValueError("spike_gradient_scale must be a scalar.")
    n_edges = connectivity["n_edges"]
    if master_weights.shape[0] is not None and master_weights.shape[0] != n_edges:
        raise ValueError(
            f"master_weights has {master_weights.shape[0]} values; "
            f"connectivity has {n_edges} edges."
        )
    if csr_weights.shape[0] is not None and csr_weights.shape[0] != n_edges:
        raise ValueError(
            f"csr_weights has {csr_weights.shape[0]} values; "
            f"connectivity has {n_edges} edges."
        )

    if connectivity["n_post"] != n_post:
        raise ValueError(
            f'Connectivity targets {connectivity["n_post"]} neurons, got n_post={n_post}.'
        )
    if connectivity["n_synapse_types"] != basis.shape[0]:
        raise ValueError(
            "Connectivity synapse types do not match the basis table: "
            f'{connectivity["n_synapse_types"]} != {basis.shape[0]}.'
        )
    if use_fixed4_forward and not connectivity["fixed4_incoming"]:
        raise ValueError(
            "use_fixed4_forward=True requires exactly four incoming edges "
            "per postsynaptic neuron."
        )

    use_packed_sm120_backward = _resolve_packed_sm120_backward(
        use_packed_sm120_backward, spikes, connectivity, basis
    )
    use_grouped_batch32_forward = spikes.shape[0] == 32 and basis.shape[1] == 4
    if use_grouped_batch32_forward:
        active_rows = tf.cast(
            tf.where(tf.reduce_any(spikes > 0, axis=0))[:, 0],
            tf.int64,
        )
    else:
        active_rows = tf.zeros([0], tf.int64)
    index_dtype = tf.dtypes.as_dtype(connectivity["index_dtype"])
    initial_currents = (
        tf.zeros([0], spikes.dtype)
        if initial_currents is None
        else tf.convert_to_tensor(initial_currents, dtype=spikes.dtype)
    )
    if initial_currents.shape.rank not in (1, 2):
        raise ValueError("initial_currents must be an empty vector or rank two.")
    if initial_currents.shape.rank == 1:
        if initial_currents.shape[0] != 0:
            raise ValueError("A rank-one initial_currents tensor must be empty.")
    elif initial_currents.shape.num_elements() == 0:
        raise ValueError("A rank-two initial_currents tensor must be nonempty.")

    return _OPS.dpointnet_csr_spike_forward(
        spikes,
        master_weights,
        connectivity["metadata_handle"],
        csr_weights,
        basis,
        spike_gradient_scale,
        active_rows,
        tf.cast(connectivity["incoming_pre_ids"], index_dtype),
        tf.cast(connectivity["incoming_edge_ids"], index_dtype),
        tf.cast(connectivity["incoming_types"], index_dtype),
        initial_currents,
        n_post=n_post,
        n_edges=connectivity["n_edges"],
        n_pairs=connectivity["n_pairs"],
        compute_spike_gradient=compute_spike_gradient,
        compute_weight_gradient=compute_weight_gradient,
        use_grouped_batch32_forward=use_grouped_batch32_forward,
        use_fixed4_forward=use_fixed4_forward,
        use_packed_sm120_backward=use_packed_sm120_backward,
    )
