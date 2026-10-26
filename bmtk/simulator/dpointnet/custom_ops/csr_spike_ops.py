import os
import inspect
import itertools
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import handle_data_util
from .._options import validate_bool_option

_LIBRARY_PATH = Path(__file__).with_name("_csr_spike_ops.so")
_ARCHITECTURE_PATH = Path(__file__).with_name("_csr_spike_ops.archs")
_OPS = None
_LOAD_ERROR = None
_RESOURCE_COUNTER = itertools.count()
_RESOURCE_CONTAINER = "bmtk_dpointnet_csr"
_ELIDE_VJP_PRODUCERS = ContextVar("dpointnet_elide_vjp_producers", default=False)


@contextmanager
def _elide_vjp_producers():
    token = _ELIDE_VJP_PRODUCERS.set(True)
    try:
        yield
    finally:
        _ELIDE_VJP_PRODUCERS.reset(token)


def _selected_physical_gpus():
    from ..distribution import data_parallel_worker_devices
    visible = tf.config.get_visible_devices("GPU")
    devices = data_parallel_worker_devices()
    if not devices:
        return visible if len(visible) == 1 else []
    selected = []
    for name in devices:
        spec = tf.DeviceSpec.from_string(name)
        if spec.device_type != "GPU" or spec.device_index >= len(visible):
            return []
        selected.append(visible[spec.device_index])
    return selected


def _read_built_architectures(architecture_path=_ARCHITECTURE_PATH):
    if not architecture_path.exists():
        return (), None
    values = {}
    for line in architecture_path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    sm_architectures = tuple(int(value) for value in values.get("sm", "").split())
    ptx_architecture = values.get("ptx")
    return sm_architectures, (int(ptx_architecture) if ptx_architecture else None)


_SM_ARCHITECTURES, _PTX_ARCHITECTURE = _read_built_architectures()


def _gpu_compatibility_error(
    sm_architectures=None,
    ptx_architecture=None,
    architecture_path=None,
):
    if architecture_path is None:
        architecture_path = _ARCHITECTURE_PATH
        if sm_architectures is None:
            sm_architectures = _SM_ARCHITECTURES
        if ptx_architecture is None:
            ptx_architecture = _PTX_ARCHITECTURE
    elif sm_architectures is None:
        sm_architectures = ()
    visible_gpus = _selected_physical_gpus()
    if not visible_gpus:
        count = len(tf.config.get_visible_devices("GPU"))
        return (
            "fused CUDA requires exactly one visible GPU or an explicit "
            f"data-parallel GPU strategy; found {count} visible GPUs"
        )
    if not sm_architectures and ptx_architecture is None:
        return f"build architecture metadata is missing at {architecture_path}"
    architectures = []
    for device in visible_gpus:
        capability = tf.config.experimental.get_device_details(device).get("compute_capability")
        if capability is None:
            return f"compute capability is unavailable for {device.name}"
        architecture = int(capability[0]) * 10 + int(capability[1])
        architectures.append(architecture)
        if architecture not in sm_architectures and not (
            ptx_architecture is not None and architecture >= ptx_architecture
        ):
            return (
                f"GPU compute capability sm_{architecture} is incompatible with "
                f"sm targets {sm_architectures} and compute_{ptx_architecture} PTX"
            )
    if len(set(architectures)) != 1:
        return "native data parallelism currently requires homogeneous GPU architectures"
    return None


def _destroy_metadata_resource(handle):
    try:
        if tf.is_symbolic_tensor(handle):
            tf.raw_ops.DestroyResourceOp(resource=handle, ignore_lookup_error=True)
        else:
            # Garbage collection can run while an unrelated function is tracing.
            with tf.init_scope():
                tf.raw_ops.DestroyResourceOp(resource=handle, ignore_lookup_error=True)
    except (tf.errors.OpError, RuntimeError):
        pass


class CsrConnectivity(Mapping):
    def __init__(self, values):
        self._values = values
        self._handles = values.pop("_replica_handles", (values["metadata_handle"],))
        self._replica_values = values.pop(
            "_replica_values",
            tuple({"metadata_handle": handle} for handle in self._handles),
        )
        self._finalizers = tuple(
            weakref.finalize(self, _destroy_metadata_resource, handle)
            for handle in self._handles
        )

    def __getitem__(self, key):
        if key in self._replica_values[0] and len(self._replica_values) > 1:
            context = tf.distribute.get_replica_context()
            if context is not None:
                replica = tf.get_static_value(context.replica_id_in_sync_group)
                if replica is None:
                    raise ValueError("CSR metadata requires a statically resolved local replica.")
                return self._replica_values[int(replica)][key]
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def close(self):
        for finalizer in self._finalizers:
            finalizer()

    @property
    def metadata_resource_handles(self):
        """Owned resident resource handles, without reading or copying their values."""
        return self._handles

    @property
    def metadata_resource_specs(self):
        """Creation-time census of owned resources; no tensor reads or copies."""
        return self._values["metadata_resource_specs"]


def _environment_flag(name):
    value = os.environ.get(name, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _validate_packed_sm120_option(value, option_name="use_packed_sm120_backward"):
    return validate_bool_option(value, option_name, allow_auto=True, unwrap_numpy=True)


def _gpu_compute_architecture():
    visible_gpus = _selected_physical_gpus()
    if not visible_gpus:
        return None
    architectures = []
    for device in visible_gpus:
        capability = tf.config.experimental.get_device_details(device).get("compute_capability")
        if capability is None:
            return None
        architectures.append(int(capability[0]) * 10 + int(capability[1]))
    return architectures[0] if len(set(architectures)) == 1 else None


def _auto_native_architecture(architecture):
    """Automatic packed-backward/accumulator policy: every SM70+ fat-binary target.

    SM70 and SM80 were excluded before the joint sweep was admitted on them; the
    kernels carry native SASS for every target and FP64-oracle tests qualify them.
    """
    return architecture is not None and architecture >= 70


def _resolve_packed_sm120_backward(option, spikes, connectivity, basis, *, diagnostics=None):
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
    if architecture is None or architecture < 61:
        description = "unavailable" if architecture is None else f"SM{architecture}"
        incompatibilities.append(
            f"GPU compute capability is {description}, not SM61 or newer"
        )
    if option is True and incompatibilities:
        raise ValueError(
            "use_packed_sm120_backward=True is incompatible: "
            + "; ".join(incompatibilities)
        )
    if diagnostics is not None:
        diagnostics.update({message: False for message in incompatibilities})
        diagnostics["explicit True or automatically admitted SM70+"] = (
            option is True or _auto_native_architecture(architecture)
        )
    return (
        option is not False and not incompatibilities
        and (option is True or _auto_native_architecture(architecture))
    )


def _resolve_packed_sm120_model_option(
    option, fused_cuda, compute_dtype, batch_size, basis_width
):
    option = _validate_packed_sm120_option(option, "use_packed_sm120_external_backward")
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
    if architecture is None or architecture < 61:
        description = "unavailable" if architecture is None else f"SM{architecture}"
        incompatibilities.append(
            f"GPU compute capability is {description}, not SM61 or newer"
        )
    if option is True and incompatibilities:
        raise ValueError(
            "use_packed_sm120_external_backward=True is incompatible: "
            + "; ".join(incompatibilities)
        )
    if incompatibilities or option is False:
        return False
    if option == "auto" and not _auto_native_architecture(architecture):
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


def _csr_index_dtype(n_edges, n_source_neurons, n_target_neurons, n_synapse_types):
    maximum_value = max(
        int(n_edges),
        int(n_source_neurons) - 1,
        int(n_target_neurons) - 1,
        int(n_synapse_types) - 1,
    )
    return tf.uint32 if maximum_value <= np.iinfo(np.uint32).max else tf.int64


def _create_metadata_resource(metadata, index_dtype):
    resource_name = f"csr_{os.getpid()}_{next(_RESOURCE_COUNTER)}"
    metadata_handle = tf.raw_ops.VarHandleOp(
        dtype=index_dtype,
        shape=metadata.shape,
        container=_RESOURCE_CONTAINER,
        shared_name=resource_name,
    )
    handle_data_util.set_handle_data(
        metadata_handle,
        handle_data_util.create_handle_data(
            tf.TensorShape(metadata.shape), index_dtype
        ),
    )
    tf.raw_ops.AssignVariableOp(
        resource=metadata_handle,
        value=tf.constant(metadata, dtype=index_dtype),
    )
    return metadata_handle


def _projection_type_representatives(basis, n_synapse_types):
    basis = np.asarray(basis)
    if (
        basis.ndim != 2
        or basis.shape[0] != n_synapse_types
        or basis.shape[1] == 0
        or not np.issubdtype(basis.dtype, np.floating)
    ):
        raise ValueError(
            "projection_basis must be a floating matrix with one row per synapse type."
        )
    representatives = np.empty(n_synapse_types, dtype=np.int64)
    first_by_bits = {}
    for index, row in enumerate(basis):
        representatives[index] = first_by_bits.setdefault(row.tobytes(), index)
    return representatives


def _uint8_type_options(connectivity):
    if not connectivity.get("uint8_types", False):
        return {}
    for name in _UINT8_TYPE_OPERATORS:
        if not _operator_accepts(name, ("uint8_types",)):
            raise RuntimeError(
                "use_uint8_synapse_types=True requires rebuilt CUDA operators; the loaded "
                "library lacks uint8_types on " + name
            )
    return {"uint8_types": True}


def _scalar_target_type_map(post_ids, synapse_types, n_post):
    if not post_ids.size:
        return None
    result = np.zeros(n_post, np.int64)
    np.maximum.at(result, post_ids, synapse_types)
    for start in range(0, post_ids.size, 1 << 20):
        stop = start + (1 << 20)
        if np.any(result[post_ids[start:stop]] != synapse_types[start:stop]):
            return None
    result.setflags(write=False)
    return result


def build_csr_connectivity(
    indices,
    synapse_types,
    n_source_neurons,
    n_target_neurons,
    n_synapse_types,
    build_compact_pairs=False,
    build_fixed4_incoming=False,
    sort_by_target=False,
    projection_basis=None,
    host_lifecycle=False,
    uint8_types=False,
    build_scalar_target_types=False,
):
    """Build canonical CSR routing and optional immutable-basis projection sharing.

    ``projection_basis`` must describe the exact effective coefficients used for
    every projection with this metadata. Rebuild without sharing for mutable or
    alternate coefficients; edge types and canonical/CSR weight order never change.
    """
    indices = np.asarray(indices)
    synapse_types = np.asarray(synapse_types)
    if type(build_scalar_target_types) is not bool:
        raise TypeError("build_scalar_target_types must be an explicit boolean.")
    if type(uint8_types) is not bool:
        raise TypeError("uint8_types must be an explicit boolean.")
    if uint8_types:
        if host_lifecycle is not True:
            raise ValueError("uint8 CSR types require the host lifecycle.")
        if not np.issubdtype(synapse_types.dtype, np.integer):
            raise TypeError("uint8 synapse_types must have an integer dtype.")
        if np.any(synapse_types < 0) or np.any(synapse_types > 255):
            raise ValueError("uint8 synapse_types must be in [0, 255]; no narrowing fallback.")
        _uint8_type_options({"uint8_types": True})
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
    scalar_target_types = (
        _scalar_target_type_map(post_ids, synapse_types, n_target_neurons)
        if build_scalar_target_types else None
    )
    index_dtype = _csr_index_dtype(
        indices.shape[0],
        n_source_neurons,
        n_target_neurons,
        n_synapse_types,
    )
    numpy_index_dtype = index_dtype.as_numpy_dtype
    projection_types = None
    if projection_basis is not None:
        if not build_compact_pairs:
            raise ValueError("projection_basis requires build_compact_pairs=True.")
        projection_types = _projection_type_representatives(
            projection_basis, n_synapse_types
        )
    if sort_by_target:
        if (
            int(n_target_neurons) < 2**64
            and int(n_synapse_types) < 2**64
            and int(n_source_neurons) * int(n_target_neurons) * int(n_synapse_types)
            <= 2**64
        ):
            keys = pre_ids.astype(np.uint64) * np.uint64(n_target_neurons)
            keys += post_ids.astype(np.uint64, copy=False)
            keys *= np.uint64(n_synapse_types)
            keys += synapse_types.astype(np.uint64, copy=False)
            edge_ids = np.argsort(keys, kind="stable").astype(
                numpy_index_dtype, copy=False
            )
        else:
            edge_ids = np.lexsort((synapse_types, post_ids, pre_ids)).astype(
                numpy_index_dtype, copy=False
            )
    else:
        edge_ids = np.argsort(pre_ids, kind="stable").astype(
            numpy_index_dtype, copy=False
        )
    sorted_pre_ids = pre_ids[edge_ids]
    counts = np.bincount(sorted_pre_ids, minlength=n_source_neurons)
    row_splits = np.empty(n_source_neurons + 1, dtype=numpy_index_dtype)
    row_splits[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_splits[1:])
    has_repeated_targets = False
    if sort_by_target and post_ids.size > 1:
        ordered_posts_for_runs = post_ids[edge_ids]
        has_repeated_targets = bool(
            np.any(
                (sorted_pre_ids[1:] == sorted_pre_ids[:-1])
                & (ordered_posts_for_runs[1:] == ordered_posts_for_runs[:-1])
            )
        )

    device = "/GPU:0" if tf.config.get_visible_devices("GPU") else "/CPU:0"
    with tf.device(device):
        post_ids = indices[edge_ids, 0].astype(numpy_index_dtype, copy=False)
        sorted_synapse_types = synapse_types[edge_ids].astype(
            numpy_index_dtype, copy=False
        )
        incoming_pre_ids = np.empty(0, dtype=numpy_index_dtype)
        incoming_edge_ids = np.empty(0, dtype=numpy_index_dtype)
        incoming_types = np.empty(0, dtype=numpy_index_dtype)
        fixed4_eligible = bool(
            not host_lifecycle and post_ids.size == 4 * n_target_neurons
            and np.all(np.bincount(indices[:, 0], minlength=n_target_neurons) == 4)
        )
        if build_fixed4_incoming and fixed4_eligible:
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
        ]
        if not host_lifecycle:
            metadata_parts.append(edge_ids)
        n_pairs = 0
        if build_compact_pairs and post_ids.size:
            pair_types = (
                sorted_synapse_types if projection_types is None
                else projection_types[sorted_synapse_types]
            )
            if (
                int(n_synapse_types) < 2**64
                and int(n_target_neurons) * int(n_synapse_types) <= 2**64
            ):
                stride = np.uint64(n_synapse_types)
                keys = post_ids.astype(np.uint64) * stride
                keys += pair_types.astype(np.uint64, copy=False)
                unique_keys, pair_ids = np.unique(keys, return_inverse=True)
                pairs = np.column_stack(
                    (unique_keys // stride, unique_keys % stride)
                ).astype(numpy_index_dtype, copy=False)
            else:
                pairs, pair_ids = np.unique(
                    np.column_stack((post_ids, pair_types)),
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
        type_byte_offset = None
        padding_bytes = 0
        if uint8_types:
            # All index sections precede the byte tail, so no index is unaligned.
            index_parts = [metadata_parts[0], *metadata_parts[2:]]
            wide = np.concatenate(index_parts)
            type_byte_offset = int(wide.nbytes)
            padding_bytes = (-int(sorted_synapse_types.size)) % index_dtype.size
            packed = np.concatenate((
                wide.view(np.uint8),
                sorted_synapse_types.astype(np.uint8),
                np.zeros(padding_bytes, np.uint8),
            ))
            metadata = packed.view(numpy_index_dtype)
        else:
            metadata = np.concatenate(metadata_parts)
        metadata_schema = (
            "csr_live_uint8_type_tail_v1" if uint8_types else
            "csr_live_without_edge_ids_v1" if host_lifecycle else "canonical_edge_ids_v1"
        )
        type_dtype = "uint8" if uint8_types else index_dtype.name
        type_payload_bytes = int(indices.shape[0]) * (1 if uint8_types else index_dtype.size)
        type_logical_min = int(synapse_types.min()) if synapse_types.size else None
        type_logical_max = int(synapse_types.max()) if synapse_types.size else None
        from ..distribution import data_parallel_worker_devices
        devices = data_parallel_worker_devices() or (device,)
        handles = []
        replica_values = []
        for worker in devices:
            with tf.device(worker):
                handles.append(_create_metadata_resource(metadata, index_dtype))
                replica_values.append(
                    {
                        "metadata_handle": handles[-1],
                        "incoming_pre_ids": tf.constant(incoming_pre_ids, dtype=index_dtype),
                        "incoming_edge_ids": tf.constant(incoming_edge_ids, dtype=index_dtype),
                        "incoming_types": tf.constant(incoming_types, dtype=index_dtype),
                    }
                )
        metadata_handle = handles[0]
        return CsrConnectivity(
            {
                "metadata_handle": metadata_handle,
                "_replica_handles": tuple(handles),
                "_replica_values": tuple(replica_values),
                "index_dtype": index_dtype.name,
                "n_edges": int(indices.shape[0]),
                "n_sources": int(n_source_neurons),
                "n_post": int(n_target_neurons),
                "n_synapse_types": int(n_synapse_types),
                "n_pairs": n_pairs,
                "metadata_shape": tuple(metadata.shape),
                "metadata_bytes": int(metadata.nbytes),
                "metadata_schema": metadata_schema,
                "metadata_resource_specs": tuple({
                    "handle": handle,
                    "dtype": index_dtype.name,
                    "shape": tuple(metadata.shape),
                    "payload_bytes": int(metadata.nbytes),
                    "logical_payload_bytes": int(metadata.nbytes) - padding_bytes,
                    "alignment_bytes": padding_bytes,
                    "role": "csr_metadata",
                    "schema": metadata_schema,
                    "type_dtype": type_dtype,
                    "type_payload_bytes": type_payload_bytes,
                    "type_logical_min": type_logical_min,
                    "type_logical_max": type_logical_max,
                } for handle in handles),
                "uint8_types": bool(uint8_types),
                "metadata_dtype": index_dtype.name,
                "type_dtype": type_dtype,
                "type_byte_offset": type_byte_offset,
                "type_payload_bytes": type_payload_bytes,
                "type_logical_min": type_logical_min,
                "type_logical_max": type_logical_max,
                "metadata_padding_bytes": padding_bytes,
                "metadata_payload_bytes": int(metadata.nbytes) - padding_bytes,
                "canonical_edge_id_offset": (
                    None if host_lifecycle else 2 * int(indices.shape[0]) + int(n_source_neurons) + 1
                ),
                "device_edge_ids": not host_lifecycle,
                **({
                    "host_edge_order": edge_ids.astype(
                        np.uint32 if indices.shape[0] <= np.iinfo(np.uint32).max else np.uint64,
                        copy=False,
                    ),
                } if host_lifecycle else {}),
                "projection_basis_shared": (
                    projection_types is not None
                    and bool(np.any(projection_types != np.arange(n_synapse_types)))
                ),
                "has_repeated_targets": has_repeated_targets,
                "sorted_by_target": bool(sort_by_target),
                **({"scalar_target_types": scalar_target_types} if build_scalar_target_types else {}),
                "fixed4_eligible": fixed4_eligible,
                "fixed4_incoming": bool(incoming_pre_ids.size),
                "incoming_pre_ids": replica_values[0]["incoming_pre_ids"],
                "incoming_edge_ids": replica_values[0]["incoming_edge_ids"],
                "incoming_types": replica_values[0]["incoming_types"],
            }
        )


def reorder_csr_values(values, connectivity):
    if connectivity.get("uint8_types"):
        raise ValueError("Typed host-lifecycle metadata has no device canonical edge IDs.")
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


def restore_csr_values(values, connectivity):
    if connectivity.get("uint8_types"):
        raise ValueError("Typed host-lifecycle metadata has no device canonical edge IDs.")
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet CUDA operator is unavailable: {cuda_op_status()}"
        )
    return _OPS.dpointnet_csr_restore(
        values,
        connectivity["metadata_handle"],
        Tindex=tf.dtypes.as_dtype(connectivity["index_dtype"]),
        n_edges=connectivity["n_edges"],
        n_sources=connectivity["n_sources"],
        n_pairs=connectivity["n_pairs"],
    )


# Op type of the VJP-only producer over CSR-ordered FP32 masters. The develop
# DpointnetCsrSpikeForward schema is unchanged, so GraphDefs serialized before
# CSR masters existed import and differentiate as before.
CSR_MASTER_FORWARD_OP = "DpointnetCsrSpikeForwardCsrMaster"


def _optional_bool_attr(op, name):
    """``op``'s boolean attribute, False when its (older) schema lacks it."""
    try:
        return bool(op.get_attr(name))
    except (ValueError, TypeError):
        return False


@ops.RegisterGradient("DpointnetCsrSpikeForward")
@ops.RegisterGradient(CSR_MASTER_FORWARD_OP)
def _fused_spike_currents_gradient(op, current_grad):
    # Resolved once from the op type: only the CSR-master proxy carries FP32
    # masters and the uint8 type tail, so develop-schema nodes and libraries
    # without these attributes take the canonical gradient operators.
    csr_master_weights = op.type == CSR_MASTER_FORWARD_OP
    return _projection_gradient(
        op.inputs, current_grad,
        csr_master_weights=csr_master_weights,
        uint8_types=csr_master_weights and _optional_bool_attr(op, "uint8_types"),
        compute_spike_gradient=op.get_attr("compute_spike_gradient"),
        compute_weight_gradient=op.get_attr("compute_weight_gradient"),
        n_post=op.get_attr("n_post"),
        index_dtype=op.get_attr("Tindex"),
        n_edges=op.get_attr("n_edges"),
        n_pairs=op.get_attr("n_pairs"),
        use_packed_sm120_backward=op.get_attr("use_packed_sm120_backward"),
        write_csr_weight_gradient=op.get_attr("write_csr_weight_gradient"),
        use_small_batch_backward=op.get_attr("use_small_batch_backward"),
    )


def _projection_gradient(
    inputs, current_grad, *, csr_master_weights, uint8_types,
    compute_spike_gradient, compute_weight_gradient, n_post, index_dtype,
    n_edges, n_pairs, use_packed_sm120_backward, write_csr_weight_gradient,
    use_small_batch_backward,
):
    type_options = {"uint8_types": True} if uint8_types else {}
    if compute_spike_gradient:
        gradient_operator = (
            _OPS.dpointnet_csr_spike_grad_csr_master
            if csr_master_weights else _OPS.dpointnet_csr_spike_grad
        )
        spike_grad, weight_grad = gradient_operator(
            inputs[0],
            current_grad,
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            Tindex=index_dtype,
            n_post=n_post,
            n_edges=n_edges,
            n_pairs=n_pairs,
            use_packed_sm120_backward=use_packed_sm120_backward,
            write_csr_weight_gradient=(
                write_csr_weight_gradient or csr_master_weights
            ),
            use_small_batch_backward=use_small_batch_backward,
            use_javier_batch32_backward=False,
            **type_options,
        )
        if not compute_weight_gradient:
            weight_grad = None
    elif compute_weight_gradient:
        spike_grad = None
        weight_grad = _OPS.dpointnet_csr_weight_grad(
            inputs[0],
            current_grad,
            inputs[2],
            inputs[4],
            **type_options,
            Tindex=index_dtype,
            n_post=n_post,
            n_edges=n_edges,
            n_pairs=n_pairs,
            use_packed_sm120_backward=use_packed_sm120_backward,
            **({"write_csr_weight_gradient": True} if csr_master_weights else {}),
        )
    else:
        spike_grad = None
        weight_grad = None
    initial_grad = current_grad if inputs[10].shape.rank == 2 else None
    return (
        spike_grad,
        (tf.cast(weight_grad, inputs[1].dtype) if weight_grad is not None else None),
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


def _elided_projection_currents(inputs, **gradient_options):
    metadata = _projection_metadata_in_current_graph(inputs[2])

    @tf.custom_gradient
    def project(*arguments):
        spikes, basis = arguments[0], arguments[3]
        batch = spikes.shape[0] if spikes.shape[0] is not None else tf.shape(spikes)[0]
        width = basis.shape[1] if basis.shape[1] is not None else tf.shape(basis)[1]
        currents = tf.zeros((batch * gradient_options["n_post"], width), spikes.dtype)

        def grad(current_grad):
            gradients = _projection_gradient(
                arguments[:2] + (metadata,) + arguments[2:],
                current_grad, **gradient_options,
            )
            return gradients[:2] + gradients[3:]

        return currents, grad

    return project(*(inputs[:2] + inputs[3:]))


def _projection_metadata_in_current_graph(metadata):
    if tf.inside_function():
        # Resource outputs on custom-gradient IdentityN look like new loop resources.
        return ops.get_default_graph().capture(metadata)
    return metadata


def fused_recurrent_accumulation_available():
    return (
        _OPS is not None
        and hasattr(_OPS, "dpointnet_csr_spike_grad_accumulate")
        and (_gpu_compute_architecture() or 0) >= 61
    )


def weight_only_accumulation_available():
    return (
        _OPS is not None
        and hasattr(_OPS, "dpointnet_csr_spike_grad_accumulate")
        and "compute_spike_gradient" in inspect.signature(
            _OPS.dpointnet_csr_spike_grad_accumulate
        ).parameters
    )


def fused_recurrent_weight_carry(
    spikes,
    weight_carrier,
    csr_weights,
    connectivity,
    basis,
    n_post,
    spike_gradient_scale,
    vjp_only=True,
    use_javier_batch32_backward=False,
    use_active_row_forward=False,
    use_forward_run_aggregation=False,
    use_device_active_queue_forward=False,
    compute_spike_gradient=True,
    csr_master_weights=False,
):
    """Currents with a differentiable identity weight carrier.

    The carrier's cotangent is the true loop-carried CSR weight accumulator.
    It is passed into the producer, not added to a materialized step gradient.
    FP32 replay uses ``vjp_only=True`` to provide only the derivative; the
    ordinary direct RNN loop uses ``vjp_only=False`` so the same call supplies
    the primal compute-dtype recurrent currents.
    ``compute_spike_gradient=False`` skips input-spike credit on the FP16
    Javier accumulator path. Weight credit and the default live recurrent
    spike-credit behavior are unchanged.
    """
    if _OPS is None or not hasattr(_OPS, "dpointnet_csr_spike_grad_accumulate"):
        raise RuntimeError("Rebuild CUDA operators for fused recurrent accumulation.")
    if weight_carrier.dtype != tf.float32:
        raise ValueError("Fused recurrent accumulation requires an FP32 carrier.")
    if not isinstance(compute_spike_gradient, bool):
        raise TypeError("compute_spike_gradient must be an explicit boolean.")
    if not compute_spike_gradient and (
        spikes.dtype != tf.float16 or not use_javier_batch32_backward
    ):
        raise ValueError(
            "Weight-only accumulation requires FP16 Javier event-weight backward."
        )
    if not compute_spike_gradient and not weight_only_accumulation_available():
        raise RuntimeError(
            "Rebuild CUDA operators for compute_spike_gradient=False; "
            "the loaded accumulator does not support weight-only gradients."
        )
    weight_dtype = tf.float32 if csr_master_weights else spikes.dtype
    if csr_weights.dtype != weight_dtype or basis.dtype != spikes.dtype:
        raise ValueError(
            "Fused recurrent accumulation requires matching compute operands."
        )
    @tf.custom_gradient
    def project(z, carrier, weights, coefficients, scale):
        currents = fused_spike_currents(
            z,
            carrier,
            weights,
            connectivity,
            coefficients,
            n_post,
            compute_spike_gradient=True,
            spike_gradient_scale=scale,
            use_packed_sm120_backward=False,
            write_csr_weight_gradient=True,
            vjp_only=vjp_only,
            use_active_row_forward=use_active_row_forward,
            use_forward_run_aggregation=use_forward_run_aggregation,
            use_device_active_queue_forward=use_device_active_queue_forward,
            **({"csr_master_weights": True} if csr_master_weights else {}),
        )
        metadata = (
            _projection_metadata_in_current_graph(connectivity["metadata_handle"])
            if vjp_only and _ELIDE_VJP_PRODUCERS.get()
            else connectivity["metadata_handle"] if tf.executing_eagerly()
            else currents.op.inputs[2]
        )
        def grad(dcurrents, dcarrier):
            if dcarrier is None:
                dcarrier = tf.zeros_like(carrier)
            if dcurrents is None:
                dcurrents = tf.zeros_like(currents)
            spike_options = {} if compute_spike_gradient else {"compute_spike_gradient": False}
            gradient_operator = (
                _OPS.dpointnet_csr_spike_grad_accumulate_csr_master
                if csr_master_weights else _OPS.dpointnet_csr_spike_grad_accumulate
            )
            dz, accumulated = gradient_operator(
                z,
                dcurrents,
                metadata,
                weights,
                coefficients,
                scale,
                dcarrier,
                Tindex=tf.as_dtype(connectivity["index_dtype"]),
                n_post=n_post,
                n_edges=connectivity["n_edges"],
                n_pairs=connectivity["n_pairs"],
                use_javier_batch32_backward=use_javier_batch32_backward,
                **spike_options,
                **_uint8_type_options(connectivity),
            )
            return (
                dz if compute_spike_gradient else None,
                accumulated, None, None, None,
            )

        return (currents, tf.identity(carrier)), grad

    return project(
        spikes,
        weight_carrier,
        csr_weights,
        basis,
        tf.cast(spike_gradient_scale, spikes.dtype),
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
    write_csr_weight_gradient=False,
    use_small_batch_backward=False,
    use_active_row_forward=False,
    use_forward_run_aggregation=False,
    use_device_active_queue_forward=False,
    vjp_only=False,
    csr_master_weights=False,
):
    """Project currents, or supply only their registered VJP for recorded replay.

    ``spikes`` may contain graded nonnegative source amplitudes, not only binary
    events. Positive values multiply the per-edge weights unchanged; counted
    Poisson inputs and local STP efficacy must not be Booleanized. With
    ``compute_spike_gradient=True``, the VJP returns dense per-source credit,
    including silent connected sources. For STP this is efficacy credit: apply
    the local release VJP before routing source-event credit across partitions.

    With ``write_csr_weight_gradient=False`` (the default), weight gradients
    already follow canonical master-edge order, including duplicate edges.
    Do not restore that gradient a second time. Direct-CSR carrier routes have
    a separate, exactly-once canonical restoration contract.

    The internal ``vjp_only`` mode returns zeros and must only be used under
    saved-primal substitution; it does not approximate the forward simulation.
    """
    if _OPS is None:
        raise RuntimeError(
            f"Fused DPointNet CUDA operator is unavailable: {cuda_op_status()}"
        )
    type_options = _uint8_type_options(connectivity)
    if type_options and not csr_master_weights:
        raise ValueError("uint8 CSR metadata requires direct CSR FP32 masters.")
    if spikes.dtype not in (tf.float16, tf.float32):
        raise TypeError(
            f"Fused DPointNet CUDA operator requires float16 or float32 spikes, got {spikes.dtype}."
        )
    if vjp_only and initial_currents is not None:
        raise ValueError("VJP-only projection requires no initial currents.")
    weight_dtype = tf.float32 if csr_master_weights else spikes.dtype
    if csr_master_weights and not vjp_only:
        raise ValueError("Canonical master weights require joint VJP-only projection.")
    if csr_weights.dtype != weight_dtype or basis.dtype != spikes.dtype:
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
    if write_csr_weight_gradient and not compute_spike_gradient:
        raise ValueError(
            "write_csr_weight_gradient=True requires recurrent spike gradients."
        )
    if use_small_batch_backward:
        if not compute_spike_gradient or spikes.shape[0] not in range(1, 9):
            raise ValueError(
                "Small-batch backward requires recurrent spike gradients and batch size 1..8."
            )
    if use_active_row_forward:
        if spikes.shape[0] not in range(1, 33) or basis.shape[1] != 4:
            raise ValueError(
                "Active-row forward requires batch size 1..32 and four basis columns."
            )
    if use_forward_run_aggregation and basis.shape[1] != 4:
        raise ValueError("Forward run aggregation requires four basis columns.")
    if use_device_active_queue_forward:
        if spikes.shape[0] not in range(1, 33) or basis.shape[1] != 4:
            raise ValueError(
                "Device active-queue forward requires batch size 1..32 and four basis columns."
            )
    use_grouped_batch32_forward = (
        use_active_row_forward or use_device_active_queue_forward or spikes.shape[0] == 32
    ) and basis.shape[1] == 4
    if use_grouped_batch32_forward and not vjp_only and not use_device_active_queue_forward:
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

    forward = (
        _OPS.dpointnet_csr_spike_forward_csr_master
        if csr_master_weights else _OPS.dpointnet_csr_spike_forward
    )
    if vjp_only and _ELIDE_VJP_PRODUCERS.get():
        return _elided_projection_currents(
            (
                spikes, master_weights, connectivity["metadata_handle"],
                csr_weights, basis, spike_gradient_scale, active_rows,
                tf.cast(connectivity["incoming_pre_ids"], index_dtype),
                tf.cast(connectivity["incoming_edge_ids"], index_dtype),
                tf.cast(connectivity["incoming_types"], index_dtype), initial_currents,
            ),
            csr_master_weights=csr_master_weights,
            uint8_types=bool(type_options),
            compute_spike_gradient=compute_spike_gradient,
            compute_weight_gradient=compute_weight_gradient,
            n_post=n_post, index_dtype=index_dtype,
            n_edges=connectivity["n_edges"], n_pairs=connectivity["n_pairs"],
            use_packed_sm120_backward=use_packed_sm120_backward,
            write_csr_weight_gradient=write_csr_weight_gradient,
            use_small_batch_backward=use_small_batch_backward,
        )
    return forward(
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
        use_forward_run_aggregation=use_forward_run_aggregation,
        use_device_active_queue_forward=use_device_active_queue_forward,
        use_packed_sm120_backward=use_packed_sm120_backward,
        write_csr_weight_gradient=write_csr_weight_gradient,
        use_small_batch_backward=use_small_batch_backward,
        vjp_only=vjp_only,
        **type_options,
    )


def combined_sweep_available():
    return (
        fused_cuda_available()
        and hasattr(_OPS, "dpointnet_combined_sweep")
        and "target_tile_size" in inspect.signature(_OPS.dpointnet_combined_sweep).parameters
    )


def _combined_sweep_accepts(attribute):
    """Whether the loaded combined sweep advertises ``attribute``; False without a library."""
    operator = getattr(_OPS, "dpointnet_combined_sweep", None)
    return (
        combined_sweep_available()
        and operator is not None
        and attribute in inspect.signature(operator).parameters
    )


def cooperative_combined_sweep_available():
    return _combined_sweep_accepts("cooperative_rows")


def fanout_combined_sweep_available():
    return _combined_sweep_accepts("fanout_rows")


def gather_combined_sweep_available():
    return _combined_sweep_accepts("gather_sources")


ADAPTIVE_ROUTING_SCHEMA = "adaptive_segment_ticket_v1"
ADAPTIVE_TRAVERSAL_CODES = {"warp_batches": 1, "block_rounds": 2, "gather": 3}
_ADAPTIVE_TRAVERSAL_NAMES = {code: name for name, code in ADAPTIVE_TRAVERSAL_CODES.items()}
_BASELINE_TRAVERSAL_NAMES = {
    10: "scatter_rows", 11: "balanced_rounds", 12: "cooperative_rows", 13: "fanout_rounds",
    14: "gather",
}
_ROUTING_CENSUS_FIELDS = (
    "dense_tile_bytes", "dense_tile_tables", "segment_bytes", "segment_tables", "mask_bytes",
    "gather_table_bytes", "gather_tables", "transient_queue_bytes", "queue_operators",
    "last_selected_tile", "traversal_code", "segments", "built_tile_bits", "operators",
)
_ADAPTIVE_TILES = (256, 512, 1024, 2048, 3072, 4096, 5120)


def _operator_accepts(name, attributes):
    operator = getattr(_OPS, name, None)
    if operator is None:
        return False
    parameters = inspect.signature(operator).parameters
    return all(attribute in parameters for attribute in attributes)


# Native operators that read CSR-ordered FP32 master weights directly.
_CSR_MASTER_OPERATORS = (
    ("dpointnet_combined_sweep_csr_master", ()),
    ("dpointnet_csr_spike_grad_csr_master", ("write_csr_weight_gradient",)),
    ("dpointnet_csr_spike_grad_accumulate_csr_master", ("write_csr_weight_gradient",)),
    ("dpointnet_csr_weight_grad", ("write_csr_weight_gradient",)),
    ("dpointnet_csr_spike_forward_csr_master", ()),
)
# Every native consumer of per-edge synapse types must read the uint8 tail.
_UINT8_TYPE_OPERATORS = (
    "dpointnet_csr_spike_forward_csr_master", "dpointnet_csr_weight_grad",
    "dpointnet_csr_spike_grad_csr_master",
    "dpointnet_csr_spike_grad_accumulate_csr_master",
    "dpointnet_combined_sweep_csr_master",
)


def csr_master_weights_available():
    """Whether the loaded library reads CSR-ordered FP32 master weights natively."""
    return combined_sweep_available() and all(
        _operator_accepts(name, attributes) for name, attributes in _CSR_MASTER_OPERATORS
    )


def uint8_synapse_types_available():
    """Whether every native per-edge type consumer accepts the uint8 type tail."""
    return csr_master_weights_available() and all(
        _operator_accepts(name, ("uint8_types",)) for name in _UINT8_TYPE_OPERATORS
    )


def adaptive_routing_available():
    """Whether the loaded library has the adaptive routing attributes and census op."""
    return (
        _combined_sweep_accepts("cooperative_rows")
        and hasattr(_OPS, "dpointnet_routing_census")
        and _operator_accepts(
            "dpointnet_combined_sweep_csr_master",
            ("adaptive_routing", "adaptive_traversal", "uint8_types"),
        )
    )


def require_adaptive_routing():
    if not adaptive_routing_available():
        raise RuntimeError(
            "use_adaptive_routing=True requires rebuilt CUDA operators: the loaded library "
            "lacks dpointnet_routing_census or the adaptive_routing/adaptive_traversal attributes. "
            "Rebuild both operators or set use_adaptive_routing=False."
        )


def scalar_input_projection_available():
    return adaptive_routing_available() and _operator_accepts(
        "dpointnet_combined_sweep_csr_master", ("scalar_source", "scalar_target_types"),
    )


def adaptive_narrow_tile(wide):
    """Narrow tile paired with ``wide``; equal to ``wide`` when none is narrower."""
    quarter = int(wide) // 4
    return 1024 if quarter >= 1024 else 512 if quarter >= 512 else 256


def adaptive_selected_tile(batch, n_post, wide_tile, multiprocessors, wide_blocks_per_sm):
    """Per-launch adaptive tile, mirroring the native rule for given device properties.

    The narrow tile is chosen when the wide grid fills at most half of the device's
    resident wide-kernel block slots (``multiprocessors * wide_blocks_per_sm``, the
    latter measured natively by an occupancy query); otherwise the wide tile is kept.
    """
    wide = validate_combined_target_tile_size(wide_tile)
    narrow = adaptive_narrow_tile(wide)
    if narrow == wide:
        return wide
    wide_blocks = int(batch) * (-(-int(n_post) // wide))
    return narrow if 2 * wide_blocks <= int(multiprocessors) * int(wide_blocks_per_sm) else wide


def adaptive_routing_plan(
    connectivities, *, target_tile_size=2048, gather_sources=False, scalar_source=None,
):
    """Deterministic per-source traversal policy of the adaptive routing schema.

    gather: low in-degree source (``combined_gather_eligible``) when target gather is
    enabled; warp_batches: rows never repeat a target and mean fanout is below 512;
    block_rounds: every other source (repeated targets, unknown repetition or high fanout).
    The tile width is chosen per launch on the device: ``wide`` unless the wide grid
    fills at most half of the resident block slots, then ``narrow``.
    """
    wide = validate_combined_target_tile_size(target_tile_size)
    traversals = []
    for conn in connectivities:
        if gather_sources and combined_gather_eligible(conn):
            traversals.append("gather")
        elif (
            conn.get("has_repeated_targets") is False
            and conn["n_sources"] > 0
            and conn["n_edges"] // conn["n_sources"] < 512
        ):
            traversals.append("warp_batches")
        else:
            traversals.append("block_rounds")
    if scalar_source is not None:
        if not isinstance(scalar_source, (int, np.integer)) or isinstance(scalar_source, bool):
            raise TypeError("scalar_source must be an integer source index or None.")
        if not 1 <= scalar_source < len(connectivities):
            raise ValueError("Scalar input projection requires a named input source.")
        if connectivities[scalar_source].get("scalar_target_types") is None:
            raise ValueError("Scalar input projection requires one synapse type per target.")
        traversals[scalar_source] = "block_rounds"
    return {
        "schema": ADAPTIVE_ROUTING_SCHEMA,
        "wide_tile": wide,
        "narrow_tile": adaptive_narrow_tile(wide),
        "traversals": traversals,
        "codes": [ADAPTIVE_TRAVERSAL_CODES[name] for name in traversals],
    }


def combined_routing_geometry(
    connectivities, *, target_tile_size=2048, cooperative_rows=False, fanout_rows=False,
    gather_sources=False, adaptive_routing=False, scalar_source=None,
):
    """Static per-source routing geometry for either schema, as the native op dispatches it.

    Disabled schema: one fixed tile and the existing traversal per source (gather,
    cooperative_rows, fanout_rounds, balanced_rounds or scatter_rows). Adaptive schema:
    the adaptive plan's traversal plus wide/narrow tiles chosen per launch. Traversal
    names match ``combined_routing_census(...)[i]["traversal"]`` after execution.
    """
    tile = validate_combined_target_tile_size(target_tile_size)
    if adaptive_routing:
        plan = adaptive_routing_plan(
            connectivities, target_tile_size=tile, gather_sources=gather_sources,
            scalar_source=scalar_source,
        )
        rule = (
            "narrow iff 2*batch*ceil(n_post/wide_tile) <= multiprocessors*resident wide blocks "
            "per multiprocessor; else wide"
        )
        return [
            {"schema": ADAPTIVE_ROUTING_SCHEMA, "traversal": name, "wide_tile": plan["wide_tile"],
             "narrow_tile": plan["narrow_tile"], "tile_rule": rule}
            for name in plan["traversals"]
        ]
    result = []
    for conn in connectivities:
        cooperative = (
            cooperative_rows and conn.get("has_repeated_targets") is False
            and conn["n_sources"] > 0 and conn["n_edges"] // conn["n_sources"] < 512
        )
        high_fanout = conn["n_sources"] > 0 and conn["n_edges"] // conn["n_sources"] >= 512
        if gather_sources and combined_gather_eligible(conn):
            name = "gather"
        elif cooperative:
            name = "cooperative_rows"
        elif high_fanout:
            name = "fanout_rounds" if fanout_rows else "balanced_rounds"
        else:
            name = "scatter_rows"
        result.append({"schema": "dense_tile_starts", "traversal": name, "tile": tile})
    return result


def combined_routing_census(connectivities):
    """Exact native census of live routing allocations per connectivity.

    Reads only allocation sizes recorded by the native operators and shared tables
    keyed by each metadata resource's current allocation; no routing data is read or
    copied. Tables are built on first execution, so call after a warm execution.
    Totals include every live operator instance (for example several compiled graphs).
    """
    if not hasattr(_OPS, "dpointnet_routing_census"):
        raise RuntimeError("Rebuild both CUDA operators for the native routing census.")
    handles = [conn["metadata_handle"] for conn in connectivities]
    with tf.device(handles[0].device or None):
        values = _OPS.dpointnet_routing_census(handles).numpy()
    result = []
    for row in values:
        item = dict(zip(_ROUTING_CENSUS_FIELDS, (int(value) for value in row)))
        code = item["traversal_code"]
        item["traversal"] = (
            _ADAPTIVE_TRAVERSAL_NAMES.get(code) or _BASELINE_TRAVERSAL_NAMES.get(code)
            or ("unexecuted" if code == 0 else f"unknown_{code}")
        )
        item["built_tiles"] = [
            tile for bit, tile in enumerate(_ADAPTIVE_TILES) if item["built_tile_bits"] & (1 << bit)
        ]
        item["routing_table_bytes"] = (
            item["dense_tile_bytes"] + item["segment_bytes"] + item["mask_bytes"]
            + item["gather_table_bytes"]
        )
        result.append(item)
    return result


# Average incoming edges per target at or below which a source is projected by a
# target-major gather (one owner thread per target, no atomics or row lookups)
# instead of the source-major scatter walk.
COMBINED_GATHER_MAX_MEAN_IN_DEGREE = 16


def combined_gather_eligible(connectivity):
    return (
        connectivity["n_edges"] > 0
        and connectivity["n_edges"] <= COMBINED_GATHER_MAX_MEAN_IN_DEGREE * connectivity["n_post"]
    )


@tf.custom_gradient
def _substitute_combined_primal(primal, *proxies):
    def grad(dcurrent):
        return (None,) + (dcurrent,) * len(proxies)

    return tf.identity(primal), grad


def validate_combined_target_tile_size(target_tile_size):
    if (
        isinstance(target_tile_size, bool)
        or not isinstance(target_tile_size, (int, np.integer))
        or target_tile_size not in (256, 512, 1024, 2048, 3072, 4096, 5120)
    ):
        raise ValueError("Combined sweep target_tile_size must be 256, 512, 1024, 2048, 3072, 4096 or 5120.")
    return int(target_tile_size)


def fused_combined_currents(
    spikes, weights, connectivities, basis, proxies, *, target_tile_size=2048,
    cooperative_rows=False, fanout_rows=False, gather_sources=False,
    csr_master_weights=False, adaptive_routing=False,
    scalar_source=None,
):
    """FP32 four-basis joint sweep with existing per-source canonical/carrier VJPs.

    ``adaptive_routing=True`` selects the adaptive_segment_ticket_v1 forward
    schedule (CSR FP32 masters and uint8 type metadata required); per-source VJPs keep
    their existing routing. The cooperative/fanout requests are superseded by the
    adaptive per-source traversal; gather_sources still admits eligible gather sources.
    """
    if len(spikes) != len(proxies):
        raise ValueError("Combined sweep requires two to eight matching sources and VJP proxies.")
    primal = combined_sweep_primal(
        spikes, weights, connectivities, basis, target_tile_size=target_tile_size,
        cooperative_rows=cooperative_rows, fanout_rows=fanout_rows,
        gather_sources=gather_sources, csr_master_weights=csr_master_weights,
        adaptive_routing=adaptive_routing, scalar_source=scalar_source,
    )
    return _substitute_combined_primal(tf.stop_gradient(primal), *proxies)


def combined_sweep_primal(
    spikes, weights, connectivities, basis, *, target_tile_size=2048,
    cooperative_rows=False, fanout_rows=False, gather_sources=False,
    csr_master_weights=False, adaptive_routing=False, scalar_source=None,
):
    """The joint sweep's non-differentiable primal currents, ``[batch * n_post, 4]``.

    ``fused_combined_currents`` attaches per-source VJP proxies to this value;
    the direct recorded adjoint consumes it as-is and calls the per-source
    native VJP operators itself.
    """
    target_tile_size = validate_combined_target_tile_size(target_tile_size)
    if type(adaptive_routing) is not bool:
        raise ValueError("adaptive_routing must be true or false.")
    if adaptive_routing:
        if not csr_master_weights:
            raise ValueError("Adaptive routing requires direct CSR FP32 master weights.")
        require_adaptive_routing()
    if type(cooperative_rows) is not bool:
        raise ValueError("cooperative_rows must be true or false.")
    if type(fanout_rows) is not bool:
        raise ValueError("fanout_rows must be true or false.")
    if type(gather_sources) is not bool:
        raise ValueError("gather_sources must be true or false.")
    if fanout_rows and not fanout_combined_sweep_available():
        raise RuntimeError("Rebuild both CUDA operators for fanout combined rows.")
    if gather_sources and not gather_combined_sweep_available():
        raise RuntimeError("Rebuild both CUDA operators for gathered combined sources.")
    if cooperative_rows and not cooperative_combined_sweep_available():
        raise RuntimeError("Rebuild both CUDA operators for cooperative combined rows.")
    if not combined_sweep_available():
        raise RuntimeError("Rebuild both CUDA operators for the combined current sweep.")
    if not 2 <= len(spikes) <= 8 or not (
        len(spikes) == len(weights) == len(connectivities)
    ):
        raise ValueError("Combined sweep requires two to eight matching sources and VJP proxies.")
    dtype = spikes[0].dtype
    index = connectivities[0]["index_dtype"]
    type_options = _uint8_type_options(connectivities[0])
    if type_options and not csr_master_weights:
        raise ValueError("uint8 CSR metadata requires direct CSR FP32 masters.")
    if adaptive_routing and not type_options:
        raise ValueError("Adaptive routing requires uint8 CSR type metadata (host lifecycle).")
    if dtype not in (tf.float16, tf.float32) or basis.dtype != dtype or basis.shape[1] != 4:
        raise ValueError("Combined sweep requires matching FP16/FP32 operands and four bases.")
    for z, weight, connectivity in zip(spikes, weights, connectivities):
        if bool(connectivity.get("uint8_types")) != bool(type_options):
            raise ValueError("Combined sweep requires matching type metadata schemas.")
        weight_dtype = tf.float32 if csr_master_weights else dtype
        if z.dtype != dtype or weight.dtype != weight_dtype or z.shape.rank != 2:
            raise ValueError("Combined sweep spike/weight dtypes and spike ranks must match.")
        if not connectivity.get("sorted_by_target", False) or connectivity["index_dtype"] != index:
            raise ValueError("Combined sweep requires target-sorted CSR with matching index dtypes.")
        if connectivity["n_post"] != connectivities[0]["n_post"]:
            raise ValueError("Combined sweep target dimensions must match.")
        if z.shape[1] != connectivity["n_sources"] or weight.shape != (connectivity["n_edges"],):
            raise ValueError(
                "Combined sweep connectivity and operand shapes must match: spikes "
                f"{z.shape} for {connectivity['n_sources']} sources, weights {weight.shape} "
                f"for {connectivity['n_edges']} edges."
            )
        if connectivity["n_synapse_types"] != basis.shape[0]:
            raise ValueError("Combined sweep type counts must match the basis.")
    schedule = {}
    if fanout_rows:
        schedule["fanout_rows"] = True
    if gather_sources:
        gathered = [combined_gather_eligible(conn) for conn in connectivities]
        if any(gathered):
            schedule["gather_sources"] = gathered
    if cooperative_rows:
        schedule["cooperative_rows"] = [
            conn.get("has_repeated_targets") is False
            and conn["n_sources"] > 0
            and conn["n_edges"] // conn["n_sources"] < 512
            for conn in connectivities
        ]
    if adaptive_routing:
        plan = adaptive_routing_plan(
            connectivities, target_tile_size=target_tile_size, gather_sources=gather_sources,
            scalar_source=scalar_source,
        )
        schedule["adaptive_routing"] = True
        schedule["adaptive_traversal"] = plan["codes"]
    if scalar_source is not None:
        if not isinstance(scalar_source, (int, np.integer)) or isinstance(scalar_source, bool):
            raise TypeError("scalar_source must be an integer source index or None.")
        if not adaptive_routing or not 1 <= scalar_source < len(connectivities):
            raise ValueError("Scalar input projection requires an adaptive named input source.")
        if not scalar_input_projection_available():
            raise RuntimeError("Rebuild both CUDA operators for scalar input projection.")
        target_types = connectivities[scalar_source].get("scalar_target_types")
        if target_types is None:
            raise ValueError("Scalar input projection requires one synapse type per target.")
        schedule["scalar_source"] = int(scalar_source)
        schedule["scalar_target_types"] = target_types.tolist()
    operator = (
        _OPS.dpointnet_combined_sweep_csr_master if csr_master_weights
        else _OPS.dpointnet_combined_sweep
    )
    return operator(
        spikes, weights, [conn["metadata_handle"] for conn in connectivities], basis,
        Tindex=tf.as_dtype(index), n_post=connectivities[0]["n_post"],
        n_edges=[conn["n_edges"] for conn in connectivities],
        n_pairs=[conn["n_pairs"] for conn in connectivities],
        target_tile_size=target_tile_size,
        **schedule,
        **type_options,
    )


ops.NotDifferentiable("DpointnetCombinedSweep")
ops.NotDifferentiable("DpointnetCombinedSweepCsrMaster")
