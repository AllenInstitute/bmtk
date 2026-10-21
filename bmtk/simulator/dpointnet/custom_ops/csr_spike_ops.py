import os
import inspect
import itertools
import weakref
from collections.abc import Mapping
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
    return architecture is not None and (architecture == 75 or architecture >= 86)


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
):
    """Build canonical CSR routing and optional immutable-basis projection sharing.

    ``projection_basis`` must describe the exact effective coefficients used for
    every projection with this metadata. Rebuild without sharing for mutable or
    alternate coefficients; edge types and canonical/CSR weight order never change.
    """
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
        metadata = np.concatenate(metadata_parts)
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
                "projection_basis_shared": (
                    projection_types is not None
                    and bool(np.any(projection_types != np.arange(n_synapse_types)))
                ),
                "has_repeated_targets": has_repeated_targets,
                "sorted_by_target": bool(sort_by_target),
                "fixed4_incoming": bool(incoming_pre_ids.size),
                "incoming_pre_ids": replica_values[0]["incoming_pre_ids"],
                "incoming_edge_ids": replica_values[0]["incoming_edge_ids"],
                "incoming_types": replica_values[0]["incoming_types"],
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


def restore_csr_values(values, connectivity):
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


@ops.RegisterGradient("DpointnetCsrSpikeForward")
def _fused_spike_currents_gradient(op, current_grad):
    compute_spike_gradient = op.get_attr("compute_spike_gradient")
    compute_weight_gradient = op.get_attr("compute_weight_gradient")
    n_post = op.get_attr("n_post")
    index_dtype = op.get_attr("Tindex")
    if compute_spike_gradient:
        gradient_operator = (
            _OPS.dpointnet_csr_spike_grad_csr_master
            if op.get_attr("csr_master_weights") else _OPS.dpointnet_csr_spike_grad
        )
        spike_grad, weight_grad = gradient_operator(
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
            write_csr_weight_gradient=(
                op.get_attr("write_csr_weight_gradient") or op.get_attr("csr_master_weights")
            ),
            use_small_batch_backward=op.get_attr("use_small_batch_backward"),
            use_javier_batch32_backward=False,
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
            **({"write_csr_weight_gradient": True} if op.get_attr("csr_master_weights") else {}),
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
        # Reuse this FuncGraph's capture; recapturing the eager handle in grad
        # breaks TensorFlow's nested while-gradient resource mapping.
        metadata = (
            connectivity["metadata_handle"]
            if tf.executing_eagerly()
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
        use_forward_run_aggregation=use_forward_run_aggregation,
        use_device_active_queue_forward=use_device_active_queue_forward,
        use_packed_sm120_backward=use_packed_sm120_backward,
        write_csr_weight_gradient=write_csr_weight_gradient,
        use_small_batch_backward=use_small_batch_backward,
        vjp_only=vjp_only,
        **({"csr_master_weights": True} if csr_master_weights else {}),
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
    csr_master_weights=False,
):
    """FP32 four-basis joint sweep with existing per-source canonical/carrier VJPs."""
    target_tile_size = validate_combined_target_tile_size(target_tile_size)
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
        len(spikes) == len(weights) == len(connectivities) == len(proxies)
    ):
        raise ValueError("Combined sweep requires two to eight matching sources and VJP proxies.")
    dtype = spikes[0].dtype
    index = connectivities[0]["index_dtype"]
    if dtype not in (tf.float16, tf.float32) or basis.dtype != dtype or basis.shape[1] != 4:
        raise ValueError("Combined sweep requires matching FP16/FP32 operands and four bases.")
    for z, weight, connectivity in zip(spikes, weights, connectivities):
        weight_dtype = tf.float32 if csr_master_weights else dtype
        if z.dtype != dtype or weight.dtype != weight_dtype or z.shape.rank != 2:
            raise ValueError("Combined sweep spike/weight dtypes and spike ranks must match.")
        if not connectivity.get("sorted_by_target", False) or connectivity["index_dtype"] != index:
            raise ValueError("Combined sweep requires target-sorted CSR with matching index dtypes.")
        if connectivity["n_post"] != connectivities[0]["n_post"]:
            raise ValueError("Combined sweep target dimensions must match.")
        if z.shape[1] != connectivity["n_sources"] or weight.shape != (connectivity["n_edges"],):
            raise ValueError("Combined sweep connectivity and operand shapes must match.")
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
    operator = (
        _OPS.dpointnet_combined_sweep_csr_master if csr_master_weights
        else _OPS.dpointnet_combined_sweep
    )
    primal = operator(
        spikes, weights, [conn["metadata_handle"] for conn in connectivities], basis,
        Tindex=tf.as_dtype(index), n_post=connectivities[0]["n_post"],
        n_edges=[conn["n_edges"] for conn in connectivities],
        n_pairs=[conn["n_pairs"] for conn in connectivities],
        target_tile_size=target_tile_size,
        **schedule,
    )
    return _substitute_combined_primal(tf.stop_gradient(primal), *proxies)


ops.NotDifferentiable("DpointnetCombinedSweep")
ops.NotDifferentiable("DpointnetCombinedSweepCsrMaster")
