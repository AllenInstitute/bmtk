"""Frozen recurrent connectivity without device-side canonical edge tables."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from .cell_models.nest_dynamics import time_steps
from .custom_ops import compact_inference_ops


RECURRENT_OPTIONS = (
    "use_pair_projection", "use_direct_csr_recurrent_gradient",
    "use_small_batch_recurrent_backward", "use_active_row_forward",
    "use_forward_run_aggregation", "use_device_active_queue_forward",
    "use_fused_recurrent_accumulation", "use_javier_recurrent_vjp",
    "use_packed_sm120_backward",
    # Joint/CSR-master routes take the standard recurrent CSR as their first source.
    "use_combined_current_sweep", "use_csr_master_weights", "use_uint8_synapse_types",
    "use_adaptive_routing", "use_scalar_input_projection", "use_locality_row_schedule",
    "use_cooperative_row_projection", "use_fanout_row_projection",
    "use_target_gather_projection", "use_fused_current_accumulation",
)


class UnsupportedCompactConnectivity(ValueError):
    """A valid configuration that needs the standard representation."""


def resolve_connectivity(requested, *, inference_only, cell_cls, params, strategy, compute_dtype=tf.float32,
                         neuron_partitioned=False):
    from .cell_models.glif3_cell import GLIF3Cell

    if requested not in ("auto", "standard", "compact"):
        raise ValueError('run.inference_connectivity must be "auto", "standard" or "compact".')
    reason = None
    if requested == "standard":
        reason = "standard connectivity explicitly requested"
    elif not inference_only:
        reason = "training or an unspecified Python-API execution mode requires standard connectivity"
    elif neuron_partitioned:
        reason = "neuron-partitioned execution uses the standard partitioned recurrent route"
    elif cell_cls is not GLIF3Cell:
        reason = "custom cell classes require their existing connectivity implementation"
    elif tf.distribute.has_strategy():
        reason = "compact CPU canonical weights must be constructed outside an enclosing strategy scope"
    elif strategy.num_replicas_in_sync != 1:
        reason = "compact inference currently requires one replica"
    elif tf.as_dtype(compute_dtype) not in (tf.float16, tf.float32):
        reason = "compact inference supports FP16/FP32 compute without changing requested precision"
    elif params.get("dynamics_mode", "nest") != "nest":
        reason = "legacy delay layouts retain standard connectivity"
    elif params.get("train_recurrent_per_type", False):
        reason = "per-type recurrent scaling requires standard connectivity"
    elif params.get("temporal_gradient_precision", "compute") != "compute":
        reason = "temporal adjoints require standard connectivity"
    elif params.get("acceleration_profile", "auto") is None or params.get("use_fused_cuda") is False:
        reason = "explicit reference/native-current override retains standard connectivity"
    elif any(name in params and params[name] is not False for name in RECURRENT_OPTIONS):
        reason = "an explicit recurrent accelerator requires standard connectivity"
    elif not compact_inference_ops.available():
        reason = compact_inference_ops.status()
    selected = "compact" if reason is None else "standard"
    if requested == "compact" and selected != "compact":
        raise ValueError(f"Compact inference requested but unsupported: {reason}")
    return {"requested": requested, "selected": selected,
            "reason": reason or "frozen inference with a compatible compact operator"}


def prepare_recurrent(network, params, dt, compute_dtype, variable_dtype):
    """Construct canonical CPU masters before entering a device strategy scope."""
    if tf.distribute.has_strategy():
        raise ValueError("Compact host weights must be prepared outside a distribution strategy scope.")
    if tf.as_dtype(compute_dtype) not in (tf.float16, tf.float32) or tf.as_dtype(variable_dtype) != tf.float32:
        raise UnsupportedCompactConnectivity("Compact inference requires FP16/FP32 compute and FP32 canonical weights.")
    synapses = network["synapses"]
    indices = np.asarray(synapses["indices"])
    neurons = int(network["n_nodes"])
    weights = np.asarray(synapses["weights"])
    syn_ids = np.asarray(synapses["syn_ids"])
    if (indices.ndim != 2 or indices.shape[1:] != (2,)
            or not np.issubdtype(indices.dtype, np.integer)
            or weights.shape != (len(indices),) or syn_ids.shape != weights.shape
            or not np.issubdtype(syn_ids.dtype, np.integer)):
        raise ValueError("Invalid compact recurrent edge shapes or integer indices.")
    if neurons < 1 or np.any(indices < 0) or np.any(indices >= neurons):
        raise ValueError("Compact recurrent endpoint outside the neuron population.")
    basis = params.get("synaptic_basis_weights")
    if basis is None:
        basis = synapses["dynamics_params"]["basis_weights"]
    elif isinstance(basis, str):
        import pickle
        with open(basis, "rb") as stream:
            basis = np.array(list(pickle.load(stream).values()))
    basis = np.asarray(basis)
    tau = params.get("tau_basis")
    if isinstance(tau, (str, Path)):
        tau = np.load(tau)
    if basis.ndim != 2 or tau is None or basis.shape[1] != np.asarray(tau).size:
        raise ValueError("Compact synaptic basis and tau_basis dimensions differ.")
    if len(basis) < 1 or np.any(syn_ids < 0) or np.any(syn_ids >= len(basis)) or not np.isfinite(basis).all():
        raise ValueError("Invalid compact synaptic type or nonfinite basis.")
    compact_basis, type_map = np.unique(
        basis.astype(tf.as_dtype(compute_dtype).as_numpy_dtype), axis=0, return_inverse=True)
    if not np.isfinite(compact_basis).all():
        raise ValueError("Compact basis conversion overflow.")
    n_types = len(compact_basis)
    if neurons * n_types > 2**32:
        raise UnsupportedCompactConnectivity("Compact target/type packing exceeds uint32.")
    raw_delays = np.asarray(synapses["delays"])
    if raw_delays.shape != weights.shape or not np.isfinite(raw_delays).all():
        raise ValueError("Invalid compact delay shape or values.")
    limit = params.get("max_delay", 5)
    maximum = float(np.max(raw_delays, initial=dt))
    maximum = maximum if limit is None or limit <= 0 else min(maximum, limit)
    if params.get("dynamics_mode", "nest") == "nest":
        steps = max(1, int(time_steps(maximum, dt)))
        delays = time_steps(np.clip(raw_delays, dt, maximum), dt)
    else:
        maximum = int(np.round(maximum))
        steps = maximum
        delays = np.round(np.clip(raw_delays, dt, maximum) / dt).astype(np.int32)
    if steps < 1 or np.any(delays < 1) or np.any(delays > steps):
        raise ValueError("Compact delay outside the allocated recurrent history.")
    rows = indices[:, 1].astype(np.int64) + (delays.astype(np.int64) - 1) * neurons
    order = np.argsort(rows, kind="stable")
    row_ptr = np.empty(neurons * steps + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(np.bincount(rows, minlength=len(row_ptr) - 1), out=row_ptr[1:])
    packed = (indices[:, 0].astype(np.uint64) * n_types + type_map[syn_ids].astype(np.uint64))[order].astype(np.uint32)
    node_params = network["node_params"]
    selective = params.get("state_precision") == "selective"
    threshold = np.asarray(node_params["V_th"], dtype=np.float64 if selective else None)
    resting = np.asarray(node_params["E_L"], dtype=np.float64 if selective else None)
    scale = (threshold - resting)[np.asarray(network["node_type_ids"])[indices[:, 0]]]
    weight_scale = params.get("recurrent_weight_scale", 1.0)
    lr_scale = params.get("lr_scale", 1.0)
    normalized = np.asarray(weights / scale * weight_scale / lr_scale, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite normalized compact recurrent weights.")
    with tf.device("/CPU:0"):
        host = tf.keras.layers.Layer(name="compact_canonical_weights", dtype="float32")
        options = dict(name="sparse_recurrent_weights", shape=normalized.shape, dtype="float32",
                       trainable=False, initializer=tf.keras.initializers.Constant(normalized))
        try:
            host.master = host.add_weight(autocast=False, **options)
        except TypeError as error:
            if "autocast" not in str(error):
                raise
            host.master = host.add_weight(experimental_autocast=False, **options)
        host.order = tf.constant(order, dtype=tf.int64)
    host.row_ptr = row_ptr
    host.targets = packed
    host.export_factor = np.asarray(scale * lr_scale / weight_scale, dtype=np.float32)
    host.initial_compute = normalized[order].astype(tf.as_dtype(compute_dtype).as_numpy_dtype)
    if not np.isfinite(host.initial_compute).all():
        raise ValueError("Compact compute-weight conversion overflow.")
    host.neurons = neurons
    host.delay_steps = steps
    host.n_types = n_types
    host.basis = compact_basis
    return host
