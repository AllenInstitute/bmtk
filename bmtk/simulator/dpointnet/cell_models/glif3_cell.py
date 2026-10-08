import os
import warnings
import tensorflow as tf
import numpy as np
import pickle as pkl
from pathlib import Path
from .nest_dynamics import (
    integration_coefficients,
    active_update,
    spike_reset,
    time_steps,
)
from bmtk.simulator.dpointnet.io_tools import io
from bmtk.simulator.dpointnet.custom_ops import (
    build_csr_connectivity,
    cuda_op_status,
    fused_dense_state,
    fused_cuda_available,
    fused_glif_state_available,
    fused_nest_state,
    fused_nest_state_available,
    fused_nest_event_vjp_available,
    fused_nest_state_history_available,
    fused_nest_type_indexed_coefficients_available,
    fused_voltage_penalty_available,
    fused_voltage_penalty_mean_step,
    fused_spike_shift,
    fused_spike_currents,
    glif_state_op_status,
    pack_nest_state_coefficients,
    pack_type_indexed_nest_state_coefficients,
    reorder_csr_values,
    restore_csr_values,
)
from bmtk.simulator.dpointnet.custom_ops.csr_spike_ops import (
    _resolve_packed_sm120_model_option,
    _validate_packed_sm120_option,
    fused_recurrent_accumulation_available,
    fused_recurrent_weight_carry,
)

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator


def make_pre_ind_table(indices, n_source_neurons):
    # Validate inputs
    if n_source_neurons <= 0:
        raise ValueError(
            f"The number of source neurons = {n_source_neurons}, must be greater than 0."
        )
    indices_np = np.asarray(indices)  # convert to np array just-in-casse
    if indices_np.ndim != 2 or indices_np.shape[1] < 2:
        raise ValueError(
            f"`indices` must have shape [n_synapses, >=2], got {indices_np.shape}."
        )
    pre_ids = indices_np[:, 1].astype(np.int64, copy=False)
    invalid = (pre_ids < 0) | (pre_ids >= n_source_neurons)
    if np.any(invalid):
        bad = int(pre_ids[np.flatnonzero(invalid)[0]])
        raise ValueError(
            f"Presynaptic index {bad} is out of bounds for `n_source_neurons={n_source_neurons}`."
        )

    if pre_ids.size == 0:
        order_np = np.empty((0,), dtype=np.int32)
        row_splits_np = np.zeros((n_source_neurons + 1,), dtype=np.int64)
    elif HAS_NUMBA:
        order_np, row_splits_np = _build_csr_order_numba(pre_ids, n_source_neurons)
    else:
        # Safe deterministic fallback if numba is unavailable.
        order_np = np.argsort(pre_ids, kind="stable")
        counts_np = np.bincount(pre_ids[order_np], minlength=n_source_neurons)
        row_splits_np = np.empty((n_source_neurons + 1,), dtype=np.int64)
        row_splits_np[0] = 0
        np.cumsum(counts_np, dtype=np.int64, out=row_splits_np[1:])

    if order_np.size <= np.iinfo(np.int32).max:
        order_tf = tf.convert_to_tensor(order_np, dtype=tf.int32)
    else:
        order_tf = tf.convert_to_tensor(order_np, dtype=tf.int64)
    row_splits_tf = tf.convert_to_tensor(row_splits_np, dtype=tf.int32)

    return tf.RaggedTensor.from_row_splits(order_tf, row_splits_tf, validate=False)


# Define a custom gradient for the spike function.
# Diverse functions can be used to define the gradient.
# Here we provide variations depending on the gradient type.
def gauss_pseudo(v_scaled, sigma, amplitude):
    dtype = v_scaled.dtype
    sigma = tf.cast(sigma, dtype)
    amplitude = tf.cast(amplitude, dtype)
    return tf.math.exp(-tf.square(v_scaled) / tf.square(sigma)) * amplitude


def pseudo_derivative(v_scaled, dampening_factor):
    dtype = v_scaled.dtype
    dampening_factor = tf.cast(dampening_factor, dtype)
    one = tf.cast(1.0, dtype)
    zero = tf.cast(0.0, dtype)
    return dampening_factor * tf.maximum(one - tf.abs(v_scaled), zero)


@tf.custom_gradient
def spike_gauss(v_scaled, sigma, amplitude):
    dtype = v_scaled.dtype
    z_ = tf.greater(v_scaled, tf.cast(0.0, dtype))
    z_ = tf.cast(z_, dtype)

    def grad(dy):
        # de_dz = tf.cast(dy, dtype)
        de_dz = dy
        dz_dv_scaled = gauss_pseudo(v_scaled, sigma, amplitude)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None, None]

    return tf.identity(z_, name="spike_gauss"), grad


@tf.custom_gradient
def spike_function(v_scaled, dampening_factor):
    dtype = v_scaled.dtype
    z_ = tf.greater(v_scaled, tf.cast(0.0, dtype))
    z_ = tf.cast(z_, dtype)

    def grad(dy):
        # de_dz = tf.cast(dy, dtype)
        de_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        de_dv_scaled = de_dz * dz_dv_scaled

        return [de_dv_scaled, None]

    return tf.identity(z_, name="spike_function"), grad


@tf.custom_gradient
def calculate_synaptic_currents(
    rec_z_buf,
    synapse_indices,
    weight_values,
    weight_values_compute,
    dense_shape,
    synaptic_basis_weights,
    syn_ids,
    pre_ind_table,
    dampening_factor,
):
    """
    Optimized synaptic current calculation with memory-efficient gradients for RSNN timestep iteration.

    Mathematical formulation:
    - Forward: I[b,post,r] = sum_pre(spike[b,pre] * W[post,pre] * basis[type[post,pre], r])
        - Grad w.r.t spike: dL/dspike[b,pre] = dampening_factor
            * sum_post sum_r(dL/dI[b,post,r] * W[post,pre] * basis[type,r])
    - Grad w.r.t W: dL/dW[post,pre] = sum_b sum_r(dL/dI[b,post,r] * spike[b,pre] * basis[type,r])

    Key optimizations:
    1. Use int32 for GPU-bound operations (segment_ids, arithmetic, unsorted_segment_sum)
    2. Use int64 for CPU-bound operations (RaggedTensor gather) and SparseTensor indices (required)
    3. Recompute cheap operations in backward pass to minimize saved activations (VRAM)
    4. Use einsum for fused multiply-sum operations on weight gradients
    """
    # Get batch size and network dimensions
    # tf.shape returns int32, dense_shape is int64 (for SparseTensor compatibility)
    batch_size = tf.cast(tf.shape(rec_z_buf)[0], dtype=tf.int64)
    n_post_neurons = dense_shape[0]  # int64
    compute_dtype = rec_z_buf.dtype  # e.g., float16 for mixed precision

    # Find non-zero spike indices
    # tf.where() returns int64 by default (GPU optimized for this operation)
    non_zero_indices = tf.where(rec_z_buf > 0)  # [num_spikes, 2], int64
    batch_indices = non_zero_indices[:, 0]  # int64
    pre_neuron_indices = non_zero_indices[
        :, 1
    ]  # keep int64 for RaggedTensor gather (CPU-optimized)

    # Retrieve connections and weights for active presynaptic neurons
    # This uses pre_ind_table (RaggedTensor), which benefits from int64 on CPU
    new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = (
        get_new_inds_table(
            synapse_indices,
            weight_values_compute,
            syn_ids,
            pre_neuron_indices,
            pre_ind_table,
        )
    )
    # new_syn_ids = tf.cast(new_syn_ids, dtype=tf.int32)  # int32 to reduce VRAM since its reused in backward pass

    # Returns: new_indices (int64), new_syn_ids (int64), post_in_degree (int32), all_synaptic_inds (int32)

    # Build segment IDs for unsorted_segment_sum using int32 for the GPU kernel.
    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)  # int64
    post_neuron_indices = new_indices[
        :, 0
    ]  # keep as int64 for compatibility, will be cast in segment_ids calculation
    num_segments = batch_size * n_post_neurons  # int64
    segment_ids = (
        batch_indices_per_connection * n_post_neurons + post_neuron_indices
    )  # int64
    segment_ids = tf.cast(segment_ids, dtype=tf.int32)
    num_segments = tf.cast(num_segments, dtype=tf.int32)

    # Compute weighted basis factors for active synapses
    # Note: basis_factors will be recomputed in grad() to save VRAM (cheap gather operation)
    # basis_factors = tf.cast(
    #     tf.gather(synaptic_basis_weights, new_syn_ids, axis=0), compute_dtype
    # )  # [n_active, n_basis], compute_dtype
    basis_factors = tf.gather(
        synaptic_basis_weights, new_syn_ids, axis=0
    )  # [n_active, n_basis]
    new_syn_ids = tf.cast(
        new_syn_ids, dtype=tf.int32
    )  # int32 to reduce VRAM since its reused in backward pass
    # Mixed precision: cast gathered subsets to compute_dtype (float16) AFTER gather.
    # This avoids creating a temporary float16 copy of the full weight table (~23M elements)
    # and only casts the ~1-2M active connections — 10-20x less cast work and no extra VRAM.
    new_weights = tf.cast(new_weights, compute_dtype)
    weighted_basis = (
        new_weights[:, tf.newaxis] * basis_factors
    )  # [n_active, n_basis], compute_dtype
    # weighted_basis = new_weights[:, tf.newaxis] * basis_factors  # [n_active, n_basis], compute_dtype --- IGNORE ---

    # if per_type_training:
    #     per_type_weights = tf.expand_dims(tf.gather(recurrent_per_type_weight_values,
    #                                                 tf.gather(connection_type_ids, all_synaptic_inds)), axis=1)
    #     new_weights = new_weights * per_type_weights

    # Sum to get currents per (batch, neuron, receptor) — float32
    i_rec_flat = tf.math.unsorted_segment_sum(weighted_basis, segment_ids, num_segments)

    # if i_rec_flat.dtype != compute_dtype:
    #     i_rec_flat = tf.cast(i_rec_flat, dtype=compute_dtype)

    def grad(dy):
        # Gradient computation - recompute cheap operations to save VRAM since this runs every timestep
        # dy arrives in compute_dtype (float16) since i_rec_flat is in compute_dtype

        # =================================================================
        # GRADIENT W.R.T. INPUT SPIKES (rec_z_buf)
        # =================================================================
        # dL/dspike[b,pre] = sum_post,r( dy[b,post,r] * W[post,pre] * basis[type,r] )
        n_syn_basis = tf.shape(synaptic_basis_weights, out_type=tf.int32)[1]
        n_post_neurons_g = dense_shape[0]  # int64
        n_pre_neurons_g = dense_shape[1]  # int64
        # weight_values_g = weight_values_compute

        def per_receptor_accum(r_id, acc):
            dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons_g])
            recurrent_weights_factors = tf.gather(
                synaptic_basis_weights[:, r_id], syn_ids, axis=0
            )
            weights_syn_receptors = weight_values_compute * recurrent_weights_factors
            sparse_w_rec = tf.sparse.SparseTensor(
                synapse_indices, weights_syn_receptors, dense_shape
            )
            de_dv_rid = tf.sparse.sparse_dense_matmul(
                dy_r, sparse_w_rec, adjoint_a=False
            )

            return r_id + 1, acc + de_dv_rid

        init_acc = tf.zeros((batch_size, n_pre_neurons_g), dtype=compute_dtype)
        _, de_dv = tf.while_loop(
            lambda r_id, _: r_id < n_syn_basis,
            per_receptor_accum,
            [tf.constant(0, dtype=tf.int32), init_acc],
            parallel_iterations=1,
        )
        de_dv *= tf.cast(dampening_factor, de_dv.dtype)

        # # Extract the gradient for this receptor type (shape: [batch_size, n_post_neurons])
        # r_id = 0
        # dy_r = tf.reshape(dy[:, r_id], [batch_size, n_post_neurons])
        # # dy_r = dy_reshaped[:, :, r_id]
        # # Compute gradient w.r.t rec_z_buf for this receptor type
        # recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
        # weights_syn_receptors = weight_values * recurrent_weights_factors
        # sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
        # de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)
        # de_dv = tf.cast(de_dv_rid, dtype=rec_z_buf.dtype)

        # =================================================================
        # GRADIENT W.R.T. WEIGHTS
        # =================================================================
        # For active synapses: dL/dW[syn] = sum_b,r( dy[b,post[syn],r] * basis[type[syn],r] )

        # Gather gradients for active connections (reuses segment_ids from forward pass)
        dnew_weights = tf.gather(dy, segment_ids)  # [n_active, n_basis]
        # Recompute basis_factors (cheap gather, saves VRAM by not storing across all timesteps)
        basis_factors_grad = tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
        # Compute weight gradients in master-weight dtype (typically float32) to avoid
        # quantizing dW in mixed precision before optimizer update.
        # basis_factors_grad = tf.cast(basis_factors_grad, weight_values.dtype)
        de_dweight_values_connection = tf.einsum(
            "cr,cr->c", dnew_weights, basis_factors_grad
        )
        # de_dweight_values_connection = tf.reduce_sum(dnew_weights * basis_factors_grad, axis=1)  # [n_active], compute_dtype
        # Accumulate to original synapse positions
        # Instead of tensor_scatter_nd_add, use unsorted_segment_sum:
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dweight_values_connection,
            segment_ids=all_synaptic_inds,
            num_segments=tf.shape(weight_values)[0],
        )
        de_dweight_values = tf.cast(de_dweight_values, dtype=weight_values.dtype)

        # de_dweight_values_connection = tf.cast(de_dweight_values_connection, dtype=weight_values.dtype)
        # de_dweight_values = _coalesce_indexed_slices_1d(
        #     values=de_dweight_values_connection,
        #     indices=all_synaptic_inds,
        #     dense_size=tf.shape(weight_values, out_type=tf.int32)[0],
        #     out_index_dtype=tf.int32
        # )

        return [
            de_dv,  # Gradient w.r.t rec_z_buf (compute_dtype)
            None,  # synapse_indices (constant)
            de_dweight_values,  # Gradient w.r.t weight_values (float32, matches master weights)
            None,  # weight_values_compute (non-trainable shadow copy)
            None,  # dense_shape[0] (constant)
            None,  # dense_shape[1] (constant)
            None,  # synaptic_basis_weights (constant)
            None,  # syn_ids (constant)
            None,  # pre_ind_table (constant)
            None,  # dampening_factor (constant)
        ]

    return i_rec_flat, grad


def get_new_inds_table(indices, weights, syn_ids, non_zero_cols, pre_ind_table):
    """Optimized function that prepares new sparse indices tensor."""
    # Gather the rows corresponding to the non_zero_cols
    selected_rows = tf.gather(pre_ind_table, non_zero_cols)
    # Flatten the selected rows to get all_inds
    all_synapse_inds = selected_rows.flat_values
    # Get the number of postsynaptic connections per active presynaptic neuron.
    # Keep as int64 — tf.repeat and downstream arithmetic use int64 segment_ids
    # for optimal GPU kernel performance (unsorted_segment_sum, gather).
    post_in_degree = selected_rows.row_lengths()  # int64
    # Gather active rows from indices/weights/syn_ids.
    # Note: gathering full active rows avoids materializing indices[:, 0]
    # for the entire synapse table, which can trigger OOM on large networks.
    new_indices = tf.gather(indices, all_synapse_inds)
    new_weights = tf.gather(weights, all_synapse_inds)
    new_syn_ids = tf.gather(syn_ids, all_synapse_inds)

    return new_indices, new_weights, new_syn_ids, post_in_degree, all_synapse_inds


@tf.custom_gradient
def calculate_input_currents(
    x_t,
    input_indices,
    input_weight_values,
    input_weight_values_compute,
    input_dense_shape,
    synaptic_basis_weights,
    input_syn_ids,
    pre_input_ind_table,
):
    """Memory-efficient input (LGN/background) synaptic current, mirroring
    ``calculate_synaptic_currents`` but for an external spike input ``x_t``.

    Forward: I[b,post,r] = sum_pre( x_t[b,pre] * W_in[post,pre] * basis[type,r] ), where
    ``x_t[b,pre]`` may be a multi-spike count (not just 0/1). Forward reads the compute-dtype
    shadow ``input_weight_values_compute``; the custom backward returns the gradient w.r.t. the
    trainable master ``input_weight_values`` and recomputes the cheap basis gather instead of
    retaining per-timestep activations. The plain-autodiff version stacked those activations
    across the whole sequence, which dominated background-input memory at high firing rates.
    No gradient flows to ``x_t`` (external input).
    """
    batch_size = tf.cast(tf.shape(x_t)[0], dtype=tf.int64)
    # input_dense_shape may be a Python/int32 tuple; force int64 for segment-id arithmetic
    n_post_neurons = tf.cast(input_dense_shape[0], dtype=tf.int64)
    compute_dtype = synaptic_basis_weights.dtype

    if x_t.dtype == tf.bool:
        non_zero_indices = tf.where(x_t)
    else:
        non_zero_indices = tf.where(x_t > 0)
    batch_indices = non_zero_indices[:, 0]
    pre_neuron_indices = non_zero_indices[:, 1]

    new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = (
        get_new_inds_table(
            input_indices,
            input_weight_values_compute,
            input_syn_ids,
            pre_neuron_indices,
            pre_input_ind_table,
        )
    )

    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
    post_neuron_indices = new_indices[:, 0]
    segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
    segment_ids = tf.cast(segment_ids, dtype=tf.int32)
    num_segments = tf.cast(batch_size * n_post_neurons, dtype=tf.int32)

    basis_factors = tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)
    new_syn_ids = tf.cast(
        new_syn_ids, dtype=tf.int32
    )  # int32 to reduce VRAM (reused in backward)
    # Per-connection presynaptic spike count (supports multi-spike inputs).
    n_pre_spikes = tf.cast(tf.gather_nd(x_t, non_zero_indices), compute_dtype)
    n_pre_per_connection = tf.repeat(n_pre_spikes, post_in_degree)
    new_weights = tf.cast(new_weights, compute_dtype)
    new_weights_final = (new_weights * n_pre_per_connection)[
        :, tf.newaxis
    ] * basis_factors
    i_in_flat = tf.math.unsorted_segment_sum(
        new_weights_final, segment_ids, num_segments
    )

    def grad(dy):
        # dL/dW_in[syn] = sum_b,r( dy[b,post[syn],r] * n_pre_spikes[syn] * basis[type[syn],r] )
        dnew_weights = tf.gather(dy, segment_ids)  # [n_active, n_basis]
        basis_factors_grad = tf.gather(
            synaptic_basis_weights, new_syn_ids, axis=0
        )  # recomputed (saves VRAM)
        de_dweight_connection = (
            tf.einsum("cr,cr->c", dnew_weights, basis_factors_grad)
            * n_pre_per_connection
        )
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dweight_connection,
            segment_ids=all_synaptic_inds,
            num_segments=tf.shape(input_weight_values)[0],
        )
        de_dweight_values = tf.cast(de_dweight_values, dtype=input_weight_values.dtype)
        return [
            None,  # x_t (external input)
            None,  # input_indices (constant)
            de_dweight_values,  # input_weight_values (master, matches trainable dtype)
            None,  # input_weight_values_compute (non-trainable shadow)
            None,  # input_dense_shape[0] (constant)
            None,  # input_dense_shape[1] (constant)
            None,  # synaptic_basis_weights (constant)
            None,  # input_syn_ids (constant)
            None,  # pre_input_ind_table (constant)
        ]

    return i_in_flat, grad


@njit(cache=True)
def _build_csr_order_numba(pre_ids, n_source_neurons):
    """O(n_syn + n_source) stable bucket build for CSR order/row_splits."""
    n_syn = pre_ids.shape[0]
    counts = np.zeros(n_source_neurons, dtype=np.int64)
    for i in range(n_syn):
        counts[pre_ids[i]] += 1

    row_splits = np.empty(n_source_neurons + 1, dtype=np.int64)
    row_splits[0] = 0
    for i in range(n_source_neurons):
        row_splits[i + 1] = row_splits[i] + counts[i]

    write_ptr = np.empty(n_source_neurons, dtype=np.int64)
    for i in range(n_source_neurons):
        write_ptr[i] = row_splits[i]

    order = np.empty(n_syn, dtype=np.int64)
    for syn_idx in range(n_syn):
        p = pre_ids[syn_idx]
        pos = write_ptr[p]
        order[pos] = syn_idx
        write_ptr[p] = pos + 1

    return order, row_splits


class SignedConstraint(tf.keras.constraints.Constraint):
    def __init__(self, positive):
        # self._positive = positive
        self.condition = positive

    def __call__(self, w):
        # condition = tf.greater(self._positive, 0)  # yields bool
        sign_corrected_w = tf.where(self.condition, tf.nn.relu(w), -tf.nn.relu(-w))
        return sign_corrected_w


def straight_through_dampen(x, dampening):
    dampening = tf.cast(dampening, x.dtype)
    tf_zero = tf.cast(0.0, x.dtype)
    tf_one = tf.cast(1.0, x.dtype)
    dampening = tf.clip_by_value(dampening, tf_zero, tf_one)
    return x * (tf_one - dampening) + tf.stop_gradient(x * dampening)


@tf.custom_gradient
def _primal_with_vjp(primal, differentiable):
    """Use an exact stored primal with the FP32 replay expression's derivative."""

    def grad(dy):
        return None, dy

    return tf.identity(primal), grad


@tf.custom_gradient
def quantized_fp32(value):
    """FP16-valued replay tensor with an FP32 identity storage Jacobian."""
    rounded = tf.cast(tf.cast(value, tf.float16), tf.float32)

    def grad(dy):
        return dy

    return rounded, grad


@tf.custom_gradient
def _range_voltage_penalty_mean(voltage, inverse_n_neurons):
    centered = tf.cast(voltage, tf.float32) - 0.5
    outside = tf.nn.relu(tf.abs(centered) - 0.5)
    mean_penalty = tf.reduce_sum(tf.square(outside), axis=1) * inverse_n_neurons

    def grad(dy):
        factor = 2.0 * outside * tf.sign(centered)
        reduction = tf.cast(dy, tf.float32)[:, None] * inverse_n_neurons
        return tf.cast(reduction * factor, voltage.dtype), None

    return mean_penalty, grad


@tf.custom_gradient
def _threshold_voltage_penalty_mean(voltage, inverse_n_neurons):
    offset = tf.cast(voltage, tf.float32) - 1.0
    mean_penalty = tf.reduce_sum(tf.square(offset), axis=1) * inverse_n_neurons

    def grad(dy):
        factor = 2.0 * offset
        reduction = tf.cast(dy, tf.float32)[:, None] * inverse_n_neurons
        return tf.cast(reduction * factor, voltage.dtype), None

    return mean_penalty, grad


def voltage_penalty_mean_step(voltage, n_neurons, penalty_mode="range"):
    """Return one timestep's neuron-mean voltage penalty per sample."""
    inverse_n_neurons = tf.math.reciprocal(tf.cast(n_neurons, tf.float32))
    if penalty_mode == "range":
        return _range_voltage_penalty_mean(voltage, inverse_n_neurons)
    if penalty_mode == "threshold":
        return _threshold_voltage_penalty_mean(voltage, inverse_n_neurons)
    raise ValueError("penalty_mode must be 'range' or 'threshold'.")


def _validate_fused_cuda_option(value):
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
    raise ValueError('use_fused_cuda must be true, false, or "auto".')


def _fused_cuda_dtype_error(compute_dtype, variable_dtype):
    compute_dtype = tf.as_dtype(compute_dtype)
    variable_dtype = tf.as_dtype(variable_dtype)
    if compute_dtype in (tf.float16, tf.float32) and variable_dtype == tf.float32:
        return None
    return (
        "the fused operator requires float16 or float32 computation and "
        "float32 variables; got "
        f"compute_dtype={compute_dtype.name}, "
        f"variable_dtype={variable_dtype.name}"
    )


def _validate_pair_projection_option(value):
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
    raise ValueError('use_pair_projection must be true, false, or "auto".')


def _validate_fixed4_forward_option(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is True or value is False:
        return value
    raise ValueError("use_fixed4_input_forward must be true or false.")


def _validate_fused_current_accumulation_option(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is True or value is False:
        return value
    raise ValueError("use_fused_current_accumulation must be true or false.")


def _validate_direct_csr_gradient_option(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if value is True or value is False:
        return value
    raise ValueError("use_direct_csr_recurrent_gradient must be true or false.")


def _resolve_pair_projection(option, fused_cuda, batch_size, n_syn_basis):
    option = _validate_pair_projection_option(option)
    incompatibilities = []
    if not fused_cuda:
        incompatibilities.append("fused CUDA currents are disabled or unavailable")
    if batch_size is None or batch_size < 1:
        incompatibilities.append("batch_size must be positive and known")
    if n_syn_basis < 1:
        incompatibilities.append("the synaptic basis must have positive width")
    if option is True and incompatibilities:
        raise ValueError(
            "use_pair_projection=True is incompatible with this model: "
            + "; ".join(incompatibilities)
        )
    return not incompatibilities and (
        option is True or (option == "auto" and batch_size == 32 and n_syn_basis == 4)
    )


def _resolve_fused_state(
    option,
    n_syn_basis,
    pseudo_gauss,
    dynamics_mode="legacy",
    compute_dtype=tf.float32,
    variable_dtype=tf.float32,
):
    option = _validate_fused_cuda_option(option)
    incompatibilities = []
    dtype_error = _fused_cuda_dtype_error(compute_dtype, variable_dtype)
    if dtype_error is not None:
        incompatibilities.append(dtype_error)
    available = (
        fused_nest_state_available()
        if dynamics_mode == "nest"
        else fused_glif_state_available()
    )
    if not available:
        incompatibilities.append(
            f"{dynamics_mode} state operator unavailable; rebuild CUDA operators: "
            + glif_state_op_status()
        )
    if n_syn_basis != 4:
        incompatibilities.append(f"the synaptic basis has {n_syn_basis} columns, not 4")
    if option is True and incompatibilities:
        raise ValueError(
            "use_fused_state=True is incompatible with this model: "
            + "; ".join(incompatibilities)
        )
    return option is not False and not incompatibilities


def _resolve_fused_state_history(option, dynamics_mode, fused_state):
    if not isinstance(option, bool):
        raise ValueError("use_fused_state_history must be a boolean.")
    if option and (
        dynamics_mode != "nest" or not fused_state or not fused_nest_state_history_available()
    ):
        raise ValueError(
            "use_fused_state_history=True requires NEST, enabled fused state, "
            "and rebuilt state/history CUDA operators."
        )
    return option


class GLIF3Cell(tf.keras.layers.Layer):
    noise_state_index = 6

    def _tracked_weight(self, initial_value, name, trainable, dtype, constraint=None):
        initial_value = np.asarray(initial_value)
        kwargs = {
            "name": name,
            "shape": initial_value.shape,
            "dtype": dtype,
            "initializer": tf.keras.initializers.Constant(initial_value),
            "trainable": trainable,
            "constraint": constraint,
        }
        try:
            return self.add_weight(autocast=False, **kwargs)
        except TypeError as exc:
            if "autocast" not in str(exc):
                raise
            return self.add_weight(experimental_autocast=False, **kwargs)

    def _untracked_variable(self, variable):
        return variable

    def __init__(
        self,
        glif_network,
        inputs,
        dt=1.0,
        gauss_std=0.5,
        dampening_factor=0.3,
        recurrent_dampening_factor=0.5,
        voltage_gradient_dampening=0.5,
        # input_weight_scale=1.0,
        recurrent_weight_scale=1.0,
        lr_scale=1.0,
        max_delay=5,
        # bkg_firing_rate=250,
        pseudo_gauss=False,
        train_recurrent=True,
        train_recurrent_per_type=True,
        # train_input=False,
        # train_noise=True,
        noise_seed=0,
        hard_reset=None,
        tau_basis=None,
        synaptic_basis_weights=None,
        use_fused_cuda=False,
        use_fused_state=False,
        use_fused_nest_event_vjp=False,
        use_pair_projection="auto",
        use_packed_sm120_backward="auto",
        use_packed_sm120_external_backward="auto",
        use_fixed4_input_forward=False,
        use_fused_current_accumulation=False,
        use_direct_csr_recurrent_gradient=False,
        use_small_batch_recurrent_backward=False,
        use_active_row_forward=False,
        use_forward_run_aggregation=False,
        use_device_active_queue_forward=False,
        use_uniform_input_delay_projection=False,
        batch_size=None,
        track_voltage_penalty=False,
        voltage_penalty_mode="range",
        return_voltage_sequences=True,
        dynamics_mode="legacy",
        state_precision="compute",
        detach_reset=True,
        detach_asc_reset=True,
        temporal_gradient_precision="compute",
        temporal_checkpoint_chunk_size=25,
        temporal_pack_spike_checkpoints=False,
        current_replay_mode=None,
        use_fused_recurrent_accumulation=False,
        use_javier_recurrent_vjp=False,
        use_prepacked_nest_coefficients=False,
        use_type_indexed_nest_coefficients=False,
        require_type_indexed_nest_coefficients=False,
        use_static_type_indexed_nest_dispatch=False,
        use_direct_state_rnn_loop=False,
        use_unity_lr_scale_fastpath=False,
        use_native_voltage_penalty=False,
        online_voltage_losses=None,
        use_fused_state_history=False,
        use_device_poisson=False,
        # current_input=False,
    ):
        super().__init__()
        self._online_voltage_losses = list(online_voltage_losses or ())

        if state_precision not in ("compute", "selective"):
            raise ValueError("state_precision must be 'compute' or 'selective'.")
        if state_precision == "selective" and (
            tf.as_dtype(self.compute_dtype) != tf.float16
            or tf.as_dtype(self.variable_dtype) != tf.float32
        ):
            raise ValueError("state_precision='selective' requires mixed_float16.")
        for name, value in (
            ("detach_reset", detach_reset),
            ("detach_asc_reset", detach_asc_reset),
            ("use_fused_nest_event_vjp", use_fused_nest_event_vjp),
            ("use_javier_recurrent_vjp", use_javier_recurrent_vjp),
            ("use_device_poisson", use_device_poisson),
            ("use_prepacked_nest_coefficients", use_prepacked_nest_coefficients),
            ("use_type_indexed_nest_coefficients", use_type_indexed_nest_coefficients),
            (
                "require_type_indexed_nest_coefficients",
                require_type_indexed_nest_coefficients,
            ),
            (
                "use_static_type_indexed_nest_dispatch",
                use_static_type_indexed_nest_dispatch,
            ),
            ("use_direct_state_rnn_loop", use_direct_state_rnn_loop),
            ("use_unity_lr_scale_fastpath", use_unity_lr_scale_fastpath),
            ("use_native_voltage_penalty", use_native_voltage_penalty),
        ):
            if value is not True and value is not False:
                raise ValueError(f"{name} must be true or false.")
        if not np.isfinite(gauss_std) or gauss_std <= 0:
            raise ValueError("gauss_std must be finite and positive.")
        if not np.isfinite(dampening_factor) or dampening_factor < 0:
            raise ValueError("dampening_factor must be finite and nonnegative.")
        self.state_precision = state_precision
        self._use_device_poisson = use_device_poisson
        self.use_fused_nest_event_vjp = use_fused_nest_event_vjp
        self._use_type_indexed_nest_coefficients = use_type_indexed_nest_coefficients
        self._require_type_indexed_nest_coefficients = require_type_indexed_nest_coefficients
        self._use_static_type_indexed_nest_dispatch = use_static_type_indexed_nest_dispatch
        self._use_prepacked_nest_coefficients = (
            use_prepacked_nest_coefficients or use_type_indexed_nest_coefficients
        )
        self._use_direct_state_rnn_loop = use_direct_state_rnn_loop
        self._use_unity_lr_scale_fastpath = use_unity_lr_scale_fastpath
        self._use_native_voltage_penalty = use_native_voltage_penalty
        self._rollout_nest_coefficients = None
        if temporal_gradient_precision not in ("compute", "float32"):
            raise ValueError(
                "temporal_gradient_precision must be 'compute' or 'float32'."
            )
        if temporal_gradient_precision == "float32":
            if state_precision != "selective":
                raise ValueError(
                    "FP32 temporal gradients require state_precision='selective'."
                )
            if (
                use_packed_sm120_backward is True
                or use_packed_sm120_external_backward is True
                or use_small_batch_recurrent_backward is True
            ):
                raise ValueError(
                    "FP32 temporal gradients require generic FP32 backward kernels; "
                    "explicit FP16 packed/small-batch backward requests are incompatible."
                )
        if (
            isinstance(temporal_checkpoint_chunk_size, bool)
            or not isinstance(temporal_checkpoint_chunk_size, (int, np.integer))
            or temporal_checkpoint_chunk_size < 1
        ):
            raise ValueError(
                "temporal_checkpoint_chunk_size must be a positive integer."
            )
        self.temporal_gradient_precision = temporal_gradient_precision
        if (
            use_fused_recurrent_accumulation is not True
            and use_fused_recurrent_accumulation is not False
        ):
            raise ValueError("use_fused_recurrent_accumulation must be true or false.")
        self.use_fused_recurrent_accumulation = use_fused_recurrent_accumulation
        self.use_javier_recurrent_vjp = use_javier_recurrent_vjp
        if use_javier_recurrent_vjp and not use_fused_recurrent_accumulation:
            raise ValueError(
                "use_javier_recurrent_vjp requires use_fused_recurrent_accumulation."
            )
        if use_fused_recurrent_accumulation:
            supported_temporal_route = (
                temporal_gradient_precision == "float32" or use_direct_state_rnn_loop
            )
            if (
                not supported_temporal_route
                or not use_direct_csr_recurrent_gradient
                or batch_size not in range(1, 33)
                or not train_recurrent
                or train_recurrent_per_type
            ):
                raise ValueError(
                    "Fused recurrent accumulation requires FP32 temporal carry or "
                    "the direct state RNN loop, batch1..32, direct CSR and trainable "
                    "per-edge recurrent weights."
                )
        if current_replay_mode not in (None, "record", "recompute"):
            raise ValueError("current_replay_mode must be None, 'record' or 'recompute'.")
        self.current_replay_mode_requested = current_replay_mode
        if current_replay_mode is None:
            current_replay_mode = (
                "record" if temporal_gradient_precision == "float32" else None
            )
        elif temporal_gradient_precision != "float32":
            raise ValueError(
                f"current_replay_mode={current_replay_mode!r} requires "
                "temporal_gradient_precision='float32'; omit it or use None "
                "for the ordinary compute route, where current replay is inactive."
            )
        if current_replay_mode == "recompute":
            warnings.warn(
                "current_replay_mode='recompute' is approximate: atomic current "
                "projection is not bitwise deterministic. Use 'record' to replay "
                "the original forward currents.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.current_replay_mode = current_replay_mode
        self.temporal_checkpoint_chunk_size = int(temporal_checkpoint_chunk_size)
        if (
            temporal_pack_spike_checkpoints is not True
            and temporal_pack_spike_checkpoints is not False
        ):
            raise ValueError("temporal_pack_spike_checkpoints must be true or false.")
        self.temporal_pack_spike_checkpoints = bool(temporal_pack_spike_checkpoints)
        self._temporal_continuous_inputs = any(
            item.get("options", {}).get("input_type", item["input_type"]) == "current"
            and item["n_inputs"] > 0
            for item in inputs.values()
        )
        self.detach_reset = detach_reset
        self.detach_asc_reset = detach_asc_reset
        self.state_dtype = (
            tf.float32
            if state_precision == "selective"
            else tf.as_dtype(self.compute_dtype)
        )
        # Keras must not narrow the heterogeneous recurrent state on cell entry.
        self._autocast = False
        self.autocast = False
        if dynamics_mode not in ("nest", "legacy"):
            raise ValueError("dynamics_mode must be 'nest' or 'legacy'")
        self.dynamics_mode = dynamics_mode
        self.spike_time_offset_steps = int(dynamics_mode == "nest")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if hard_reset is None:
            hard_reset = dynamics_mode == "nest"
        if (
            use_small_batch_recurrent_backward is not True
            and use_small_batch_recurrent_backward is not False
        ):
            raise ValueError(
                "use_small_batch_recurrent_backward must be true or false."
            )
        self._use_small_batch_recurrent_backward = use_small_batch_recurrent_backward
        if use_active_row_forward is not True and use_active_row_forward is not False:
            raise ValueError("use_active_row_forward must be true or false.")
        self._use_active_row_forward = use_active_row_forward
        if (
            use_forward_run_aggregation is not True
            and use_forward_run_aggregation is not False
        ):
            raise ValueError("use_forward_run_aggregation must be true or false.")
        self._use_forward_run_aggregation = use_forward_run_aggregation
        if (
            use_device_active_queue_forward is not True
            and use_device_active_queue_forward is not False
        ):
            raise ValueError("use_device_active_queue_forward must be true or false.")
        self._use_device_active_queue_forward = use_device_active_queue_forward
        if (
            use_uniform_input_delay_projection is not True
            and use_uniform_input_delay_projection is not False
        ):
            raise ValueError(
                "use_uniform_input_delay_projection must be true or false."
            )
        self._use_uniform_input_delay_projection = (
            use_uniform_input_delay_projection
        )
        self.__seq_idx = 0
        use_pair_projection = _validate_pair_projection_option(use_pair_projection)
        self._use_packed_sm120_backward = _validate_packed_sm120_option(
            use_packed_sm120_backward
        )
        self._use_fixed4_input_forward = _validate_fixed4_forward_option(
            use_fixed4_input_forward
        )
        self._use_fused_current_accumulation = (
            _validate_fused_current_accumulation_option(use_fused_current_accumulation)
        )
        self._use_direct_csr_recurrent_gradient = _validate_direct_csr_gradient_option(
            use_direct_csr_recurrent_gradient
        )
        use_fused_cuda = _validate_fused_cuda_option(use_fused_cuda)
        fused_dtype_error = _fused_cuda_dtype_error(
            self.compute_dtype, self.variable_dtype
        )
        fused_available = fused_cuda_available() and fused_dtype_error is None
        if use_fused_cuda is True and not fused_available:
            unavailable_reason = fused_dtype_error or cuda_op_status()
            raise RuntimeError(
                "use_fused_cuda=True but the fused DPointNet CUDA operator "
                f"is unavailable: {unavailable_reason}"
            )
        self._use_fused_cuda = fused_available and (
            use_fused_cuda is True or use_fused_cuda == "auto"
        )
        if self._use_fused_current_accumulation and not self._use_fused_cuda:
            raise ValueError(
                "use_fused_current_accumulation=True requires fused CUDA currents."
            )
        if self._use_direct_csr_recurrent_gradient and not self._use_fused_cuda:
            raise ValueError(
                "use_direct_csr_recurrent_gradient=True requires fused CUDA currents."
            )
        if use_small_batch_recurrent_backward and (
            not self._use_fused_cuda or batch_size not in range(1, 9)
        ):
            raise ValueError(
                "Small-batch recurrent backward requires fused CUDA and batch size 1..8."
            )
        if use_active_row_forward and (
            not self._use_fused_cuda or batch_size not in range(1, 33)
        ):
            raise ValueError(
                "Active-row forward requires fused CUDA and batch size 1..32."
            )
        if use_forward_run_aggregation and not self._use_fused_cuda:
            raise ValueError(
                "use_forward_run_aggregation=True requires fused CUDA currents."
            )
        if use_device_active_queue_forward and (
            not self._use_fused_cuda or batch_size not in range(1, 33)
        ):
            raise ValueError(
                "use_device_active_queue_forward=True requires fused CUDA and batch size 1..32."
            )
        if use_fused_cuda == "auto" and not self._use_fused_cuda:
            unavailable_reason = fused_dtype_error or cuda_op_status()
            io.log_warning(
                "DPointNet fused CUDA currents are unavailable; using the "
                f"TensorFlow fallback. Status: {unavailable_reason}"
            )
        elif self._use_fused_cuda:
            io.log_info(f"DPointNet fused CUDA currents enabled ({cuda_op_status()}).")

        _node_params = dict(glif_network["node_params"])
        if state_precision == "selective":
            _node_params = {
                name: np.asarray(value, dtype=np.float64)
                for name, value in _node_params.items()
            }

        voltage_scale = _node_params["V_th"] - _node_params["E_L"]

        ## TODO: Don't update the dictionary, just make adjusted_asc_amps a variable
        _node_params["asc_amps"] = _node_params["asc_amps"] / voltage_scale[..., None]

        self._node_type_ids = np.array(glif_network["node_type_ids"])
        (
            self._nest_unique_type_ids,
            self._nest_type_first_indices_np,
            self._nest_type_indices_np,
        ) = np.unique(self._node_type_ids, return_index=True, return_inverse=True)
        self._nest_type_indices = tf.constant(self._nest_type_indices_np, dtype=tf.int64)
        self._nest_type_first_indices = tf.constant(
            self._nest_type_first_indices_np, dtype=tf.int32
        )
        self._nest_type_count = int(len(self._nest_unique_type_ids))
        self._dt = tf.constant(dt, self.state_dtype)
        self._recurrent_dampening = tf.constant(
            recurrent_dampening_factor, self.compute_dtype
        )
        self._dampening_factor = tf.constant(dampening_factor, self.state_dtype)
        self._voltage_gradient_dampening = tf.constant(
            voltage_gradient_dampening, self.state_dtype
        )
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = tf.constant(lr_scale, dtype=self.compute_dtype)
        self._lr_scale_is_unity = bool(np.asarray(lr_scale).item() == 1.0)
        if self._use_unity_lr_scale_fastpath and not self._lr_scale_is_unity:
            raise ValueError("use_unity_lr_scale_fastpath=True requires lr_scale=1.0.")

        self._noise_seed_base = tf.constant(int(noise_seed), dtype=tf.int64)
        self.noise_seed = tf.Variable(
            int(noise_seed), trainable=False, dtype=tf.int64, name="noise_seed"
        )
        self.noise_stream = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._hard_reset = hard_reset
        if voltage_penalty_mode not in ("range", "threshold"):
            raise ValueError("voltage_penalty_mode must be 'range' or 'threshold'.")
        self._track_voltage_penalty = bool(track_voltage_penalty)
        self._voltage_penalty_mode = voltage_penalty_mode
        self._return_voltage_sequences = bool(return_voltage_sequences)
        if not self._return_voltage_sequences and not self._track_voltage_penalty:
            raise ValueError(
                "Disabling voltage sequences requires track_voltage_penalty=True."
            )
        # self._current_input = current_input
        self._n_neurons = int(glif_network["n_nodes"])
        self._gauss_std = tf.constant(gauss_std, self.state_dtype)

        # Determine the membrane time decay constant
        tau = _node_params["C_m"] / _node_params["g"]
        membrane_decay = np.exp(-dt / tau)
        current_factor = (
            -np.expm1(-dt / tau) if state_precision == "selective"
            else 1 - membrane_decay
        ) / _node_params["g"]

        # Determine the dynamic parameters for each synaptic basis function.
        if tau_basis is None:
            raise ValueError(
                f"Invalid tau_basis = {tau_basis}, please pass in a numpy array or a path to a npy file."
            )
        if isinstance(tau_basis, (str, Path)):
            tau_path = tau_basis
            tau_basis = np.load(tau_path)
        elif isinstance(tau_basis, (list, tuple)):
            tau_basis = np.array(tau_basis)
        if state_precision == "selective":
            tau_basis = np.asarray(tau_basis, dtype=np.float64)

        self._n_syn_basis = tau_basis.size
        syn_decay_np = np.exp(-dt / tau_basis)
        syn_decay_np = np.tile(syn_decay_np, self._n_neurons)
        self.syn_decay = tf.constant(
            syn_decay_np[None, :], dtype=self.state_dtype
        )  # expand the dimension for processing different receptor types
        psc_initial_np = np.e / tau_basis
        psc_initial_np = np.tile(psc_initial_np, self._n_neurons)
        self.psc_initial = tf.constant(
            psc_initial_np[None, :], dtype=self.state_dtype
        )  # expand the dimension for processing different receptor types

        network_max_delay = np.max(glif_network["synapses"]["delays"], initial=dt)
        if max_delay is None or max_delay <= 0:
            self.max_delay = int(np.round(network_max_delay))
        else:
            self.max_delay = int(np.round(np.min([network_max_delay, max_delay])))
        self._delay_limit_ms = self.max_delay
        if dynamics_mode == "nest":
            self._delay_limit_ms = float(
                network_max_delay
                if max_delay is None or max_delay <= 0
                else min(network_max_delay, max_delay)
            )
            self.max_delay = max(1, int(time_steps(self._delay_limit_ms, dt)))

        # Gather the neuron parameters for every neuron
        t_ref_per_neuron = _node_params["t_ref"][self._node_type_ids]
        t_ref_steps = np.ceil(t_ref_per_neuron / dt).astype(np.int16)
        if dynamics_mode == "nest":
            t_ref_steps = time_steps(t_ref_per_neuron, dt).astype(np.int16)
        t_ref_steps = np.maximum(t_ref_steps, 1)
        max_ref_steps = int(np.max(t_ref_steps))
        if max_ref_steps > 127:
            self._refractory_state_dtype = tf.int16
            print(
                f"Warning: max refractory period is {max_ref_steps} steps, which exceeds int8 capacity. Using int16 for refractory state."
            )
        else:
            self._refractory_state_dtype = tf.int8
        self.t_ref_steps = tf.constant(t_ref_steps, dtype=self._refractory_state_dtype)

        self.asc_amps = tf.Variable(
            tf.cast(
                tf.gather(_node_params["asc_amps"], indices=self._node_type_ids),
                self.state_dtype,
            ),
            trainable=False,
            dtype=self.state_dtype,
        )

        # def _gather(prop):
        #     return tf.gather(prop, self._node_type_ids)

        # def _f(_v, trainable=False, dtype=None):
        #     if dtype is None:
        #         dtype = self.compute_dtype
        #     return tf.Variable(
        #         tf.cast(_gather(_v), dtype),
        #         trainable=trainable,
        #         dtype=dtype,
        #     )

        # self.asc_amps_2 = _f(_node_params['asc_amps'], trainable=False)

        if state_precision == "selective":
            rates = _node_params["k"][self._node_type_ids]
            if not np.isfinite(rates).all() or (rates <= 0).any():
                raise ValueError("Adaptation rates must be finite and positive.")
            self.asc_decay = tf.constant(np.exp(-dt * rates), self.state_dtype)
        else:
            _k = tf.cast(_node_params["k"], self.compute_dtype)
            _k = tf.gather(_k, self._node_type_ids)
            _k = tf.math.log(_k / (1.0 - _k))
            _k = tf.Variable(tf.cast(_k, self.compute_dtype), trainable=False)
            self.asc_decay = tf.exp(-self._dt * tf.nn.sigmoid(_k.read_value()))
        self.v_th = tf.constant(1.0, dtype=self.state_dtype)
        self.v_reset = tf.constant(0.0, dtype=self.state_dtype)

        # Cast before constructing the Variable: tf.Variable does not auto-cast a
        # float32 initial value to a float16 compute_dtype under mixed precision.
        self.decay = tf.Variable(
            tf.cast(tf.gather(membrane_decay, self._node_type_ids), self.state_dtype),
            trainable=False,
            dtype=self.state_dtype,
        )
        self.current_factor = tf.Variable(
            tf.cast(tf.gather(current_factor, self._node_type_ids), self.state_dtype),
            trainable=False,
            dtype=self.state_dtype,
        )

        if dynamics_mode == "nest":
            _, _, psc_voltage, rise_voltage = integration_coefficients(
                dt, _node_params["C_m"], _node_params["g"], tau_basis
            )
            self.psc_voltage = tf.constant(
                psc_voltage[self._node_type_ids], dtype=self.state_dtype
            )
            self.rise_voltage = tf.constant(
                rise_voltage[self._node_type_ids], dtype=self.state_dtype
            )
            rates = np.asarray(_node_params["k"])[self._node_type_ids]
            if not np.isfinite(rates).all() or (rates <= 0).any():
                raise ValueError(
                    "NEST adaptation decay rates must be finite and positive"
                )
            self.asc_decay = tf.constant(np.exp(-dt * rates), dtype=self.state_dtype)
            self.asc_mean = tf.constant(
                -np.expm1(-dt * rates) / (dt * rates), dtype=self.state_dtype
            )
            self.asc_refractory_decay = tf.constant(
                np.asarray(_node_params.get("asc_r", np.ones_like(_node_params["k"])))[
                    self._node_type_ids
                ]
                * np.exp(-rates * t_ref_per_neuron[:, None]),
                dtype=self.state_dtype,
            )
            reset = (
                np.asarray(_node_params["V_reset"]) - _node_params["E_L"]
            ) / voltage_scale
            self.v_reset = tf.constant(
                reset[self._node_type_ids], dtype=self.state_dtype
            )
            initial = (
                np.asarray(_node_params.get("V_m", _node_params["E_L"]))
                - _node_params["E_L"]
            ) / voltage_scale
            self.initial_voltage = tf.constant(
                initial[self._node_type_ids], dtype=self.state_dtype
            )
            initial_asc = (
                np.asarray(
                    _node_params.get("asc_init", np.zeros_like(_node_params["k"]))
                )
                / voltage_scale[:, None]
            )
            self.initial_asc = tf.constant(
                initial_asc[self._node_type_ids].reshape(-1), dtype=self.state_dtype
            )

        ## TODO: This shouldn't be stored in a separate pickle.
        # path = os.path.join(glif_network["data_dir"], 'tf_data', 'syn_id_to_syn_weights_dict.pkl')
        # path = 'GLIF_network/tf_data/syn_id_to_syn_weights_dict.pkl'
        # with open(path, "rb") as f:
        #     syn_id_to_syn_weights_dict = pkl.load(f)
        # synaptic_basis_weights_ = np.array(list(syn_id_to_syn_weights_dict.values()))
        # synaptic_basis_weights_ = tf.constant(synaptic_basis_weights_, dtype=self.compute_dtype)

        if synaptic_basis_weights is None:
            _synaptic_basis_weights = glif_network["synapses"]["dynamics_params"][
                "basis_weights"
            ]
        elif isinstance(synaptic_basis_weights, str):
            with open(synaptic_basis_weights, "rb") as f:
                syn_id_to_syn_weights_dict = pkl.load(f)
            _synaptic_basis_weights = np.array(
                list(syn_id_to_syn_weights_dict.values())
            )
        elif isinstance(synaptic_basis_weights, (list, np.ndarray)):
            _synaptic_basis_weights = np.array(synaptic_basis_weights)
        else:
            raise NotImplementedError()

        self.synaptic_basis_weights = tf.constant(
            _synaptic_basis_weights, dtype=self.compute_dtype
        )
        self._use_fused_state = _resolve_fused_state(
            use_fused_state,
            self._n_syn_basis,
            self._pseudo_gauss,
            dynamics_mode,
            compute_dtype=self.compute_dtype,
            variable_dtype=self.variable_dtype,
        )
        self._use_fused_state_history = _resolve_fused_state_history(
            use_fused_state_history, dynamics_mode, self._use_fused_state
        )
        if use_fused_nest_event_vjp and (
            dynamics_mode != "nest" or not self._use_fused_state
            or not fused_nest_event_vjp_available()
        ):
            raise ValueError(
                "use_fused_nest_event_vjp=True requires NEST dynamics, fused state "
                "and rebuilt compatible NEST event-VJP CUDA operators."
            )
        if use_native_voltage_penalty and not fused_voltage_penalty_available():
            raise ValueError(
                "use_native_voltage_penalty=True requires rebuilt compatible "
                "voltage-penalty CUDA operators."
            )
        if use_prepacked_nest_coefficients and (
            dynamics_mode != "nest" or not self._use_fused_state
        ):
            raise ValueError(
                "use_prepacked_nest_coefficients=True requires NEST dynamics "
                "and fused state."
            )
        if use_type_indexed_nest_coefficients and (
            dynamics_mode != "nest"
            or not self._use_fused_state
            or not fused_nest_type_indexed_coefficients_available()
        ):
            raise ValueError(
                "use_type_indexed_nest_coefficients=True requires NEST dynamics, "
                "fused state and rebuilt compatible NEST type-indexed CUDA operators."
            )
        if require_type_indexed_nest_coefficients and not use_type_indexed_nest_coefficients:
            raise ValueError(
                "require_type_indexed_nest_coefficients=True requires "
                "use_type_indexed_nest_coefficients=True."
            )
        if self._use_fused_state:
            io.log_info(
                f"DPointNet fused {dynamics_mode} state transition enabled "
                f"(use_fused_state={use_fused_state!r})."
            )
        self._use_pair_projection = _resolve_pair_projection(
            use_pair_projection,
            self._use_fused_cuda,
            batch_size,
            self._n_syn_basis,
        )
        self._use_packed_sm120_external_backward = _resolve_packed_sm120_model_option(
            use_packed_sm120_external_backward,
            self._use_fused_cuda,
            self.compute_dtype,
            batch_size,
            self._n_syn_basis,
        )
        if self._use_pair_projection:
            io.log_info(
                "DPointNet pair-projected recurrent backward enabled "
                f"(use_pair_projection={use_pair_projection!r})."
            )
        elif self._use_fused_cuda:
            io.log_info(
                "DPointNet general recurrent backward selected "
                f"(use_pair_projection={use_pair_projection!r}, "
                f"batch_size={batch_size}, n_syn_basis={self._n_syn_basis})."
            )

        # TODO: Allow option to not have recurrent connectivity (eg. in case only want to train feedforward network)
        ### Network recurrent connectivity ###
        indices = np.array(
            glif_network["synapses"]["indices"]
        )  # NOTE: These are the tf indices, not SONATA, and in the form [trg, src]
        weights = np.array(glif_network["synapses"]["weights"])
        dense_shape = np.array(glif_network["synapses"]["dense_shape"])
        syn_ids = np.array(glif_network["synapses"]["syn_ids"])
        delays = np.array(glif_network["synapses"]["delays"])
        weights = (
            weights / voltage_scale[self._node_type_ids[indices[:, 0]]]
        )  # Scale down the recurrent weights
        # Per-edge factor to invert the load-time scaling on export (recover physical syn_weight):
        # physical = internal * voltage_scale[target] * lr_scale / recurrent_weight_scale.
        self._recurrent_export_factor = (
            voltage_scale[self._node_type_ids[indices[:, 0]]]
            * lr_scale
            / recurrent_weight_scale
        ).astype(np.float32)
        delays = np.round(np.clip(delays, dt, self._delay_limit_ms) / dt).astype(
            np.int32
        )  # Use the maximum delay to clip the synaptic delays
        if dynamics_mode == "nest":
            delays = time_steps(
                np.clip(glif_network["synapses"]["delays"], dt, self._delay_limit_ms),
                dt,
            )
        indices[:, 1] = indices[:, 1] + self._n_neurons * (
            delays - 1
        )  # Introduce the delays in the presynaptic neuron indices

        # the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
        self.recurrent_dense_shape = dense_shape[0], self.max_delay * dense_shape[1]

        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(
            indices, dtype=tf.int64, trainable=False
        )  # dtype necessary for sparse dense matmul
        if self._use_fused_cuda:
            self.recurrent_fused_connectivity = build_csr_connectivity(
                indices,
                syn_ids,
                self.recurrent_dense_shape[1],
                self._n_neurons,
                self.synaptic_basis_weights.shape[0],
                build_compact_pairs=self._use_pair_projection,
                sort_by_target=self._use_forward_run_aggregation,
            )
            self.pre_ind_table = None
        else:
            self.recurrent_fused_connectivity = None
            self.pre_ind_table = make_pre_ind_table(
                indices, n_source_neurons=self.recurrent_dense_shape[1]
            )

        recurrent_weight_positive = tf.constant(weights >= 0, dtype=tf.bool)
        if use_fused_recurrent_accumulation and (
            not self._use_fused_cuda
            or self._n_syn_basis != 4
            or self.recurrent_fused_connectivity["index_dtype"] != "uint32"
            or not self.recurrent_fused_connectivity["n_pairs"]
            or not fused_recurrent_accumulation_available()
        ):
            raise ValueError(
                "Fused recurrent accumulation requires rebuilt SM86+ CUDA operators "
                "and four-basis uint32 compact-pair connectivity."
            )

        if train_recurrent:
            if train_recurrent_per_type:
                individual_training = False
                per_type_training = True
            else:
                individual_training = True
                per_type_training = False
        else:
            individual_training = False
            per_type_training = False

        if self._use_direct_csr_recurrent_gradient and not individual_training:
            raise ValueError(
                "use_direct_csr_recurrent_gradient=True requires individually "
                "trainable recurrent edge weights."
            )

        self.recurrent_weight_values = self._tracked_weight(
            weights * recurrent_weight_scale / lr_scale,
            name="sparse_recurrent_weights",
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=individual_training,
            dtype=self.variable_dtype,
        )  # shape = (n_synapses,)

        if self.variable_dtype != self.compute_dtype or individual_training:
            # Keep a non-trainable compute-lane shadow. Two reasons:
            #  (1) mixed precision: avoids casting the full weight table every timestep;
            #  (2) the recurrent @tf.custom_gradient reads these weights inside the RNN
            #      while_loop. If that read is the *trainable* variable (esp. a MirroredVariable
            #      under MirroredStrategy), TF requires the grad to take a `variables` kwarg and
            #      otherwise errors. Reading a non-trainable shadow avoids that. The shadow is
            #      kept in sync via refresh_recurrent_weight_shadow() after each optimizer step.
            recurrent_weight_values_compute = tf.Variable(
                tf.cast(self.recurrent_weight_values, self.compute_dtype),
                name="sparse_recurrent_weights_compute",
                trainable=False,
                dtype=self.compute_dtype,
            )
            # Keep untracked so older checkpoints remain loadable with assert_consumed().
            self.recurrent_weight_values_compute = self._untracked_variable(
                recurrent_weight_values_compute
            )
        else:
            self.recurrent_weight_values_compute = self.recurrent_weight_values
        if self._use_fused_cuda:
            recurrent_csr_weights = tf.Variable(
                reorder_csr_values(
                    self.recurrent_weight_values_compute,
                    self.recurrent_fused_connectivity,
                ),
                name="sparse_recurrent_weights_csr_compute",
                trainable=False,
                dtype=self.compute_dtype,
            )
            self.recurrent_csr_weight_values_compute = self._untracked_variable(
                recurrent_csr_weights
            )
        else:
            self.recurrent_csr_weight_values_compute = None

        self.syn_ids = tf.constant(
            syn_ids, dtype=tf.int64
        )  # this needs to be int64 for efficiency
        # self.recurrent_weights_factors = tf.gather(self.synaptic_basis_weights, self.syn_ids, axis=0) # TensorShape([23525415, 5])
        io.log_debug(
            f" > Added recurrent synapses: indices = {len(indices)}; trainable = {individual_training}"
        )
        del indices, weights, dense_shape, delays, syn_ids, recurrent_weight_positive

        ### Inputs
        # TODO: Inputs needs to be a list since order is extremely important
        self.inputs_idx = np.zeros(len(inputs) + 1, dtype=int)
        self.inputs = {}
        self._input_history_size = 0

        for idx, (input_name, input_network) in enumerate(inputs.items()):
            # TODO: Use a named tuple instead of dict
            input_props = {}
            n_input_nodes = input_network["n_inputs"]
            input_dense_shape = (self._n_neurons, n_input_nodes)

            input_props["input_dim"] = n_input_nodes
            input_indices = np.array(input_network["indices"], dtype=np.int64)
            input_props["history_start"] = self._input_history_size
            input_props["uniform_delay_steps"] = None
            if dynamics_mode == "nest":
                input_delays = np.asarray(input_network["delays"], dtype=float)
                if not np.isfinite(input_delays).all() or (input_delays < dt).any():
                    raise ValueError(
                        "NEST input delays must be finite and at least one timestep"
                    )
                input_steps = time_steps(input_delays, dt)
                history_steps = int(input_steps.max(initial=1)) - 1
                uniform_delay = (
                    input_steps.size > 0
                    and np.all(input_steps == input_steps.flat[0])
                )
                if self._use_uniform_input_delay_projection and uniform_delay:
                    input_props["uniform_delay_steps"] = int(input_steps.flat[0])
                    input_dense_shape = (self._n_neurons, n_input_nodes)
                else:
                    input_indices[:, 1] += n_input_nodes * (input_steps - 1)
                    input_dense_shape = (
                        self._n_neurons,
                        n_input_nodes * (history_steps + 1),
                    )
                self._input_history_size += n_input_nodes * history_steps
            input_props["history_stop"] = self._input_history_size
            input_props["input_dense_shape"] = input_dense_shape
            input_weights = np.array(input_network["weights"])
            input_syn_ids = np.array(input_network["syn_ids"])
            input_weights = (
                input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0]]]
            )
            input_props["input_indices"] = tf.Variable(
                input_indices, trainable=False, dtype=tf.int64
            )

            input_options = input_network.get("options", {})
            input_type = input_options.get("input_type", input_network["input_type"])
            # Per-edge factor to invert the load-time scaling on export (recover physical syn_weight):
            # physical = internal * voltage_scale[target] * lr_scale / weight_scale.
            input_props["export_factor"] = (
                voltage_scale[self._node_type_ids[input_indices[:, 0]]]
                * lr_scale
                / input_options.get("weight_scale", 1.0)
            ).astype(np.float32)
            input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.bool)
            input_trainable = input_options.get("trainable", False)

            input_props["input_weight_values"] = self._tracked_weight(
                input_weights * input_options.get("weight_scale", 1.0) / lr_scale,
                name=f"{input_name}_input_weights",
                constraint=SignedConstraint(input_weight_positive),
                trainable=input_trainable,
                dtype=self.variable_dtype,
            )
            # Non-trainable compute-dtype shadow (mirrors the recurrent weights): the input-current
            # @tf.custom_gradient reads this in the forward (avoids casting in fp16 and avoids reading
            # a trainable MirroredVariable inside the RNN while_loop), while the gradient targets the
            # master above. Kept in sync via refresh_recurrent_weight_shadow() after each step.
            if self.variable_dtype != self.compute_dtype or input_trainable:
                _input_weight_compute = tf.Variable(
                    tf.cast(input_props["input_weight_values"], self.compute_dtype),
                    name=f"{input_name}_input_weights_compute",
                    trainable=False,
                    dtype=self.compute_dtype,
                )
                input_props["input_weight_values_compute"] = self._untracked_variable(
                    _input_weight_compute
                )
            else:
                input_props["input_weight_values_compute"] = input_props[
                    "input_weight_values"
                ]
            input_props["input_syn_ids"] = tf.constant(
                input_syn_ids, dtype=tf.int64
            )  # for efficiency this needs to be in int64
            input_props["input_type"] = input_type
            if input_type == "spikes":
                end_indx = self.inputs_idx[idx] + n_input_nodes
            elif input_type in ("poisson_spikes_internal", "noisy_current"):
                firing_rate = input_options.get("firing_rate", 250.0)
                input_props["spike_prob"] = tf.constant(
                    firing_rate * dt / 1000.0, dtype=self.compute_dtype
                )
                input_props["spike_prob_value"] = float(
                    np.asarray(firing_rate * dt / 1000.0, dtype=np.float64)
                )
                if self._use_device_poisson:
                    from scipy.stats import poisson

                    rate = float(
                        np.asarray(
                            input_props["spike_prob_value"],
                            dtype=tf.as_dtype(self.compute_dtype).as_numpy_dtype,
                        )
                    )
                    if not np.isfinite(rate) or rate < 0:
                        raise ValueError(
                            "Device Poisson rate must be finite and nonnegative."
                        )
                    cutoff = int(poisson.ppf(np.nextafter(1.0, 0.0), rate))
                    levels = poisson.cdf(np.arange(cutoff + 1), rate)
                    levels[-1] = 1.0
                    input_props["poisson_cdf"] = tf.constant(levels, tf.float64)
                end_indx = self.inputs_idx[idx]
            elif input_type == "current":
                end_indx = self.inputs_idx[idx] + n_input_nodes
            else:
                raise ValueError(f"Unknown input type {input_type}")
            if input_type in ("spikes", "poisson_spikes_internal", "noisy_current"):
                if self._use_fused_cuda:
                    input_props["fused_connectivity"] = build_csr_connectivity(
                        input_indices,
                        input_syn_ids,
                        input_dense_shape[1],
                        self._n_neurons,
                        self.synaptic_basis_weights.shape[0],
                        build_compact_pairs=(
                            self._use_packed_sm120_external_backward is not False
                            and input_trainable
                        ),
                        build_fixed4_incoming=self._use_fixed4_input_forward,
                        sort_by_target=self._use_forward_run_aggregation,
                    )
                    input_props["use_fixed4_forward"] = input_props[
                        "fused_connectivity"
                    ]["fixed4_incoming"]
                    input_props["use_packed_sm120_backward"] = (
                        self._use_packed_sm120_external_backward
                        if input_trainable
                        else False
                    )
                    input_props["pre_input_ind_table"] = None
                    input_csr_weights = tf.Variable(
                        reorder_csr_values(
                            input_props["input_weight_values_compute"],
                            input_props["fused_connectivity"],
                        ),
                        name=f"{input_name}_input_weights_csr_compute",
                        trainable=False,
                        dtype=self.compute_dtype,
                    )
                    input_props["csr_weight_values_compute"] = self._untracked_variable(
                        input_csr_weights
                    )
                else:
                    input_props["fused_connectivity"] = None
                    input_props["csr_weight_values_compute"] = None
                    input_props["pre_input_ind_table"] = make_pre_ind_table(
                        input_indices,
                        n_source_neurons=input_dense_shape[1],
                    )

            io.log_debug(
                f' > Added "{input_name}" input synapses: indices = {len(input_indices)}, trainble = {input_trainable}'
            )
            self.inputs_idx[idx + 1] = end_indx

            # if input_name == 'bkg':
            #     input_props['spike_prob'] = tf.constant(bkg_firing_rate * 0.001, dtype=self.compute_dtype)

            self.inputs[input_name] = input_props

        """
        lgn_inputs = inputs[0]
        # self.input_dim = inputs['lgn']['n_inputs']
        self.input_dim = lgn_inputs['n_inputs']
        self.lgn_input_dense_shape = (self._n_neurons, self.input_dim)
        input_indices = np.array(lgn_inputs['indices'])
        input_weights = np.array(lgn_inputs['weights'])
        input_syn_ids = np.array(lgn_inputs['syn_ids'])
        input_weights = input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0]]]

        self.input_indices = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

        input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.bool)
        self.input_weight_values = tf.Variable(
            input_weights * input_weight_scale / lr_scale,
            name="sparse_input_weights",
            constraint=SignedConstraint(input_weight_positive),
            trainable=train_input,
            dtype=self.variable_dtype
        )
        self.input_syn_ids = tf.constant(input_syn_ids, dtype=tf.int64) # for efficiency this needs to be in int64
        if not self._current_input:
            self.pre_input_ind_table = make_pre_ind_table(input_indices, n_source_neurons=self.lgn_input_dense_shape[1])

        print(f"    > # LGN input synapses {len(input_indices)}")
        del input_indices, input_weights, input_syn_ids, input_weight_positive #, input_delays

        ### BKG input connectivity ###
        bkg_inputs = inputs[1]
        self.bkg_spike_prob = tf.constant(bkg_firing_rate * 0.001, dtype=self.compute_dtype)
        self.bkg_input_dense_shape = (self._n_neurons, bkg_inputs["n_inputs"],)
        bkg_input_indices = np.array(bkg_inputs['indices'])
        bkg_input_weights = np.array(bkg_inputs['weights'])
        bkg_input_syn_ids = np.array(bkg_inputs['syn_ids'])
        # Scale down the background input weights
        bkg_input_weights = (bkg_input_weights/voltage_scale[self._node_type_ids[bkg_input_indices[:, 0]]])
        # # Introduce the delays in the postsynaptic neuron indices
        # bkg_input_delays = np.array(bkg_input['delays'])
        # bkg_input_delays = np.round(np.clip(bkg_input_delays, dt, self.max_delay)/dt).astype(np.int32)
        # bkg_input_indices[:, 1] = bkg_input_indices[:, 1] + self._n_neurons * (bkg_input_delays - 1)
        self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int64)
        # self.bkg_input_indices = tf.Variable(bkg_input_indices, trainable=False, dtype=tf.int32)
        self.pre_bkg_ind_table = make_pre_ind_table(bkg_input_indices, n_source_neurons=self.bkg_input_dense_shape[1])

        # Define Tensorflow variables
        # bkg_input_weight_positive = tf.Variable(
        #     bkg_input_weights >= 0.0, name="bkg_input_weights_sign", trainable=False)
        # bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.int8)
        bkg_input_weight_positive = tf.constant(bkg_input_weights >= 0, dtype=tf.bool)
        self.bkg_input_weights = tf.Variable(
            bkg_input_weights * input_weight_scale / lr_scale,
            name="rest_of_brain_weights",
            constraint=SignedConstraint(bkg_input_weight_positive),
            trainable=train_noise,
            dtype=self.variable_dtype
        )

        self.bkg_input_syn_ids = tf.constant(bkg_input_syn_ids, dtype=tf.int64)
        # self.bkg_input_weights_factors = tf.gather(self.synaptic_basis_weights, bkg_input_syn_ids, axis=0)

        print(f"    > # BKG input synapses {len(bkg_input_indices)}")
        del bkg_input_indices, bkg_input_weights, bkg_input_syn_ids, bkg_input_weight_positive #, bkg_input_delays
        """

    def refresh_recurrent_weight_shadow(self):
        """Sync the compute-dtype shadow of the recurrent weights with the trained master.

        Under mixed precision the forward uses ``recurrent_weight_values_compute`` (a
        non-trainable float16 copy) while the optimizer updates the float32 master
        ``recurrent_weight_values``. This must be called after each optimizer step, else the
        forward keeps using stale weights and recurrent-weight training has no effect.
        Matches the reference V1_GLIF_model.
        """
        if (
            self.recurrent_weight_values_compute is not self.recurrent_weight_values
            and self.recurrent_weight_values.trainable
        ):
            self.recurrent_weight_values_compute.assign(
                tf.cast(self.recurrent_weight_values, self.compute_dtype)
            )
        recurrent_csr_shadow = getattr(
            self, "recurrent_csr_weight_values_compute", None
        )
        if recurrent_csr_shadow is not None and self.recurrent_weight_values.trainable:
            recurrent_csr_shadow.assign(
                reorder_csr_values(
                    tf.cast(self.recurrent_weight_values, self.compute_dtype),
                    self.recurrent_fused_connectivity,
                )
            )

        # Also sync any trainable input-weight shadows (e.g. trainable background weights),
        # which the input-current custom gradient reads in its forward pass.
        for input_net in self.inputs.values():
            master = input_net.get("input_weight_values")
            shadow = input_net.get("input_weight_values_compute")
            if shadow is None or shadow is master or not master.trainable:
                continue
            shadow.assign(tf.cast(master, self.compute_dtype))
            csr_shadow = input_net.get("csr_weight_values_compute")
            if csr_shadow is not None:
                csr_shadow.assign(
                    reorder_csr_values(
                        tf.cast(master, self.compute_dtype),
                        input_net["fused_connectivity"],
                    )
                )

    def close_fused_cuda(self):
        connectivity = getattr(self, "recurrent_fused_connectivity", None)
        if connectivity is not None:
            connectivity.close()
        for input_net in self.inputs.values():
            connectivity = input_net.get("fused_connectivity")
            if connectivity is not None:
                connectivity.close()

    def calculate_i_rec_with_custom_grad(
        self, rec_z_buf, projection_values=None, recurrent_weight_carrier=None
    ):
        basis, weight = (
            (self.synaptic_basis_weights,
             self.recurrent_csr_weight_values_compute if self._use_fused_cuda
             else self.recurrent_weight_values_compute)
            if projection_values is None else projection_values
        )
        if self._use_fused_cuda:
            if recurrent_weight_carrier is not None:
                if projection_values is not None:
                    raise ValueError(
                        "Weight-carrier recurrent accumulation is not compatible "
                        "with alternate projection values."
                    )
                return fused_recurrent_weight_carry(
                    rec_z_buf,
                    recurrent_weight_carrier,
                    weight,
                    self.recurrent_fused_connectivity,
                    basis,
                    self._n_neurons,
                    self._recurrent_dampening,
                    vjp_only=False,
                    use_javier_batch32_backward=self.use_javier_recurrent_vjp,
                    use_active_row_forward=(
                        rec_z_buf.shape[0] != 32 and self._use_active_row_forward
                    ),
                    use_forward_run_aggregation=(
                        rec_z_buf.shape[0] != 32
                        and self._use_forward_run_aggregation
                        and self.recurrent_fused_connectivity.get(
                            "has_repeated_targets", False
                        )
                    ),
                    use_device_active_queue_forward=(
                        rec_z_buf.shape[0] != 32
                        and self._use_device_active_queue_forward
                    ),
                )
            return fused_spike_currents(
                rec_z_buf,
                self.recurrent_weight_values,
                weight,
                self.recurrent_fused_connectivity,
                basis,
                self._n_neurons,
                compute_spike_gradient=True,
                compute_weight_gradient=self.recurrent_weight_values.trainable,
                spike_gradient_scale=self._recurrent_dampening,
                use_packed_sm120_backward=self._use_packed_sm120_backward,
                write_csr_weight_gradient=(self._use_direct_csr_recurrent_gradient),
                use_small_batch_backward=self._use_small_batch_recurrent_backward,
                use_active_row_forward=self._use_active_row_forward,
                use_forward_run_aggregation=(
                    self._use_forward_run_aggregation
                    and self.recurrent_fused_connectivity.get(
                        "has_repeated_targets", False
                    )
                ),
                use_device_active_queue_forward=self._use_device_active_queue_forward,
            )
        return calculate_synaptic_currents(
            rec_z_buf,
            self.recurrent_indices,
            self.recurrent_weight_values,
            weight,
            self.recurrent_dense_shape,
            basis,
            self.syn_ids,
            self.pre_ind_table,
            self._recurrent_dampening,
        )

    def restore_segmented_variable_gradients(self, variables, gradients):
        master = self.recurrent_weight_values
        master_value = master.value
        transformed = []
        found_recurrent = False
        for variable, gradient in zip(variables, gradients):
            if variable is master or variable is master_value:
                gradient = restore_csr_values(
                    gradient, self.recurrent_fused_connectivity
                )
                found_recurrent = True
            transformed.append(gradient)
        if not found_recurrent:
            raise RuntimeError(
                "Direct CSR recurrent gradient could not locate its FP32 master."
            )
        return tuple(transformed)

    def calculate_input_current_from_firing_probabilities(
        self, x_t, input_net, projection_values=None
    ):
        """Input current when the input is firing PROBABILITIES (rate/'current' input) rather
        than discrete spikes. Computes I[:, post, r] = sum_pre w[post,pre]*basis[type,r]*prob[pre]
        via a per-receptor sparse-dense matmul. Ported from the reference V1_GLIF_model;
        uses the per-input dict (input_net) like calculate_input_current_from_spikes.
        """
        batch_size = tf.shape(x_t)[0]
        input_indices = input_net["input_indices"]
        basis, input_weight_values = (
            (self.synaptic_basis_weights, input_net["input_weight_values"])
            if projection_values is None else projection_values
        )
        input_syn_ids = input_net["input_syn_ids"]
        input_dense_shape = input_net["input_dense_shape"]

        i_in = tf.TensorArray(dtype=self.compute_dtype, size=self._n_syn_basis)
        for r_id in range(self._n_syn_basis):
            input_weights_factors = tf.gather(
                basis[:, r_id], input_syn_ids, axis=0
            )
            weights_syn_receptors = (
                tf.cast(input_weight_values, self.compute_dtype) * input_weights_factors
            )
            sparse_w_in = tf.sparse.SparseTensor(
                input_indices, weights_syn_receptors, input_dense_shape
            )
            i_receptor = tf.sparse.sparse_dense_matmul(
                sparse_w_in, tf.cast(x_t, self.compute_dtype), adjoint_b=True
            )
            i_in = i_in.write(r_id, i_receptor)
        i_in = i_in.stack()
        i_in = tf.transpose(i_in)  # -> [batch, n_neurons, n_syn_basis] after reshape
        i_in_flat = tf.reshape(i_in, [batch_size * self._n_neurons, self._n_syn_basis])
        return i_in_flat

    def calculate_input_current_from_spikes(
        self, x_t, input_net, initial_currents=None, projection_values=None
    ):
        basis, weight = (
            (self.synaptic_basis_weights,
             input_net["csr_weight_values_compute"] if self._use_fused_cuda
             else input_net["input_weight_values_compute"])
            if projection_values is None else projection_values
        )
        if self._use_fused_cuda:
            return fused_spike_currents(
                tf.cast(x_t, self.compute_dtype),
                input_net["input_weight_values"],
                weight,
                input_net["fused_connectivity"],
                basis,
                self._n_neurons,
                compute_spike_gradient=False,
                compute_weight_gradient=input_net["input_weight_values"].trainable,
                use_fixed4_forward=input_net.get("use_fixed4_forward", False),
                initial_currents=initial_currents,
                use_active_row_forward=self._use_active_row_forward,
                use_forward_run_aggregation=(
                    self._use_forward_run_aggregation
                    and input_net["fused_connectivity"].get(
                        "has_repeated_targets", False
                    )
                ),
                use_device_active_queue_forward=self._use_device_active_queue_forward,
                use_packed_sm120_backward=input_net.get(
                    "use_packed_sm120_backward", False
                ),
            )
        # Memory-efficient input current via the @tf.custom_gradient module function: the forward
        # reads the compute-dtype shadow and the backward recomputes the cheap basis gather (instead
        # of retaining per-timestep activations), while the gradient targets the trainable master.
        return calculate_input_currents(
            x_t,
            input_net["input_indices"],
            input_net["input_weight_values"],
            weight,
            input_net["input_dense_shape"],
            basis,
            input_net["input_syn_ids"],
            input_net["pre_input_ind_table"],
        )

    def update_psc(self, psc, psc_rise, rec_inputs):
        dtype = psc.dtype
        if self.state_precision == "selective":
            psc, psc_rise, rec_inputs = (
                tf.cast(value, self.state_dtype)
                for value in (psc, psc_rise, rec_inputs)
            )
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        return tf.cast(new_psc, dtype), tf.cast(new_psc_rise, dtype)

    def _dense_update_impl(
        self, batch_size, prev_z, v, r, asc, psc_rise, psc, rec_inputs
    ):
        # new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
        # new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        # new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
        prev_z = tf.cast(prev_z, v.dtype)
        psc = tf.cast(psc, v.dtype)

        # Calculate the ASC variables
        asc = tf.reshape(asc, (batch_size, self._n_neurons, 2))
        # new_asc = self.asc_decay * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = (
            self.asc_decay * asc
            + tf.expand_dims(
                tf.stop_gradient(prev_z) if self.detach_asc_reset else prev_z, axis=-1
            )
            * self.asc_amps
        )
        new_asc = tf.reshape(new_asc, (batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current
        input_current = tf.reshape(
            psc, (batch_size, self._n_neurons, self._n_syn_basis)
        )
        input_current = tf.reduce_sum(input_current, -1)
        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1)  # + self.gathered_g
        # Compute membrane update in variable_dtype (fp32 under mixed policy) for
        # more stable threshold crossings, then store state in compute_dtype.
        # decayed_v = self.decay * v
        # reset_current = prev_z * self.v_gap
        # new_v = decayed_v + self.current_factor * c1 + reset_current
        # new_v = self.decay * v + self.current_factor * c1 - prev_z
        # new_v = self.decay * v + self.current_factor * c1 - tf.stop_gradient(prev_z)
        # Damp only the voltage self-loop. We intentionally leave
        # current_factor * c1 untouched so recurrent/input pathways keep full credit.
        dampened_v = straight_through_dampen(v, self._voltage_gradient_dampening)
        new_v = (
            self.decay * dampened_v
            + self.current_factor * c1
            - (tf.stop_gradient(prev_z) if self.detach_reset else prev_z)
        )
        # new_v = self.decay * dampened_v + self.current_factor * c1 - prev_z
        # Update the voltage according to the LIF equation and the refractory period
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        refractory_dtype = r.dtype
        prev_spike = tf.cast(prev_z, dtype=refractory_dtype)
        t_ref_steps = tf.cast(self.t_ref_steps, dtype=refractory_dtype)
        new_r = tf.stop_gradient(
            tf.maximum(
                r + prev_spike * t_ref_steps - tf.cast(1, refractory_dtype),
                tf.cast(0, refractory_dtype),
            )
        )  # prevent gradients from flowing through the refractory state

        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            refractory_mask = tf.greater(new_r, 0)
            new_v = tf.where(refractory_mask, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # new_v_state = tf.cast(new_v, self.compute_dtype)
        return new_v, new_r, new_asc, new_psc_rise, new_psc

    def advance_noise_seed(self):
        """Select a fresh logical-rollout stream outside recomputed chunks."""
        stream_id = self.noise_stream.assign_add(tf.constant(1, dtype=tf.int64))
        self.noise_seed.assign(self._noise_seed_base + stream_id)

    def reset_voltage_penalty_state(self, state):
        if self._online_voltage_losses:
            return tuple(state[:-1]) + (tf.zeros_like(state[-1]),)
        return tuple(state)

    def calculate_noise_current(
        self, batch_size, noise_step, input_net, initial_currents=None, noise_seed=None,
        projection_values=None,
    ):
        rest_of_brain = GLIF3Cell.sample_noise_spikes(
            self, batch_size, noise_step, input_net, noise_seed=noise_seed
        )
        return self.calculate_input_current_from_spikes(
            rest_of_brain, input_net, initial_currents=initial_currents,
            **({} if projection_values is None else {"projection_values": projection_values}),
        )

    def sample_noise_spikes(self, batch_size, noise_step, input_net, noise_seed=None):
        if getattr(self, "_use_device_poisson", False):
            rollout_seed = getattr(self, "_rollout_noise_seed", None)
            base_seed = tf.cast(
                (
                    noise_seed
                    if noise_seed is not None
                    else rollout_seed if rollout_seed is not None else self.noise_seed
                ),
                tf.int32,
            )
            replica_context = tf.distribute.get_replica_context()
            replica_id = (
                0
                if replica_context is None
                else replica_context.replica_id_in_sync_group
            )
            seed = tf.stack(
                [
                    base_seed + tf.cast(replica_id, tf.int32) * 1000003,
                    tf.cast(noise_step[0], tf.int32),
                ]
            )
            shape = [
                batch_size,
                input_net.get("input_dim", input_net["input_dense_shape"][1]),
            ]
            uniform = tf.random.stateless_uniform(shape, seed=seed, dtype=tf.float64)
            counts = tf.searchsorted(
                input_net["poisson_cdf"], tf.reshape(uniform, [-1]), side="right"
            )
            return tf.reshape(counts, shape)
        # StatelessRandomPoisson only has a CPU kernel. Building every input on
        # the host keeps the sampler off the device stream: a device-resident
        # seed, rate or shape costs an in-order round trip every time step.
        with tf.device("/CPU:0"):
            step_seed = tf.cast(noise_step[0], tf.int32)
            rollout_seed = getattr(self, "_rollout_noise_seed", None)
            base_seed = tf.cast(
                noise_seed if noise_seed is not None
                else rollout_seed if rollout_seed is not None
                else self.noise_seed,
                tf.int32,
            )
            replica_context = tf.distribute.get_replica_context()
            if replica_context is None:
                replica_id = tf.constant(0, dtype=tf.int32)
            else:
                replica_id = tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
            noise_seed = tf.stack(
                [base_seed + replica_id * tf.constant(1000003, dtype=tf.int32), step_seed],
                axis=0,
            )
            static_batch = tf.get_static_value(batch_size)
            poisson_shape = tf.stack(
                [
                    (tf.constant(int(static_batch), tf.int32)
                     if static_batch is not None
                     else tf.cast(batch_size, tf.int32)),
                    tf.cast(
                        input_net.get("input_dim", input_net["input_dense_shape"][1]),
                        tf.int32,
                    ),
                ],
                axis=0,
            )
            rest_of_brain = tf.random.stateless_poisson(
                shape=poisson_shape,
                seed=noise_seed,
                lam=(
                    tf.constant(input_net["spike_prob_value"], dtype=self.compute_dtype)
                    if "spike_prob_value" in input_net
                    else input_net["spike_prob"]
                ),
                dtype=tf.int32,
            )
        return rest_of_brain

    def validate_state_precision(self, states):
        if self.state_precision == "selective" and (
            states[1].dtype != tf.float32
            or states[3].dtype != tf.float32
            or any(states[i].dtype != tf.float16 for i in (0, 4, 5))
            or (self._input_history_size and states[7].dtype != tf.float16)
        ):
            raise ValueError(
                "Selective precision requires FP32 voltage/ASC and FP16 "
                "synaptic/history initial state; explicitly convert a "
                "compute-policy state before continuing."
            )

    def _capture_adjoint_projection_values(self):
        values = [
            self.recurrent_csr_weight_values_compute
            if self._use_fused_cuda else self.recurrent_weight_values_compute
        ]
        for net in self.inputs.values():
            if net["input_type"] == "current":
                value = tf.cast(net["input_weight_values"], self.compute_dtype)
            else:
                value = (
                    net["csr_weight_values_compute"]
                    if self._use_fused_cuda else net["input_weight_values_compute"]
                )
            values.append(value)
        return tf.identity(self.synaptic_basis_weights), tuple(tf.identity(v) for v in values)

    def _prepare_adjoint_projection_context(self, saved_values=None):
        basis = tf.cast(
            self.synaptic_basis_weights if saved_values is None else saved_values[0],
            tf.float32,
        )

        def frame(index, master, shadow, csr, continuous=False):
            saved_weight = None if saved_values is None else saved_values[1][index]
            if continuous:
                weight = _primal_with_vjp(
                    tf.cast(
                        tf.cast(master, self.compute_dtype) if saved_weight is None else saved_weight,
                        tf.float32,
                    ),
                    tf.cast(master, tf.float32),
                )
                return basis, None, None, weight
            if saved_weight is not None:
                if self._use_fused_cuda:
                    csr = saved_weight
                else:
                    shadow = saved_weight
            return (
                basis,
                tf.cast(shadow, tf.float32) if not self._use_fused_cuda else None,
                tf.cast(csr, tf.float32) if self._use_fused_cuda else None,
                None,
            )

        return (
            frame(
                0,
                self.recurrent_weight_values,
                self.recurrent_weight_values_compute,
                getattr(self, "recurrent_csr_weight_values_compute", None),
            ),
            *(
                frame(
                    index + 1,
                    net["input_weight_values"],
                    net["input_weight_values_compute"],
                    net.get("csr_weight_values_compute"),
                    net["input_type"] == "current",
                )
                for index, net in enumerate(self.inputs.values())
            ),
        )

    def _adjoint_projection(
        self,
        spikes,
        input_net=None,
        continuous=False,
        context=None,
        vjp_only=False,
        recurrent_weight_carrier=None,
    ):
        """FP32 linear VJP at the quantized projection weights and spike values."""
        basis = tf.cast(self.synaptic_basis_weights, tf.float32)
        recurrent = input_net is None
        if recurrent:
            master = self.recurrent_weight_values
            shadow = self.recurrent_weight_values_compute
            indices, syn_ids, shape = (
                self.recurrent_indices,
                self.syn_ids,
                self.recurrent_dense_shape,
            )
            table = self.pre_ind_table if not self._use_fused_cuda else None
            connectivity = (
                self.recurrent_fused_connectivity if self._use_fused_cuda else None
            )
            csr = (
                self.recurrent_csr_weight_values_compute
                if self._use_fused_cuda
                else None
            )
        else:
            master = input_net["input_weight_values"]
            shadow = input_net["input_weight_values_compute"]
            indices, syn_ids, shape = (
                input_net["input_indices"],
                input_net["input_syn_ids"],
                input_net["input_dense_shape"],
            )
            table = input_net.get("pre_input_ind_table")
            connectivity = input_net.get("fused_connectivity")
            csr = input_net.get("csr_weight_values_compute")
        if context is not None:
            basis, shadow, csr, weight = context
        elif continuous:
            weight = _primal_with_vjp(
                tf.cast(tf.cast(master, self.compute_dtype), tf.float32),
                tf.cast(master, tf.float32),
            )
        if continuous:
            receptors = []
            for receptor in range(self._n_syn_basis):
                values = weight * tf.gather(basis[:, receptor], syn_ids)
                matrix = tf.sparse.SparseTensor(indices, values, shape)
                receptors.append(
                    tf.transpose(
                        tf.sparse.sparse_dense_matmul(matrix, spikes, adjoint_b=True)
                    )
                )
            return tf.reshape(tf.stack(receptors, axis=-1), (-1, self._n_syn_basis))
        if self._use_fused_cuda:
            if recurrent_weight_carrier is not None:
                if not recurrent or not vjp_only:
                    raise ValueError(
                        "The weight carrier is for recurrent VJP-only replay."
                    )
                return fused_recurrent_weight_carry(
                    spikes,
                    recurrent_weight_carrier,
                    tf.cast(csr, tf.float32),
                    connectivity,
                    basis,
                    self._n_neurons,
                    self._recurrent_dampening,
                )
            return fused_spike_currents(
                spikes,
                master,
                tf.cast(csr, tf.float32),
                connectivity,
                basis,
                self._n_neurons,
                compute_spike_gradient=recurrent,
                compute_weight_gradient=master.trainable,
                spike_gradient_scale=(
                    tf.cast(self._recurrent_dampening, tf.float32) if recurrent else 1.0
                ),
                use_packed_sm120_backward=False,
                vjp_only=vjp_only,
                write_csr_weight_gradient=recurrent
                and self._use_direct_csr_recurrent_gradient,
            )
        if recurrent:
            return calculate_synaptic_currents(
                spikes,
                indices,
                master,
                tf.cast(shadow, tf.float32),
                shape,
                basis,
                syn_ids,
                table,
                tf.cast(self._recurrent_dampening, tf.float32),
            )
        return calculate_input_currents(
            spikes,
            indices,
            master,
            tf.cast(shadow, tf.float32),
            shape,
            basis,
            syn_ids,
            table,
        )

    def _adjoint_currents(
        self,
        inputs,
        states,
        projection_context=None,
        noise_seed=None,
        vjp_only=False,
        recurrent_weight_carrier=None,
    ):
        recurrent = self._adjoint_projection(
            states[0],
            context=None if projection_context is None else projection_context[0],
            vjp_only=vjp_only,
            recurrent_weight_carrier=recurrent_weight_carrier,
        )
        if recurrent_weight_carrier is not None:
            recurrent, recurrent_weight_carrier = recurrent
        currents = [recurrent]
        history = []
        batch = tf.shape(inputs)[0]
        for index, net in enumerate(self.inputs.values()):
            internal = net["input_type"] in ("poisson_spikes_internal", "noisy_current")
            spikes = (
                tf.cast(
                    self.sample_noise_spikes(
                        batch, states[6], net, noise_seed=noise_seed
                    ),
                    tf.float32,
                )
                if internal
                else quantized_fp32(
                    inputs[:, self.inputs_idx[index] : self.inputs_idx[index + 1]]
                )
            )
            if self.dynamics_mode == "nest":
                start, stop = net["history_start"], net["history_stop"]
                if stop > start:
                    current_spikes = spikes
                    prior_history = states[7][:, start:stop]
                    updated_history = tf.concat([current_spikes, prior_history], axis=1)
                    history.append(updated_history[:, : stop - start])
                    uniform_delay_steps = net.get("uniform_delay_steps")
                    if uniform_delay_steps is None:
                        spikes = updated_history
                    elif uniform_delay_steps > 1:
                        offset = (uniform_delay_steps - 2) * net["input_dim"]
                        spikes = prior_history[:, offset : offset + net["input_dim"]]
            currents.append(
                self._adjoint_projection(
                    spikes,
                    net,
                    continuous=net["input_type"] == "current",
                    vjp_only=vjp_only,
                    context=(
                        None
                        if projection_context is None
                        else projection_context[index + 1]
                    ),
                )
            )
        values = tf.reshape(
            tf.add_n(currents), (batch, self._n_neurons * self._n_syn_basis)
        )
        scaled_values = (
            values
            if self._use_unity_lr_scale_fastpath
            else values * tf.cast(self._lr_scale, tf.float32)
        )
        result = scaled_values, history
        return (
            result + (recurrent_weight_carrier,)
            if recurrent_weight_carrier is not None
            else result
        )

    def prepare_rollout_nest_coefficients(self):
        if not (
            self._use_prepacked_nest_coefficients
            and self.dynamics_mode == "nest"
            and self._use_fused_state
        ):
            return None
        coefficients = pack_nest_state_coefficients(
            tf.constant(self._n_neurons, dtype=tf.int32),
            self.state_dtype,
            syn_decay=self.syn_decay,
            psc_initial=self.psc_initial,
            asc_decay=self.asc_decay,
            asc_amps=self.asc_amps,
            decay=self.decay,
            current_factor=self.current_factor,
            asc_mean=self.asc_mean,
            asc_refractory_decay=self.asc_refractory_decay,
            psc_voltage=self.psc_voltage,
            rise_voltage=self.rise_voltage,
            v_reset=self.v_reset,
            voltage_gradient_dampening=self._voltage_gradient_dampening,
        )
        kernel_coefficients = (
            tf.transpose(coefficients)
            if tf.as_dtype(self.state_dtype) == tf.float32
            else coefficients
        )
        if self._use_type_indexed_nest_coefficients:
            type_table_bytes = (
                self._nest_type_count
                * 28
                * np.dtype(tf.as_dtype(self.state_dtype).as_numpy_dtype).itemsize
            )
            if self._use_static_type_indexed_nest_dispatch:
                if type_table_bytes > 48 * 1024:
                    raise ValueError(
                        "Static type-indexed NEST dispatch requires the compact "
                        "coefficient table to fit in CUDA shared memory."
                    )
                type_coefficients = tf.gather(
                    coefficients, self._nest_type_first_indices
                )
                type_indices = self._nest_type_indices
                type_identity = tf.constant(True)
            else:
                type_coefficients, type_indices, type_identity = (
                    pack_type_indexed_nest_state_coefficients(
                        coefficients,
                        self._nest_type_indices,
                        self._nest_type_first_indices,
                    )
                )
                type_identity = tf.logical_and(
                    type_identity, tf.constant(type_table_bytes <= 48 * 1024)
                )
            return (
                coefficients,
                kernel_coefficients,
                type_coefficients,
                type_indices,
                type_identity,
                self._require_type_indexed_nest_coefficients,
            )
        return coefficients, kernel_coefficients

    def call(self, inputs, states):
        if self.temporal_gradient_precision == "float32":
            raise ValueError(
                "FP32 temporal gradients require TemporalAdjointRunner or ExplicitStateRNN; "
                "a direct cell tape cannot carry FP32 gradients through FP16 state."
            )
        return self._call_impl(inputs, states)

    _supports_recorded_currents = True

    def _project_step_currents(
        self,
        inputs,
        states,
        noise_seed=None,
        projection_values=None,
        recurrent_weight_carrier=None,
    ):
        """Original compute-dtype math, optionally at immutable rollout weights."""
        def frame(index):
            return (
                {} if projection_values is None else
                {"projection_values": (projection_values[0], projection_values[1][index])}
            )

        # A static batch keeps reshape shapes host constants; a device int64
        # shape forces a device-to-host copy every time step.
        batch_size = (
            int(inputs.shape[0])
            if inputs.shape[0] is not None
            else tf.cast(tf.shape(inputs)[0], dtype=tf.int64)
        )
        z_buf, noise_step = states[0], states[6]
        new_input_history = []
        i_rec = self.calculate_i_rec_with_custom_grad(
            tf.cast(z_buf, self.compute_dtype),
            recurrent_weight_carrier=recurrent_weight_carrier,
            **frame(0),
        )
        if recurrent_weight_carrier is not None:
            i_rec, recurrent_weight_carrier = i_rec

        rec_inputs = i_rec
        extern_currents = []
        for idx, input_net in enumerate(self.inputs.values()):
            if self.dynamics_mode == "legacy" and input_net["input_type"] in (
                "poisson_spikes_internal",
                "noisy_current",
            ):
                if self._use_fused_current_accumulation:
                    rec_inputs = self.calculate_noise_current(
                        batch_size,
                        noise_step,
                        input_net,
                        initial_currents=rec_inputs,
                        noise_seed=noise_seed,
                        **frame(idx + 1),
                    )
                else:
                    extern_currents.append(
                        self.calculate_noise_current(
                            batch_size, noise_step, input_net, noise_seed=noise_seed,
                            **frame(idx + 1),
                        )
                    )
                continue

            input_spikes = inputs[:, self.inputs_idx[idx] : self.inputs_idx[idx + 1]]
            if self.dynamics_mode == "nest":
                if input_net["input_type"] in (
                    "poisson_spikes_internal",
                    "noisy_current",
                ):
                    input_spikes = self.sample_noise_spikes(
                        batch_size, noise_step, input_net, noise_seed=noise_seed
                    )
                input_spikes = tf.cast(input_spikes, self.compute_dtype)
                history_start, history_stop = (
                    input_net["history_start"],
                    input_net["history_stop"],
                )
                if history_stop > history_start:
                    current_input_spikes = input_spikes
                    prior_history = tf.cast(
                        states[7][:, history_start:history_stop],
                        self.compute_dtype,
                    )
                    updated_history = tf.concat(
                        [current_input_spikes, prior_history], axis=1
                    )
                    new_input_history.append(
                        updated_history[:, : history_stop - history_start]
                    )
                    uniform_delay_steps = input_net.get("uniform_delay_steps")
                    if uniform_delay_steps is None:
                        input_spikes = updated_history
                    elif uniform_delay_steps > 1:
                        offset = (uniform_delay_steps - 2) * input_net["input_dim"]
                        input_spikes = prior_history[
                            :, offset : offset + input_net["input_dim"]
                        ]
            if input_net["input_type"] == "current":
                extern_currents.append(
                    self.calculate_input_current_from_firing_probabilities(
                        input_spikes, input_net, **frame(idx + 1)
                    )
                )
            else:
                if self._use_fused_current_accumulation:
                    rec_inputs = self.calculate_input_current_from_spikes(
                        input_spikes,
                        input_net,
                        initial_currents=rec_inputs,
                        **frame(idx + 1),
                    )
                else:
                    extern_currents.append(
                        self.calculate_input_current_from_spikes(
                            input_spikes, input_net, **frame(idx + 1)
                        )
                    )

        if extern_currents:
            rec_inputs = rec_inputs + tf.add_n(extern_currents)
        # Reshape i_rec_flat back to [batch_size, num_neurons]
        rec_inputs = tf.reshape(
            rec_inputs, [batch_size, self._n_neurons * self._n_syn_basis]
        )
        # Scale with the learning rate
        if not self._use_unity_lr_scale_fastpath:
            rec_inputs = rec_inputs * self._lr_scale
        if recurrent_weight_carrier is not None:
            return rec_inputs, new_input_history, recurrent_weight_carrier
        return rec_inputs, new_input_history

    def _replay_input_history(self, inputs, states, noise_seed=None):
        history = []
        if self.dynamics_mode != "nest":
            return history
        for idx, net in enumerate(self.inputs.values()):
            start, stop = net["history_start"], net["history_stop"]
            if stop <= start:
                continue
            spikes = (
                self.sample_noise_spikes(tf.shape(inputs)[0], states[6], net, noise_seed=noise_seed)
                if net["input_type"] in ("poisson_spikes_internal", "noisy_current")
                else inputs[:, self.inputs_idx[idx]:self.inputs_idx[idx + 1]]
            )
            spikes = tf.concat(
                [tf.cast(spikes, self.compute_dtype),
                 tf.cast(states[7][:, start:stop], self.compute_dtype)], axis=1
            )
            history.append(spikes[:, :stop - start])
        return history

    def _call_impl(
        self,
        inputs,
        states,
        adjoint_replay=False,
        projection_context=None,
        noise_seed=None,
        recorded_currents=None,
        capture_currents=False,
        projection_values=None,
        recurrent_weight_carrier=None,
    ):
        if self._online_voltage_losses and not adjoint_replay:
            self.validate_state_precision(states)
            inputs = tf.cast(inputs, self.compute_dtype)
            # Keep selective voltage/ASC and the online accumulator in FP32.
            states = tuple(
                tf.cast(value, self.state_dtype if index in (1, 3) else self.compute_dtype)
                if value.dtype.is_floating else value
                for index, value in enumerate(states[:-1])
            ) + (tf.cast(states[-1], tf.float32),)
        if self.temporal_gradient_precision == "float32" and not adjoint_replay:
            inputs = tf.stop_gradient(inputs)
            states = tuple(tf.stop_gradient(value) for value in states)
        # A static batch keeps reshape shapes host constants; a device int64
        # shape forces a device-to-host copy every time step.
        batch_size = (
            int(inputs.shape[0])
            if inputs.shape[0] is not None
            else tf.cast(tf.shape(inputs)[0], dtype=tf.int64)
        )
        expected_states = 8 if self._input_history_size else 7
        expected_states += bool(self._online_voltage_losses)
        if len(states) != expected_states:
            raise ValueError(
                f"Expected {expected_states} states including external delay history, got {len(states)}"
            )
        if not adjoint_replay:
            self.validate_state_precision(states)
        z_buf, v, r, asc, psc_rise, psc, noise_step = states[:7]
        prev_z = z_buf[:, :self._n_neurons]
        if recorded_currents is None:
            primal_weight_carrier = (
                None if adjoint_replay else recurrent_weight_carrier
            )
            projected = self._project_step_currents(
                inputs,
                states,
                noise_seed,
                **(
                    {}
                    if primal_weight_carrier is None
                    else {"recurrent_weight_carrier": primal_weight_carrier}
                ),
                **(
                    {}
                    if projection_values is None
                    else {"projection_values": projection_values}
                ),
            )
            if primal_weight_carrier is None:
                rec_inputs, new_input_history = projected
            else:
                rec_inputs, new_input_history, recurrent_weight_carrier = projected
        else:
            rec_inputs = tf.ensure_shape(
                tf.cast(recorded_currents, self.compute_dtype),
                [inputs.shape[0], self._n_neurons * self._n_syn_basis],
            )
            new_input_history = (
                [] if adjoint_replay else self._replay_input_history(inputs, states, noise_seed)
            )
        forward_currents = rec_inputs
        if self.temporal_gradient_precision == "float32" and not adjoint_replay:
            rec_inputs = tf.stop_gradient(rec_inputs)
        if adjoint_replay:
            reference = self._adjoint_currents(
                inputs,
                states,
                projection_context,
                noise_seed,
                vjp_only=True,
                recurrent_weight_carrier=recurrent_weight_carrier,
            )
            reference_currents, reference_history = reference[:2]
            if recurrent_weight_carrier is not None:
                recurrent_weight_carrier = reference[2]
            rec_inputs = _primal_with_vjp(
                tf.stop_gradient(tf.cast(rec_inputs, tf.float32)), reference_currents
            )
            new_input_history = reference_history

        if self.dynamics_mode == "nest" and self._use_fused_state:
            rollout_coefficients = (
                self._rollout_nest_coefficients
                if self._use_prepacked_nest_coefficients
                else None
            )
            if self._use_prepacked_nest_coefficients and rollout_coefficients is None:
                rollout_coefficients = self.prepare_rollout_nest_coefficients()
            coefficient_kwargs = {}
            if rollout_coefficients is not None:
                coefficient_kwargs = {
                    "packed_coefficients": rollout_coefficients[0],
                    "packed_kernel_coefficients": rollout_coefficients[1],
                }
                if len(rollout_coefficients) > 2:
                    coefficient_kwargs.update(
                        {
                            "packed_type_coefficients": rollout_coefficients[2],
                            "type_indices": rollout_coefficients[3],
                            "type_indexed_identity": rollout_coefficients[4],
                            "require_type_indexed_coefficients": rollout_coefficients[5],
                        }
                    )
            fused_result = (
                fused_nest_state(
                    v,
                    r,
                    asc,
                    psc_rise,
                    psc,
                    rec_inputs,
                    z_buf,
                    syn_decay=self.syn_decay,
                    psc_initial=self.psc_initial,
                    asc_decay=self.asc_decay,
                    asc_amps=self.asc_amps,
                    decay=self.decay,
                    current_factor=self.current_factor,
                    asc_mean=self.asc_mean,
                    asc_refractory_decay=self.asc_refractory_decay,
                    psc_voltage=self.psc_voltage,
                    rise_voltage=self.rise_voltage,
                    t_ref_steps=self.t_ref_steps,
                    dt=self._dt,
                    v_reset=self.v_reset,
                    v_th=self.v_th,
                    dampening=self._dampening_factor,
                    voltage_gradient_dampening=self._voltage_gradient_dampening,
                    hard_reset=self._hard_reset,
                    detach_reset=self.detach_reset,
                    detach_asc_reset=self.detach_asc_reset,
                    pseudo_gauss=self._pseudo_gauss,
                    gauss_std=self._gauss_std,
                    return_pre_reset_voltage=bool(self._online_voltage_losses),
                    use_fused_event_vjp=self.use_fused_nest_event_vjp,
                    fuse_history=self._use_fused_state_history,
                    **coefficient_kwargs,
                )
            )
            new_z, new_v, new_r, new_asc, new_psc_rise, new_psc, new_z_buf = fused_result[:7]
            if self._online_voltage_losses:
                floor_voltage, floor_active = fused_result[7], r <= 0
        elif self.dynamics_mode == "nest":
            new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
            new_v, new_r, adaptation, active = active_update(
                straight_through_dampen(v, self._voltage_gradient_dampening),
                r,
                tf.reshape(asc, [batch_size, self._n_neurons, 2]),
                tf.reshape(
                    tf.cast(psc, v.dtype),
                    [batch_size, self._n_neurons, self._n_syn_basis],
                ),
                tf.reshape(
                    tf.cast(psc_rise, v.dtype),
                    [batch_size, self._n_neurons, self._n_syn_basis],
                ),
                decay=self.decay,
                current_factor=self.current_factor,
                asc_decay=self.asc_decay,
                asc_mean=self.asc_mean,
                psc_voltage=self.psc_voltage,
                rise_voltage=self.rise_voltage,
                reset_voltage=self.v_reset,
                hard_reset=self._hard_reset,
            )
            new_z = (
                spike_gauss(new_v - self.v_th, self._gauss_std, self._dampening_factor)
                if self._pseudo_gauss
                else spike_function(new_v - self.v_th, self._dampening_factor)
            )
            new_z = tf.where(active, new_z, tf.zeros_like(new_z))
            if self._online_voltage_losses:
                floor_voltage, floor_active = new_v, active
            new_v, new_r, adaptation = spike_reset(
                new_v,
                new_r,
                adaptation,
                new_z,
                reset_voltage=self.v_reset,
                refractory_steps=self.t_ref_steps,
                asc_amplitudes=self.asc_amps,
                asc_refractory_decay=self.asc_refractory_decay,
                hard_reset=self._hard_reset,
                detach_reset=self.detach_reset,
                detach_asc_reset=self.detach_asc_reset,
            )
            new_asc = tf.reshape(adaptation, [batch_size, self._n_neurons * 2])
            new_z = tf.cast(new_z, z_buf.dtype)
            new_z_buf = tf.concat([new_z, z_buf[:, : -self._n_neurons]], axis=1)
        elif self._use_fused_state:
            new_v, new_r, new_asc, new_psc_rise, new_psc = fused_dense_state(
                prev_z,
                v,
                r,
                asc,
                psc_rise,
                psc,
                rec_inputs,
                syn_decay=self.syn_decay,
                psc_initial=self.psc_initial,
                asc_decay=self.asc_decay,
                asc_amps=self.asc_amps,
                decay=self.decay,
                current_factor=self.current_factor,
                t_ref_steps=self.t_ref_steps,
                dt=self._dt,
                v_reset=self.v_reset,
                voltage_gradient_dampening=self._voltage_gradient_dampening,
                hard_reset=self._hard_reset,
                detach_reset=self.detach_reset,
                detach_asc_reset=self.detach_asc_reset,
            )
            new_z, new_z_buf = fused_spike_shift(
                new_v - self.v_th,
                new_r > 0,
                z_buf,
                self._dampening_factor,
                pseudo_gauss=self._pseudo_gauss,
                gauss_std=self._gauss_std,
            )
        else:
            new_v, new_r, new_asc, new_psc_rise, new_psc = self._dense_update_impl(
                batch_size, prev_z, v, r, asc, psc_rise, psc, rec_inputs
            )
            v_sc = new_v - self.v_th
            if self._pseudo_gauss:
                new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
            else:
                new_z = spike_function(v_sc, self._dampening_factor)
            refractory_active = tf.greater(new_r, 0)
            new_z = tf.where(refractory_active, tf.zeros_like(new_z), new_z)
            new_z = tf.cast(new_z, z_buf.dtype)
            new_z_buf = tf.concat([new_z, z_buf[:, : -self._n_neurons]], axis=1)

        if adjoint_replay:
            new_psc_rise = quantized_fp32(new_psc_rise)
            new_psc = quantized_fp32(new_psc)
        if self._online_voltage_losses and self.dynamics_mode == "legacy":
            floor_voltage, floor_active = new_v, new_r <= 0
        if self._return_voltage_sequences:
            outputs = tf.concat([tf.cast(new_z, new_v.dtype), new_v], axis=-1)
        else:
            voltage_penalty = (
                fused_voltage_penalty_mean_step(new_v, self._voltage_penalty_mode)
                if self._use_native_voltage_penalty
                else voltage_penalty_mean_step(
                    new_v, self._n_neurons, self._voltage_penalty_mode
                )
            )
            outputs = (
                (new_z, voltage_penalty)
                if self.dynamics_mode == "nest" or self.state_precision == "selective" or self._online_voltage_losses
                else tf.concat(
                    [tf.cast(new_z, tf.float32), voltage_penalty[:, None]], axis=-1
                )
            )
        new_state = (
            new_z_buf,
            new_v,
            new_r,
            new_asc,
            new_psc_rise,
            new_psc,
            noise_step + 1,
        )
        if self._input_history_size:
            new_state = new_state + (tf.concat(new_input_history, axis=1),)
        if self._online_voltage_losses:
            penalties = tf.stack(
                [loss.step(floor_voltage, floor_active) for loss in self._online_voltage_losses],
                axis=-1,
            )
            new_state += (tf.cast(states[-1], tf.float32) + penalties,)
        if capture_currents:
            return outputs, new_state, tf.stop_gradient(forward_currents)
        if recurrent_weight_carrier is not None:
            return outputs, new_state, recurrent_weight_carrier
        return outputs, new_state

    @property
    def output_size(self):
        if (
            self.dynamics_mode == "nest" or self.state_precision == "selective" or self._online_voltage_losses
        ) and not self._return_voltage_sequences:
            return (tf.TensorShape([self._n_neurons]), tf.TensorShape([]))
        return (
            self._n_neurons * 2
            if self._return_voltage_sequences
            else self._n_neurons + 1
        )

    @property
    def state_size(self):
        state_size = (
            self._n_neurons * self.max_delay,  # z buffer
            self._n_neurons,  # v
            self._n_neurons,  # r
            self._n_neurons * 2,  # asc
            self._n_neurons * self._n_syn_basis,  # psc rise
            self._n_neurons * self._n_syn_basis,  # psc
            tf.TensorShape([]),  # noise timestep
        )
        if getattr(self, "_input_history_size", 0):
            state_size = state_size + (self._input_history_size,)
        if self._online_voltage_losses:
            state_size += (len(self._online_voltage_losses),)
        return state_size

    def zero_state(self, batch_size, dtype, with_names=False):
        selective = getattr(self, "state_precision", "compute") == "selective"
        if selective:
            dtype = self.compute_dtype
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        state_dtype = self.state_dtype if selective else dtype
        v0 = tf.zeros((batch_size, self._n_neurons), state_dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), self._refractory_state_dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), state_dtype)
        if getattr(self, "dynamics_mode", "legacy") == "nest":
            v0 = v0 + tf.cast(self.initial_voltage, state_dtype)
            asc = asc + tf.cast(self.initial_asc, state_dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        noise_step0 = tf.zeros((batch_size,), dtype=tf.int32)
        state = (z0_buf, v0, r0, asc, psc_rise0, psc0, noise_step0)
        state_names = (
            "z0_buf",
            "v0",
            "r0",
            "asc",
            "psc_rise0",
            "psc0",
            "noise_step0",
        )
        if getattr(self, "_input_history_size", 0):
            state = state + (tf.zeros((batch_size, self._input_history_size), dtype),)
            state_names = state_names + ("input_history0",)
        if getattr(self, "_online_voltage_losses", ()):
            state += (tf.zeros((batch_size, len(self._online_voltage_losses)), tf.float32),)
            state_names += ("online_voltage_penalty0",)
        if with_names:
            return state, state_names
        return state
