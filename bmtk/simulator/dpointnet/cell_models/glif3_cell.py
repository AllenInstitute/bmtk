import os
import tensorflow as tf
import numpy as np
import pickle as pkl
from pathlib import Path
from bmtk.simulator.dpointnet.io_tools import io


try:
    from numba import njit
    HAS_NUMBA = True
except Exception:
    HAS_NUMBA = False


def make_pre_ind_table(indices, n_source_neurons):
    # Validate inputs
    if n_source_neurons <= 0:
        raise ValueError(f'The number of source neurons = {n_source_neurons}, must be greater than 0.')
    indices_np = np.asarray(indices)  # convert to np array just-in-casse
    if indices_np.ndim != 2 or indices_np.shape[1] < 2:
        raise ValueError(f'`indices` must have shape [n_synapses, >=2], got {indices_np.shape}.')
    pre_ids = indices_np[:, 1].astype(np.int64, copy=False)
    invalid = (pre_ids < 0) | (pre_ids >= n_source_neurons)
    if np.any(invalid):
        bad = int(pre_ids[np.flatnonzero(invalid)[0]])
        raise ValueError(f'Presynaptic index {bad} is out of bounds for `n_source_neurons={n_source_neurons}`.')

    if pre_ids.size == 0:
        order_np = np.empty((0,), dtype=np.int32)
        row_splits_np = np.zeros((n_source_neurons + 1,), dtype=np.int64)
    elif HAS_NUMBA:
        order_np, row_splits_np = _build_csr_order_numba(pre_ids, n_source_neurons)
    else:
        # Safe deterministic fallback if numba is unavailable.
        order_np = np.argsort(pre_ids, kind='stable')
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

    return tf.identity(z_, name='spike_gauss'), grad


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
def calculate_synaptic_currents(rec_z_buf, synapse_indices, weight_values, weight_values_compute,
                                dense_shape, synaptic_basis_weights, syn_ids, pre_ind_table):
    
    """
    Optimized synaptic current calculation with memory-efficient gradients for RSNN timestep iteration.

    Mathematical formulation:
    - Forward: I[b,post,r] = sum_pre(spike[b,pre] * W[post,pre] * basis[type[post,pre], r])
    - Grad w.r.t spike: dL/dspike[b,pre] = sum_post sum_r(dL/dI[b,post,r] * W[post,pre] * basis[type,r])
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
    pre_neuron_indices = non_zero_indices[:, 1]  # keep int64 for RaggedTensor gather (CPU-optimized)

    # Retrieve connections and weights for active presynaptic neurons
    # This uses pre_ind_table (RaggedTensor), which benefits from int64 on CPU
    new_indices, new_weights, new_syn_ids, post_in_degree, all_synaptic_inds = get_new_inds_table(
        synapse_indices, weight_values_compute, syn_ids, pre_neuron_indices, pre_ind_table
    )
    # new_syn_ids = tf.cast(new_syn_ids, dtype=tf.int32)  # int32 to reduce VRAM since its reused in backward pass

    # Returns: new_indices (int64), new_syn_ids (int64), post_in_degree (int32), all_synaptic_inds (int32)

    # Build segment IDs for unsorted_segment_sum using int32 for the GPU kernel.
    batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)  # int64
    post_neuron_indices = new_indices[:, 0]  # keep as int64 for compatibility, will be cast in segment_ids calculation
    num_segments = batch_size * n_post_neurons  # int64
    segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices  # int64
    segment_ids = tf.cast(segment_ids, dtype=tf.int32)
    num_segments = tf.cast(num_segments, dtype=tf.int32)

    # Compute weighted basis factors for active synapses
    # Note: basis_factors will be recomputed in grad() to save VRAM (cheap gather operation)
    # basis_factors = tf.cast(
    #     tf.gather(synaptic_basis_weights, new_syn_ids, axis=0), compute_dtype
    # )  # [n_active, n_basis], compute_dtype
    basis_factors = tf.gather(synaptic_basis_weights, new_syn_ids, axis=0)  # [n_active, n_basis]
    new_syn_ids = tf.cast(new_syn_ids, dtype=tf.int32)  # int32 to reduce VRAM since its reused in backward pass
    # Mixed precision: cast gathered subsets to compute_dtype (float16) AFTER gather.
    # This avoids creating a temporary float16 copy of the full weight table (~23M elements)
    # and only casts the ~1-2M active connections — 10-20x less cast work and no extra VRAM.
    new_weights = tf.cast(new_weights, compute_dtype)
    weighted_basis = new_weights[:, tf.newaxis] * basis_factors  # [n_active, n_basis], compute_dtype
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
            recurrent_weights_factors = tf.gather(synaptic_basis_weights[:, r_id], syn_ids, axis=0)
            weights_syn_receptors = weight_values_compute * recurrent_weights_factors
            sparse_w_rec = tf.sparse.SparseTensor(synapse_indices, weights_syn_receptors, dense_shape)
            de_dv_rid = tf.sparse.sparse_dense_matmul(dy_r, sparse_w_rec, adjoint_a=False)

            return r_id + 1, acc + de_dv_rid

        init_acc = tf.zeros((batch_size, n_pre_neurons_g), dtype=compute_dtype)
        _, de_dv= tf.while_loop(
            lambda r_id, _: r_id < n_syn_basis,
            per_receptor_accum,
            [tf.constant(0, dtype=tf.int32), init_acc],
            parallel_iterations=1,
        )

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

        # IMPORTANT: The gradient should reflect the same dampening that was applied
        # to rec_z_buf in the forward pass. However, we don't have access to the
        # dampening factor here, so the dampening must be applied BEFORE calling
        # this function, not within it.

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
        de_dweight_values_connection = tf.einsum('cr,cr->c', dnew_weights, basis_factors_grad)
        # de_dweight_values_connection = tf.reduce_sum(dnew_weights * basis_factors_grad, axis=1)  # [n_active], compute_dtype
        # Accumulate to original synapse positions
        # Instead of tensor_scatter_nd_add, use unsorted_segment_sum:
        de_dweight_values = tf.math.unsorted_segment_sum(
            data=de_dweight_values_connection,
            segment_ids=all_synaptic_inds,
            num_segments=tf.shape(weight_values)[0]
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
            de_dv,              # Gradient w.r.t rec_z_buf (compute_dtype)
            None,               # synapse_indices (constant)
            de_dweight_values,  # Gradient w.r.t weight_values (float32, matches master weights)
            None,               # weight_values_compute (non-trainable shadow copy)
            None,               # dense_shape[0] (constant)
            None,               # dense_shape[1] (constant)
            None,               # synaptic_basis_weights (constant)
            None,               # syn_ids (constant)
            None                # pre_ind_table (constant)
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




class GLIF3Cell(tf.keras.layers.Layer):
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
            hard_reset=False,
            tau_syns=None,
            # current_input=False,
        ):
        super().__init__()
        _node_params = dict(glif_network['node_params'])

        voltage_scale = _node_params['V_th'] - _node_params['E_L']
        
        ## TODO: Don't update the dictionary, just make adjusted_asc_amps a variable
        _node_params["asc_amps"] = (_node_params["asc_amps"] / voltage_scale[..., None])

        self._node_type_ids = np.array(glif_network['node_type_ids'])
        self._dt = tf.constant(dt, self.compute_dtype)
        self._recurrent_dampening = tf.constant(recurrent_dampening_factor, self.compute_dtype)
        self._dampening_factor = tf.constant(dampening_factor, self.compute_dtype)
        self._voltage_gradient_dampening = tf.constant(voltage_gradient_dampening, self.compute_dtype)
        self._pseudo_gauss = pseudo_gauss
        self._lr_scale = tf.constant(lr_scale, dtype=self.compute_dtype)

        ## TODO: noise_seed and noise_stream variables don't appear to be be changed, try making them constant
        self.noise_seed = tf.Variable(int(noise_seed), trainable=False, dtype=tf.int64, name='noise_seed')
        self.noise_stream = tf.Variable(0, trainable=False, dtype=tf.int64)
        self._hard_reset = hard_reset
        # self._current_input = current_input
        self._n_neurons = int(glif_network["n_nodes"])
        self._gauss_std = tf.constant(gauss_std, self.compute_dtype)

        # Determine the membrane time decay constant
        tau = _node_params["C_m"] / _node_params["g"]
        membrane_decay = np.exp(-dt / tau)
        current_factor = (1 - membrane_decay) / _node_params["g"]

        # Determine the synaptic dynamic parameters for each of the 4 basis receptors.
        if tau_syns is None:
            raise ValueError(f'Invalid tau_syns = {tau_syns}, please pass in a numpy array or a path to a npy file.')
        if isinstance(tau_syns, (str, Path)):
            # tau_syns may be passed in either as a vector or stored in an npy file.
            tau_path = tau_syns
            tau_syns = np.load(tau_path)
        elif isinstance(tau_syns, (list, tuple)):
            tau_syns = np.array(tau_syns)
        
        self._n_syn_basis = tau_syns.size
        syn_decay_np = np.exp(-dt / tau_syns)
        syn_decay_np = np.tile(syn_decay_np, self._n_neurons)
        self.syn_decay = tf.constant(syn_decay_np[None, :], dtype=self.compute_dtype) # expand the dimension for processing different receptor types
        psc_initial_np = np.e / tau_syns
        psc_initial_np = np.tile(psc_initial_np, self._n_neurons)
        self.psc_initial = tf.constant(psc_initial_np[None, :], dtype=self.compute_dtype) # expand the dimension for processing different receptor types

        self.max_delay = int(np.round(np.min([np.max(glif_network["synapses"]["delays"]), max_delay])))
        

        # Gather the neuron parameters for every neuron
        t_ref_per_neuron = _node_params["t_ref"][self._node_type_ids]
        t_ref_steps = np.ceil(t_ref_per_neuron / dt).astype(np.int16)
        t_ref_steps = np.maximum(t_ref_steps, 1)
        max_ref_steps = int(np.max(t_ref_steps))
        if max_ref_steps > 127:
            self._refractory_state_dtype = tf.int16
            print(f"Warning: max refractory period is {max_ref_steps} steps, which exceeds int8 capacity. Using int16 for refractory state.")
        else:
            self._refractory_state_dtype = tf.int8
        self.t_ref_steps = tf.constant(t_ref_steps, dtype=self._refractory_state_dtype)
        
        self.asc_amps = tf.Variable(
            tf.cast(tf.gather(_node_params['asc_amps'], indices=self._node_type_ids), self.compute_dtype),
            trainable=False,
            dtype=self.compute_dtype
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
        
        _k = tf.cast(_node_params['k'], self.compute_dtype)
        _k = tf.gather(_k, self._node_type_ids)
        _k = tf.math.log(_k /(1.0 - _k))
        _k = tf.cast(_k, self.compute_dtype)
        _k = tf.Variable(_k, trainable=False)  # TODO: Is this necessary??
        k = tf.nn.sigmoid(_k.read_value())

        self.asc_decay = tf.exp(-self._dt * k)
        self.v_th = tf.constant(1.0, dtype=self.compute_dtype)
        self.v_reset = tf.constant(0.0, dtype=self.compute_dtype)

        # Cast before constructing the Variable: tf.Variable does not auto-cast a
        # float32 initial value to a float16 compute_dtype under mixed precision.
        self.decay = tf.Variable(
            tf.cast(tf.gather(membrane_decay, self._node_type_ids), self.compute_dtype),
            trainable=False, dtype=self.compute_dtype)
        self.current_factor = tf.Variable(
            tf.cast(tf.gather(current_factor, self._node_type_ids), self.compute_dtype),
            trainable=False, dtype=self.compute_dtype)

        ## TODO: This shouldn't be stored in a separate pickle.
        # path = os.path.join(glif_network["data_dir"], 'tf_data', 'syn_id_to_syn_weights_dict.pkl')
        # path = 'GLIF_network/tf_data/syn_id_to_syn_weights_dict.pkl'
        # with open(path, "rb") as f:
        #     syn_id_to_syn_weights_dict = pkl.load(f)
        # synaptic_basis_weights_ = np.array(list(syn_id_to_syn_weights_dict.values()))
        # synaptic_basis_weights_ = tf.constant(synaptic_basis_weights_, dtype=self.compute_dtype)

        _synaptic_basis_weights = glif_network["synapses"]['dynamics_params']['basis_weights']
        self.synaptic_basis_weights = tf.constant(_synaptic_basis_weights, dtype=self.compute_dtype)

        # TODO: Allow option to not have recurrent connectivity (eg. in case only want to train feedforward network)
        ### Network recurrent connectivity ###
        indices = np.array(glif_network["synapses"]["indices"]) # NOTE: These are the tf indices, not SONATA, and in the form [trg, src] 
        weights = np.array(glif_network["synapses"]["weights"])
        dense_shape = np.array(glif_network["synapses"]["dense_shape"])
        syn_ids = np.array(glif_network["synapses"]["syn_ids"])
        delays = np.array(glif_network["synapses"]["delays"])
        weights = (weights/voltage_scale[self._node_type_ids[indices[:, 0]]])  # Scale down the recurrent weights
        delays = np.round(np.clip(delays, dt, self.max_delay)/dt).astype(np.int32) # Use the maximum delay to clip the synaptic delays
        indices[:, 1] = indices[:, 1] + self._n_neurons * (delays - 1) # Introduce the delays in the presynaptic neuron indices


        # the first column (presynaptic neuron) has size n_neurons and the second column (postsynaptic neuron) has size max_delay*n_neurons
        self.recurrent_dense_shape = dense_shape[0], self.max_delay * dense_shape[1]

        # Define the Tensorflow variables
        self.recurrent_indices = tf.Variable(indices, dtype=tf.int64, trainable=False) #dtype necessary for sparse dense matmul
        self.pre_ind_table = make_pre_ind_table(indices, n_source_neurons=self.recurrent_dense_shape[1]) # dtype int32
        
        recurrent_weight_positive = tf.constant(weights >= 0, dtype=tf.bool)

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

        self.recurrent_weight_values = tf.Variable(
            weights * recurrent_weight_scale / lr_scale,
            name="sparse_recurrent_weights",
            constraint=SignedConstraint(recurrent_weight_positive),
            trainable=individual_training,
            dtype=self.variable_dtype
        ) # shape = (n_synapses,)

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
            self.recurrent_weight_values_compute = self._no_dependency(recurrent_weight_values_compute)
        else:
            self.recurrent_weight_values_compute = self.recurrent_weight_values

        self.syn_ids = tf.constant(syn_ids, dtype=tf.int64) # this needs to be int64 for efficiency
        # self.recurrent_weights_factors = tf.gather(self.synaptic_basis_weights, self.syn_ids, axis=0) # TensorShape([23525415, 5])
        io.log_debug(f' > Added recurrent synapses: indices = {len(indices)}; trainable = {individual_training}')
        
        # print(f"    > # Recurrent synapses: {len(indices)}")

        del indices, weights, dense_shape, delays, syn_ids, recurrent_weight_positive

        ### Inputs
        # TODO: Inputs needs to be a list since order is extremely important
        self.inputs_idx = np.zeros(len(inputs) + 1, dtype=int)
        self.inputs = {}
        for idx, (input_name, input_network) in enumerate(inputs.items()):
            # TODO: Use a named tuple instead of dict
            input_props = {}
            n_input_nodes = input_network['n_inputs']
            input_dense_shape = (self._n_neurons, n_input_nodes)

            input_props['input_dim'] = n_input_nodes
            input_props['input_dense_shape'] = input_dense_shape
            input_indices = np.array(input_network['indices'])
            input_weights = np.array(input_network['weights'])
            input_syn_ids = np.array(input_network['syn_ids'])
            input_weights = input_weights / voltage_scale[self._node_type_ids[input_indices[:, 0]]]
            input_props['input_indices'] = tf.Variable(input_indices, trainable=False, dtype=tf.int64)

            input_type = input_network['input_type']
            input_options = input_network.get('options', {})
            input_weight_positive = tf.constant(input_weights >= 0, dtype=tf.bool)
            input_trainable = input_options.get('trainable', False)

            input_props['input_weight_values'] = tf.Variable(
                input_weights * input_options.get('weight_scale', 1.0) / lr_scale,
                name=f'{input_name}_input_weights',
                constraint=SignedConstraint(input_weight_positive),
                trainable=input_trainable,
                dtype=self.variable_dtype
            )
            input_props['input_syn_ids'] = tf.constant(input_syn_ids, dtype=tf.int64) # for efficiency this needs to be in int64
            # if not self._current_input:
            #     input_props['pre_input_ind_table'] = make_pre_ind_table(
            #         input_indices, 
            #         n_source_neurons=input_dense_shape[1]
            #     )
            input_props['input_type'] = input_type
            if input_type == 'spikes':
                input_props['pre_input_ind_table'] = make_pre_ind_table(
                    input_indices, 
                    n_source_neurons=input_dense_shape[1]
                )
                end_indx = self.inputs_idx[idx] + n_input_nodes
            elif input_type == 'current':
                end_indx = self.inputs_idx[idx] + n_input_nodes
            else:
                raise ValueError(f'Unknown input type {input_type}')

            io.log_debug(f' > Added "{input_name}" input synapses: indices = {len(input_indices)}, trainble = {input_trainable}')
            self.inputs_idx[idx+1] = end_indx

            # if input_name == 'bkg':
            #     input_props['spike_prob'] = tf.constant(bkg_firing_rate * 0.001, dtype=self.compute_dtype)

            self.inputs[input_name] = input_props

        '''
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
        '''

    def refresh_recurrent_weight_shadow(self):
        """Sync the compute-dtype shadow of the recurrent weights with the trained master.

        Under mixed precision the forward uses ``recurrent_weight_values_compute`` (a
        non-trainable float16 copy) while the optimizer updates the float32 master
        ``recurrent_weight_values``. This must be called after each optimizer step, else the
        forward keeps using stale weights and recurrent-weight training has no effect.
        Matches the reference V1_GLIF_model.
        """
        if self.recurrent_weight_values_compute is self.recurrent_weight_values:
            return
        if not self.recurrent_weight_values.trainable:
            return
        self.recurrent_weight_values_compute.assign(
            tf.cast(self.recurrent_weight_values, self.compute_dtype)
        )

    def calculate_i_rec_with_custom_grad(self, rec_z_buf):
        return calculate_synaptic_currents(
            rec_z_buf,
            self.recurrent_indices,
            self.recurrent_weight_values,
            self.recurrent_weight_values_compute,
            self.recurrent_dense_shape,
            self.synaptic_basis_weights,
            self.syn_ids,
            self.pre_ind_table
        )

    def calculate_input_current_from_spikes(self, x_t, input_net):
        batch_size = tf.cast(tf.shape(x_t)[0], dtype=tf.int64) # int64
        n_post_neurons = input_net['input_dense_shape'][0]
        # n_post_neurons = self.lgn_input_dense_shape[0]
        
        # Find the indices of non-zero inputs
        if x_t.dtype == tf.bool:
            non_zero_indices = tf.where(x_t)
        else:
            non_zero_indices = tf.where(x_t > 0)

        batch_indices = non_zero_indices[:, 0]  # int64
        pre_neuron_indices = non_zero_indices[:, 1] #int64

        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(
            input_net['input_indices'],
            input_net['input_weight_values'],
            input_net['input_syn_ids'],
            pre_neuron_indices,
            input_net['pre_input_ind_table']
            
            
            # self.input_indices,
            # self.input_weight_values,
            # self.input_syn_ids,
            # pre_neuron_indices,
            # self.pre_input_ind_table
        )

        # Expand batch_indices to match the length of inds_flat
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree) #int64
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0] #int64
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        segment_ids = tf.cast(segment_ids, dtype=tf.int32)
        num_segments = tf.cast(batch_size * n_post_neurons, dtype=tf.int32)
        # Compute weighted basis factors
        basis_factors = tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)  # shape (n_active_connections, n_syn_basis)
        # Accumulate currents in variable_dtype for better numerical fidelity, then cast once.
        n_pre_spikes = tf.cast(tf.gather_nd(x_t, non_zero_indices), dtype=self.compute_dtype)
        new_weights = tf.cast(new_weights, self.compute_dtype)
        new_weights_final = (new_weights * tf.repeat(n_pre_spikes, post_in_degree))[:, tf.newaxis] * basis_factors
        # new_weights_final = new_weights[:, tf.newaxis] * basis_factors

        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(new_weights_final, segment_ids, num_segments)

        # Cast back to compute_dtype to keep downstream state buffers compact.
        # if i_in_flat.dtype != self.compute_dtype:
        #     i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat

    def _dense_update_impl(self, batch_size, prev_z, v, r, asc, psc_rise, psc, rec_inputs):
        # new_psc, new_psc_rise = self.update_psc(psc, psc_rise, rec_inputs)
        new_psc_rise = psc_rise * self.syn_decay + rec_inputs * self.psc_initial
        new_psc = psc * self.syn_decay + self._dt * self.syn_decay * psc_rise
        
        # Calculate the ASC variables
        asc = tf.reshape(asc, (batch_size, self._n_neurons, 2))
        # new_asc = self.asc_decay * asc + tf.expand_dims(prev_z, axis=-1) * self.asc_amps
        new_asc = self.asc_decay * asc + tf.expand_dims(tf.stop_gradient(prev_z), axis=-1) * self.asc_amps
        new_asc = tf.reshape(new_asc, (batch_size, self._n_neurons * 2))
        # Calculate the postsynaptic current
        input_current = tf.reshape(psc, (batch_size, self._n_neurons, self._n_syn_basis))
        input_current = tf.reduce_sum(input_current, -1)
        # Add all the postsynaptic current sources
        c1 = input_current + tf.reduce_sum(asc, axis=-1) # + self.gathered_g
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
        new_v = self.decay * dampened_v + self.current_factor * c1 - tf.stop_gradient(prev_z)
        # new_v = self.decay * dampened_v + self.current_factor * c1 - prev_z
        # Update the voltage according to the LIF equation and the refractory period
        # New r is a variable that accounts for the refractory period in which a neuron cannot spike
        # prev_spike = tf.cast(prev_z > 0, self._refractory_state_dtype)
        prev_spike = tf.cast(prev_z, dtype=self._refractory_state_dtype)
        # new_r = tf.maximum(r + prev_spike * self.t_ref_steps - 1, 0)
        new_r = tf.stop_gradient(tf.maximum(r + prev_spike * self.t_ref_steps - 1, 0)) # prevent gradients from flowing through the refractory state

        if self._hard_reset:
            # Here we keep the voltage at the reset value during the refractory period
            refractory_mask = tf.greater(new_r, 0)
            new_v = tf.where(refractory_mask, self.v_reset, new_v)
            # Here we make a hard reset and let the voltage freely evolve but we do not let the
            # neuron spike during the refractory period

        # new_v_state = tf.cast(new_v, self.compute_dtype)
        return new_v, new_r, new_asc, new_psc_rise, new_psc

    def calculate_noise_current(self, batch_size, noise_step, input_net):
        # n_post_neurons = self.bkg_input_dense_shape[0]
        n_post_neurons = input_net['input_dense_shape'][0]
        step_seed = tf.cast(noise_step[0], tf.int32)
        base_seed = tf.cast(self.noise_seed, tf.int32)
        replica_context = tf.distribute.get_replica_context()
        if replica_context is None:
            replica_id = tf.constant(0, dtype=tf.int32)
        else:
            replica_id = tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
        noise_seed = tf.stack(
            [base_seed + replica_id * tf.constant(1000003, dtype=tf.int32), step_seed],
            axis=0,
        )
        poisson_shape = tf.stack(
            [tf.cast(batch_size, tf.int32), tf.cast(input_net['input_dense_shape'][1], tf.int32)],
            axis=0,
        )
        rest_of_brain = tf.random.stateless_poisson(
            shape=poisson_shape,
            seed=noise_seed,
            # lam=self.bkg_spike_prob,
            lam=input_net['spike_prob'],
            dtype=tf.int32,
        )

        # Keep noise indexing in int64
        non_zero_indices = tf.where(rest_of_brain > 0)

        batch_indices = non_zero_indices[:, 0] #int64
        pre_neuron_indices = non_zero_indices[:, 1] #int64
        # Get the indices into self.recurrent_indices for each pre_neuron_index
        # self.pre_ind_table is a RaggedTensor or a list of lists mapping pre_neuron_index to indices in recurrent_indices
        new_indices, new_weights, new_syn_ids, post_in_degree, _ = get_new_inds_table(
            input_net['input_indices'],
            input_net['input_weight_values'],
            input_net['input_syn_ids'],
            pre_neuron_indices,
            input_net['pre_input_ind_table'],
            # self.bkg_input_indices,
            # self.bkg_input_weights,
            # self.bkg_input_syn_ids,
            # pre_neuron_indices,
            # self.pre_bkg_ind_table
        )
        # Expand batch_indices to match the length of inds_flat
        batch_indices_per_connection = tf.repeat(batch_indices, post_in_degree)
        # Get post-synaptic neuron indices
        post_neuron_indices = new_indices[:, 0]
        # Compute segment IDs
        segment_ids = batch_indices_per_connection * n_post_neurons + post_neuron_indices
        segment_ids = tf.cast(segment_ids, dtype=tf.int32)
        num_segments = tf.cast(batch_size * n_post_neurons, dtype=tf.int32)

        # # Alternative (slower): Gather spike counts once per active input and broadcast by connection degree.
        # # Keeping it here only as reference.
        # active_spike_counts = tf.gather_nd(rest_of_brain, non_zero_indices)
        # n_pre_spikes = tf.cast(tf.repeat(active_spike_counts, post_in_degree), dtype=self.variable_dtype)

        presynaptic_indices = tf.stack(
            [batch_indices_per_connection, new_indices[:, 1]],
            axis=1
        ) # gather works better with int64
        n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.compute_dtype)
        # n_pre_spikes = tf.cast(tf.gather_nd(rest_of_brain, presynaptic_indices), dtype=self.compute_dtype)

        # Compute weighted basis factors
        basis_factors = tf.gather(self.synaptic_basis_weights, new_syn_ids, axis=0)
        # Accumulate currents in variable_dtype for better numerical fidelity, then cast once.
        new_weights = tf.cast(new_weights, dtype=self.compute_dtype) # shape (n_active_connections,)
        # new_weights = tf.cast(new_weights * n_pre_spikes, self.compute_dtype)
        # new_weights_final = new_weights[:, tf.newaxis] * basis_factors
        # new_weights_final = tf.cast(new_weights * n_pre_spikes, dtype=self.compute_dtype)
        new_weights_final = (new_weights * n_pre_spikes)[:, tf.newaxis] * basis_factors

        # Calculate input currents
        i_in_flat = tf.math.unsorted_segment_sum(new_weights_final, segment_ids, num_segments)

        # # Cast back to compute_dtype to keep downstream state buffers compact.
        # if i_in_flat.dtype != self.compute_dtype:
        #     i_in_flat = tf.cast(i_in_flat, dtype=self.compute_dtype)

        return i_in_flat

    def call(self, inputs, states):
        # lgn_inputs = inputs[:, :self.input_dim]

        batch_size = tf.cast(tf.shape(inputs)[0], dtype=tf.int64)

        z_buf, v, r, asc, psc_rise, psc, noise_step = states
        prev_z = z_buf[:, :self._n_neurons]
        
        rec_z_buf = straight_through_dampen(z_buf, 1.0 - self._recurrent_dampening)
        i_rec = self.calculate_i_rec_with_custom_grad(rec_z_buf)
        
        extern_currents = []
        for idx, input_net in enumerate(self.inputs.values()):
            input_spikes = inputs[:, self.inputs_idx[idx]:self.inputs_idx[idx+1]]
            if input_net['input_type'] == 'current':
                extern_currents.append(self.calculate_input_current_from_firing_probabilities(input_spikes))
            else:
                extern_currents.append(self.calculate_input_current_from_spikes(input_spikes, input_net))
        
        rec_inputs = i_rec + tf.add_n(extern_currents)

        # Reshape i_rec_flat back to [batch_size, num_neurons]
        rec_inputs = tf.reshape(rec_inputs, [batch_size, self._n_neurons * self._n_syn_basis])

        # Scale with the learning rate
        rec_inputs = rec_inputs * self._lr_scale
        new_v, new_r, new_asc, new_psc_rise, new_psc = self._dense_update_impl(
            batch_size, prev_z, v, r, asc, psc_rise, psc, rec_inputs
        )

        # Generate spikes from a high-fidelity membrane lane before state quantization.
        # v_sc = (new_v - self.v_th) / self.normalizer # normalized is 1 for scaled voltage
        v_sc = new_v - self.v_th
        if self._pseudo_gauss:
            new_z = spike_gauss(v_sc, self._gauss_std, self._dampening_factor)
        else:
            new_z = spike_function(v_sc, self._dampening_factor)
        
        # Generate the new spikes if the refractory period is concluded
        refractory_active = tf.greater(new_r, 0)
        new_z = tf.where(refractory_active, tf.zeros_like(new_z), new_z)

        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer

        outputs = (
            new_z,
            new_v,
            # new_v * self.voltage_scale + self.voltage_offset,
            # (input_current + tf.reduce_sum(asc, axis=-1)) * self.voltage_scale,
        )
        new_noise_step = noise_step + 1
        new_state = (new_z_buf, new_v, new_r, new_asc, new_psc_rise, new_psc, new_noise_step)
        return outputs, new_state

        # Generate the new spikes if the refractory period is concluded
        refractory_active = tf.greater(new_r, 0)
        new_z = tf.where(refractory_active, tf.zeros_like(new_z), new_z)

        # Add current spikes to the buffer
        new_z_buf = tf.concat([new_z, z_buf[:, :-self._n_neurons]], axis=1)  # Shift buffer
        return inputs, [states]

    @property
    def state_size(self):
        return (
            self._n_neurons * self.max_delay,     # z buffer
            self._n_neurons,                      # v
            self._n_neurons,                      # r
            self._n_neurons * 2,                  # asc
            self._n_neurons * self._n_syn_basis,  # psc rise
            self._n_neurons * self._n_syn_basis,  # psc
            tf.TensorShape([]),                   # noise timestep counter
        )

    def zero_state(self, batch_size, dtype, with_names=False):
        z0_buf = tf.zeros((batch_size, self._n_neurons * self.max_delay), dtype)
        v0 = tf.zeros((batch_size, self._n_neurons), dtype)
        r0 = tf.zeros((batch_size, self._n_neurons), self._refractory_state_dtype)
        asc = tf.zeros((batch_size, self._n_neurons * 2), dtype)
        psc_rise0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        psc0 = tf.zeros((batch_size, self._n_neurons * self._n_syn_basis), dtype)
        noise_step0 = tf.zeros((batch_size,), tf.int32)

        if with_names:
            return (z0_buf, v0, r0, asc, psc_rise0, psc0, noise_step0), ('z0_buf', 'v0', 'r0', 'asc', 'psc_rise0', 'psc0', 'noise_step0')
        else:
            return z0_buf, v0, r0, asc, psc_rise0, psc0, noise_step0

