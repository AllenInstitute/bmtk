import numpy as np
import tensorflow as tf

from . import loss_utils


def _connection_type_ids(network, data_dir=''):
    """Per-synapse connection-type id, replicating other_v1_utils.connection_type_ids.

    A connection type is the (pre cell type, post cell type) pair. The returned array is
    aligned with ``network['synapses']['weights']`` ordering.
    """
    pop_names = loss_utils.get_pop_names(network, data_dir=data_dir)
    cell_types = np.array([loss_utils.pop_name_to_cell_type(p) for p in pop_names])
    _, pop_ids_cells = np.unique(cell_types, return_inverse=True)

    indices = np.asarray(network['synapses']['indices'])
    n_nodes = network['n_nodes']
    pre_cells = indices[:, 0]
    post_cells = indices[:, 1] % n_nodes

    pop_ids_synapses = pop_ids_cells[pre_cells] * 1000 + pop_ids_cells[post_cells]
    _, connection_type_ids = np.unique(pop_ids_synapses, return_inverse=True)
    return connection_type_ids


def _build_group_order(group_ids, n_groups):
    """Stable bucket ordering by group id with CSR row_splits."""
    order = np.argsort(group_ids, kind='stable')
    counts = np.bincount(group_ids, minlength=n_groups)
    row_splits = np.empty(n_groups + 1, dtype=np.int64)
    row_splits[0] = 0
    np.cumsum(counts, dtype=np.int64, out=row_splits[1:])
    return order.astype(np.int64, copy=False), row_splits


def _sort_initial_values_by_group(initial_values, order, row_splits):
    """Sort initial values independently inside each group segment."""
    out = np.empty(initial_values.shape[0], dtype=initial_values.dtype)
    for g in range(row_splits.shape[0] - 1):
        start, end = row_splits[g], row_splits[g + 1]
        out[start:end] = np.sort(initial_values[order[start:end]])
    return out


class EMDWeightRegularization:
    """Earth Mover's Distance (Wasserstein-1) regularizer on the recurrent weights.

    Penalizes the per-connection-type EMD between the current and initial recurrent
    synaptic weight distributions, averaged over all connection types. Because the
    initial distribution is captured at construction, the loss is exactly 0 at init.
    Ported from V1_GLIF_model loss_functions.EarthMoversDistanceRegularizer.
    """

    def __init__(self, rnn, cost=10.0, data_dir='', dtype=None, **kwargs):
        self._rnn = rnn
        self._network = rnn.recurrent_network
        # The trainable recurrent weights (kept in the master/fp32 dtype under mixed precision).
        self._weights = rnn.cell.recurrent_weight_values
        self._dtype = dtype or self._weights.dtype
        self._cost = tf.constant(cost, dtype=self._dtype)

        # Capture the initial weights directly from the cell variable. These are already in the
        # normalized (weights / voltage_scale) space the reference EMD uses, and guarantee the
        # loss is 0 at init regardless of any weight/lr scaling applied in the cell.
        initial_value_np = self._weights.numpy().astype(np.float32).copy()

        group_ids = _connection_type_ids(self._network, data_dir=data_dir).astype(np.int64, copy=False)
        n_groups = int(np.max(group_ids) + 1) if group_ids.size else 0

        # Presort the initial values per group so the call only needs to sort the current weights.
        if group_ids.size == 0:
            order_np = np.empty((0,), dtype=np.int64)
            row_splits_np = np.zeros((1,), dtype=np.int64)
            sorted_initial_np = np.empty((0,), dtype=np.float32)
        else:
            order_np, row_splits_np = _build_group_order(group_ids, n_groups)
            sorted_initial_np = _sort_initial_values_by_group(initial_value_np, order_np, row_splits_np)

        order_dtype = tf.int32 if order_np.size <= np.iinfo(np.int32).max else tf.int64
        order_tf = tf.convert_to_tensor(order_np, dtype=order_dtype)
        row_splits_tf = tf.convert_to_tensor(row_splits_np, dtype=tf.int32)
        sorted_initial_tf = tf.convert_to_tensor(sorted_initial_np, dtype=self._dtype)

        self.num_unique = tf.constant(n_groups, dtype=tf.int32)
        # Avoids RaggedTensor.from_value_rowids -> DenseBincount (raises under GPU deterministic
        # mode in TF 2.15); build row_splits directly from per-type counts instead.
        self._group_indices = tf.RaggedTensor.from_row_splits(order_tf, row_splits_tf, validate=False)
        self._sorted_initial_values = tf.RaggedTensor.from_row_splits(sorted_initial_tf, row_splits_tf, validate=False)

    @staticmethod
    def module():
        return 'EMDWeightRegularization'

    @tf.function(jit_compile=False)  # jit_compile=True uses a lot of memory.
    def _compute(self, x):
        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        emd_losses = tf.TensorArray(self._dtype, size=self.num_unique)
        for i in tf.range(self.num_unique):
            x_i = tf.gather(x, self._group_indices[i])
            y_i = self._sorted_initial_values[i]  # already presorted at init
            emd = tf.reduce_mean(tf.abs(tf.sort(x_i) - y_i))
            emd_losses = emd_losses.write(i, emd)
        reg_loss = tf.reduce_mean(emd_losses.stack())
        return reg_loss * self._cost

    def __call__(self, **kwargs):
        # Weight regularizer: independent of activity (spikes/voltages). Reads the current
        # trainable recurrent weights so gradients flow back to them.
        return self._compute(self._weights)
