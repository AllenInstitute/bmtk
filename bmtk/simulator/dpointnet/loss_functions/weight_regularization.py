import numpy as np
import tensorflow as tf
from contextlib import contextmanager
from contextvars import ContextVar

from . import loss_utils


_weight_regularization_scope = ContextVar("weight_regularization_scope", default=None)


@contextmanager
def weight_regularization_scope():
    """Share weight losses only within one training loss/gradient evaluation."""
    token = _weight_regularization_scope.set(object())
    try:
        yield
    finally:
        _weight_regularization_scope.reset(token)


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


def _grouped_emd_custom_gradient(
    values, group_order, sorted_initial_values, group_slices, sensitivity_denominators
):
    """Mean per-group Wasserstein-1 distance with one final scatter VJP."""

    @tf.custom_gradient
    def emd(x):
        grouped_x = tf.gather(x, group_order)
        group_losses = []
        positions = []
        sensitivities = []
        for start, end in group_slices:
            x_i = grouped_x[start:end]
            order = tf.argsort(x_i, stable=True)
            deviation = tf.gather(x_i, order) - sorted_initial_values[start:end]
            group_losses.append(tf.reduce_mean(tf.abs(deviation)))
            positions.append(order + tf.cast(start, order.dtype))
            denominator = tf.constant(
                sensitivity_denominators[len(sensitivities)], dtype=x.dtype
            )
            sensitivities.append(tf.sign(deviation) / denominator)
        loss = tf.reduce_mean(tf.stack(group_losses))
        targets = tf.gather(group_order, tf.concat(positions, axis=0))
        derivative = tf.concat(sensitivities, axis=0)

        def grad(upstream):
            return tf.scatter_nd(
                tf.cast(targets[:, tf.newaxis], tf.int64),
                derivative * tf.cast(upstream, derivative.dtype),
                tf.cast(tf.shape(x), tf.int64),
            )

        return loss, grad

    return emd(values)


def _javier_grouped_emd(values, group_order, sorted_initial_values, group_slices, n_groups):
    """Javier-derived grouped EMD with analytic VJP.

    Adapted from Javier's ``v1_model_utils/loss_functions.py`` ``_grouped_emd``
    at commit 2c52ec10; credited to Javier. DPointNet supplies canonical
    recurrent-edge ordering via ``group_order`` and pre-sorted initial values.
    """

    @tf.custom_gradient
    def emd(x):
        grouped_x = tf.gather(x, group_order)
        group_losses, positions, sensitivities = [], [], []
        for start, end in group_slices:
            x_i = grouped_x[start:end]
            order = tf.argsort(x_i)
            deviation = tf.gather(x_i, order) - sorted_initial_values[start:end]
            group_losses.append(tf.reduce_mean(tf.abs(deviation)))
            positions.append(order + start)
            sensitivities.append(
                tf.sign(deviation) / float((end - start) * n_groups)
            )
        reg_loss = tf.reduce_mean(tf.stack(group_losses))
        targets = tf.gather(group_order, tf.concat(positions, axis=0))
        derivative = tf.concat(sensitivities, axis=0)

        def grad(upstream):
            return tf.scatter_nd(
                targets[:, tf.newaxis], derivative * upstream, tf.shape(x)
            )

        return reg_loss, grad

    return emd(values)


class EMDWeightRegularization:
    """Earth Mover's Distance (Wasserstein-1) regularizer on the recurrent weights.

    Penalizes the per-connection-type EMD between the current and initial recurrent
    synaptic weight distributions, averaged over all connection types. Because the
    initial distribution is captured at construction, the loss is exactly 0 at init.
    Ported from V1_GLIF_model loss_functions.EarthMoversDistanceRegularizer.
    """

    _use_grouped_custom_gradient = False
    _use_javier_grouped_emd = False
    _deduplicate_within_graph = False

    def __init__(
        self,
        rnn,
        cost=10.0,
        data_dir='',
        dtype=None,
        use_grouped_custom_gradient=False,
        use_javier_grouped_emd=False,
        deduplicate_within_graph=False,
        **kwargs
    ):
        self._rnn = rnn
        self._network = rnn.recurrent_network
        # The trainable recurrent weights (kept in the master/fp32 dtype under mixed precision).
        self._weights = rnn.cell.recurrent_weight_values
        self._dtype = dtype or self._weights.dtype
        self._cost = tf.constant(cost, dtype=self._dtype)
        self._use_grouped_custom_gradient = bool(use_grouped_custom_gradient)
        self._use_javier_grouped_emd = bool(use_javier_grouped_emd)
        self._deduplicate_within_graph = bool(deduplicate_within_graph)
        self._graph_cache = {}

        # Capture the initial weights directly from the cell variable. These are already in the
        # normalized (weights / voltage_scale) space the reference EMD uses, and guarantee the
        # loss is 0 at init regardless of any weight/lr scaling applied in the cell.
        initial_value_np = self._weights.numpy().astype(np.float32).copy()

        group_ids = _connection_type_ids(self._network, data_dir=data_dir).astype(
            np.int64, copy=False
        )
        n_groups = int(np.max(group_ids) + 1) if group_ids.size else 0

        # Presort the initial values per group so the call only needs to sort the current weights.
        if group_ids.size == 0:
            order_np = np.empty((0,), dtype=np.int64)
            row_splits_np = np.zeros((1,), dtype=np.int64)
            sorted_initial_np = np.empty((0,), dtype=np.float32)
        else:
            order_np, row_splits_np = _build_group_order(group_ids, n_groups)
            sorted_initial_np = _sort_initial_values_by_group(
                initial_value_np, order_np, row_splits_np
            )

        order_dtype = tf.int32 if order_np.size <= np.iinfo(np.int32).max else tf.int64
        self._n_groups = n_groups
        self.num_unique = tf.constant(n_groups, dtype=tf.int32)
        self._group_order = tf.Variable(
            order_np,
            dtype=order_dtype,
            trainable=False,
            name="emd_group_order",
        )
        self._row_splits = tf.Variable(
            row_splits_np,
            dtype=tf.int64,
            trainable=False,
            name="emd_row_splits",
        )
        self._sorted_initial_values = tf.Variable(
            sorted_initial_np,
            dtype=self._dtype,
            trainable=False,
            name="emd_sorted_initial_values",
        )
        self._group_slices = tuple(
            (int(row_splits_np[i]), int(row_splits_np[i + 1]))
            for i in range(n_groups)
        )
        self._group_sensitivity_denominators = tuple(
            int((end - start) * n_groups)
            for start, end in self._group_slices
        )

    @staticmethod
    def module():
        return "EMDWeightRegularization"

    @tf.function(jit_compile=False)  # jit_compile=True uses a lot of memory.
    def _compute(self, x):
        if x.dtype != self._dtype:
            x = tf.cast(x, self._dtype)
        if len(x.shape) > 1 and x.shape[1] == 1:
            x = tf.squeeze(x, axis=1)

        if self._n_groups == 0:
            return tf.reduce_sum(x) * tf.cast(0.0, self._dtype)

        denominators = getattr(
            self,
            "_group_sensitivity_denominators",
            tuple(
                int((end - start) * self._n_groups)
                for start, end in self._group_slices
            ),
        )
        if self._use_javier_grouped_emd:
            reg_loss = _javier_grouped_emd(
                x,
                tf.convert_to_tensor(self._group_order),
                tf.convert_to_tensor(self._sorted_initial_values),
                self._group_slices,
                self._n_groups,
            )
            return reg_loss * self._cost

        if self._use_grouped_custom_gradient:
            reg_loss = _grouped_emd_custom_gradient(
                x,
                tf.convert_to_tensor(self._group_order),
                tf.convert_to_tensor(self._sorted_initial_values),
                self._group_slices,
                denominators,
            )
            return reg_loss * self._cost

        grouped_x = tf.gather(x, self._group_order)
        emd_losses = tf.TensorArray(self._dtype, size=self._n_groups)
        for i in tf.range(self._n_groups):
            start = self._row_splits[i]
            end = self._row_splits[i + 1]
            x_i = grouped_x[start:end]
            y_i = self._sorted_initial_values[start:end]
            emd = tf.reduce_mean(tf.abs(tf.sort(x_i) - y_i))
            emd_losses = emd_losses.write(i, emd)
        reg_loss = tf.reduce_mean(emd_losses.stack())
        return reg_loss * self._cost

    def __call__(self, **kwargs):
        # Weight regularizer: independent of activity (spikes/voltages). Reads the current
        # trainable recurrent weights so gradients flow back to them.
        scope = _weight_regularization_scope.get()
        if self._deduplicate_within_graph and scope is not None and tf.inside_function():
            graph = tf.compat.v1.get_default_graph()
            cache_key = (graph, scope)
            cached = self._graph_cache.get(cache_key)
            if cached is None:
                cached = self._compute(tf.convert_to_tensor(self._weights))
                self._graph_cache[cache_key] = cached
            return cached
        return self._compute(tf.convert_to_tensor(self._weights))
