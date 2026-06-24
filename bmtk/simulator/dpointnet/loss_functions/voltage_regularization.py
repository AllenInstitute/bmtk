import tensorflow as tf

from . import loss_utils


class VoltageRegularization:
    def __init__(self, rnn, voltage_cost=1e-5, dtype=tf.float32, core_mask=None, penalty_mode="range", **kwargs):
        self._rnn = rnn
        self._voltage_cost = tf.constant(voltage_cost, dtype=tf.float32)
        # self._cell = cell
        self._dtype = dtype
        self._penalty_mode = penalty_mode
        # Resolve core mask from an explicit mask or a core_radius (matches reference loss_core_radius).
        self._core_mask = loss_utils.resolve_core_mask(
            rnn.recurrent_network, core_mask, kwargs.get('core_radius'),
            kwargs.get('data_dir', 'GLIF_network')
        )

    @staticmethod
    def module():
        return 'VoltageRegularization'

    @tf.function(jit_compile=True)
    def _safe_global_mean(self, penalty):
        return tf.reduce_mean(tf.cast(penalty, tf.float32))

    @tf.function(jit_compile=True)
    def _compute_range_loss(self, voltages):
        penalty = tf.square(tf.nn.relu(tf.abs(voltages - 0.5) - 0.5))
        return self._safe_global_mean(penalty)

    @tf.function(jit_compile=True)
    def _compute_threshold_loss(self, voltages):
        penalty = tf.square(voltages - 1.0)
        return self._safe_global_mean(penalty)

    def __call__(self, voltages, **kwargs):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)

        if self._penalty_mode == "range":
            voltage_loss = self._compute_range_loss(voltages)
        elif self._penalty_mode == "threshold":
            voltage_loss = self._compute_threshold_loss(voltages)
        else:
            raise ValueError(f'Unknown voltage penalty_mode "{self._penalty_mode}". Options: range, threshold.')

        return voltage_loss * self._voltage_cost
