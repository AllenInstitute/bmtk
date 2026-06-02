import tensorflow as tf

from . import loss_utils


class VoltageRegularization:
    def __init__(self, rnn, voltage_cost=1e-5, dtype=tf.float32, core_mask=None, **kwargs):
        self._rnn = rnn
        self._voltage_cost = voltage_cost
        # self._cell = cell
        self._dtype = dtype
        # Resolve core mask from an explicit mask or a core_radius (matches reference loss_core_radius).
        self._core_mask = loss_utils.resolve_core_mask(
            rnn.recurrent_network, core_mask, kwargs.get('core_radius'),
            kwargs.get('data_dir', 'GLIF_network')
        )

    @staticmethod
    def module():
        return 'VoltageRegularization'

    def __call__(self, voltages, **kwargs):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)

        v_tot = tf.square(voltages - 1.0)
        voltage_loss = tf.reduce_mean(v_tot)
        return voltage_loss * self._voltage_cost
