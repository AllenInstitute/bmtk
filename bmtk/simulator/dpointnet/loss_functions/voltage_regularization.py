import tensorflow as tf


class VoltageRegularization:
    def __init__(self, rnn, voltage_cost=1e-5, dtype=tf.float32, core_mask=None, **kwargs):
        self._rnn = rnn
        self._voltage_cost = voltage_cost
        # self._cell = cell
        self._dtype = dtype
        self._core_mask = core_mask

    @staticmethod
    def module():
        return 'VoltageRegularization'

    def __call__(self, voltages, **kwargs):
        if self._core_mask is not None:
            voltages = tf.boolean_mask(voltages, self._core_mask, axis=2)

        v_tot = tf.square(voltages - 1.0)
        voltage_loss = tf.reduce_mean(v_tot)
        return voltage_loss * self._voltage_cost
