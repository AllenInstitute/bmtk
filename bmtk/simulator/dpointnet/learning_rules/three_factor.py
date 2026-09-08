import tensorflow as tf

from .eprop import EPropLearningRule


class ThreeFactorLearningRule(EPropLearningRule):
    """Configurable GLIF3 eligibility multiplied by a local third factor."""

    _SIGNAL_MODES = ('combined', 'spike', 'voltage')

    def __init__(
            self,
            signal='combined',
            spike_signal_scale=1.0,
            voltage_signal_scale=1.0,
            **kwargs):
        if signal not in self._SIGNAL_MODES:
            raise ValueError(
                f'Unknown three-factor signal "{signal}". Supported signals: '
                f'{", ".join(self._SIGNAL_MODES)}.'
            )
        self.signal = signal
        self.spike_signal_scale = float(spike_signal_scale)
        self.voltage_signal_scale = float(voltage_signal_scale)
        super().__init__(name='three_factor', **kwargs)

    @classmethod
    def module(cls):
        return 'three_factor'

    def _local_factor(self, observations, pseudo_derivative):
        spike_factor = (
            tf.cast(observations.spike_learning_signal, tf.float32)
            * pseudo_derivative
            * tf.cast(self.spike_signal_scale, tf.float32)
        )
        voltage_factor = (
            tf.cast(observations.voltage_learning_signal, tf.float32)
            * tf.cast(self.voltage_signal_scale, tf.float32)
        )
        if self.signal == 'spike':
            return spike_factor
        if self.signal == 'voltage':
            return voltage_factor
        return spike_factor + voltage_factor

    def get_config(self):
        config = super().get_config()
        config.update({
            'signal': self.signal,
            'spike_signal_scale': self.spike_signal_scale,
            'voltage_signal_scale': self.voltage_signal_scale,
        })
        return config
