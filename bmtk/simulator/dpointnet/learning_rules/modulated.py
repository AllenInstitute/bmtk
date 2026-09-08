import tensorflow as tf

from .eprop import EPropLearningRule


class ModulatedEligibilityLearningRule(EPropLearningRule):
    """Eligibility updates driven by a finite set of broadcast modulators."""

    _SIGNALS = ("spike", "voltage", "combined")
    _PROJECTIONS = ("fixed_balanced_partition", "cell_class_partition")

    def __init__(
        self,
        n_channels=1,
        signal="combined",
        channel_projection="fixed_balanced_partition",
        **kwargs,
    ):
        if not isinstance(n_channels, int) or n_channels <= 0:
            raise ValueError("n_channels must be a positive integer.")
        if signal not in self._SIGNALS:
            raise ValueError(
                f'Unknown modulator signal "{signal}". Supported: '
                f'{", ".join(self._SIGNALS)}.'
            )
        if channel_projection not in self._PROJECTIONS:
            raise ValueError(
                f'Unknown channel projection "{channel_projection}". Supported: '
                f'{", ".join(self._PROJECTIONS)}.'
            )
        self.n_channels = n_channels
        self.signal = signal
        self.channel_projection = channel_projection
        super().__init__(name="modulated_eligibility", **kwargs)

    @classmethod
    def module(cls):
        return "modulated_eligibility"

    def _channel_ids(self):
        n_neurons = self.rnn._cell._n_neurons
        if self.channel_projection == "fixed_balanced_partition":
            return tf.math.mod(tf.range(n_neurons, dtype=tf.int32), self.n_channels)

        node_type_ids = tf.convert_to_tensor(
            self.rnn._cell._node_type_ids, dtype=tf.int64
        )
        _, class_ids = tf.unique(node_type_ids)
        return tf.math.mod(class_ids, self.n_channels)

    def _raw_modulator(self, observations, pseudo_derivative):
        spike_signal = tf.cast(
            observations.spike_learning_signal, tf.float32
        ) * tf.cast(pseudo_derivative, tf.float32)
        voltage_signal = tf.cast(observations.voltage_learning_signal, tf.float32)
        if self.signal == "spike":
            return spike_signal
        if self.signal == "voltage":
            return voltage_signal
        return spike_signal + voltage_signal

    def _local_factor(self, observations, pseudo_derivative):
        raw_modulator = self._raw_modulator(observations, pseudo_derivative)
        channel_ids = self._channel_ids()
        projection = tf.one_hot(channel_ids, self.n_channels, dtype=tf.float32)
        channel_sums = tf.einsum("btn,nk->btk", raw_modulator, projection)
        channel_counts = tf.maximum(tf.reduce_sum(projection, axis=0), 1.0)
        channel_modulators = channel_sums / channel_counts
        return tf.gather(channel_modulators, channel_ids, axis=2)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "name": self.module(),
                "n_channels": self.n_channels,
                "signal": self.signal,
                "channel_projection": self.channel_projection,
            }
        )
        return config
