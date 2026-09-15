import numpy as np
import tensorflow as tf

from .inputs_base import InputsGeneratorMod


class PoissonSpikes(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        input_network.input_type = "spikes"
        firing_rate = np.asarray(kwargs.get("firing_rate", 250.0), dtype=float)
        if firing_rate.ndim > 1 or firing_rate.size == 0:
            raise ValueError(
                "firing_rate must be a scalar or non-empty one-dimensional sequence."
            )
        if np.any(firing_rate < 0.0):
            raise ValueError("firing_rate values must be non-negative.")
        self._firing_rates = firing_rate.reshape(-1)
        self._seed = kwargs.get("seed", None)
        self._n_nodes = input_network.n_nodes

    @staticmethod
    def module():
        return "poisson_spikes"

    @staticmethod
    def input_type():
        return "spikes"

    def create_generator(self, seq_len=None, dt=1.0, dtype=tf.float32, **kwargs):
        _seq_len = seq_len or self.rnn.adjusted_seq_len
        rng = np.random.default_rng(self._seed)
        np_dtype = tf.as_dtype(dtype).as_numpy_dtype

        def _generator():
            while True:
                firing_rate = float(rng.choice(self._firing_rates))
                lam = firing_rate * dt / 1000.0
                spikes = rng.poisson(lam=lam, size=(_seq_len, self._n_nodes)).astype(
                    np_dtype
                )
                yield spikes, {"firing_rate": firing_rate}

        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(_seq_len, self._n_nodes), dtype=dtype),
                {"firing_rate": tf.TensorSpec(shape=(), dtype=tf.float32)},
            ),
        )
