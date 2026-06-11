import numpy as np
import tensorflow as tf

from .inputs_base import InputsGeneratorMod


class PoissonSpikes(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        input_network.input_type = 'spikes'
        self._firing_rate = kwargs.get('firing_rate', 250.0)
        self._seed = kwargs.get('seed', None)
        self._n_nodes = input_network.n_nodes

    @staticmethod
    def module():
        return 'poisson_spikes'

    @staticmethod
    def input_type():
        return 'spikes'

    def create_generator(self, seq_len=None, dt=1.0, dtype=tf.float32, **kwargs):
        _seq_len = seq_len or self.rnn.adjusted_seq_len
        lam = self._firing_rate * dt / 1000.0
        rng = np.random.default_rng(self._seed)
        np_dtype = tf.as_dtype(dtype).as_numpy_dtype

        def _generator():
            while True:
                spikes = rng.poisson(lam=lam, size=(_seq_len, self._n_nodes)).astype(
                    np_dtype
                )
                yield spikes, {'firing_rate': self._firing_rate}

        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(_seq_len, self._n_nodes), dtype=dtype),
                {'firing_rate': tf.TensorSpec(shape=(), dtype=tf.float32)},
            ),
        )