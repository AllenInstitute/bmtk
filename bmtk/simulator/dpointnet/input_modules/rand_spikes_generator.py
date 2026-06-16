from glob import glob
import numpy as np
import tensorflow as tf

from bmtk.simulator.dpointnet.io_tools import io
from .inputs_base import InputsGeneratorMod



class BernoulliSpikes(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, firing_rate, **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        self._n_nodes = self.input_network.n_nodes
        self.input_network.options['input_type'] = 'spikes'

        self._firing_rate = firing_rate
        self._dtype = kwargs.get('dtype', None)
        self._dt = kwargs.get('dt', None)
        self._lam = None       
        self.rng = np.random.default_rng(seed=3000)

    @staticmethod
    def module():
        return 'bernoulli_spikes'
    
    @staticmethod
    def input_type():
        return 'spikes'
    
    def create_generator(self, seq_len=None, dt=1.0, dtype=tf.float32, **kwargs):
        self._dt = self._dt or dt
        _dtype = self._dtype or dtype
        _seq_len = seq_len or self.rnn.adjusted_seq_len
        self._lam = self._firing_rate*self._dt/1000.0

        def _generator():
            while True:
                spikes = self.rng.random((_seq_len, self._n_nodes)) <= self._lam  # .astype(np.bool)
                
                # spikes = np.random.rand(_seq_len, self._n_nodes) <= self._lam  # .astype(np.bool)
                yield spikes, {'firing_rate': self._firing_rate}

        data_set = tf.data.Dataset.from_generator(
            _generator, 
            output_signature=(
                tf.TensorSpec(shape=(_seq_len, self._n_nodes), dtype=_dtype),
                {
                    'firing_rate': tf.TensorSpec(shape=(), dtype=tf.float32),
                }
            )
        )
        return data_set


class RandomSpikesGenerator(BernoulliSpikes):
    @staticmethod
    def module():
        return 'random'
