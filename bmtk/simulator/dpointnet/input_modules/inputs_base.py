import tensorflow as tf


class InputsGeneratorMod:
    def __init__(self, rnn, name, input_network, **kwargs):
        self.rnn = rnn
        self.name = name
        self.input_network = input_network
        self.population_name = input_network.name

    @staticmethod
    def module():
        raise NotImplementedError()
       
    @staticmethod
    def input_type():
        raise NotImplementedError()
    
    def create_generator(self, seq_len, dt=1.0, dtype=tf.float32):
        raise NotImplementedError()
