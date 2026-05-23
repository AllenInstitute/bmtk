from .inputs_base import InputsGeneratorMod


class NoisyCurrent(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        input_network.input_type = 'current'

    @staticmethod
    def module():
        return 'noisy_current'
    
    @staticmethod
    def input_type():
        return 'spikes'
    
    def create_generator(self, **kwargs):
        return None
