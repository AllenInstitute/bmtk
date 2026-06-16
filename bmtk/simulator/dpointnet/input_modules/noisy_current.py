from .inputs_base import InputsGeneratorMod


class PoissonSpikesInternal(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, firing_rate, **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        input_network.input_type = 'current'
        self.input_network.options['input_type'] = 'poisson_spikes_internal'
        self.input_network.options['firing_rate'] = firing_rate

    @staticmethod
    def module():
        return 'poisson_spikes_internal'
    
    @staticmethod
    def input_type():
        return 'spikes'
    
    def create_generator(self, **kwargs):
        return None


class NoisyCurrent(PoissonSpikesInternal):
    @staticmethod
    def module():
        return 'noisy_current'
