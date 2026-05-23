import tensorflow as tf
from copy import copy

from bmtk.simulator.dpointnet.data_iterator import DataIterator


class InitStateFromInputModule:
    def __init__(self, rnn, **kwargs):
        self.rnn = rnn
        self._input_mods = {}

        for name, mod in self.rnn.parse_input_mods_from_config(kwargs['inputs']):
            self._input_mods[name] = mod

        self._init_state = None
        self._spikes_itrs = None

    @property
    def init_state(self):
        if self._init_state is None:
            self._init_state = self.rnn.cell.zero_state(
                self.rnn.adjusted_batch_size, 
                self.rnn.dtype,
                with_names=False
            )
        return self._init_state

    @property
    def spikes_itrs(self):
        if self._spikes_itrs is None:
            self._spikes_itrs = DataIterator(
                input_mods=list(self._input_mods.values()), 
                batch_size=self.rnn.adjusted_batch_size,
                seq_len=self.rnn.seq_len,
                ordered_populations=self.rnn.ordered_inputs_populations
            )

        return self._spikes_itrs

    def get_state(self, **kwargs):
        spikes_inputs, _ = self.spikes_itrs.next_spikes()
        self.state_model = self.rnn.state_only_model
        state_out = self.state_model([spikes_inputs, self.init_state])
        return state_out
