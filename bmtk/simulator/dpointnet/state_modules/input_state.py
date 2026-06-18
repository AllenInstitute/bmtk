import tensorflow as tf
from copy import copy

from bmtk.simulator.dpointnet.data_iterator import DataIterator
from bmtk.simulator.dpointnet.io_tools import io


class InitStateFromInputModule:
    def __init__(self, rnn, **kwargs):
        self.rnn = rnn
        self._input_mods = {}

        for name, mod in self.rnn.parse_input_mods_from_config(kwargs['inputs']):
            self._input_mods[name] = mod

        self._init_state = None
        self._init_state_batch_size = None
        self._spikes_itrs = None
        self._spikes_itrs_batch_size = None
        self._last_state = None

    def init_state(self, batch_size=None):
        batch_size = batch_size or self.rnn.adjusted_batch_size
        if self._init_state is None or self._init_state_batch_size != batch_size:
            self._init_state = self.rnn.cell.zero_state(
                batch_size,
                self.rnn.dtype,
                with_names=False
            )
            self._init_state_batch_size = batch_size
        return self._init_state

    def spikes_itrs(self, batch_size=None):
        batch_size = batch_size or self.rnn.adjusted_batch_size
        if self._spikes_itrs is None or self._spikes_itrs_batch_size != batch_size:
            if self._spikes_itrs is not None:
                self._spikes_itrs.close()
            self._spikes_itrs = DataIterator(
                input_mods=list(self._input_mods.values()), 
                batch_size=batch_size,
                seq_len=self.rnn.seq_len,
                ordered_populations=self.rnn.ordered_inputs_populations
            )
            self._spikes_itrs_batch_size = batch_size

        return self._spikes_itrs

    def get_state(self, max_retries=8, batch_size=None, **kwargs):
        batch_size = batch_size or self.rnn.adjusted_batch_size
        last_err = None
        self.state_model = self.rnn.state_only_model
        for attempt in range(max_retries):
            try:
                spikes_inputs, _ = self.spikes_itrs(batch_size).next_spikes()
                state_out = self.state_model([spikes_inputs, self.init_state(batch_size)])
                self._last_state = state_out
                return state_out
            except (tf.errors.InvalidArgumentError, tf.errors.UnknownError) as exc:
                last_err = exc
                io.log_debug(
                    f'{self.__class__.__name__}: get_state() raised {type(exc).__name__} '
                    f'(attempt {attempt + 1}/{max_retries}); rebuilding iterator and retrying.'
                )
                try:
                    self.spikes_itrs(batch_size).close()
                    self.spikes_itrs(batch_size).build()
                except Exception:
                    pass

        if self._last_state is not None:
            io.log_warning(
                f'{self.__class__.__name__}: get_state() failed after {max_retries} attempts; '
                'reusing the last successfully generated initial state.'
            )
            return self._last_state

        raise last_err
