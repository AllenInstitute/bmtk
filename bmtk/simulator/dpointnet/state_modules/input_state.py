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
        self._spikes_itrs = None
        self._last_state = None
        self._reuse_last_state = False

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

    def get_state(self, max_retries=8, **kwargs):
        if self._reuse_last_state and self._last_state is not None:
            return self._last_state

        last_err = None
        self.state_model = self.rnn.state_only_model
        for attempt in range(max_retries):
            try:
                spikes_inputs, _ = self.spikes_itrs.next_spikes()
                state_out = self.state_model([spikes_inputs, self.init_state])
                self._last_state = state_out
                if last_err is not None:
                    self._reuse_last_state = True
                    io.log_warning(
                        f'{self.__class__.__name__}: recovered after transient get_state() '
                        'failure; reusing this initial state for subsequent epochs.'
                    )
                return state_out
            except (tf.errors.InvalidArgumentError, tf.errors.UnknownError) as exc:
                last_err = exc
                io.log_debug(
                    f'{self.__class__.__name__}: get_state() raised {type(exc).__name__} '
                    f'(attempt {attempt + 1}/{max_retries}); rebuilding iterator and retrying.'
                )
                try:
                    self.spikes_itrs.close()
                    self.spikes_itrs.build()
                except Exception:
                    pass

        if self._last_state is not None:
            self._reuse_last_state = True
            io.log_warning(
                f'{self.__class__.__name__}: get_state() failed after {max_retries} attempts; '
                'reusing the last successfully generated initial state.'
            )
            return self._last_state

        raise last_err
