import tensorflow as tf
from copy import copy

from bmtk.simulator.dpointnet.data_iterator import DataIterator
from bmtk.simulator.dpointnet.io_tools import io


def _complete_noise_state(state_out, initial_state, sequence_length):
    state_out = tuple(state_out)
    initial_state = tuple(initial_state)
    if len(state_out) == len(initial_state):
        return state_out
    if len(state_out) != len(initial_state) - 1:
        raise ValueError(
            f"State-only rollout returned {len(state_out)} states; "
            f"expected {len(initial_state)}."
        )
    if len(state_out) > 6 and tf.as_dtype(state_out[6].dtype).is_integer:
        raise ValueError(
            "Rollout is missing external delay history, not the noise step"
        )
    noise_step = initial_state[6] + tf.cast(sequence_length, tf.int32)
    return state_out[:6] + (noise_step,) + state_out[6:]


class InitStateFromInputModule:
    def __init__(self, rnn, **kwargs):
        self.rnn = rnn
        self._input_mods = {}

        for name, mod in self.rnn.parse_input_mods_from_config(kwargs["inputs"]):
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
                batch_size, self.rnn.dtype, with_names=False
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
                ordered_populations=self.rnn.ordered_inputs_populations,
                recover_input_errors=True,
            )
            self._spikes_itrs_batch_size = batch_size

        return self._spikes_itrs

    def get_state(self, max_retries=8, batch_size=None, **kwargs):
        batch_size = batch_size or self.rnn.adjusted_batch_size
        last_err = None
        diagnostic_path = None
        self.state_model = self.rnn.state_only_model
        for attempt in range(max_retries):
            try:
                spikes_inputs, _ = self.spikes_itrs(batch_size).next_spikes()
                self.rnn.cell.advance_noise_seed()
                initial_state = self.init_state(batch_size)
                state_out = self.state_model(
                    self.rnn.model_inputs(spikes_inputs, initial_state)
                )
                state_out = _complete_noise_state(
                    state_out,
                    initial_state,
                    tf.shape(spikes_inputs)[1],
                )
                self._last_state = state_out
                if last_err is not None:
                    details = (
                        f" Details: {diagnostic_path}."
                        if diagnostic_path is not None else ""
                    )
                    io.log_info(
                        "Initial-state input generation recovered; continuing normally."
                        + details
                    )
                return state_out
            except (tf.errors.InvalidArgumentError, tf.errors.UnknownError) as exc:
                last_err = exc
                diagnostic_path = io.save_exception(
                    exc,
                    f"{self.__class__.__name__}: initial-state generation "
                    f"(attempt {attempt + 1}/{max_retries}, batch_size={batch_size})",
                )
                io.log_debug(
                    f"{self.__class__.__name__}: get_state() raised {type(exc).__name__} "
                    f"(attempt {attempt + 1}/{max_retries}); rebuilding iterator and retrying."
                )
                try:
                    self.spikes_itrs(batch_size).close()
                    self.spikes_itrs(batch_size).build()
                except tf.errors.OpError as rebuild_error:
                    io.save_exception(
                        rebuild_error,
                        f"{self.__class__.__name__}: iterator rebuild "
                        f"(attempt {attempt + 1}/{max_retries})",
                    )

        if self._last_state is not None:
            details = (
                f" Details: {diagnostic_path}."
                if diagnostic_path is not None else ""
            )
            io.log_warning(
                "Initial-state input generation was interrupted; continuing with "
                "the previous initial state." + details
            )
            return self._last_state

        io.log_error(
            "Unable to generate an initial state; no previous state is available."
            + (f" Details: {diagnostic_path}." if diagnostic_path is not None else "")
        )
        raise last_err
