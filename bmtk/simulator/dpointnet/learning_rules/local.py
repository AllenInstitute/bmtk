import tensorflow as tf

from .eprop import EPropLearningRule


class LocalTraceLearningRule(EPropLearningRule):
    """Weight-surface support for rules using only pre/post spike traces."""

    def _edge_spikes(self, surface, observations):
        indices = tf.cast(surface.indices, tf.int32)
        post_ids = indices[:, 0]
        raw_pre_ids = indices[:, 1]
        post_spikes = tf.gather(
            tf.cast(observations.spikes, tf.float32), post_ids, axis=2
        )

        if surface.source != "recurrent":
            input_props = self.rnn._cell.inputs[surface.source]
            input_index = list(self.rnn._cell.inputs).index(surface.source)
            begin = self.rnn._cell.inputs_idx[input_index]
            end = begin + input_props["input_dim"]
            pre_spikes = tf.cast(observations.input_spikes[:, :, begin:end], tf.float32)
            return tf.gather(pre_spikes, raw_pre_ids, axis=2), post_spikes

        n_neurons = self.rnn._cell._n_neurons
        max_delay = self.rnn._cell.max_delay
        delays = raw_pre_ids // n_neurons
        pre_ids = raw_pre_ids % n_neurons
        batch_size = tf.shape(observations.spikes, out_type=tf.int32)[0]
        seq_len = tf.shape(observations.spikes, out_type=tf.int32)[1]
        initial_history = tf.reverse(
            tf.reshape(
                observations.initial_state[0],
                [batch_size, max_delay, n_neurons],
            ),
            axis=[1],
        )
        history = tf.concat(
            [initial_history, tf.cast(observations.spikes, tf.float32)], axis=1
        )
        history = tf.transpose(history, [1, 2, 0])
        source_times = (
            tf.range(seq_len, dtype=tf.int32)[:, tf.newaxis]
            + max_delay
            - 1
            - delays[tf.newaxis, :]
        )
        tiled_pre_ids = tf.broadcast_to(pre_ids[tf.newaxis, :], tf.shape(source_times))
        pre_spikes = tf.gather_nd(
            history, tf.stack([source_times, tiled_pre_ids], axis=-1)
        )
        return tf.transpose(pre_spikes, [2, 0, 1]), post_spikes

    @staticmethod
    def _trace(spikes, decay):
        by_time = tf.transpose(spikes, [1, 0, 2])
        initial = tf.zeros_like(by_time[0])
        traced = tf.scan(lambda state, value: decay * state + value, by_time, initial)
        return tf.transpose(traced, [1, 0, 2])


class PairSTDPLearningRule(LocalTraceLearningRule):
    """Pair-based STDP using only delayed pre- and postsynaptic spikes."""

    _WEIGHT_DEPENDENCE = ("additive", "multiplicative")

    def __init__(
        self,
        tau_pre_ms=20.0,
        tau_post_ms=20.0,
        a_plus=1.0,
        a_minus=1.0,
        weight_dependence="additive",
        **kwargs,
    ):
        if tau_pre_ms <= 0.0 or tau_post_ms <= 0.0:
            raise ValueError("STDP trace time constants must be greater than zero.")
        if weight_dependence not in self._WEIGHT_DEPENDENCE:
            raise ValueError(
                f'Unknown STDP weight dependence "{weight_dependence}". Supported: '
                f'{", ".join(self._WEIGHT_DEPENDENCE)}.'
            )
        self.tau_pre_ms = float(tau_pre_ms)
        self.tau_post_ms = float(tau_post_ms)
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.weight_dependence = weight_dependence
        super().__init__(name="pair_stdp", **kwargs)

    @classmethod
    def module(cls):
        return "pair_stdp"

    def compute_updates(self, observations):
        dt = tf.cast(self.rnn._cell._dt, tf.float32)
        pre_decay = tf.exp(-dt / tf.cast(self.tau_pre_ms, tf.float32))
        post_decay = tf.exp(-dt / tf.cast(self.tau_post_ms, tf.float32))
        updates = []
        for surface in self.weight_surfaces:
            pre_spikes, post_spikes = self._edge_spikes(surface, observations)
            pre_trace = self._trace(pre_spikes, pre_decay)
            post_trace = self._trace(post_spikes, post_decay)
            potentiation = self.a_plus * pre_trace * post_spikes
            depression = self.a_minus * pre_spikes * post_trace
            if self.weight_dependence == "multiplicative":
                scale = 1.0 / (1.0 + tf.abs(tf.cast(surface.variable, tf.float32)))
                potentiation *= scale[tf.newaxis, tf.newaxis, :]
                depression *= scale[tf.newaxis, tf.newaxis, :]
            weight_delta = tf.reduce_mean(
                tf.reduce_sum(potentiation - depression, axis=1), axis=0
            )
            updates.append((surface, tf.cast(-weight_delta, surface.variable.dtype)))
        return tuple(updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "name": self.module(),
                "tau_pre_ms": self.tau_pre_ms,
                "tau_post_ms": self.tau_post_ms,
                "a_plus": self.a_plus,
                "a_minus": self.a_minus,
                "weight_dependence": self.weight_dependence,
            }
        )
        return config


class LocalRateHomeostasisLearningRule(LocalTraceLearningRule):
    """Strict-local homeostasis from presynaptic and postsynaptic rate traces."""

    _UPDATE_MODES = (
        "pre_trace_times_post_rate_error",
        "multiplicative_post_rate_scaling",
    )

    def __init__(
        self,
        target_rate_hz,
        tau_pre_ms=20.0,
        tau_rate_ms=100.0,
        update="pre_trace_times_post_rate_error",
        **kwargs,
    ):
        if target_rate_hz < 0.0:
            raise ValueError("target_rate_hz must be non-negative.")
        if tau_pre_ms <= 0.0 or tau_rate_ms <= 0.0:
            raise ValueError(
                "Homeostasis trace time constants must be greater than zero."
            )
        if update not in self._UPDATE_MODES:
            raise ValueError(
                f'Unknown local-rate update "{update}". Supported: '
                f'{", ".join(self._UPDATE_MODES)}.'
            )
        self.target_rate_hz = float(target_rate_hz)
        self.tau_pre_ms = float(tau_pre_ms)
        self.tau_rate_ms = float(tau_rate_ms)
        self.update = update
        super().__init__(name="local_rate_homeostasis", **kwargs)

    @classmethod
    def module(cls):
        return "local_rate_homeostasis"

    def compute_updates(self, observations):
        dt = tf.cast(self.rnn._cell._dt, tf.float32)
        pre_decay = tf.exp(-dt / tf.cast(self.tau_pre_ms, tf.float32))
        rate_decay = tf.exp(-dt / tf.cast(self.tau_rate_ms, tf.float32))
        target_probability = tf.cast(self.target_rate_hz, tf.float32) * dt / 1000.0
        updates = []
        for surface in self.weight_surfaces:
            pre_spikes, post_spikes = self._edge_spikes(surface, observations)
            post_rate = (1.0 - rate_decay) * self._trace(post_spikes, rate_decay)
            rate_error = target_probability - post_rate
            if self.update == "pre_trace_times_post_rate_error":
                local_drive = self._trace(pre_spikes, pre_decay) * rate_error
            else:
                weight = tf.cast(surface.variable, tf.float32)
                local_drive = weight[tf.newaxis, tf.newaxis, :] * rate_error
            weight_delta = tf.reduce_mean(tf.reduce_sum(local_drive, axis=1), axis=0)
            updates.append((surface, tf.cast(-weight_delta, surface.variable.dtype)))
        return tuple(updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "name": self.module(),
                "target_rate_hz": self.target_rate_hz,
                "tau_pre_ms": self.tau_pre_ms,
                "tau_rate_ms": self.tau_rate_ms,
                "update": self.update,
            }
        )
        return config
