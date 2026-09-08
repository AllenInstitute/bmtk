import tensorflow as tf

from .eprop import EPropLearningRule


def _find_target(targets, key):
    if isinstance(targets, dict):
        if key in targets:
            return targets[key]
        for value in targets.values():
            found = _find_target(value, key)
            if found is not None:
                return found
    elif isinstance(targets, (list, tuple)):
        for value in targets:
            found = _find_target(value, key)
            if found is not None:
                return found
    return None


class NeuronLocalThreeFactorLearningRule(EPropLearningRule):
    """Eligibility updates with target errors local to each postsynaptic neuron."""

    _MODULATORS = ("postsynaptic_rate_error", "postsynaptic_voltage_error")
    _CUE_MEMORY_SIGNALS = ("spike", "surrogate")
    _CUE_MEMORY_MODES = ("postsynaptic_gate", "synaptic_eligibility")
    _TARGET_SCOPES = ("all_readout", "target_pool_only")

    def __init__(
        self,
        modulator="postsynaptic_rate_error",
        pool_a_start=0,
        pool_a_end=100,
        pool_b_start=100,
        pool_b_end=200,
        cue_duration_ms=20.0,
        response_window_ms=50.0,
        high_target_rate_hz=20.0,
        low_target_rate_hz=0.0,
        target_rate_hz=None,
        tau_rate_ms=20.0,
        high_target_voltage_offset=0.0,
        low_target_voltage_offset=-1.0,
        cue_memory_tau_ms=50.0,
        cue_memory_gain=0.0,
        cue_memory_floor=1.0,
        cue_memory_signal="surrogate",
        cue_memory_mode="postsynaptic_gate",
        target_scope="all_readout",
        **kwargs,
    ):
        if modulator not in self._MODULATORS:
            raise ValueError(
                f'Unknown neuron-local modulator "{modulator}". Supported: '
                f'{", ".join(self._MODULATORS)}.'
            )
        if not (0 <= pool_a_start < pool_a_end <= pool_b_start < pool_b_end):
            raise ValueError("Readout pools must be non-empty, ordered, and disjoint.")
        if cue_duration_ms < 0.0 or response_window_ms <= 0.0:
            raise ValueError(
                "cue_duration_ms must be non-negative and response_window_ms positive."
            )
        if high_target_rate_hz < 0.0 or low_target_rate_hz < 0.0:
            raise ValueError("Target firing rates must be non-negative.")
        if target_rate_hz is not None and target_rate_hz < 0.0:
            raise ValueError("target_rate_hz must be non-negative.")
        if tau_rate_ms <= 0.0:
            raise ValueError("tau_rate_ms must be greater than zero.")
        if cue_memory_tau_ms <= 0.0:
            raise ValueError("cue_memory_tau_ms must be greater than zero.")
        if cue_memory_gain < 0.0 or cue_memory_floor < 0.0:
            raise ValueError(
                "cue_memory_gain and cue_memory_floor must be non-negative."
            )
        if cue_memory_signal not in self._CUE_MEMORY_SIGNALS:
            raise ValueError(
                f'Unknown cue-memory signal "{cue_memory_signal}". Supported: '
                f'{", ".join(self._CUE_MEMORY_SIGNALS)}.'
            )
        if cue_memory_mode not in self._CUE_MEMORY_MODES:
            raise ValueError(
                f'Unknown cue-memory mode "{cue_memory_mode}". Supported: '
                f'{", ".join(self._CUE_MEMORY_MODES)}.'
            )
        if target_scope not in self._TARGET_SCOPES:
            raise ValueError(
                f'Unknown target scope "{target_scope}". Supported: '
                f'{", ".join(self._TARGET_SCOPES)}.'
            )
        self.modulator = modulator
        self.pool_a_start = int(pool_a_start)
        self.pool_a_end = int(pool_a_end)
        self.pool_b_start = int(pool_b_start)
        self.pool_b_end = int(pool_b_end)
        self.cue_duration_ms = float(cue_duration_ms)
        self.response_window_ms = float(response_window_ms)
        self.high_target_rate_hz = float(high_target_rate_hz)
        self.low_target_rate_hz = float(low_target_rate_hz)
        self.target_rate_hz = None if target_rate_hz is None else float(target_rate_hz)
        self.tau_rate_ms = float(tau_rate_ms)
        self.high_target_voltage_offset = float(high_target_voltage_offset)
        self.low_target_voltage_offset = float(low_target_voltage_offset)
        self.cue_memory_tau_ms = float(cue_memory_tau_ms)
        self.cue_memory_gain = float(cue_memory_gain)
        self.cue_memory_floor = float(cue_memory_floor)
        self.cue_memory_signal = cue_memory_signal
        self.cue_memory_mode = cue_memory_mode
        self.target_scope = target_scope
        super().__init__(name="neuron_local_three_factor", **kwargs)

    @classmethod
    def module(cls):
        return "neuron_local_three_factor"

    def build(self, rnn):
        if self.pool_b_end > rnn._cell._n_neurons:
            raise ValueError("Readout pools exceed the number of network neurons.")
        super().build(rnn)

    @staticmethod
    def _trace(spikes, decay):
        by_time = tf.transpose(spikes, [1, 0, 2])
        initial = tf.zeros_like(by_time[0])
        traced = tf.scan(lambda state, value: decay * state + value, by_time, initial)
        return tf.transpose(traced, [1, 0, 2])

    def _target_tensors(self, observations):
        labels = _find_target(observations.targets, "class_label")
        delays = _find_target(observations.targets, "delay_ms")
        batch_size = tf.shape(observations.spikes, out_type=tf.int32)[0]
        seq_len = tf.shape(observations.spikes, out_type=tf.int32)[1]
        if labels is None and delays is None and self.target_rate_hz is not None:
            high_pool = tf.ones(
                [batch_size, self.rnn._cell._n_neurons], dtype=tf.float32
            )
            return (
                high_pool,
                tf.ones([1, 1, self.rnn._cell._n_neurons], dtype=tf.float32),
                tf.ones([batch_size, seq_len, 1], dtype=tf.float32),
                self.target_rate_hz,
            )
        if labels is None or delays is None:
            raise ValueError(
                "neuron_local_three_factor requires class_label and delay_ms targets."
            )
        labels = tf.reshape(tf.cast(labels, tf.int32), [-1])
        delays = tf.reshape(tf.cast(delays, tf.float32), [-1])
        if labels.shape.rank != 1 or delays.shape.rank != 1:
            raise ValueError("class_label and delay_ms targets must be vectors.")
        tf.debugging.assert_equal(tf.shape(labels)[0], batch_size)
        tf.debugging.assert_equal(tf.shape(delays)[0], batch_size)

        neuron_ids = tf.range(self.rnn._cell._n_neurons, dtype=tf.int32)
        in_pool_a = tf.logical_and(
            neuron_ids >= self.pool_a_start, neuron_ids < self.pool_a_end
        )
        in_pool_b = tf.logical_and(
            neuron_ids >= self.pool_b_start, neuron_ids < self.pool_b_end
        )
        readout_mask = tf.cast(tf.logical_or(in_pool_a, in_pool_b), tf.float32)
        target_pool_b = tf.cast(labels[:, tf.newaxis] == 1, tf.float32)
        high_pool = tf.where(
            target_pool_b > 0.0,
            tf.cast(in_pool_b[tf.newaxis, :], tf.float32),
            tf.cast(in_pool_a[tf.newaxis, :], tf.float32),
        )
        if self.target_scope == "target_pool_only":
            readout_mask = high_pool
        else:
            readout_mask = readout_mask[tf.newaxis, :]

        dt = tf.cast(self.rnn._cell._dt, tf.float32)
        time_ms = tf.cast(tf.range(seq_len), tf.float32) * dt
        response_start = self.cue_duration_ms + delays
        response_mask = tf.logical_and(
            time_ms[tf.newaxis, :] >= response_start[:, tf.newaxis],
            time_ms[tf.newaxis, :]
            < response_start[:, tf.newaxis] + self.response_window_ms,
        )
        return (
            high_pool,
            readout_mask[:, tf.newaxis, :],
            tf.cast(response_mask[:, :, tf.newaxis], tf.float32),
            self.high_target_rate_hz,
        )

    def _cue_memory(self, observations, pseudo_derivative):
        dt = tf.cast(self.rnn._cell._dt, tf.float32)
        decay = tf.exp(-dt / tf.cast(self.cue_memory_tau_ms, tf.float32))
        if self.cue_memory_signal == "spike":
            signal = tf.cast(observations.spikes, tf.float32)
        else:
            signal = tf.cast(pseudo_derivative, tf.float32)
        seq_len = tf.shape(signal, out_type=tf.int32)[1]
        time_ms = tf.cast(tf.range(seq_len), tf.float32) * dt
        cue_mask = tf.cast(
            time_ms[tf.newaxis, :, tf.newaxis] < self.cue_duration_ms,
            tf.float32,
        )
        return (1.0 - decay) * self._trace(signal * cue_mask, decay)

    def _local_factor(self, observations, pseudo_derivative):
        high_pool, readout_mask, response_mask, high_target_rate_hz = (
            self._target_tensors(observations)
        )
        if self.modulator == "postsynaptic_rate_error":
            dt = tf.cast(self.rnn._cell._dt, tf.float32)
            decay = tf.exp(-dt / tf.cast(self.tau_rate_ms, tf.float32))
            spikes = tf.cast(observations.spikes, tf.float32)
            observed_rate = (1.0 - decay) * self._trace(spikes, decay)
            high_probability = high_target_rate_hz * dt / 1000.0
            low_probability = self.low_target_rate_hz * dt / 1000.0
            target = low_probability + high_pool * (high_probability - low_probability)
            error = observed_rate - target[:, tf.newaxis, :]
            local_factor = error * tf.cast(pseudo_derivative, tf.float32)
        else:
            threshold = tf.cast(self.rnn._cell.v_th, tf.float32)
            low_target = threshold + self.low_target_voltage_offset
            high_target = threshold + self.high_target_voltage_offset
            target = low_target + high_pool * (high_target - low_target)
            local_factor = (
                tf.cast(observations.voltages, tf.float32) - target[:, tf.newaxis, :]
            )
        if self.cue_memory_mode == "postsynaptic_gate" and (
            self.cue_memory_gain > 0.0 or self.cue_memory_floor != 1.0
        ):
            memory = self._cue_memory(observations, pseudo_derivative)
            local_factor *= self.cue_memory_floor + self.cue_memory_gain * memory
        return local_factor * readout_mask * response_mask

    def _synaptic_cue_gradients(self, surface, observations, pseudo_derivative):
        indices = tf.cast(surface.indices, tf.int32)
        n_edges = tf.shape(indices, out_type=tf.int32)[0]
        chunk_size = tf.constant(self.edge_chunk_size, dtype=tf.int32)
        batch_size = tf.shape(observations.spikes, out_type=tf.int32)[0]
        seq_len = tf.shape(observations.spikes, out_type=tf.int32)[1]
        local_factor = self._local_factor(observations, pseudo_derivative)
        if self.learning_signal_clip is not None:
            clip = tf.cast(self.learning_signal_clip, local_factor.dtype)
            local_factor = tf.clip_by_value(local_factor, -clip, clip)
        local_factor_by_time = tf.transpose(local_factor, [1, 2, 0])
        post_memory_signal = (
            tf.cast(observations.spikes, tf.float32)
            if self.cue_memory_signal == "spike"
            else tf.cast(pseudo_derivative, tf.float32)
        )
        post_memory_by_time = tf.transpose(post_memory_signal, [1, 2, 0])
        dt = tf.cast(self.rnn._cell._dt, tf.float32)
        decay = tf.exp(-dt / tf.cast(self.cue_memory_tau_ms, tf.float32))

        if surface.source == "recurrent":
            n_neurons = self.rnn._cell._n_neurons
            max_delay = self.rnn._cell.max_delay
            initial_history = tf.reverse(
                tf.reshape(
                    observations.initial_state[0],
                    [batch_size, max_delay, n_neurons],
                ),
                axis=[1],
            )
            pre_history = tf.concat(
                [initial_history, tf.cast(observations.spikes, tf.float32)],
                axis=1,
            )
            pre_by_time = tf.transpose(pre_history, [1, 2, 0])
        else:
            input_props = self.rnn._cell.inputs[surface.source]
            input_index = list(self.rnn._cell.inputs).index(surface.source)
            begin = self.rnn._cell.inputs_idx[input_index]
            end = begin + input_props["input_dim"]
            pre_by_time = tf.transpose(
                tf.cast(observations.input_spikes[:, :, begin:end], tf.float32),
                [1, 2, 0],
            )
            max_delay = 0

        gradients = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            infer_shape=False,
            element_shape=tf.TensorShape([None]),
        )

        def chunk_condition(start, chunk_index, result):
            del chunk_index, result
            return start < n_edges

        def chunk_body(start, chunk_index, result):
            end = tf.minimum(start + chunk_size, n_edges)
            chunk_indices = indices[start:end]
            post_ids = chunk_indices[:, 0]
            raw_pre_ids = chunk_indices[:, 1]
            if surface.source == "recurrent":
                delays = raw_pre_ids // self.rnn._cell._n_neurons
                pre_ids = raw_pre_ids % self.rnn._cell._n_neurons
            else:
                delays = tf.zeros_like(raw_pre_ids)
                pre_ids = raw_pre_ids
            edge_count = end - start
            cue_eligibility = tf.zeros([batch_size, edge_count], dtype=tf.float32)
            gradient = tf.zeros([edge_count], dtype=tf.float32)

            def time_condition(time, eligibility, edge_gradient):
                del eligibility, edge_gradient
                return time < seq_len

            def time_body(time, eligibility, edge_gradient):
                post_factor = tf.transpose(
                    tf.gather(local_factor_by_time[time], post_ids)
                )
                if surface.source == "recurrent":
                    source_times = time + max_delay - 1 - delays
                    pre_spikes = tf.transpose(
                        tf.gather_nd(
                            pre_by_time,
                            tf.stack([source_times, pre_ids], axis=1),
                        )
                    )
                    source_event_times = time - 1 - delays
                    memory_times = tf.maximum(source_event_times, 0)
                    post_memory = tf.transpose(
                        tf.gather_nd(
                            post_memory_by_time,
                            tf.stack([memory_times, post_ids], axis=1),
                        )
                    )
                else:
                    pre_spikes = tf.transpose(tf.gather(pre_by_time[time], pre_ids))
                    source_event_times = tf.fill(tf.shape(pre_ids), time)
                    post_memory = tf.transpose(
                        tf.gather(post_memory_by_time[time], post_ids)
                    )
                cue_event = tf.logical_and(
                    source_event_times >= 0,
                    tf.cast(source_event_times, tf.float32) * dt < self.cue_duration_ms,
                )
                new_eligibility = (
                    decay * eligibility
                    + pre_spikes
                    * post_memory
                    * tf.cast(cue_event[tf.newaxis, :], tf.float32)
                )
                edge_gradient += tf.reduce_mean(post_factor * new_eligibility, axis=0)
                return time + 1, new_eligibility, edge_gradient

            _, _, gradient = tf.while_loop(
                time_condition,
                time_body,
                (tf.constant(0, tf.int32), cue_eligibility, gradient),
                parallel_iterations=1,
            )
            return end, chunk_index + 1, result.write(chunk_index, gradient)

        _, _, gradients = tf.while_loop(
            chunk_condition,
            chunk_body,
            (tf.constant(0, tf.int32), tf.constant(0, tf.int32), gradients),
            parallel_iterations=1,
        )
        gradient = gradients.concat()
        if self.gradient_clip_norm is not None:
            gradient = tf.clip_by_norm(gradient, self.gradient_clip_norm)
        return tf.cast(gradient, surface.variable.dtype)

    def compute_updates(self, observations):
        pseudo_derivative = self._surrogate_derivative(observations)
        if self.cue_memory_mode == "synaptic_eligibility":
            return tuple(
                (
                    surface,
                    tf.cast(self.cue_memory_floor, surface.variable.dtype)
                    * self._edge_gradients(surface, observations, pseudo_derivative)
                    + tf.cast(self.cue_memory_gain, surface.variable.dtype)
                    * self._synaptic_cue_gradients(
                        surface, observations, pseudo_derivative
                    ),
                )
                for surface in self.weight_surfaces
            )
        return tuple(
            (
                surface,
                self._edge_gradients(surface, observations, pseudo_derivative),
            )
            for surface in self.weight_surfaces
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "name": self.module(),
                "modulator": self.modulator,
                "pool_a_start": self.pool_a_start,
                "pool_a_end": self.pool_a_end,
                "pool_b_start": self.pool_b_start,
                "pool_b_end": self.pool_b_end,
                "cue_duration_ms": self.cue_duration_ms,
                "response_window_ms": self.response_window_ms,
                "high_target_rate_hz": self.high_target_rate_hz,
                "low_target_rate_hz": self.low_target_rate_hz,
                "target_rate_hz": self.target_rate_hz,
                "tau_rate_ms": self.tau_rate_ms,
                "high_target_voltage_offset": self.high_target_voltage_offset,
                "low_target_voltage_offset": self.low_target_voltage_offset,
                "cue_memory_tau_ms": self.cue_memory_tau_ms,
                "cue_memory_gain": self.cue_memory_gain,
                "cue_memory_floor": self.cue_memory_floor,
                "cue_memory_signal": self.cue_memory_signal,
                "cue_memory_mode": self.cue_memory_mode,
                "target_scope": self.target_scope,
            }
        )
        return config
