"""Exact compute-temporal NEST adjoints from retained forward-state records."""

from contextlib import nullcontext
from types import SimpleNamespace

import tensorflow as tf
from tensorflow.python.eager import record

from .acceleration import record_rollout_acceleration
from .cell_models.glif3_cell import GLIF3Cell, _primal_with_vjp
from .cell_models.nest_dynamics import spike_reset
from .custom_ops.csr_spike_ops import _elide_vjp_producers
from .cell_models.state_rnn import (
    _internal_poisson_inputs, _prepare_cpu_poisson, _prepared_noise_step,
    _prepared_poisson_admitted, _rollout_context, _weight_device_scope,
    PREPARED_POISSON_AUTO_CEILING_BYTES, PREPARED_POISSON_BUDGET_BYTES,
)
from .segmented_recompute import _pack_spikes, _unpack_spikes
from .spike_statistics import SpikeStatistics, SpikeStatisticsRunner, _StatisticsCollector


@tf.function(jit_compile=True)
def _pack_record_flags(flags):
    shape = tf.shape(flags)
    packed = _pack_spikes(tf.reshape(flags, (-1, flags.shape[-1])))
    return tf.reshape(packed, (shape[0], shape[1], (flags.shape[-1] + 62) // 63))


@tf.function(jit_compile=True)
def _unpack_record_flags(packed, width, dtype):
    shape = tf.shape(packed)
    flags = _unpack_spikes(tf.reshape(packed, (-1, (width + 62) // 63)), width, dtype)
    return tf.reshape(flags, (shape[0], shape[1], width))


@tf.function(jit_compile=True)
def _restore_record_flags(initial_history, packed, neurons, length):
    history = tf.stop_gradient(initial_history)
    events = tf.ensure_shape(
        _unpack_record_flags(packed, 2 * neurons, history.dtype),
        (length, history.shape[0], 2 * neurons),
    )
    values = []
    for index in range(length):
        values.append(tf.concat((history, events[index]), axis=1))
        history = tf.concat((events[index, :, neurons:], history[:, :-neurons]), axis=1)
    return tf.stop_gradient(tf.stack(values))


@tf.function(jit_compile=True)
def _restore_asc(initial, flags, decay, amplitudes, refractory_decay, neurons, time_major=False):
    initial = tf.stop_gradient(initial)
    flags = tf.stop_gradient(flags)
    dtype = initial.dtype
    decay, amplitudes, refractory_decay = (
        tf.reshape(tf.cast(value, dtype), (neurons, 2))
        for value in (decay, amplitudes, refractory_decay)
    )
    values = []
    adaptation = tf.reshape(initial, (-1, neurons, 2))
    for index in range(flags.shape[0 if time_major else 1]):
        values.append(tf.reshape(adaptation, tf.shape(initial)))
        events = flags[index] if time_major else flags[:, index]
        active = events[:, :neurons] == 0
        adaptation = tf.where(active[..., None], adaptation * decay, adaptation)
        zeros = tf.zeros_like(events[:, :neurons])
        _, _, adaptation = spike_reset(
            zeros, zeros, adaptation, events[:, neurons:],
            reset_voltage=zeros, refractory_steps=zeros,
            asc_amplitudes=amplitudes, asc_refractory_decay=refractory_decay,
            hard_reset=True,
        )
    return tf.stop_gradient(tf.stack(values, axis=0 if time_major else 1))


def _records_supported(cell):
    if type(cell) is not GLIF3Cell:
        return ("unmodified GLIF3Cell",)
    conditions = {
        "NEST hard forward": cell.dynamics_mode == "nest" and cell._hard_reset,
        "compute temporal gradients": cell.temporal_gradient_precision == "compute",
        "native joint projection": cell._use_fused_cuda and cell._use_combined_current_sweep,
        "native state/history": cell._use_fused_state and cell._use_fused_state_history,
        "direct state loop": cell._use_direct_state_rnn_loop,
        "compact full-neuron output": not cell._return_voltage_sequences,
        "spike/count input sources": all(net["input_type"] != "current" for net in cell.inputs.values()),
    }
    return tuple(name for name, met in conditions.items() if not met)


class _RecordedSequence(tf.keras.layers.Layer):
    def __init__(self, cell, **kwargs):
        super().__init__(dtype=cell.compute_dtype, **kwargs)
        self.cell = cell
        self._autocast = False
        self.autocast = False

    def call(self, values):
        inputs, *state = values
        return self.run(inputs, tuple(state))

    def compute_output_spec(self, values):
        batch, length = values[0].shape[:2]
        neurons = self.cell._n_neurons
        flags = 2 * neurons
        shapes = (
            (batch, length, neurons), (batch, length),
            (length, batch, neurons), (length, batch, 0),
            (length, batch, (flags + 62) // 63),
            (length, batch, self.cell._input_history_size),
        )
        dtypes = (
            self.cell.compute_dtype, tf.float32, values[2].dtype,
            values[4].dtype, tf.int64, self.cell.compute_dtype,
        )
        return tuple(
            tf.keras.KerasTensor(shape, dtype=dtype)
            for shape, dtype in zip(shapes, dtypes)
        ) + tuple(
            tf.keras.KerasTensor(state.shape, dtype=state.dtype) for state in values[1:]
        )

    def recorded_step(self, inputs, states, retained, weight_carrier=()):
        cell = self.cell
        neurons = cell._n_neurons
        batch = inputs.shape[0]
        history_width = neurons * cell.max_delay
        voltage, adaptation, flags, input_history = retained
        voltage = tf.ensure_shape(voltage, (batch, neurons))
        adaptation = tf.ensure_shape(adaptation, (batch, 2 * neurons))
        input_history = tf.ensure_shape(input_history, (batch, cell._input_history_size))
        flags = tf.ensure_shape(flags, (batch, history_width + 2 * neurons))
        old_history = flags[:, :history_width]
        old_refractory = flags[:, history_width:history_width + neurons]
        spikes = flags[:, history_width + neurons:]
        states = (
            _primal_with_vjp(old_history, states[0]), states[1],
            tf.cast(old_refractory, states[2].dtype),
            _primal_with_vjp(adaptation, states[3]), *states[4:],
        )
        if cell._input_history_size:
            states = states[:7] + (_primal_with_vjp(input_history, states[7]),) + states[8:]
        new_voltage = tf.where(spikes > 0, tf.cast(cell.v_reset, voltage.dtype), voltage)
        new_history = tf.concat((spikes, old_history[:, :-neurons]), axis=1)
        native_outputs = (
            voltage, new_voltage, tf.zeros_like(states[2]),
            tf.zeros_like(states[3]), tf.zeros_like(states[4]),
            tf.zeros_like(states[5]), spikes, new_history,
        )
        options = {"recorded_nest_outputs": native_outputs}
        if weight_carrier:
            options["recurrent_weight_carrier"] = weight_carrier[0]
        return cell._call_impl(inputs, states, **options)

    def run(self, inputs, state, retained=None):
        cell = self.cell
        capture = retained is None
        batch, length = inputs.shape[:2]
        neurons = cell._n_neurons
        history_width = neurons * cell.max_delay
        flag_width = 2 * neurons
        shapes = [(batch, neurons), (batch,)]
        dtypes = [tf.as_dtype(cell.compute_dtype), tf.float32]
        if capture:
            shapes.extend((
                (batch, neurons), (batch, 0),
                (batch, flag_width), (batch, cell._input_history_size),
            ))
            dtypes.extend((state[1].dtype, state[3].dtype, state[0].dtype, tf.as_dtype(cell.compute_dtype)))
        else:
            flags = _restore_record_flags(state[0], retained[2], neurons, length)
            adaptation = _restore_asc(
                state[3], tf.ensure_shape(flags[:, :, history_width:], (length, batch, 2 * neurons)),
                cell.asc_decay, cell.asc_amps, cell.asc_refractory_decay, neurons,
                time_major=True,
            )
            retained = (
                retained[0], adaptation, flags, retained[3],
            )
        list_shapes = tuple(
            tf.constant([-1 if size is None else size for size in shape], tf.int32)
            for shape in shapes
        )
        arrays = tuple(tf.raw_ops.EmptyTensorList(
            element_shape=shape, max_num_elements=-1, element_dtype=dtype,
        ) for dtype, shape in zip(dtypes, list_shapes))
        with _rollout_context(cell), _elide_vjp_producers():
            prepare = cell._use_prepared_cpu_poisson
            budget = PREPARED_POISSON_BUDGET_BYTES
            if prepare == "auto":
                prepare = _prepared_poisson_admitted(cell, inputs)
                budget = PREPARED_POISSON_AUTO_CEILING_BYTES
            tables = _prepare_cpu_poisson(cell, inputs, state, budget) if prepare and _internal_poisson_inputs(cell) else {}
            record_rollout_acceleration(getattr(cell, "acceleration_report", None), {
                "use_direct_state_rnn_loop": True,
                "use_prepared_cpu_poisson": bool(tables),
                "temporal_path": "recorded_forward" if capture else "recorded_vjp",
            })
            carrier = (tf.identity(cell.recurrent_weight_values),) if cell.use_fused_recurrent_accumulation else ()
            noise_index = ()
            if tables:
                with _weight_device_scope(cell):
                    noise_index = (tf.constant(0, tf.int64),)

            def step(index, states, outputs, *weight_carrier):
                step_noise_index = weight_carrier[-1:] if tables else ()
                weight_carrier = weight_carrier[:-1] if tables else weight_carrier
                with _prepared_noise_step(cell, tables, step_noise_index[0]) if tables else nullcontext():
                    if capture:
                        options = {"capture_nest_record": True}
                        if weight_carrier:
                            options["recurrent_weight_carrier"] = weight_carrier[0]
                        result = cell._call_impl(inputs[:, index], states, **options)
                    else:
                        result = self.recorded_step(
                            inputs[:, index], states,
                            tuple(value[index] for value in retained), weight_carrier,
                        )
                output, next_state = result[:2]
                if weight_carrier:
                    weight_carrier = (result[2],)
                values = tuple(output)
                if capture:
                    pre_reset, _ = result[-1]
                    adaptation = tf.zeros((tf.shape(inputs)[0], 0), states[3].dtype)
                    flags = tf.concat((
                        tf.cast(states[2] > 0, states[0].dtype), output[0],
                    ), axis=1)
                    input_history = states[7] if cell._input_history_size else tf.zeros((tf.shape(inputs)[0], 0), cell.compute_dtype)
                    values += (pre_reset, adaptation, flags, input_history)
                if tables:
                    with _weight_device_scope(cell):
                        step_noise_index = (step_noise_index[0] + tf.constant(1, tf.int64),)
                return (index + 1, tuple(next_state), tuple(
                    tf.raw_ops.TensorListPushBack(input_handle=array, tensor=value)
                    for array, value in zip(outputs, values)
                ), *weight_carrier, *step_noise_index)

            _, state, arrays, *_ = tf.while_loop(
                lambda index, *_: index < tf.shape(inputs)[1],
                step, (tf.constant(0), tuple(state), arrays, *carrier, *noise_index),
                parallel_iterations=1,
            )
        stacked = tuple(
            tf.raw_ops.TensorListStack(
                input_handle=array, element_shape=list_shape,
                element_dtype=dtype, num_elements=-1,
            )
            for array, list_shape, dtype in zip(arrays, list_shapes, dtypes)
        )
        sequences = tuple(
            tf.ensure_shape(
                tf.transpose(value, [1, 0] + list(range(2, len(shape) + 1)))
                if index < 2 else value,
                (batch, length, *shape[1:]) if index < 2 else (length, batch, *shape[1:]),
            )
            for index, (value, shape) in enumerate(zip(stacked, shapes))
        )
        if capture:
            sequences = sequences[:4] + (_pack_record_flags(sequences[4]), sequences[5])
        return sequences + tuple(state)

    def chunk_vjp(self, inputs, boundary, retained, sequence_cotangents,
                  state_cotangents, variables):
        cell = self.cell
        batch = inputs.shape[0]
        neurons = cell._n_neurons
        history_width = neurons * cell.max_delay
        flags = _restore_record_flags(boundary[0], retained[2], neurons, inputs.shape[1])
        adaptation = _restore_asc(
            boundary[3], tf.ensure_shape(
                flags[:, :, history_width:], (inputs.shape[1], batch, 2 * neurons),
            ),
            cell.asc_decay, cell.asc_amps, cell.asc_refractory_decay, neurons,
            time_major=True,
        )
        retained = (retained[0], adaptation, flags, retained[3])
        floating_indices = tuple(
            index for index, value in enumerate(boundary) if value.dtype.is_floating
        )
        cotangents = tuple(state_cotangents[index] for index in floating_indices)
        weight_gradients = tuple(tf.zeros_like(value) for value in variables)
        carrier_enabled = cell.use_fused_recurrent_accumulation
        master = cell.recurrent_weight_values
        candidates = (master, master.value)
        for candidate in tuple(candidates):
            if isinstance(candidate, tf.distribute.DistributedValues):
                candidates += tuple(candidate.values)
        recurrent_index = next((
            index for index, value in enumerate(variables)
            if any(value is candidate for candidate in candidates)
        ), None) if carrier_enabled else None
        if carrier_enabled and master.trainable and recurrent_index is None:
            raise RuntimeError("Recorded VJP could not locate the recurrent master.")
        direct_indices = tuple(
            index for index in range(len(variables)) if index != recurrent_index
        )
        direct_variables = tuple(variables[index] for index in direct_indices)
        with _rollout_context(cell), _elide_vjp_producers():
            prepare = cell._use_prepared_cpu_poisson
            budget = PREPARED_POISSON_BUDGET_BYTES
            if prepare == "auto":
                prepare = _prepared_poisson_admitted(cell, inputs)
                budget = PREPARED_POISSON_AUTO_CEILING_BYTES
            tables = _prepare_cpu_poisson(cell, inputs, boundary, budget) if prepare and _internal_poisson_inputs(cell) else {}
            carrier = (tf.identity(master),) if carrier_enabled else ()
            last = tf.shape(inputs)[1] - 1
            noise_index = ()
            if tables:
                with _weight_device_scope(cell):
                    noise_index = (tf.cast(last, tf.int64),)
            record_rollout_acceleration(getattr(cell, "acceleration_report", None), {
                "use_direct_state_rnn_loop": True,
                "use_prepared_cpu_poisson": bool(tables),
                "temporal_path": "recorded_vjp",
            })

            def step(index, cotangents, accumulated, *noise_index):
                states = list(boundary)
                states[6] = boundary[6] + tf.cast(index, boundary[6].dtype)
                states = tuple(tf.ensure_shape(
                    value, (batch, *value.shape[1:]),
                ) for value in states)
                floating = tuple(states[position] for position in floating_indices)
                with tf.GradientTape() as tape:
                    tape.watch(floating + direct_variables + carrier)
                    with _prepared_noise_step(cell, tables, noise_index[0]) if tables else nullcontext():
                        result = self.recorded_step(
                            inputs[:, index], states,
                            tuple(value[index] for value in retained), carrier,
                        )
                output, next_state = result[:2]
                next_floating = tuple(next_state[position] for position in floating_indices)
                output_cotangents = tuple(
                    tf.zeros_like(value) if gradient is None else gradient[:, index]
                    for value, gradient in zip(output, sequence_cotangents)
                )
                carrier_cotangents = (
                    (accumulated[recurrent_index],)
                    if recurrent_index is not None else
                    tuple(tf.zeros_like(value) for value in carrier)
                )
                gradients = tape.gradient(
                    tuple(output) + next_floating + ((result[2],) if carrier else ()),
                    floating + direct_variables + carrier,
                    output_gradients=output_cotangents + cotangents + carrier_cotangents,
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                next_cotangents = gradients[:len(floating)]
                next_accumulated = list(accumulated)
                for position, gradient in zip(
                    direct_indices, gradients[len(floating):len(floating) + len(direct_variables)],
                ):
                    next_accumulated[position] = accumulated[position] + gradient
                if recurrent_index is not None:
                    next_accumulated[recurrent_index] = gradients[-1]
                if tables:
                    with _weight_device_scope(cell):
                        noise_index = (noise_index[0] - tf.constant(1, tf.int64),)
                return index - 1, tuple(next_cotangents), tuple(next_accumulated), *noise_index

            _, cotangents, weight_gradients, *_ = tf.while_loop(
                lambda index, *_: index >= 0,
                step, (last, cotangents, weight_gradients, *noise_index),
                parallel_iterations=1,
            )
        return cotangents + weight_gradients


class _RecordingCollector:
    def __init__(self, runner, sampled_traces):
        view = SimpleNamespace(
            sequence_specs=runner.sequence_specs[:2], count_windows=runner.count_windows,
            sequence_length=runner.sequence_length, chunk_size=runner.chunk_size,
            n_chunks=runner.n_chunks, remainder_size=runner.remainder_size,
            _assemble_chunks=runner._assemble_chunks,
        )
        self.statistics = _StatisticsCollector(view, sampled_traces)
        self.runner = runner

    def initialize(self, inputs):
        return self.statistics.initialize(inputs), tuple(
            tf.TensorArray(spec.dtype, size=self.runner.n_chunks, infer_shape=self.runner.remainder_size == 0)
            for spec in self.runner.sequence_specs[2:]
        )

    def append(self, carry, index, sequences):
        statistics, records = carry
        return self.statistics.append(statistics, index, sequences[:2]), tuple(
            array.write(index, value) for array, value in zip(records, sequences[2:])
        )

    def finish(self, carry, inputs):
        return self.statistics.finish(carry[0], inputs)

    def cache(self, carry):
        return carry[1]


class RecordedAdjointRunner(SpikeStatisticsRunner):
    """Retain exact NEST membrane/event history, avoiding membrane/projection replay.

    Attached adaptation-reset credit uses exact prior ASC values reconstructed
    within each chunk from its boundary state and recorded events.
    """

    def __init__(self, core_model, cell, sequence_length, chunk_size, **kwargs):
        unsupported = _records_supported(cell)
        if unsupported:
            raise ValueError("Recorded adjoint requires: " + ", ".join(unsupported))
        if kwargs.get("differentiate_inputs", False):
            raise ValueError("Recorded adjoint currently requires differentiate_inputs=False.")
        if "n_sequence_outputs" in kwargs:
            raise ValueError("Recorded adjoint owns its recording-model output layout.")
        self.recording_layer = _RecordedSequence(cell)
        recording_model = tf.keras.Model(
            core_model.inputs, self.recording_layer(core_model.inputs),
        )
        super().__init__(
            recording_model, sequence_length, chunk_size, n_sequence_outputs=6, **kwargs
        )

    def __call__(self, inputs, initial_state, sampled_traces=()):
        collector = _RecordingCollector(self, sampled_traces)
        output_count = len(self.count_windows) + len(sampled_traces) + 1
        checks = (
            tf.debugging.assert_equal(tf.shape(inputs)[1], self.sequence_length),
            tf.debugging.assert_equal(
                tf.reduce_all((initial_state[0] == 0) | (initial_state[0] == 1)), True,
                message="Recorded adjoints require binary initial spike history.",
            ),
        )

        @tf.custom_gradient
        def rollout(inputs, *initial_state):
            with tf.control_dependencies(checks), record.stop_recording():
                statistics, state, (boundaries, retained) = self._forward(
                    inputs, initial_state, checks[0], collector=collector,
                )

            def grad(*upstream, variables=None):
                variables = tuple(variables or ())

                def recorded_vjp(index, chunk, boundary, sequence_cotangents,
                                 state_cotangents, variables):
                    return self.recording_layer.chunk_vjp(
                        chunk, boundary, tuple(array.read(index) for array in retained),
                        sequence_cotangents, state_cotangents, variables,
                    )

                input_gradient, state_gradients, weights = self._backward(
                    inputs, boundaries, state, (), upstream[output_count:], variables,
                    sequence_gradient_fn=collector.statistics.gradient_fn(upstream[:output_count], inputs),
                    chunk_vjp_fn=recorded_vjp,
                )
                arguments = (input_gradient,) + state_gradients
                return (arguments, list(weights)) if variables else arguments

            return statistics + state, grad

        flat = rollout(inputs, *tuple(initial_state))
        count_end = len(self.count_windows)
        trace_end = count_end + len(sampled_traces)
        return SpikeStatistics(
            tuple(flat[:count_end]), tuple(flat[count_end:trace_end]),
            tuple(flat[trace_end:output_count]), tuple(flat[output_count:]),
        )
