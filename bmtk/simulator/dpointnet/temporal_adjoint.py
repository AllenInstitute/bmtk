"""Quantized forward rollouts with independently FP32 temporal VJPs."""

import tensorflow as tf
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.ops import handle_data_util

from .segmented_recompute import _pack_spikes, _unpack_spikes
from .custom_ops import csr_spike_ops


def _floating32(value):
    return tf.cast(value, tf.float32) if value.dtype.is_floating else value


class _HostCurrentTape:
    """Per-rollout anonymous CPU resource; loop carries only an ordering scalar."""

    def __init__(self, dtype, batch_size, width, chunk_size, boundaries,
                 handle=None, flow=None):
        self.dtype = tf.as_dtype(dtype)
        if self.dtype not in (tf.float16, tf.float32):
            raise ValueError("Recorded currents require float16 or float32.")
        self.batch_size = batch_size
        self.width = width
        self.chunk_size = chunk_size
        self.boundaries = boundaries
        self.elements = chunk_size * width
        self.words = (self.elements * self.dtype.size + 3) // 4
        self.padded_elements = self.words * 4 // self.dtype.size
        self.ops = csr_spike_ops._OPS
        if self.ops is not None and not hasattr(self.ops, "dpointnet_current_tape_create"):
            raise RuntimeError("Rebuild custom operators for the CPU current-tape resource.")
        self.backend = "tensor_resource" if self.ops is not None else "integer_table"
        with tf.device("/CPU:0"):
            self.handle = (
                (self.ops.dpointnet_current_tape_create(
                    tf.size(boundaries) - 1, T=self.dtype
                ) if self.ops is not None else tf.raw_ops.AnonymousMutableHashTableOfTensors(
                    key_dtype=tf.int64, value_dtype=tf.int32, value_shape=[self.words]
                )) if handle is None else handle
            )
            if handle is None:
                # Control-flow autodiff queries captured resources; this opaque
                # tape is not a differentiable variable, regardless of storage.
                data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
                data.is_set = True
                entry = data.shape_and_type.add()
                entry.dtype = tf.int32.as_datatype_enum
                entry.shape.CopyFrom(tf.TensorShape([self.words]).as_proto())
                handle_data_util.set_handle_data(self.handle, data)
            self.flow = tf.constant(0, tf.int32) if flow is None else flow

    def write(self, index, currents, flow):
        with tf.device("/CPU:0"):
            if self.ops is not None:
                return self.ops.dpointnet_current_tape_write(
                    self.handle, index, tf.identity(currents, name="original_current_tape_copy"), flow
                )
            tf.debugging.assert_equal(index, flow, "Current tape writes must be sequential")
            tf.debugging.assert_equal(
                tf.raw_ops.LookupTableSizeV2(table_handle=self.handle),
                tf.cast(index * self.batch_size, tf.int64),
                "Current tape writes must be sequential",
            )
            values = tf.identity(currents, name="original_current_tape_copy")
            values = tf.transpose(values, [1, 0, 2])
            values = tf.reshape(values, [self.batch_size, -1])
            values = tf.pad(values, [[0, 0], [0, self.padded_elements - tf.shape(values)[1]]])
            if self.dtype == tf.float16:
                values = tf.reshape(values, [self.batch_size, self.words, 2])
            packed = tf.bitcast(values, tf.int32)
            keys = tf.cast(index * self.batch_size + tf.range(self.batch_size), tf.int64)
            inserted = tf.raw_ops.LookupTableInsertV2(
                table_handle=self.handle, keys=keys, values=packed
            )
            with tf.control_dependencies([inserted]):
                return tf.identity(index + 1)

    def completed(self, flow):
        return _HostCurrentTape(
            self.dtype, self.batch_size, self.width, self.chunk_size, self.boundaries,
            handle=self.handle, flow=flow,
        )

    def read(self, index):
        with tf.device("/CPU:0"):
            index = tf.cast(index, tf.int32)
            if self.ops is not None:
                return tf.ensure_shape(self.ops.dpointnet_current_tape_read(
                    self.handle, index, self.flow, T=self.dtype
                ), [None, None, self.width])
            checks = [
                tf.debugging.assert_greater_equal(index, 0),
                tf.debugging.assert_less(index, self.flow, "Current chunk not recorded"),
                tf.debugging.assert_greater_equal(
                    tf.raw_ops.LookupTableSizeV2(table_handle=self.handle),
                    tf.cast((index + 1) * self.batch_size, tf.int64),
                    "Current chunk not recorded",
                ),
            ]
            with tf.control_dependencies(checks):
                keys = tf.cast(index * self.batch_size + tf.range(self.batch_size), tf.int64)
                packed = tf.raw_ops.LookupTableFindV2(
                    table_handle=self.handle, keys=keys,
                    default_value=tf.zeros([self.words], tf.int32),
                )
            values = tf.reshape(tf.bitcast(packed, self.dtype),
                                [self.batch_size, self.padded_elements])
            length = self.boundaries[index + 1] - self.boundaries[index]
            values = tf.reshape(values[:, :length * self.width],
                                [self.batch_size, length, self.width])
            return tf.transpose(values, [1, 0, 2])


class TemporalAdjointRunner:
    """Carry FP32 cotangents through FP32 replay of the quantized forward map.

    Only checkpoint primals use the cell's mixed storage. Replay activations and
    temporal cotangents are FP32; this is additional backward working memory.
    """

    def __init__(self, cell, chunk_size=25, pack_spike_checkpoints=False):
        if getattr(cell, "temporal_gradient_precision", None) != "float32":
            raise ValueError(
                "TemporalAdjointRunner requires temporal_gradient_precision='float32'."
            )
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer.")
        self.cell = cell
        self.chunk_size = chunk_size
        self.pack_spike_checkpoints = bool(pack_spike_checkpoints)
        self.current_replay_mode = getattr(cell, "current_replay_mode", "record")
        if self.current_replay_mode not in ("record", "recompute"):
            raise ValueError("current_replay_mode must be 'record' or 'recompute'.")
        supports_currents = getattr(cell, "_supports_recorded_currents", False)
        if self.current_replay_mode == "recompute" and (
            not supports_currents
            or not callable(getattr(cell, "_capture_adjoint_projection_values", None))
        ):
            raise ValueError("Current recompute requires projection snapshot support.")
        self.record_currents = supports_currents and self.current_replay_mode == "record"
        self.current_tape_device = "/CPU:0"

    def _validate(self, inputs, state):
        if inputs.shape.rank != 3:
            raise ValueError(
                "Temporal adjoint inputs must have shape [batch, time, inputs]."
            )
        if self.cell._temporal_continuous_inputs and inputs.dtype != tf.float32:
            raise ValueError(
                "FP32 temporal continuous-input gradients require float32 inputs."
            )
        self.cell.validate_state_precision(state)
        tf.debugging.assert_positive(
            tf.shape(inputs)[1], message="The rollout must not be empty."
        )

    def _loop(
        self,
        inputs,
        initial_state,
        replay=False,
        projection_context=None,
        noise_seed=None,
        recorded_currents=None,
        capture_currents=False,
        projection_values=None,
    ):
        length = tf.shape(inputs)[1]
        shapes = tf.nest.flatten(self.cell.output_size)
        if isinstance(self.cell.output_size, tuple):
            dtypes = [
                tf.float32 if replay else tf.as_dtype(self.cell.compute_dtype),
                tf.float32,
            ]
        else:
            dtypes = [tf.float32]
        arrays = tuple(
            tf.TensorArray(
                dtype,
                size=length,
                clear_after_read=False,
                element_shape=tf.TensorShape([inputs.shape[0]]).concatenate(shape),
            )
            for dtype, shape in zip(dtypes, shapes)
        )
        currents = (
            tf.TensorArray(
                tf.as_dtype(self.cell.compute_dtype), size=length,
                clear_after_read=False, infer_shape=False,
            ) if capture_currents else tf.constant(0)
        )

        carry_weights = replay and getattr(
            self.cell, "use_fused_recurrent_accumulation", False
        )
        if carry_weights and capture_currents:
            raise ValueError("Weight-carrier replay cannot capture primal currents.")
        carrier = (
            (tf.identity(self.cell.recurrent_weight_values),) if carry_weights else ()
        )

        def step(index, state, output_arrays, current_array, *weight_carrier):
            kwargs = {"adjoint_replay": replay}
            if projection_context is not None:
                kwargs["projection_context"] = projection_context
            if projection_values is not None:
                kwargs["projection_values"] = projection_values
            if noise_seed is not None:
                kwargs["noise_seed"] = noise_seed
            if recorded_currents is not None:
                kwargs["recorded_currents"] = recorded_currents[index]
            if capture_currents:
                output, next_state, current = self.cell._call_impl(
                    inputs[:, index], state, capture_currents=True, **kwargs
                )
                current_array = current_array.write(index, current)
            elif carry_weights:
                output, next_state, next_carrier = self.cell._call_impl(
                    inputs[:, index],
                    state,
                    recurrent_weight_carrier=weight_carrier[0],
                    **kwargs
                )
                weight_carrier = (next_carrier,)
            else:
                output, next_state = self.cell._call_impl(inputs[:, index], state, **kwargs)
            flat = tf.nest.flatten(output)
            output_arrays = tuple(
                array.write(index, value) for array, value in zip(output_arrays, flat)
            )
            return (
                index + 1,
                tuple(next_state),
                output_arrays,
                current_array,
            ) + weight_carrier

        result = tf.while_loop(
            lambda index, *_: index < length,
            step,
            (tf.constant(0), tuple(initial_state), arrays, currents) + carrier,
            parallel_iterations=1,
        )
        _, state, arrays, currents = result[:4]
        outputs = tuple(
            tf.ensure_shape(
                tf.transpose(
                    array.stack(), [1, 0] + list(range(2, len(tf.TensorShape(shape)) + 2))
                ),
                tf.TensorShape(inputs.shape[:2]).concatenate(shape),
            )
            for array, shape in zip(arrays, shapes)
        )
        if capture_currents:
            return outputs, state, currents.stack()
        return outputs, state

    def _forward(self, inputs, initial_state, probe_steps=()):
        """Return (sequences, final_state, boundaries, saved, seed, tape, values).

        ``saved`` contains boundary TensorArrays; ``tape`` is None in recompute.
        ``values`` holds the invocation's quantized basis and projection weights.
        Keep this seven-field private cache layout compatible with replay gates.
        """
        self._validate(inputs, initial_state)
        length = tf.shape(inputs)[1]
        noise_seed = (
            tf.identity(self.cell.noise_seed)
            if hasattr(self.cell, "noise_seed")
            else None
        )
        capture_projection = getattr(self.cell, "_capture_adjoint_projection_values", None)
        projection_values = capture_projection() if capture_projection is not None else None
        probes = tf.constant(tuple(probe_steps), tf.int32)
        tf.debugging.assert_greater_equal(probes, 0)
        tf.debugging.assert_less_equal(probes, length)
        boundaries = tf.sort(
            tf.unique(
                tf.concat(
                    [tf.range(0, length, self.chunk_size), [length], probes], axis=0
                )
            ).y
        )
        count = tf.size(boundaries) - 1
        state_arrays = tuple(
            tf.TensorArray(
                tf.int32 if self.pack_spike_checkpoints and index == 0 else value.dtype,
                size=count + 1,
                clear_after_read=False,
            ).write(
                0,
                (
                    _pack_spikes(value)
                    if self.pack_spike_checkpoints and index == 0
                    else value
                ),
            )
            for index, value in enumerate(initial_state)
        )
        output_dtypes = (
            (tf.as_dtype(self.cell.compute_dtype), tf.float32)
            if isinstance(self.cell.output_size, tuple)
            else (tf.float32,)
        )
        outputs = tuple(
            tf.TensorArray(
                dtype,
                size=length,
                clear_after_read=False,
                element_shape=tf.TensorShape([inputs.shape[0]]).concatenate(shape),
            )
            for dtype, shape in zip(
                output_dtypes, tf.nest.flatten(self.cell.output_size)
            )
        )
        current_tape = (
            _HostCurrentTape(
                self.cell.compute_dtype, tf.shape(inputs)[0],
                self.cell._n_neurons * self.cell._n_syn_basis,
                self.chunk_size, boundaries,
            ) if self.record_currents else None
        )

        def chunk(index, state, saved, sequences, tape_flow):
            start, stop = boundaries[index], boundaries[index + 1]
            result = self._loop(
                inputs[:, start:stop], state, noise_seed=noise_seed,
                capture_currents=self.record_currents,
            )
            chunk_outputs, state = result[:2]
            if self.record_currents:
                tape_flow = current_tape.write(index, result[2], tape_flow)
            sequences = tuple(
                array.scatter(
                    tf.range(start, stop),
                    tf.transpose(value, [1, 0] + list(range(2, value.shape.rank))),
                )
                for array, value in zip(sequences, chunk_outputs)
            )
            saved = tuple(
                array.write(
                    index + 1,
                    (
                        _pack_spikes(value)
                        if self.pack_spike_checkpoints and j == 0
                        else value
                    ),
                )
                for j, (array, value) in enumerate(zip(saved, state))
            )
            return index + 1, state, saved, sequences, tape_flow

        _, state, state_arrays, outputs, tape_flow = tf.while_loop(
            lambda index, *_: index < count,
            chunk,
            (tf.constant(0), tuple(initial_state), state_arrays, outputs, tf.constant(0)),
            parallel_iterations=1,
        )
        sequences = tuple(
            tf.ensure_shape(
                tf.transpose(
                    value.stack(), [1, 0] + list(range(2, value.element_shape.rank + 1))
                ),
                tf.TensorShape(inputs.shape[:2]).concatenate(shape),
            )
            for value, shape in zip(outputs, tf.nest.flatten(self.cell.output_size))
        )
        if self.record_currents:
            current_tape = current_tape.completed(tape_flow)
        return sequences, state, boundaries, state_arrays, noise_seed, current_tape, projection_values

    def _read_current_chunk(self, current_tape, index):
        if not self.record_currents:
            return None
        with tf.device(self.current_tape_device):
            currents = current_tape.read(index)
        with tf.device(self.cell.recurrent_weight_values.handle.device or None):
            return tf.identity(currents)

    def _replay_cached_chunk(self, inputs, initial_state, cache, index):
        """Diagnostic: return replay outputs/state, original outputs/boundary state.

        Each chunk starts at its own original checkpoint, as in backward; this
        is not a free-running rollout. Call within the cache's tracing scope.
        """
        sequences, _, boundaries, saved, seed, tape, values = cache
        if values is None:
            raise ValueError("Cached replay diagnostics require projection snapshots.")
        if (
            self.record_currents != (self.current_replay_mode == "record")
            or (tape is not None) != self.record_currents
        ):
            raise ValueError("Cache and runner replay mode disagree; do not toggle record_currents.")
        index = tf.convert_to_tensor(index, tf.int32)
        tf.debugging.assert_greater_equal(index, 0)
        tf.debugging.assert_less(index, tf.size(boundaries) - 1)

        def read_state(position):
            state = tuple(array.read(position) for array in saved)
            if self.pack_spike_checkpoints:
                state = (
                    _unpack_spikes(
                        state[0], initial_state[0].shape[-1], initial_state[0].dtype
                    ),
                ) + state[1:]
            return state

        start, stop = boundaries[index], boundaries[index + 1]
        outputs, state = self._loop(
            tf.cast(inputs[:, start:stop], tf.float32),
            tuple(_floating32(value) for value in read_state(index)),
            replay=True,
            projection_context=self.cell._prepare_adjoint_projection_context(
                saved_values=values
            ),
            projection_values=values,
            noise_seed=seed,
            recorded_currents=self._read_current_chunk(tape, index),
        )
        return (
            outputs, state,
            tuple(value[:, start:stop] for value in sequences),
            read_state(index + 1),
        )

    def _backward(
        self,
        inputs,
        initial_state,
        cache,
        output_gradients,
        final_gradients,
        variables,
        capture=False,
    ):
        sequences, final_state, boundaries, saved, noise_seed, current_tape, projection_values = cache
        count = tf.size(boundaries) - 1
        floating_indices = tuple(
            i for i, value in enumerate(initial_state) if value.dtype.is_floating
        )
        cotangents = tuple(
            (
                tf.zeros(tf.shape(value), tf.float32)
                if gradient is None
                else tf.cast(gradient, tf.float32)
            )
            for value, gradient in zip(final_state, final_gradients)
        )
        input_gradients = tf.TensorArray(tf.float32, size=tf.shape(inputs)[1])
        variable_gradients = tuple(tf.zeros_like(value) for value in variables)
        captured = tuple(
            tf.TensorArray(
                tf.float32, size=count + 1 if capture else 1, clear_after_read=False
            ).write(count if capture else 0, cotangents[i])
            for i in floating_indices
        )
        output_gradients = tuple(
            (
                tf.zeros_like(value)
                if gradient is None
                else gradient
            )
            for value, gradient in zip(sequences, output_gradients)
        )

        def reverse(index, carry, weight_gradients, input_array, state_arrays):
            start, stop = boundaries[index], boundaries[index + 1]
            primal = tuple(array.read(index) for array in saved)
            if self.pack_spike_checkpoints:
                primal = (
                    _unpack_spikes(
                        primal[0], initial_state[0].shape[-1], initial_state[0].dtype
                    ),
                ) + primal[1:]
            replay_state = tuple(_floating32(value) for value in primal)
            chunk_inputs = tf.cast(inputs[:, start:stop], tf.float32)
            current_chunk = self._read_current_chunk(current_tape, index)
            floating_state = tuple(replay_state[i] for i in floating_indices)
            with tf.GradientTape() as tape:
                tape.watch((chunk_inputs, *floating_state))
                tape.watch(variables)
                prepare = getattr(
                    self.cell, "_prepare_adjoint_projection_context", None
                )
                projection_context = (
                    prepare(saved_values=projection_values) if projection_values is not None
                    else prepare() if prepare is not None else None
                )
                output, state = self._loop(
                    chunk_inputs,
                    replay_state,
                    replay=True,
                    projection_context=projection_context,
                    noise_seed=noise_seed,
                    recorded_currents=current_chunk,
                    projection_values=projection_values,
                )
            gradients = tape.gradient(
                output + tuple(state[i] for i in floating_indices),
                (chunk_inputs, *floating_state, *variables),
                output_gradients=tuple(tf.cast(g[:, start:stop], tf.float32) for g in output_gradients)
                + tuple(carry[i] for i in floating_indices),
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            input_array = input_array.scatter(
                tf.range(start, stop), tf.transpose(gradients[0], (1, 0, 2))
            )
            next_carry = list(carry)
            for position, state_index in enumerate(floating_indices):
                next_carry[state_index] = gradients[position + 1]
            weight_gradients = tuple(
                previous + tf.convert_to_tensor(gradient)
                for previous, gradient in zip(
                    weight_gradients, gradients[1 + len(floating_indices) :]
                )
            )
            if capture:
                state_arrays = tuple(
                    array.write(index, next_carry[i])
                    for array, i in zip(state_arrays, floating_indices)
                )
            return (
                index - 1,
                tuple(next_carry),
                weight_gradients,
                input_array,
                state_arrays,
            )

        _, cotangents, variable_gradients, input_gradients, captured = tf.while_loop(
            lambda index, *_: index >= 0,
            reverse,
            (count - 1, cotangents, variable_gradients, input_gradients, captured),
            parallel_iterations=1,
        )
        if (
            self.cell._use_direct_csr_recurrent_gradient
            and self.cell.recurrent_weight_values.trainable
        ):
            variable_gradients = self.cell.restore_segmented_variable_gradients(
                variables, variable_gradients
            )
        diagnostics = tuple(array.stack() for array in captured) if capture else ()
        return (
            tf.transpose(input_gradients.stack(), (1, 0, 2)),
            cotangents,
            variable_gradients,
            diagnostics,
        )

    def __call__(self, inputs, initial_state):
        inputs = tf.convert_to_tensor(inputs)
        initial_state = tuple(initial_state)
        self._validate(inputs, initial_state)

        @tf.custom_gradient
        def rollout(*arguments):
            cache = self._forward(arguments[0], arguments[1:])
            sequences, final_state = cache[:2]
            count = len(sequences)

            def grad(*gradients, variables=None):
                variable_list = tuple(variables or ())
                dx, ds, dw, _ = self._backward(
                    arguments[0],
                    arguments[1:],
                    cache,
                    gradients[:count],
                    gradients[count:],
                    variable_list,
                )
                result = (
                    (
                        tf.cast(dx, arguments[0].dtype)
                        if arguments[0].dtype.is_floating
                        else None
                    ),
                    *(
                        tf.cast(g, x.dtype) if x.dtype.is_floating else None
                        for x, g in zip(arguments[1:], ds)
                    ),
                )
                return (result, list(dw)) if variables is not None else result

            return sequences + tuple(final_state), grad

        result = rollout(inputs, *initial_state)
        count = 2 if isinstance(self.cell.output_size, tuple) else 1
        outputs = tuple(result[:count]) if count == 2 else result[0]
        return outputs, tuple(result[count:])

    def differentiate(self, inputs, initial_state, loss_fn, probe_steps=()):
        """Return loss and FP32 VJPs before external half-state API boundaries.

        ``loss_fn(sequence_outputs, final_state)`` receives FP32 copies of all
        floating outputs. Probe steps are state timestamps before that step.
        """
        inputs = tf.convert_to_tensor(inputs)
        initial_state = tuple(initial_state)
        cache = self._forward(inputs, initial_state, probe_steps)
        sequences, final_state, boundaries = cache[:3]
        floating_indices = tuple(
            i for i, value in enumerate(final_state) if value.dtype.is_floating
        )
        sequence32 = tuple(_floating32(value) for value in sequences)
        state32 = tuple(_floating32(value) for value in final_state)
        watched = sequence32 + tuple(state32[i] for i in floating_indices)
        with tf.GradientTape() as tape:
            tape.watch(watched)
            loss = loss_fn(
                sequence32 if len(sequence32) == 2 else sequence32[0], state32
            )
        gradients = tape.gradient(
            loss, watched, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )
        final_gradients = [None] * len(final_state)
        for i, value in zip(floating_indices, gradients[len(sequences) :]):
            final_gradients[i] = value
        variables = tuple(
            value
            for value in (
                self.cell.recurrent_weight_values,
                *(net["input_weight_values"] for net in self.cell.inputs.values()),
            )
            if value.trainable
        )
        variables = tuple(
            value.value if not callable(getattr(value, "value", None)) else value
            for value in variables
        )
        dx, ds, dw, captured = self._backward(
            inputs,
            initial_state,
            cache,
            gradients[: len(sequences)],
            final_gradients,
            variables,
            capture=True,
        )
        indices = tf.searchsorted(boundaries, tf.constant(tuple(probe_steps), tf.int32))
        return {
            "loss": loss,
            "outputs": sequence32 if len(sequence32) == 2 else sequence32[0],
            "final_state": final_state,
            "input_gradients": dx,
            "initial_state_gradients": tuple(ds[i] for i in floating_indices),
            "variable_gradients": dw,
            "probe_steps": tf.constant(tuple(probe_steps), tf.int32),
            "state_cotangents": tuple(tf.gather(value, indices) for value in captured),
            "floating_state_indices": floating_indices,
        }
