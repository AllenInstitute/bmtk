import tensorflow as tf
from tensorflow.python.eager import record


class SegmentedRecomputeRunner:
    """Run an RNN in recomputed temporal chunks while preserving exact BPTT."""

    def __init__(
        self,
        core_model,
        sequence_length,
        chunk_size,
        n_sequence_outputs,
        differentiate_inputs=False,
    ):
        self.core_model = core_model
        self.sequence_length = int(sequence_length)
        self.chunk_size = int(chunk_size)
        self.n_sequence_outputs = int(n_sequence_outputs)
        self.differentiate_inputs = bool(differentiate_inputs)
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if not 1 <= self.chunk_size <= self.sequence_length:
            raise ValueError(
                "chunk_size must satisfy 1 <= chunk_size <= sequence_length."
            )

        output_specs = tuple(tf.nest.flatten(core_model.output))
        if not 1 <= self.n_sequence_outputs < len(output_specs):
            raise ValueError(
                "n_sequence_outputs must leave at least one final-state output."
            )
        self.sequence_specs = output_specs[: self.n_sequence_outputs]
        self.n_full_chunks = self.sequence_length // self.chunk_size
        self.remainder_size = self.sequence_length % self.chunk_size
        self.chunk_sizes = tuple(
            min(self.chunk_size, self.sequence_length - start)
            for start in range(0, self.sequence_length, self.chunk_size)
        )
        self.n_chunks = len(self.chunk_sizes)

    def _run_chunk(self, inputs, *state):
        return tuple(tf.nest.flatten(self.core_model((inputs, *state))))

    def _slice_chunk(self, inputs, chunk_index, chunk_length, time_check=None):
        start = chunk_index * self.chunk_size
        dependencies = () if time_check is None else (time_check,)
        with tf.control_dependencies(dependencies):
            chunk = tf.identity(inputs[:, start : start + chunk_length, ...])
        if inputs.shape.rank is not None:
            chunk_shape = inputs.shape.as_list()
            chunk_shape[1] = chunk_length
            chunk = tf.ensure_shape(chunk, chunk_shape)
        return chunk

    def _assemble_chunks(self, array, output_spec, inputs):
        chunks = tuple(array.read(index) for index in range(self.n_chunks))
        sequence = chunks[0] if self.n_chunks == 1 else tf.concat(chunks, axis=1)
        output_shape = output_spec.shape
        output_rank = getattr(output_shape, "rank", None)
        if output_rank is None and output_shape is not None:
            output_rank = len(output_shape)
        if output_rank is not None:
            output_shape = (
                output_shape.as_list()
                if hasattr(output_shape, "as_list")
                else list(output_shape)
            )
            output_shape[0] = inputs.shape[0]
            output_shape[1] = self.sequence_length
            sequence = tf.ensure_shape(sequence, output_shape)
        return sequence

    def _forward(self, inputs, initial_state, time_check):
        sequence_arrays = tuple(
            tf.TensorArray(
                dtype=spec.dtype,
                size=self.n_chunks,
                infer_shape=self.remainder_size == 0,
                clear_after_read=True,
            )
            for spec in self.sequence_specs
        )
        state_arrays = tuple(
            tf.TensorArray(dtype=value.dtype, size=self.n_chunks, clear_after_read=True)
            for value in initial_state
        )

        def run_full_chunk(chunk_index, arrays, states, history):
            chunk = self._slice_chunk(inputs, chunk_index, self.chunk_size, time_check)
            history = tuple(
                array.write(chunk_index, value) for array, value in zip(history, states)
            )
            flat_outputs = self._run_chunk(chunk, *states)
            chunk_sequences = flat_outputs[: self.n_sequence_outputs]
            next_state = tuple(flat_outputs[self.n_sequence_outputs :])
            arrays = tuple(
                array.write(chunk_index, value)
                for array, value in zip(arrays, chunk_sequences)
            )
            return chunk_index + 1, arrays, next_state, history

        _, sequence_arrays, state, state_arrays = tf.while_loop(
            lambda chunk_index, _arrays, _state, _history: (
                chunk_index < self.n_full_chunks
            ),
            run_full_chunk,
            (
                tf.constant(0),
                sequence_arrays,
                tuple(initial_state),
                state_arrays,
            ),
            parallel_iterations=1,
        )

        if self.remainder_size:
            remainder_index = tf.constant(self.n_full_chunks)
            state_arrays = tuple(
                array.write(remainder_index, value)
                for array, value in zip(state_arrays, state)
            )
            chunk = self._slice_chunk(
                inputs, remainder_index, self.remainder_size, time_check
            )
            flat_outputs = self._run_chunk(chunk, *state)
            chunk_sequences = flat_outputs[: self.n_sequence_outputs]
            state = tuple(flat_outputs[self.n_sequence_outputs :])
            sequence_arrays = tuple(
                array.write(remainder_index, value)
                for array, value in zip(sequence_arrays, chunk_sequences)
            )

        sequences = tuple(
            self._assemble_chunks(array, spec, inputs)
            for array, spec in zip(sequence_arrays, self.sequence_specs)
        )
        return sequences, state, state_arrays

    def _backward(
        self,
        inputs,
        state_arrays,
        final_state,
        sequence_gradients,
        final_state_gradients,
        variables,
    ):
        inputs_are_differentiable = self.differentiate_inputs and (
            inputs.dtype.is_floating or inputs.dtype.is_complex
        )
        input_gradients = (
            (
                tf.TensorArray(
                    dtype=inputs.dtype,
                    size=self.n_chunks,
                    infer_shape=self.remainder_size == 0,
                    clear_after_read=True,
                ),
            )
            if inputs_are_differentiable
            else ()
        )
        state_dtypes = tuple(array.dtype for array in state_arrays)
        state_cotangents = tuple(
            (
                (
                    gradient
                    if gradient is not None
                    else tf.zeros_like(final_state[index])
                )
                if dtype.is_floating or dtype.is_complex
                else tf.zeros(tf.shape(final_state[index]), tf.float32)
            )
            for index, (dtype, gradient) in enumerate(
                zip(state_dtypes, final_state_gradients)
            )
        )
        variable_gradients = tuple(tf.zeros_like(value) for value in variables)

        def reverse_chunk(
            chunk_index,
            chunk_length,
            input_history,
            output_state_cotangents,
            accumulated_variable_gradients,
        ):
            start = chunk_index * self.chunk_size
            chunk = self._slice_chunk(inputs, chunk_index, chunk_length)
            boundary_state = tuple(array.read(chunk_index) for array in state_arrays)
            differentiable_state_indices = tuple(
                index
                for index, value in enumerate(boundary_state)
                if value.dtype.is_floating or value.dtype.is_complex
            )
            differentiable_state = tuple(
                boundary_state[index] for index in differentiable_state_indices
            )

            with tf.GradientTape() as tape:
                if inputs_are_differentiable:
                    tape.watch(chunk)
                tape.watch(differentiable_state)
                tape.watch(variables)
                recomputed = self._run_chunk(chunk, *boundary_state)

            chunk_sequence_gradients = tuple(
                (
                    None
                    if gradient is None
                    else gradient[:, start : start + chunk_length, ...]
                )
                for gradient in sequence_gradients
            )
            state_output_gradients = tuple(
                (
                    output_state_cotangents[index]
                    if dtype.is_floating or dtype.is_complex
                    else None
                )
                for index, dtype in enumerate(state_dtypes)
            )
            input_targets = (chunk,) if inputs_are_differentiable else ()
            targets = input_targets + differentiable_state + variables
            gradients = tape.gradient(
                recomputed,
                targets,
                output_gradients=(chunk_sequence_gradients + state_output_gradients),
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )

            target_offset = int(inputs_are_differentiable)
            if inputs_are_differentiable:
                chunk_gradient = gradients[0]
            state_gradient_values = gradients[
                target_offset : target_offset + len(differentiable_state)
            ]
            variable_gradient_values = gradients[
                target_offset + len(differentiable_state) :
            ]
            next_state_cotangents = list(output_state_cotangents)
            for index, gradient in zip(
                differentiable_state_indices, state_gradient_values
            ):
                next_state_cotangents[index] = gradient
            next_variable_gradients = tuple(
                accumulated + gradient
                for accumulated, gradient in zip(
                    accumulated_variable_gradients, variable_gradient_values
                )
            )
            if inputs_are_differentiable:
                input_history = (input_history[0].write(chunk_index, chunk_gradient),)
            return (
                input_history,
                tuple(next_state_cotangents),
                next_variable_gradients,
            )

        if self.remainder_size:
            input_gradients, state_cotangents, variable_gradients = reverse_chunk(
                tf.constant(self.n_full_chunks),
                self.remainder_size,
                input_gradients,
                state_cotangents,
                variable_gradients,
            )

        def run_reverse_full_chunk(
            chunk_index,
            input_history,
            output_state_cotangents,
            accumulated_variable_gradients,
        ):
            input_history, state_values, variable_values = reverse_chunk(
                chunk_index,
                self.chunk_size,
                input_history,
                output_state_cotangents,
                accumulated_variable_gradients,
            )
            return (
                chunk_index - 1,
                input_history,
                state_values,
                variable_values,
            )

        _, input_gradients, state_cotangents, variable_gradients = tf.while_loop(
            lambda chunk_index, _inputs, _states, _variables: chunk_index >= 0,
            run_reverse_full_chunk,
            (
                tf.constant(self.n_full_chunks - 1),
                input_gradients,
                state_cotangents,
                variable_gradients,
            ),
            parallel_iterations=1,
        )
        if inputs_are_differentiable:
            input_chunks = tuple(
                input_gradients[0].read(index) for index in range(self.n_chunks)
            )
            input_gradient = (
                input_chunks[0]
                if self.n_chunks == 1
                else tf.concat(input_chunks, axis=1)
            )
            input_gradient = tf.ensure_shape(input_gradient, inputs.shape)
        else:
            input_gradient = None
        initial_state_gradients = tuple(
            gradient if dtype.is_floating or dtype.is_complex else None
            for gradient, dtype in zip(state_cotangents, state_dtypes)
        )
        return input_gradient, initial_state_gradients, variable_gradients

    def __call__(self, inputs, initial_state):
        time_check = tf.debugging.assert_equal(
            tf.shape(inputs)[1],
            self.sequence_length,
            message="Segmented input length does not match sequence_length.",
        )

        @tf.custom_gradient
        def segmented_rollout(inputs, *initial_state):
            with record.stop_recording():
                sequences, state, state_arrays = self._forward(
                    inputs, initial_state, time_check
                )
            flat_outputs = sequences + state

            def grad(*output_gradients, variables=None):
                sequence_gradients = output_gradients[: self.n_sequence_outputs]
                final_state_gradients = output_gradients[self.n_sequence_outputs :]
                variables = tuple(variables or ())
                input_gradient, state_gradients, variable_gradients = self._backward(
                    inputs,
                    state_arrays,
                    state,
                    sequence_gradients,
                    final_state_gradients,
                    variables,
                )
                argument_gradients = (input_gradient,) + state_gradients
                if variables:
                    return argument_gradients, list(variable_gradients)
                return argument_gradients

            return flat_outputs, grad

        return segmented_rollout(inputs, *tuple(initial_state))
