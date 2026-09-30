import inspect

import tensorflow as tf


class ExplicitStateRNN(tf.keras.layers.RNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._autocast = False
        self.autocast = False
        if getattr(self.cell, "temporal_gradient_precision", "compute") == "float32":
            if self.unroll or self.go_backwards or self.stateful:
                raise ValueError(
                    "FP32 temporal gradients require a forward, non-stateful, non-unrolled RNN."
                )

    def call(self, sequences, initial_state=None, mask=None, training=False):
        if self.stateful:
            raise ValueError(
                "ExplicitStateRNN requires explicit state, not stateful=True"
            )
        if isinstance(sequences, (list, tuple)):
            sequences, *initial_state = sequences
        if isinstance(mask, (list, tuple)):
            mask = mask[0]
        if initial_state is None:
            initial_state = self.cell.zero_state(
                tf.shape(sequences)[1 if getattr(self, "time_major", False) else 0],
                self.compute_dtype,
            )
        states = [tf.convert_to_tensor(value) for value in initial_state]
        self.cell.validate_state_precision(states)
        rollout_coefficients = (
            self.cell.prepare_rollout_nest_coefficients()
            if getattr(self.cell, "_use_prepacked_nest_coefficients", False)
            else None
        )
        previous_rollout_coefficients = getattr(
            self.cell, "_rollout_nest_coefficients", None
        )
        self.cell._rollout_nest_coefficients = rollout_coefficients
        previous_rollout_noise_seed = getattr(self.cell, "_rollout_noise_seed", None)
        if hasattr(self.cell, "noise_seed"):
            # One host copy per rollout instead of a device read every step.
            with tf.device("/CPU:0"):
                self.cell._rollout_noise_seed = tf.cast(
                    tf.identity(self.cell.noise_seed), tf.int32
                )
        if getattr(self.cell, "temporal_gradient_precision", "compute") == "float32":
            if mask is not None or getattr(self, "time_major", False):
                raise ValueError(
                    "FP32 temporal gradients require unmasked batch-major input."
                )
            from ..temporal_adjoint import TemporalAdjointRunner

            runner = TemporalAdjointRunner(
                self.cell,
                chunk_size=self.cell.temporal_checkpoint_chunk_size,
                pack_spike_checkpoints=self.cell.temporal_pack_spike_checkpoints,
            )
            try:
                output, states = runner(sequences, states)
            finally:
                self.cell._rollout_nest_coefficients = previous_rollout_coefficients
                self.cell._rollout_noise_seed = previous_rollout_noise_seed
            if not self.return_sequences:
                output = tf.nest.map_structure(lambda value: value[:, -1], output)
            return (output, *states) if self.return_state else output
        try:
            compact_unroll = self.unroll and isinstance(self.cell.output_size, tuple)
            use_direct_loop = (
                getattr(self.cell, "_use_direct_state_rnn_loop", False)
                and mask is None
                and not self.go_backwards
                and not self.unroll
                and not getattr(self, "time_major", False)
            )
            carry_weights = (
                use_direct_loop
                and getattr(self.cell, "temporal_gradient_precision", "compute")
                == "compute"
                and getattr(self.cell, "use_fused_recurrent_accumulation", False)
            )
            if (
                getattr(self.cell, "use_fused_recurrent_accumulation", False)
                and getattr(self.cell, "temporal_gradient_precision", "compute")
                == "compute"
                and not use_direct_loop
            ):
                raise ValueError(
                    "Ordinary fused recurrent accumulation requires the direct "
                    "state RNN loop."
                )
            if use_direct_loop:
                cell_kwargs = (
                    {"training": training}
                    if "training" in inspect.signature(self.cell.call).parameters
                    else {}
                )
                length = tf.shape(sequences)[1]
                flat_shapes = tf.nest.flatten(self.cell.output_size)
                if isinstance(self.cell.output_size, tuple):
                    flat_dtypes = [tf.as_dtype(self.cell.compute_dtype), tf.float32]
                else:
                    flat_dtypes = [
                        tf.as_dtype(
                            self.cell.state_dtype
                            if getattr(self.cell, "_return_voltage_sequences", False)
                            else tf.float32
                        )
                    ]
                element_shapes = [
                    tf.TensorShape([sequences.shape[0]]).concatenate(shape)
                    for shape in flat_shapes
                ]
                # Appending needs no write index: TensorArray writes make the
                # gradient loop pop an accumulated int32 index every step,
                # which is a device-to-host round trip on GPU.
                list_shapes = [
                    tf.constant(
                        [-1 if size is None else size for size in shape.as_list()],
                        tf.int32,
                    )
                    for shape in element_shapes
                ]
                arrays = tuple(
                    tf.raw_ops.EmptyTensorList(
                        element_shape=list_shape,
                        max_num_elements=-1,
                        element_dtype=dtype,
                    )
                    for dtype, list_shape in zip(flat_dtypes, list_shapes)
                )
                carrier = (
                    (tf.identity(self.cell.recurrent_weight_values),)
                    if carry_weights
                    else ()
                )

                def direct_step(index, state_values, output_arrays, *weight_carrier):
                    if carry_weights:
                        output, next_states, next_carrier = self.cell._call_impl(
                            sequences[:, index],
                            state_values,
                            recurrent_weight_carrier=weight_carrier[0],
                        )
                        weight_carrier = (next_carrier,)
                    else:
                        output, next_states = self.cell(
                            sequences[:, index], state_values, **cell_kwargs
                        )
                    output_arrays = tuple(
                        tf.raw_ops.TensorListPushBack(input_handle=array, tensor=value)
                        for array, value in zip(output_arrays, tf.nest.flatten(output))
                    )
                    return (
                        index + 1,
                        tuple(next_states),
                        output_arrays,
                    ) + weight_carrier

                result = tf.while_loop(
                    lambda index, *_: index < length,
                    direct_step,
                    (tf.constant(0), tuple(states), arrays) + carrier,
                    parallel_iterations=1,
                )
                _, states, arrays = result[:3]
                stacked = tuple(
                    tf.ensure_shape(
                        tf.raw_ops.TensorListStack(
                            input_handle=array,
                            element_shape=list_shape,
                            element_dtype=dtype,
                            num_elements=-1,
                        ),
                        tf.TensorShape([sequences.shape[1]]).concatenate(element_shape),
                    )
                    for array, dtype, list_shape, element_shape in zip(
                        arrays, flat_dtypes, list_shapes, element_shapes
                    )
                )
                flat_sequence = tuple(
                    tf.transpose(
                        value,
                        [1, 0] + list(range(2, len(tf.TensorShape(shape)) + 2)),
                    )
                    for value, shape in zip(stacked, flat_shapes)
                )
                sequence = tf.nest.pack_sequence_as(self.cell.output_size, flat_sequence)
                last = tf.nest.map_structure(lambda value: value[:, -1], sequence)
            elif hasattr(self, "inner_loop") and not compact_unroll:
                last, sequence, states = self.inner_loop(
                    sequences, states, mask, training=training
                )
            else:
                cell_kwargs = (
                    {"training": training}
                    if "training" in inspect.signature(self.cell.call).parameters
                    else {}
                )

                def step(inputs, states):
                    output, next_states = self.cell(inputs, states, **cell_kwargs)
                    if compact_unroll:
                        spikes, penalty = output
                        output = tf.concat(
                            [tf.cast(spikes, tf.float32), penalty[:, None]], axis=-1
                        )
                    return output, next_states

                last, sequence, states = tf.keras.backend.rnn(
                    step,
                    sequences,
                    states,
                    go_backwards=self.go_backwards,
                    mask=mask,
                    unroll=self.unroll,
                    input_length=sequences.shape[
                        0 if getattr(self, "time_major", False) else 1
                    ],
                    time_major=getattr(self, "time_major", False),
                    zero_output_for_mask=self.zero_output_for_mask,
                    return_all_outputs=self.return_sequences,
                )
        finally:
            self.cell._rollout_nest_coefficients = previous_rollout_coefficients
            self.cell._rollout_noise_seed = previous_rollout_noise_seed
        output = sequence if self.return_sequences else last
        if compact_unroll:
            output = (
                tf.cast(output[..., :-1], self.cell.compute_dtype),
                output[..., -1],
            )
        return (output, *states) if self.return_state else output

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        batch, length = sequences_shape[:2]
        prefix = (batch, length) if self.return_sequences else (batch,)
        output_shape = tf.nest.map_structure(
            lambda size: prefix + tuple(tf.TensorShape(size).as_list()),
            self.cell.output_size,
        )
        if not self.return_state:
            return output_shape
        state_shapes = [
            (batch,)
            + (tuple(size.as_list()) if isinstance(size, tf.TensorShape) else (size,))
            for size in self.cell.state_size
        ]
        return (output_shape, *state_shapes)

    def compute_output_spec(
        self, sequences, initial_state=None, mask=None, training=False
    ):
        shapes = self.compute_output_shape(sequences.shape)
        output_shape = shapes[0] if self.return_state else shapes
        if isinstance(self.cell.output_size, tuple):
            output = (
                tf.keras.KerasTensor(output_shape[0], dtype=self.cell.compute_dtype),
                tf.keras.KerasTensor(output_shape[1], dtype="float32"),
            )
        else:
            output = tf.keras.KerasTensor(
                output_shape,
                dtype=(
                    self.cell.state_dtype
                    if self.cell._return_voltage_sequences
                    else "float32"
                ),
            )
        if not self.return_state:
            return output
        initial_state = (
            initial_state
            if initial_state is not None
            else self.cell.zero_state(1, self.compute_dtype)
        )
        return (
            output,
            *(
                tf.keras.KerasTensor(shape, dtype=state.dtype)
                for shape, state in zip(shapes[1:], initial_state)
            ),
        )
