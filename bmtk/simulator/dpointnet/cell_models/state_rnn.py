import inspect

import tensorflow as tf


class ExplicitStateRNN(tf.keras.layers.RNN):
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
        compact_unroll = self.unroll and not self.cell._return_voltage_sequences
        if hasattr(self, "inner_loop") and not compact_unroll:
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
        output = (
            tf.keras.KerasTensor(output_shape, dtype=self.cell.compute_dtype)
            if self.cell._return_voltage_sequences
            else (
                tf.keras.KerasTensor(output_shape[0], dtype=self.cell.compute_dtype),
                tf.keras.KerasTensor(output_shape[1], dtype="float32"),
            )
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
