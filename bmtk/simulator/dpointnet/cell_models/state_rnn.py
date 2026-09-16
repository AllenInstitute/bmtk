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
        if hasattr(self, "inner_loop"):
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
                return self.cell(inputs, states, **cell_kwargs)

            last, sequence, states = tf.keras.backend.rnn(
                step,
                sequences,
                states,
                go_backwards=self.go_backwards,
                mask=mask,
                unroll=self.unroll,
                input_length=sequences.shape[0 if self.time_major else 1],
                time_major=self.time_major,
                zero_output_for_mask=self.zero_output_for_mask,
                return_all_outputs=self.return_sequences,
            )
        output = sequence if self.return_sequences else last
        return (output, *states) if self.return_state else output

    def compute_output_shape(self, sequences_shape, initial_state_shape=None):
        batch, length = sequences_shape[:2]
        output_shape = (
            (batch, length, self.cell.output_size)
            if self.return_sequences
            else (batch, self.cell.output_size)
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
        if not self.return_state:
            return tf.keras.KerasTensor(shapes, dtype=self.compute_dtype)
        initial_state = (
            initial_state
            if initial_state is not None
            else self.cell.zero_state(1, self.compute_dtype)
        )
        output_dtype = (
            self.cell.compute_dtype
            if self.cell._return_voltage_sequences
            else "float32"
        )
        return (
            tf.keras.KerasTensor(shapes[0], dtype=output_dtype),
            *(
                tf.keras.KerasTensor(shape, dtype=state.dtype)
                for shape, state in zip(shapes[1:], initial_state)
            ),
        )
