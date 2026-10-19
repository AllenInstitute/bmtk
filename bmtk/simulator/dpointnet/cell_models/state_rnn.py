import inspect
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import tensorflow as tf

_ROLLOUT_CONSTANTS = ContextVar("dpointnet_rollout_constants", default=())
_PREPARED_NOISE = ContextVar("dpointnet_prepared_cpu_poisson", default=())


def _weight_device(cell):
    """Device of the compute weight shadow, tolerating distributed variables.

    A Keras2 ``DistributedVariable`` rejects ``.handle`` outside the replica
    context; its primary component or replica values still name the device.
    """
    variable = cell.recurrent_weight_values_compute
    for candidate in (variable, getattr(variable, "primary", None)):
        if candidate is None:
            continue
        try:
            return candidate.handle.device
        except (ValueError, AttributeError):
            continue
    values = getattr(variable, "values", None)
    if values:
        return values[0].device
    return getattr(variable, "device", None)


def _weight_device_scope(cell):
    device = _weight_device(cell)
    return tf.device(device) if device else nullcontext()


def _prepared_noise_for(cell, input_net):
    for owner, values in reversed(_PREPARED_NOISE.get()):
        if owner is cell:
            return values.get(id(input_net))
    return None


@contextmanager
def _prepared_noise_step(cell, tables, index):
    with _weight_device_scope(cell):
        values = {
            key: tf.cast(tf.gather(table, index), tf.int32)
            for key, table in tables.items()
        }
    token = _PREPARED_NOISE.set(_PREPARED_NOISE.get() + ((cell, values),))
    try:
        yield
    finally:
        _PREPARED_NOISE.reset(token)


def _internal_poisson_inputs(cell):
    return [
        net for net in cell.inputs.values()
        if net["input_type"] in ("poisson_spikes_internal", "noisy_current")
    ]


PREPARED_POISSON_BUDGET_BYTES = 64 * 1024**2
# Automatic admission is decided statically within the budget above; the
# runtime guard for auto only rejects pathological rollouts far beyond it.
PREPARED_POISSON_AUTO_CEILING_BYTES = 1024**3


def _poisson_width(net):
    return net["input_dim"] if "input_dim" in net else net["input_dense_shape"][1]


def _prepared_poisson_admitted(cell, sequences):
    """Static admission for automatic prepared Poisson from declared/static shapes."""
    internal = _internal_poisson_inputs(cell)
    batch = sequences.shape[0]
    if batch is None:
        batch = getattr(cell, "_declared_batch_size", None)
    length = sequences.shape[1]
    if length is None:
        length = getattr(cell, "_declared_sequence_length", None)
    if not internal or batch is None or length is None:
        return False
    entries = int(length) * int(batch) * sum(int(_poisson_width(net)) for net in internal)
    return entries * 8 <= PREPARED_POISSON_BUDGET_BYTES


def _prepare_cpu_poisson(cell, sequences, states, budget_bytes=PREPARED_POISSON_BUDGET_BYTES):
    internal = _internal_poisson_inputs(cell)
    with tf.device("/CPU:0"):
        shape = tf.shape(sequences, out_type=tf.int64)
        batch, length = shape[0], shape[1]
        entries = length * batch * sum(
            net.get("input_dim", net["input_dense_shape"][1]) for net in internal
        )
        with tf.control_dependencies([
            tf.debugging.assert_less_equal(
                entries * 8, tf.constant(budget_bytes, tf.int64),
                message=f"Prepared CPU Poisson exceeds the {budget_bytes >> 20} MiB rollout count budget.",
            )
        ]):
            length = tf.identity(length)
    tables = {}
    for net in internal:
        width = net.get("input_dim", net["input_dense_shape"][1])
        with tf.device("/CPU:0"):
            start = tf.identity(states[6])
            counts = tf.map_fn(
                lambda index: cell.sample_noise_spikes(batch, start + index, net),
                tf.range(tf.cast(length, start.dtype), dtype=start.dtype),
                fn_output_signature=tf.TensorSpec((sequences.shape[0], width), tf.int32),
                parallel_iterations=1,
            )
        with _weight_device_scope(cell):
            # The int32 GatherV2 registration keeps its parameters on the host.
            tables[id(net)] = tf.cast(counts, tf.int64)
    return tables


def _rollout_constant(cell, name):
    for owner, values in reversed(_ROLLOUT_CONSTANTS.get()):
        if owner is cell and name in values:
            return values[name]
    return getattr(cell, name)


@contextmanager
def _rollout_context(cell):
    missing = object()
    previous_coefficients = getattr(cell, "_rollout_nest_coefficients", missing)
    previous_noise_seed = getattr(cell, "_rollout_noise_seed", missing)
    constants = (
        cell.prepare_rollout_constants()
        if hasattr(cell, "prepare_rollout_constants")
        else {}
    )
    token = _ROLLOUT_CONSTANTS.set(_ROLLOUT_CONSTANTS.get() + ((cell, constants),))
    try:
        cell._rollout_nest_coefficients = (
            cell.prepare_rollout_nest_coefficients()
            if getattr(cell, "_use_prepacked_nest_coefficients", False)
            else None
        )
        if hasattr(cell, "noise_seed"):
            # One host copy per rollout instead of a device read every step.
            with tf.device("/CPU:0"):
                cell._rollout_noise_seed = tf.cast(
                    tf.identity(cell.noise_seed), tf.int32
                )
        yield
    finally:
        _ROLLOUT_CONSTANTS.reset(token)
        for name, previous in (
            ("_rollout_nest_coefficients", previous_coefficients),
            ("_rollout_noise_seed", previous_noise_seed),
        ):
            if previous is missing:
                if hasattr(cell, name):
                    delattr(cell, name)
            else:
                setattr(cell, name, previous)


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
        float32_temporal = (
            getattr(self.cell, "temporal_gradient_precision", "compute") == "float32"
        )
        if float32_temporal and getattr(self.cell, "_use_prepared_cpu_poisson", False) is True:
            raise ValueError("Prepared CPU Poisson requires compute-temporal gradients.")
        if float32_temporal and (mask is not None or getattr(self, "time_major", False)):
            raise ValueError(
                "FP32 temporal gradients require unmasked batch-major input."
            )
        if initial_state is None:
            initial_state = self.cell.zero_state(
                tf.shape(sequences)[1 if getattr(self, "time_major", False) else 0],
                self.compute_dtype,
            )
        states = [tf.convert_to_tensor(value) for value in initial_state]
        self.cell.validate_state_precision(states)
        with _rollout_context(self.cell):
            if float32_temporal:
                from ..temporal_adjoint import TemporalAdjointRunner

                runner = TemporalAdjointRunner(
                    self.cell,
                    chunk_size=self.cell.temporal_checkpoint_chunk_size,
                    pack_spike_checkpoints=self.cell.temporal_pack_spike_checkpoints,
                )
                output, states = runner(sequences, states)
                if not self.return_sequences:
                    output = tf.nest.map_structure(lambda value: value[:, -1], output)
                return (output, *states) if self.return_state else output
            compact_unroll = self.unroll and isinstance(self.cell.output_size, tuple)
            use_direct_loop = (
                getattr(self.cell, "_use_direct_state_rnn_loop", False)
                and mask is None
                and not self.go_backwards
                and not self.unroll
                and not getattr(self, "time_major", False)
            )
            prepare_noise = getattr(self.cell, "_use_prepared_cpu_poisson", False)
            prepared_route = (
                use_direct_loop and not float32_temporal and self.cell.dynamics_mode == "nest"
            )
            if prepare_noise is True and not prepared_route:
                raise ValueError("Prepared CPU Poisson requires the direct NEST compute-temporal loop.")
            noise_budget = PREPARED_POISSON_BUDGET_BYTES
            if prepare_noise == "auto":
                # Automatic selection keeps the ordinary per-step sampler wherever
                # the exact prepared route or its static budget is not established.
                prepare_noise = prepared_route and _prepared_poisson_admitted(self.cell, sequences)
                noise_budget = PREPARED_POISSON_AUTO_CEILING_BYTES
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
                length = sequences.shape[1] or tf.shape(sequences)[1]
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
                noise_tables = (
                    _prepare_cpu_poisson(self.cell, sequences, states, noise_budget)
                    if prepare_noise else {}
                )
                noise_index = ()
                if prepare_noise:
                    with _weight_device_scope(self.cell):
                        noise_index = (tf.constant(0, tf.int64),)

                def direct_step(index, state_values, output_arrays, *weight_carrier):
                    step_noise_index = weight_carrier[-1:] if prepare_noise else ()
                    weight_carrier = weight_carrier[:-1] if prepare_noise else weight_carrier
                    with (
                        _prepared_noise_step(self.cell, noise_tables, step_noise_index[0])
                        if prepare_noise else nullcontext()
                    ):
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
                    if prepare_noise:
                        with _weight_device_scope(self.cell):
                            step_noise_index = (step_noise_index[0] + tf.constant(1, tf.int64),)
                    output_arrays = tuple(
                        tf.raw_ops.TensorListPushBack(input_handle=array, tensor=value)
                        for array, value in zip(output_arrays, tf.nest.flatten(output))
                    )
                    return (
                        index + 1,
                        tuple(next_states),
                        output_arrays,
                    ) + weight_carrier + step_noise_index

                result = tf.while_loop(
                    lambda index, *_: index < length,
                    direct_step,
                    (tf.constant(0), tuple(states), arrays) + carrier + noise_index,
                    parallel_iterations=32,
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
