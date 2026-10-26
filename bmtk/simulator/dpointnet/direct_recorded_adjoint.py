"""Exact NEST recorded adjoint whose reverse loop calls native primitives directly.

The forward stores, per time step, the FP32 pre-reset membrane and one
bit-packed word row holding the step's spikes and its pre-step refractory
mask. Nothing else is kept: no chunk-boundary states, no stacked spike
sequences and no external input history (delayed input frames are re-sliced
from the replay). The reverse loop then calls, per step, the native NEST
history/events backward, the native sparse weight/source VJPs and the native
voltage-penalty VJP, and routes the spike cotangent itself. Attached
adaptation-reset credit uses exact prior ASC values re-walked per chunk from a
snapshot taken every ``chunk_size`` steps, as the chunked recorded adjoint does.

Two layers are deliberately separate so that a neuron-partitioned driver can
reuse the arithmetic without this module's storage policy:

* :class:`DirectRecordedStep` holds the per-rollout constants and offers
  transport-independent one-step primitives: ``forward`` (sources in, record
  and new state out), ``reverse`` (record, cotangents and sources in; previous
  cotangents, dense per-population source cotangents and caller-owned weight
  accumulators out) and ``advance_asc`` (exact ASC re-walk). Sources and
  source cotangents are explicit per-population tensors; nothing is fused
  across the projection/state boundary.
* :class:`DirectRecordedAdjointRunner` is the single-replica driver: it keeps
  the records in TensorLists, streams count/trace statistics and runs the
  full-horizon reverse loop behind a ``tf.custom_gradient``.

Structure adapted from Javier's ``v1_model_utils/recorded_adjoint.py`` reverse
step (credit: Javier); DPointNet keeps its NEST dynamics, event map, dtype
boundaries, independent per-source VJPs and canonical master ownership.
"""

import tensorflow as tf
from tensorflow.python.eager import record

from .acceleration import record_rollout_acceleration
from .cell_models.glif3_cell import GLIF3Cell
from .cell_models.state_rnn import (
    _internal_poisson_inputs, _prepare_cpu_poisson, _prepared_poisson_admitted,
    _rollout_constant, _rollout_context, _weight_device, _weight_device_scope,
    PREPARED_POISSON_AUTO_CEILING_BYTES, PREPARED_POISSON_BUDGET_BYTES,
)
from .custom_ops import csr_spike_ops, glif_state_ops
from .custom_ops.glif_state_ops import (
    NestDirectDispatch, advance_asc_record, direct_reverse_ops_available,
    direct_reverse_record, nest_record_forward_available, pack_step_record,
    record_counts, record_traces, shift_history_record, step_record_ops_available,
    unpack_step_record,
)
from .spike_statistics import SpikeStatistics, _window

RECORD_FORMAT = "direct_step_record_v1"


def direct_records_supported(cell):
    """Names of the conditions the cell fails for the direct recorded adjoint."""
    if type(cell) is not GLIF3Cell:
        return ("unmodified GLIF3Cell",)
    conditions = {
        "NEST hard forward": cell.dynamics_mode == "nest" and cell._hard_reset,
        "compute temporal gradients": cell.temporal_gradient_precision == "compute",
        "native joint projection": cell._use_fused_cuda and cell._use_combined_current_sweep,
        "CSR master weights": bool(cell._csr_master_weights),
        "native recurrent accumulation": bool(cell.use_fused_recurrent_accumulation),
        "native state/history": cell._use_fused_state and cell._use_fused_state_history,
        "native event VJP": bool(cell.use_fused_nest_event_vjp),
        "prepacked NEST coefficients": bool(cell._use_prepacked_nest_coefficients),
        "native voltage penalty": bool(cell._use_native_voltage_penalty),
        "native step records": step_record_ops_available() and direct_reverse_ops_available(),
        "direct state loop": cell._use_direct_state_rnn_loop,
        "compact full-neuron output": not cell._return_voltage_sequences,
        "no online voltage losses": not cell._online_voltage_losses,
        "spike/count input sources": all(
            net["input_type"] != "current" for net in cell.inputs.values()
        ),
        "non-uniform input delay projection": all(
            net.get("uniform_delay_steps") is None for net in cell.inputs.values()
        ),
        "positive recurrent delay": cell.max_delay >= 1,
    }
    return tuple(name for name, met in conditions.items() if not met)


# Automatic admission keeps the per-step record payload (pre-reset membrane,
# packed words, the two-level ASC snapshots) within this share of the device's
# currently free memory; the rest of the update (projection, states, masters,
# optimizer) is the same on every checkpointed route.
DIRECT_RECORD_MEMORY_FRACTION = 0.6


def direct_record_bytes(cell, batch, length, chunk_size, free_bytes=None):
    """``(fits, reason)``: advisory estimate of the direct route's record payload.

    The payload is ``length * batch * neurons`` state-dtype membrane values
    plus ``2 * ceil(neurons / 32)`` int32 words per step and the per-chunk and
    per-block ASC snapshots. ``free_bytes`` defaults to the current device
    report. ``fits`` is ``None`` when the payload or the device memory is
    unknown, ``False`` when it exceeds ``DIRECT_RECORD_MEMORY_FRACTION`` of the
    free memory. The estimate never disables the route; callers warn.
    """
    neurons = getattr(cell, "_n_neurons", None)
    if not neurons or not isinstance(batch, int) or batch <= 0 or not isinstance(length, int) or length <= 0:
        return None, "record payload unknown (static batch, sequence length and neuron count required)"
    selective = getattr(cell, "state_precision", "compute") == "selective"
    state_bytes = tf.as_dtype(cell.state_dtype if selective else cell.compute_dtype).size
    words = 2 * (-(-neurons // 32))
    chunk_size = max(1, min(int(chunk_size), length))
    n_chunks = -(-length // chunk_size)
    asc_block = -(-chunk_size // 2)
    snapshots = n_chunks + asc_block + -(-chunk_size // asc_block) + 2
    payload = (
        length * batch * neurons * state_bytes
        + (length + int(getattr(cell, "max_delay", 1))) * batch * words * 4
        + snapshots * batch * 2 * neurons * state_bytes
    )
    if free_bytes is None:
        free_bytes = device_free_bytes()
        if free_bytes is None:
            return None, f"record payload {payload / 2**30:.2f} GiB; no queryable GPU memory report"
    limit = DIRECT_RECORD_MEMORY_FRACTION * free_bytes
    reason = (
        f"record payload {payload / 2**30:.2f} GiB versus {limit / 2**30:.2f} GiB advisory "
        f"({DIRECT_RECORD_MEMORY_FRACTION:.0%} of {free_bytes / 2**30:.2f} GiB free)"
    )
    return payload <= limit, reason


def device_free_bytes():
    """Free bytes on the selected GPU from the CUDA runtime TensorFlow loaded, or None."""
    import ctypes
    import ctypes.util

    if not tf.config.list_physical_devices("GPU"):
        return None
    for name in ("libcudart.so.12", "libcudart.so", ctypes.util.find_library("cudart")):
        if not name:
            continue
        try:
            runtime = ctypes.CDLL(name)
        except OSError:
            continue
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        try:
            device = int(tf.config.list_logical_devices("GPU")[0].name.rsplit(":", 1)[1])
            if runtime.cudaSetDevice(device) != 0:
                continue
            if runtime.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)) != 0:
                continue
        except (AttributeError, IndexError, ValueError):
            continue
        return int(free.value)
    return None


def _metadata(connectivity):
    return csr_spike_ops._projection_metadata_in_current_graph(connectivity["metadata_handle"])


class DirectRecordedStep:
    """Per-rollout constants and one-step primitives of the direct recorded adjoint.

    Construct inside ``_rollout_context(cell)`` (prepacked coefficients and the
    rollout noise seed live there) with the static batch and the initial state
    dtypes. ``populations`` lists the recurrent population first, then the
    cell's named inputs in declaration order; every primitive takes and
    returns per-population tuples in that order.
    """

    def __init__(self, cell, batch, initial_state):
        unsupported = direct_records_supported(cell)
        if unsupported:
            raise ValueError("Direct recorded adjoint requires: " + ", ".join(unsupported))
        self.cell = cell
        self.batch = int(batch)
        self.neurons = int(cell._n_neurons)
        self.basis_count = int(cell._n_syn_basis)
        self.delay = int(cell.max_delay)
        self.history_width = self.neurons * self.delay
        self.spike_dtype = tf.as_dtype(cell.compute_dtype)
        self.state_dtype = tf.as_dtype(initial_state[1].dtype)
        self.refractory_dtype = tf.as_dtype(initial_state[2].dtype)
        self.words = 2 * ((self.neurons + 31) // 32)
        self.has_input_history = bool(cell._input_history_size)
        self.n_states = 8 if self.has_input_history else 7
        if len(initial_state) != self.n_states:
            raise ValueError(f"Expected {self.n_states} states, got {len(initial_state)}.")
        self.populations = [{
            "name": "<recurrent>", "recurrent": True, "width": self.history_width,
            "connectivity": cell._recurrent_projection_connectivity(),
            "variable": cell.recurrent_weight_values,
            "trainable": bool(cell.recurrent_weight_values.trainable),
        }]
        for index, (name, net) in enumerate(cell.inputs.items()):
            width = int(net["input_dim"])
            self.populations.append({
                "name": name, "recurrent": False, "net": net, "width": width,
                "poisson": net["input_type"] in ("poisson_spikes_internal", "noisy_current"),
                "columns": (int(cell.inputs_idx[index]), int(cell.inputs_idx[index + 1])),
                "history_steps": (int(net["history_stop"]) - int(net["history_start"])) // width,
                "history": (int(net["history_start"]), int(net["history_stop"])),
                "trainable": bool(net["input_weight_values"].trainable),
                "connectivity": net["fused_connectivity"],
                "variable": net["input_weight_values"],
            })
        self.penalty_mode = cell._voltage_penalty_mode
        self.sweep_options = dict(
            target_tile_size=cell.combined_current_target_tile_size,
            cooperative_rows=cell._use_cooperative_row_projection,
            fanout_rows=cell._use_fanout_row_projection,
            gather_sources=cell._use_target_gather_projection,
            csr_master_weights=True,
            **({"adaptive_routing": True} if cell._adaptive_routing else {}),
            **({"scalar_source": cell._scalar_input_source}
               if cell._use_scalar_input_projection else {}),
        )
        rollout = cell._rollout_nest_coefficients
        if rollout is None:
            raise ValueError("Direct recorded adjoint requires prepacked NEST coefficients.")
        self.nest_arguments = dict(
            syn_decay=cell.syn_decay, psc_initial=cell.psc_initial, asc_decay=cell.asc_decay,
            asc_amps=cell.asc_amps, decay=cell.decay, current_factor=cell.current_factor,
            asc_mean=cell.asc_mean, asc_refractory_decay=cell.asc_refractory_decay,
            psc_voltage=cell.psc_voltage, rise_voltage=cell.rise_voltage,
            t_ref_steps=_rollout_constant(cell, "t_ref_steps"), dt=_rollout_constant(cell, "_dt"),
            v_reset=cell.v_reset, v_th=_rollout_constant(cell, "v_th"),
            dampening=_rollout_constant(cell, "_dampening_factor"),
            voltage_gradient_dampening=_rollout_constant(cell, "_voltage_gradient_dampening"),
            hard_reset=cell._hard_reset, hard_reset_gradient_mode=cell.hard_reset_gradient_mode,
            detach_reset=cell.detach_reset, detach_asc_reset=cell.detach_asc_reset,
            pseudo_gauss=cell._pseudo_gauss, gauss_std=_rollout_constant(cell, "_gauss_std"),
            return_pre_reset_voltage=True, use_fused_event_vjp=cell.use_fused_nest_event_vjp,
            fuse_history=cell._use_fused_state_history,
            hoist_type_indexed_guard=cell._guarded_type_indexed_dispatch,
            packed_coefficients=rollout[0], packed_kernel_coefficients=rollout[1],
        )
        if len(rollout) > 2:
            self.nest_arguments.update(
                packed_type_coefficients=rollout[2], type_indices=rollout[3],
                type_indexed_identity=rollout[4], require_type_indexed_coefficients=rollout[5],
            )
        T, S = self.state_dtype, self.spike_dtype
        # Cell constants are host tensors; a GPU op inside a time loop would
        # otherwise receive them again every iteration. Materialize every
        # per-step operand on the compute device once per rollout.
        self.device = _weight_device(cell) or getattr(cell.recurrent_weight_values, "device", None)

        def resident(value, dtype):
            with self.device_scope():
                return tf.identity(tf.cast(value, dtype))

        self.nest_arguments.update(
            t_ref_steps=resident(self.nest_arguments["t_ref_steps"], self.refractory_dtype),
            dt=resident(self.nest_arguments["dt"], T), v_th=resident(self.nest_arguments["v_th"], T),
            dampening=resident(self.nest_arguments["dampening"], T),
            gauss_std=resident(self.nest_arguments["gauss_std"], T),
            voltage_gradient_dampening=resident(self.nest_arguments["voltage_gradient_dampening"], T),
        )
        self.dispatch = NestDirectDispatch(
            rollout, T,
            hoist_type_indexed_guard=self.nest_arguments["hoist_type_indexed_guard"],
            hard_reset=cell._hard_reset, hard_reset_gradient_mode=cell.hard_reset_gradient_mode,
            detach_reset=cell.detach_reset, detach_asc_reset=cell.detach_asc_reset,
            pseudo_gauss=cell._pseudo_gauss,
        )
        self.basis = resident(_rollout_constant(cell, "synaptic_basis_weights"), S)
        lr_scale = _rollout_constant(cell, "_lr_scale")
        static_scale = tf.get_static_value(lr_scale)
        self.scales_currents = static_scale is None or float(static_scale) != 1.0
        self.lr_scale = resident(lr_scale, S)
        self.recurrent_scale = resident(_rollout_constant(cell, "_recurrent_dampening"), S)
        self.dt = self.nest_arguments["dt"]
        self.v_th = self.nest_arguments["v_th"]
        self.t_ref_steps = self.nest_arguments["t_ref_steps"]
        self.v_reset = resident(self.nest_arguments["v_reset"], T)
        self.dampening = self.nest_arguments["dampening"]
        self.gauss_std = self.nest_arguments["gauss_std"]
        with self.device_scope():
            coefficients = self.dispatch.coefficients
            # Compact copy of the ASC decay/amplitude/refractory-decay columns.
            self.asc_table = tf.identity(tf.concat(
                [coefficients[:, 8:12], coefficients[:, 16:18]], axis=1,
            ))

    def device_scope(self):
        """A fresh compute-device scope (``tf.device`` scopes are single use)."""
        return tf.device(self.device) if self.device else _weight_device_scope(self.cell)

    # -- records ------------------------------------------------------------------

    def unpack(self, words):
        """``(spikes, blocked)`` of a recorded step: compute-dtype spikes, refractory-dtype mask."""
        return unpack_step_record(words, self.neurons, self.spike_dtype, self.refractory_dtype)

    # -- forward ------------------------------------------------------------------

    def forward(self, sources, state, record_row=None):
        """One recorded step.

        ``sources`` are the per-population compute-dtype spike tensors the
        projection reads (recurrent delayed history first). Returns
        ``(spikes, new_state, (pre_reset_voltage, words), penalty)`` where the
        record is the FP32 pre-reset membrane and the packed spike/pre-step
        refractory words and ``penalty`` is the native per-sample mean voltage
        penalty of the post-reset membrane. ``new_state`` holds the seven base
        states; the caller appends external input history when the cell has it.

        ``record_row=(buffer, row)`` names a ``[time, batch, neurons]`` resource
        variable and a host ``int32`` row: the pre-reset membrane is then written
        into that row by the state kernel, which also packs ``words``, and the
        returned ``pre_reset_voltage`` is ``None``. Values are identical either
        way; the in-place form saves the separate pack kernel and row copy.
        """
        history, v, r, asc, rise, psc, noise_step = state[:7]
        currents = csr_spike_ops.combined_sweep_primal(
            list(sources), [population["variable"] for population in self.populations],
            [population["connectivity"] for population in self.populations],
            self.basis, **self.sweep_options,
        )
        currents = tf.reshape(currents, [self.batch, self.neurons * self.basis_count])
        if self.scales_currents:
            currents = currents * self.lr_scale
        if record_row is None:
            pre_reset, new_v, new_r, new_asc, new_rise, new_psc, spikes, new_history = self.dispatch.forward(
                v, r, asc, rise, psc, currents, history, self.t_ref_steps, self.dt, self.v_th,
            )
            words = pack_step_record(spikes, r)
        else:
            if not nest_record_forward_available():
                raise RuntimeError("In-place step records require the rebuilt record forward operators.")
            buffer, row = record_row
            pre_reset = None
            new_v, new_r, new_asc, new_rise, new_psc, spikes, new_history, words = self.dispatch.forward_record(
                v, r, asc, rise, psc, currents, history, self.t_ref_steps, self.dt, self.v_th,
                buffer, row,
            )
        penalty = glif_state_ops._OPS.dpointnet_voltage_penalty_forward(new_v, mode=self.penalty_mode)
        new_state = (new_history, new_v, new_r, new_asc, new_rise, new_psc, noise_step + 1)
        return spikes, new_state, (pre_reset, words), penalty

    def advance_asc(self, asc, words):
        """Exact ASC after the recorded step ``words`` whose pre-step ASC is ``asc``."""
        return advance_asc_record(asc, words, self.asc_table)

    def shift_history(self, history, words):
        """The previous step's delay window from this step's and the oldest record's ``words``."""
        return shift_history_record(history, words, self.neurons)

    # -- reverse ------------------------------------------------------------------

    def reverse(self, record, previous_asc, carry, spike_cotangent, penalty_cotangent,
                sources, source_credit, accumulators, trace=None):
        """One reverse step.

        ``record`` is the step's ``(pre_reset_voltage, words)``, or
        ``(record_buffer, words, row)`` with a ``[time, batch, neurons]``
        voltage buffer (tensor or resource variable) read at ``row``;
        ``previous_asc`` the exact ASC before the step; ``carry`` the cotangents
        of the step's outputs ``(voltage, asc, rise, psc, new_history)`` with an
        optional sixth entry, a second new-history cotangent (normally the
        following step's recurrent source credit) that the native backward sums
        with the fifth; ``spike_cotangent``
        the loss cotangent of the step's spikes (compute dtype) and
        ``penalty_cotangent`` that of its ``[batch]`` penalty; ``trace`` optionally
        adds sampled-trace cotangents ``(column, indices, mask, groups)`` onto
        ``spike_cotangent`` inside the native prelude. ``sources`` are
        the per-population spike tensors the step projected, ``source_credit``
        per-population booleans requesting dense source cotangents (silent
        sources included) and ``accumulators`` the caller-owned per-population
        FP32 weight-gradient accumulators (``None`` to skip a population).

        Returns ``(previous_carry, source_cotangents, accumulators)`` where the
        previous carry holds the cotangents of the step's inputs
        ``(voltage, asc, rise, psc, old_history)`` **without** the recurrent
        source credit, which is ``source_cotangents[0]``; pass it as the sixth
        carry entry of the previous step (or add it) to complete the history
        cotangent.
        """
        pre_reset, words = record[:2]
        row = record[2] if len(record) > 2 else 0
        grad_v, grad_asc, grad_rise, grad_psc, grad_history = carry[:5]
        history_addend = carry[5] if len(carry) > 5 else None
        # The penalty acts on the post-reset membrane: the reset value where the
        # neuron fired, otherwise the recorded pre-reset value (blocked neurons
        # were already clamped to the reset in the forward).
        threshold, blocked, grad_v, _, spike_cotangent = direct_reverse_record(
            pre_reset, words, grad_v, penalty_cotangent, self.v_th, self.v_reset,
            spike_base=spike_cotangent, refractory_dtype=self.refractory_dtype,
            penalty_mode=self.penalty_mode, row=row, trace=trace,
        )
        v_grad, asc_grad, rise_grad, psc_grad, currents_grad, old_history_grad = self.dispatch(
            threshold, blocked, self.dt, tf.zeros_like(threshold), grad_v,
            grad_asc, grad_rise, grad_psc, previous_asc, self.v_th, self.dampening, self.gauss_std,
            spike_cotangent, grad_history, history_addend,
        )
        if self.scales_currents:
            currents_grad = currents_grad * self.lr_scale
        currents_grad = tf.reshape(currents_grad, [self.batch * self.neurons, self.basis_count])
        source_cotangents, new_accumulators = [], []
        for population, source, credit, accumulator in zip(
            self.populations, sources, source_credit, accumulators,
        ):
            connectivity = population["connectivity"]
            attributes = dict(
                Tindex=tf.as_dtype(connectivity["index_dtype"]), n_post=self.neurons,
                n_edges=connectivity["n_edges"], n_pairs=connectivity["n_pairs"],
                **csr_spike_ops._uint8_type_options(connectivity),
            )
            if population["recurrent"]:
                if not credit:
                    raise ValueError("The recurrent population always receives source credit.")
                if accumulator is None:
                    accumulator = tf.zeros_like(population["variable"])
                source_grad, accumulated = csr_spike_ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master(
                    source, currents_grad, _metadata(connectivity), population["variable"],
                    self.basis, self.recurrent_scale, accumulator,
                    use_javier_batch32_backward=self.cell.use_javier_recurrent_vjp,
                    compute_spike_gradient=True, **attributes,
                )
                source_cotangents.append(source_grad)
                new_accumulators.append(accumulated)
                continue
            if credit:
                source_grad, weight_grad = csr_spike_ops._OPS.dpointnet_csr_spike_grad_csr_master(
                    source, currents_grad, _metadata(connectivity), population["variable"],
                    self.basis, tf.ones((), self.spike_dtype),
                    use_packed_sm120_backward=False, write_csr_weight_gradient=True,
                    use_small_batch_backward=False, use_javier_batch32_backward=False,
                    **attributes,
                )
                source_cotangents.append(source_grad)
            else:
                source_cotangents.append(None)
                weight_grad = None if accumulator is None else csr_spike_ops._OPS.dpointnet_csr_weight_grad(
                    source, currents_grad, _metadata(connectivity), self.basis,
                    use_packed_sm120_backward=csr_spike_ops._resolve_packed_sm120_backward(
                        population["net"].get("use_packed_sm120_backward", False),
                        source, connectivity, self.basis,
                    ),
                    write_csr_weight_gradient=True, **attributes,
                )
            new_accumulators.append(None if accumulator is None else accumulator + weight_grad)
        previous = (v_grad, asc_grad, rise_grad, psc_grad, old_history_grad)
        return previous, tuple(source_cotangents), tuple(new_accumulators)


class DirectRecordedAdjointRunner:
    """Streamed statistics plus exact direct recorded adjoint for one GLIF3Cell.

    ``__call__(inputs, initial_state, sampled_traces)`` has the
    ``SpikeStatisticsRunner`` contract: ``SpikeStatistics(counts, traces,
    sequences, final_state)`` whose only sequence is the per-step mean voltage
    penalty ``[batch, time]``. Weight gradients reach the enclosing tape through
    the custom gradient's variable list; input gradients are not provided.
    """

    def __init__(self, core_model, cell, sequence_length, chunk_size, count_windows=None,
                 differentiate_inputs=False, **kwargs):
        unsupported = direct_records_supported(cell)
        if unsupported:
            raise ValueError("Direct recorded adjoint requires: " + ", ".join(unsupported))
        if differentiate_inputs:
            raise ValueError("Direct recorded adjoint does not differentiate inputs.")
        unknown = set(kwargs) - {"pack_spike_checkpoints", "variable_gradient_transform", "n_sequence_outputs"}
        if unknown:
            raise TypeError(f"Unexpected runner options: {sorted(unknown)}")
        if kwargs.get("variable_gradient_transform") is not None:
            raise ValueError("Direct recorded adjoint returns CSR-master gradients without a transform.")
        self.core_model = core_model
        self.cell = cell
        self.sequence_length = int(sequence_length)
        self.chunk_size = int(chunk_size)
        if self.sequence_length <= 0 or not 1 <= self.chunk_size <= self.sequence_length:
            raise ValueError("chunk_size must satisfy 1 <= chunk_size <= sequence_length.")
        self.n_chunks = -(-self.sequence_length // self.chunk_size)
        self.remainder_size = self.sequence_length % self.chunk_size
        # Exact prior-ASC values are re-walked per chunk in two passes: coarse
        # sub-snapshots every ``asc_block`` steps, then one block at a time, so
        # the resident buffers hold about asc_block + chunk_size / asc_block
        # tensors instead of chunk_size, for one extra walk of the chunk's
        # first blocks per chunk.
        self.asc_block = -(-self.chunk_size // 2)
        self.differentiate_inputs = False
        self.count_windows = tuple(
            _window(window, self.sequence_length)
            for window in (((0, self.sequence_length),) if count_windows is None else count_windows)
        )
        if not self.count_windows:
            raise ValueError("Statistics require at least one count window.")
        self.n_sequence_outputs = 2

    # -- input sources --------------------------------------------------------------

    def _prepare_noise(self, step, inputs, initial_state):
        """Prepared CPU Poisson tables, cast once to the compute dtype on the device."""
        cell = step.cell
        prepare = cell._use_prepared_cpu_poisson
        budget = PREPARED_POISSON_BUDGET_BYTES
        if prepare == "auto":
            prepare = _prepared_poisson_admitted(cell, inputs)
            budget = PREPARED_POISSON_AUTO_CEILING_BYTES
        if not (prepare and _internal_poisson_inputs(cell)):
            return {}
        tables = _prepare_cpu_poisson(cell, inputs, initial_state, budget)
        with step.device_scope():
            return {key: tf.identity(tf.cast(table, step.spike_dtype)) for key, table in tables.items()}

    def _input_frame(self, step, population, inputs, time, tables, noise_steps, clock=None):
        """The population's spikes/counts emitted at ``time`` in the compute dtype.

        Prepared Poisson tables are indexed by the relative time, as in
        ``_prepared_noise_step``; ``noise_steps`` is the initial ``[batch]``
        noise-step state for the unprepared sampler. ``clock`` is an optional
        device-resident copy of ``time`` so the table gather needs no host
        index upload (whose completion polling stalls the dispatch thread).
        """
        if population["poisson"]:
            table = tables.get(id(population["net"]))
            if table is not None:
                frame = tf.gather(table, time if clock is None else clock)
            else:
                frame = tf.cast(step.cell.sample_noise_spikes(
                    step.batch, noise_steps + tf.cast(time, noise_steps.dtype), population["net"],
                ), step.spike_dtype)
        else:
            start, stop = population["columns"]
            frame = tf.cast(inputs[:, time, start:stop], step.spike_dtype)
        # A tensor time index drops the static width the sweep validates.
        return tf.ensure_shape(frame, (step.batch, population["width"]))

    def _reverse_source(self, step, population, inputs, time, tables, noise_steps, initial_state):
        """``x(time), x(time - 1), ...`` concatenated as the forward projected them."""
        frames = [self._input_frame(step, population, inputs, time, tables, noise_steps)]
        start, _ = population["history"]
        width = population["width"]
        for k in range(1, population["history_steps"] + 1):
            def replay(k=k):
                return self._input_frame(step, population, inputs, time - k, tables, noise_steps)

            def initial(k=k):
                with tf.device("/CPU:0"):
                    begin = tf.stack([0, start + (k - 1 - time) * width])
                return tf.cast(tf.slice(initial_state[7], begin, [step.batch, width]), step.spike_dtype)

            with tf.device("/CPU:0"):
                replayed = time - k >= 0
            frames.append(tf.cond(replayed, replay, initial))
        return frames[0] if len(frames) == 1 else tf.concat(frames, axis=1)

    # -- statistics -------------------------------------------------------------------

    def _trace_plan(self, sampled_traces, step):
        """One merged gather/scatter plan for every sampled-trace specification.

        All sampled entries are addressed as flat ``[batch, neurons]`` indices
        so one step costs one gather (forward) and one scatter (reverse) for all
        specifications together; the per-specification outputs are split by
        their ``rows * per_trial`` extents.
        """
        if not sampled_traces:
            return None
        indices, masks, extents, groups, offset = [], [], [], [], 0
        for spec in sampled_traces:
            begin, end = spec.batch_range
            rows = end - begin
            per_trial, max_count = int(spec.neuron_mask.shape[1]), int(spec.neuron_mask.shape[2])
            ids = tf.cast(spec.sample_ids, tf.int64)
            with tf.control_dependencies([
                tf.debugging.assert_greater_equal(ids, tf.cast(0, ids.dtype)),
                tf.debugging.assert_less(ids, tf.cast(step.neurons, ids.dtype)),
            ]):
                ids = tf.identity(ids)
            samples = tf.broadcast_to(tf.range(begin, end, dtype=tf.int64)[:, None], tf.shape(ids))
            indices.append(tf.reshape(tf.stack([samples, ids], axis=-1), [-1, 2]))
            masks.append(tf.reshape(spec.neuron_mask, [-1]))
            extents.append(rows * per_trial)
            # Flat entry k of this specification belongs to output row k // max_count.
            # int64: TensorFlow keeps int32 GPU tensors in host memory, which would
            # re-upload the segment ids every time step.
            groups.append(tf.repeat(tf.range(rows * per_trial, dtype=tf.int64) + offset, max_count))
            offset += rows * per_trial
        # Integer index arithmetic lands on the host; materialize the plan on the
        # compute device once so the loops do not re-upload it every step.
        with step.device_scope():
            plan = dict(
                indices=tf.identity(tf.concat(indices, axis=0)),
                mask=tf.identity(tf.cast(tf.concat(masks, axis=0), step.spike_dtype)),
                mask32=tf.identity(tf.cast(tf.concat(masks, axis=0), tf.float32)),
                groups=tf.identity(tf.concat(groups, axis=0)),
            )
        plan.update(extents=tuple(extents), total=offset)
        return plan

    # -- record storage ------------------------------------------------------------------

    def _scratch(self, step, name, shape, dtype):
        """Persistent untracked scratch variable, created once per shape/dtype."""
        key = (name, tuple(shape), tf.as_dtype(dtype).name)
        buffer = getattr(self, "_scratch_buffers", {}).get(key)
        if buffer is None:
            with tf.init_scope(), step.device_scope():
                buffer = tf.Variable(tf.zeros(shape, dtype), trainable=False, name=name)
            buffer = self.cell._untracked_variable(buffer)
            self._scratch_buffers = {**getattr(self, "_scratch_buffers", {}), key: buffer}
        return buffer

    def _record_buffer(self, step, length):
        """Persistent ``[length, batch, neurons]`` voltage-record scratch variable."""
        return self._scratch(
            step, "direct_recorded_voltage_scratch", [length, step.batch, step.neurons], step.state_dtype,
        )

    # -- forward driver ------------------------------------------------------------------

    def _forward(self, inputs, initial_state, sampled_traces, time_check):
        cell = self.cell
        batch, length = int(inputs.shape[0]), self.sequence_length
        with _rollout_context(cell):
            step = DirectRecordedStep(cell, batch, initial_state)
            neurons = step.neurons
            tables = self._prepare_noise(step, inputs, initial_state)
            record_rollout_acceleration(getattr(cell, "acceleration_report", None), {
                "use_direct_state_rnn_loop": True,
                "use_prepared_cpu_poisson": bool(tables),
                "temporal_path": "direct_recorded_forward",
                "record_format": RECORD_FORMAT,
            })
            noise_steps = initial_state[6]
            # Per-step FP32 records as separate allocations land in oversized free
            # chunks (about 1.5x their size on the full network); a loop-carried
            # contiguous buffer makes Grappler's memory optimizer swap loop state to
            # the host. A persistent untracked scratch variable, written by row,
            # avoids both; it is sized once per (length, batch) and reused.
            scratch = self._record_buffer(step, length)
            buffered = scratch is not None
            in_place = buffered and nest_record_forward_available()
            # The scratch variable is written from the closure; the loop variable then
            # only carries the TensorList alternative (an empty placeholder otherwise).
            voltages = tf.zeros([0], step.state_dtype) if buffered else tf.raw_ops.EmptyTensorList(
                element_shape=tf.constant([batch, neurons], tf.int32), max_num_elements=length,
                element_dtype=step.state_dtype,
            )
            # Word records are indexed by time + delay: the first ``delay`` entries hold
            # the initial history slots (oldest first), so the reverse loop reads any
            # delayed spike frame without a conditional.
            words = tf.raw_ops.EmptyTensorList(
                element_shape=tf.constant([batch, step.words], tf.int32),
                max_num_elements=length + step.delay, element_dtype=tf.int32,
            )
            no_refractory = tf.zeros([batch, neurons], step.refractory_dtype)
            for slot in reversed(range(step.delay)):
                initial_slot = tf.cast(initial_state[0][:, slot * neurons:(slot + 1) * neurons], step.spike_dtype)
                words = tf.raw_ops.TensorListPushBack(
                    input_handle=words, tensor=pack_step_record(initial_slot, no_refractory),
                )
            asc_snapshots = self._scratch(
                step, "direct_recorded_asc_snapshot_scratch", [self.n_chunks, batch, 2 * neurons],
                step.state_dtype,
            )
            penalties = tf.TensorArray(tf.float32, size=length, element_shape=(batch,))
            trace_plan = self._trace_plan(sampled_traces, step)

            def body(time, clock, state, voltages, words, penalties):
                sources = [tf.cast(state[0], step.spike_dtype)]
                new_input_history = []
                for population in step.populations[1:]:
                    source = self._input_frame(step, population, inputs, time, tables, noise_steps, clock)
                    if population["history_steps"]:
                        start, stop = population["history"]
                        source = tf.concat(
                            [source, tf.cast(state[7][:, start:stop], step.spike_dtype)], axis=1,
                        )
                        new_input_history.append(source[:, : stop - start])
                    sources.append(source)
                # With the record forward the state kernel writes the pre-reset row
                # itself (host loop counter as the row; no device index upload).
                spikes, new_state, (pre_reset, record), penalty = step.forward(
                    sources, state, record_row=(scratch.handle, time) if in_place else None,
                )
                if step.has_input_history:
                    new_state += (tf.concat(new_input_history, axis=1),)
                if buffered and not in_place:
                    tf.raw_ops.ResourceScatterUpdate(
                        resource=scratch.handle, indices=clock[None], updates=pre_reset[None],
                    )
                elif not buffered:
                    voltages = tf.raw_ops.TensorListPushBack(input_handle=voltages, tensor=pre_reset)
                words = tf.raw_ops.TensorListPushBack(input_handle=words, tensor=record)
                penalties = penalties.write(time, penalty)
                return time + 1, clock + 1, new_state, voltages, words, penalties

            def chunk(index, clock, state, voltages, words, penalties):
                tf.raw_ops.ResourceScatterUpdate(
                    resource=asc_snapshots.handle, indices=(clock // self.chunk_size)[None],
                    updates=state[3][None],
                )
                stop = tf.minimum((index + 1) * self.chunk_size, length)
                _, clock, state, voltages, words, penalties = tf.while_loop(
                    lambda time, *_: time < stop, body,
                    (index * self.chunk_size, clock, state, voltages, words, penalties),
                    parallel_iterations=1,
                )
                return index + 1, clock, state, voltages, words, penalties

            with tf.control_dependencies([time_check]):
                state = tuple(tf.identity(value) for value in initial_state)
            # Device-resident int64 step clock beside the host loop counter: device
            # index consumers read it without a per-step host-to-device upload.
            with step.device_scope():
                clock = tf.identity(tf.zeros([], tf.int64))
            _, _, state, voltages, words, penalties = tf.while_loop(
                lambda index, *_: index < self.n_chunks, chunk,
                (tf.constant(0), clock, state, voltages, words, penalties),
                parallel_iterations=1,
            )
            # Statistics come from the packed records after the loop: a stacked
            # [time, batch, words] view costs one reduction per window instead of
            # several small ops per step. The initial-history rows stay in the
            # stack and the windows shift by ``delay`` instead: TensorFlow's int32
            # StridedSlice is a host-memory kernel on GPUs, so slicing them off
            # would round-trip the whole record through the host.
            with step.device_scope():
                stacked = tf.raw_ops.TensorListStack(
                    input_handle=words, element_shape=tf.constant([batch, step.words], tf.int32),
                    element_dtype=tf.int32, num_elements=length + step.delay,
                )
                counts = tuple(
                    record_counts(stacked, neurons, begin + step.delay, end + step.delay)
                    for begin, end in self.count_windows
                )
                trace_outputs = ()
                if trace_plan is not None:
                    offset = 0
                    for spec, extent in zip(sampled_traces, trace_plan["extents"]):
                        traces = record_traces(
                            stacked, trace_plan["indices"], trace_plan["mask32"], trace_plan["groups"],
                            neurons, trace_plan["total"], spec.window[0] + step.delay,
                            spec.window[1] + step.delay,
                        )
                        trace_outputs += (tf.cast(traces[offset:offset + extent], step.spike_dtype),)
                        offset += extent
        penalty_sequence = tf.transpose(penalties.stack())
        records = (scratch if buffered else voltages, words, asc_snapshots, tables, noise_steps, buffered)
        return counts, trace_outputs, (penalty_sequence,), tuple(state), records

    # -- reverse driver ------------------------------------------------------------------

    def _backward(self, inputs, initial_state, final_state, records, sampled_traces,
                  count_gradients, trace_gradients, penalty_gradient, state_gradients, variables):
        cell = self.cell
        batch, length = int(inputs.shape[0]), self.sequence_length
        voltages, words, asc_snapshots, tables, noise_steps, buffered = records
        with _rollout_context(cell):
            step = DirectRecordedStep(cell, batch, initial_state)
            neurons, delay, S, T = step.neurons, step.delay, step.spike_dtype, step.state_dtype
            record_rollout_acceleration(getattr(cell, "acceleration_report", None), {
                "temporal_path": "direct_recorded_vjp",
            })
            trace_plan = self._trace_plan(sampled_traces, step)

            # Loss cotangents of the step's spikes, independent of the recurrence.
            constant_count_gradient, windowed = None, []
            for gradient, (begin, end) in zip(count_gradients, self.count_windows):
                value = tf.zeros([batch, neurons], tf.float32) if gradient is None else gradient
                if (begin, end) == (0, length):
                    constant_count_gradient = (
                        value if constant_count_gradient is None else constant_count_gradient + value
                    )
                else:
                    windowed.append((value, begin, end))
            # The constant window cotangent is rounded to the compute dtype once; the
            # native prelude adds each step's sampled-trace cotangents onto it with
            # one rounding per entry, as the reference's compute-dtype sum does.
            constant_spike_cotangent = (
                tf.zeros([batch, neurons], S) if constant_count_gradient is None
                else tf.cast(constant_count_gradient, S)
            )
            # Trace cotangents padded to the full horizon, time-major: zero
            # contributions outside a window cost one small scatter but no
            # host-side conditional per step.
            trace_columns = None
            if trace_plan is not None:
                trace_columns = tf.transpose(tf.concat([
                    tf.zeros([extent, length], tf.float32) if gradient is None
                    else tf.pad(tf.cast(gradient, tf.float32), ((0, 0), (spec.window[0], length - spec.window[1])))
                    for gradient, spec, extent in zip(trace_gradients, sampled_traces, trace_plan["extents"])
                ], axis=0))
            penalty_gradient = (
                tf.zeros([batch, length], tf.float32) if penalty_gradient is None else penalty_gradient
            )

            def host_mask(time, begin, end):
                with tf.device("/CPU:0"):
                    return tf.cast((time >= begin) & (time < end), tf.float32)

            def spike_base(time):
                """Window cotangents of the step's spikes before the trace additions."""
                parts = [constant_spike_cotangent]
                for value, begin, end in windowed:
                    parts.append(tf.cast(value * host_mask(time, begin, end), S))
                return tf.add_n(parts) if len(parts) > 1 else parts[0]

            def trace_at(time):
                if trace_plan is None:
                    return None
                return (trace_columns[time], trace_plan["indices"], trace_plan["mask32"], trace_plan["groups"])

            def voltage_record(time):
                """``(pre_reset, words, row)`` for ``DirectRecordedStep.reverse``."""
                if buffered:
                    # The scratch variable itself: the native prelude reads row ``time``
                    # in place (a full read would copy the copy-on-read variable).
                    return (voltages, words_at(time), time)
                return (tf.raw_ops.TensorListGetItem(
                    input_handle=voltages, index=time,
                    element_shape=tf.constant([batch, neurons], tf.int32), element_dtype=T,
                ), words_at(time), 0)

            def words_at(time):
                """Record of ``time``; ``-delay <= time < 0`` are the initial history slots."""
                return tf.raw_ops.TensorListGetItem(
                    input_handle=words, index=time + delay,
                    element_shape=tf.constant([batch, step.words], tf.int32), element_dtype=tf.int32,
                )

            def spikes_at(time):
                return step.unpack(words_at(time))[0]

            populations = step.populations
            source_credit = tuple(population["recurrent"] for population in populations)
            accumulate = tuple(population["trainable"] for population in populations)

            block = self.asc_block
            blocks_per_chunk = -(-self.chunk_size // block)

            def walk_asc(begin, end, asc, buffer, stride):
                """Advance ``asc`` over ``[begin, end]`` storing ``asc(begin + j * stride)`` at ``j``.

                Every step writes slot ``ceil((time - begin) / stride)``; the
                block-boundary step writes a slot last, so no conditional is
                needed. ``buffer`` must have ``ceil((end - begin) / stride) + 1``
                slots; the value at ``end`` itself is stored too.
                """
                def body(time, asc, buffer):
                    index = (time - begin + stride - 1) // stride
                    return time + 1, step.advance_asc(asc, words_at(time)), buffer.write(index, asc)

                _, asc, buffer = tf.while_loop(
                    lambda time, *_: time < end, body, (begin, asc, buffer), parallel_iterations=1,
                )
                return buffer.write((end - begin + stride - 1) // stride, asc)

            def chunk_snapshot(chunk_index):
                row = tf.raw_ops.ResourceGather(
                    resource=asc_snapshots.handle, indices=chunk_index[None], dtype=T,
                )
                return tf.reshape(row, [batch, 2 * neurons])

            def reverse_step(time, carry, window, accumulators, asc_buffer, block_start):
                sources = [window]
                for population, keep in zip(populations[1:], accumulate[1:]):
                    sources.append(
                        self._reverse_source(step, population, inputs, time, tables, noise_steps, initial_state)
                        if keep else tf.zeros([batch, 0], S)
                    )
                carry, credit, accumulators = step.reverse(
                    voltage_record(time), asc_buffer.read(time - block_start), carry,
                    spike_base(time), penalty_gradient[:, time], sources, source_credit,
                    tuple(acc if keep else None for acc, keep in zip(accumulators, accumulate)),
                    trace=trace_at(time),
                )
                carry = carry + (credit[0],)
                accumulators = tuple(
                    tf.zeros([0], tf.float32) if acc is None else acc for acc in accumulators
                )
                # At time 0 the shifted-in frame is never read; clamp it to the oldest record.
                window = step.shift_history(window, words_at(tf.maximum(time - 1 - delay, -delay)))
                return time - 1, carry, window, accumulators, asc_buffer, block_start

            def reverse_block(block_index, carry, window, accumulators, coarse, chunk_start, chunk_stop):
                block_start = chunk_start + block_index * block
                block_stop = tf.minimum(block_start + block, chunk_stop)
                # The last block's value at block_stop - 1 is written by the walk's
                # final write; the walk stops one step short of the block end.
                fine = walk_asc(
                    block_start, block_stop - 1, coarse.read(block_index),
                    tf.TensorArray(T, size=block, element_shape=(batch, 2 * neurons)), 1,
                )
                _, carry, window, accumulators, _, _ = tf.while_loop(
                    lambda time, *_: time >= block_start, reverse_step,
                    (block_stop - 1, carry, window, accumulators, fine, block_start),
                    parallel_iterations=1,
                )
                return block_index - 1, carry, window, accumulators, coarse, chunk_start, chunk_stop

            def reverse_chunk(chunk_index, carry, window, accumulators):
                start = chunk_index * self.chunk_size
                stop = tf.minimum(start + self.chunk_size, length)
                last_block = (stop - start - 1) // block
                # Coarse sub-snapshots at every block start up to the last block only.
                coarse = walk_asc(
                    start, start + last_block * block, chunk_snapshot(chunk_index),
                    tf.TensorArray(T, size=blocks_per_chunk + 1, element_shape=(batch, 2 * neurons)), block,
                )
                _, carry, window, accumulators, _, _, _ = tf.while_loop(
                    lambda block_index, *_: block_index >= 0, reverse_block,
                    (last_block, carry, window, accumulators, coarse, start, stop),
                    parallel_iterations=1,
                )
                return chunk_index - 1, carry, window, accumulators

            def floating_gradient(gradient, value):
                return tf.zeros_like(value) if gradient is None else tf.cast(gradient, value.dtype)

            carry = (
                floating_gradient(state_gradients[1], final_state[1]),
                floating_gradient(state_gradients[3], final_state[3]),
                floating_gradient(state_gradients[4], final_state[4]),
                floating_gradient(state_gradients[5], final_state[5]),
                floating_gradient(state_gradients[0], final_state[0]),
                tf.zeros_like(final_state[0]),
            )
            # The delay window of the last step as one [batch, neurons * delay] tensor
            # (newest slot first), as the forward's history state holds it.
            window = tf.concat(
                [spikes_at(tf.constant(length - 1 - k)) for k in range(1, delay + 1)], axis=1,
            )
            accumulators = tuple(
                tf.zeros_like(population["variable"]) if keep else tf.zeros([0], tf.float32)
                for population, keep in zip(populations, accumulate)
            )
            _, carry, _, accumulators = tf.while_loop(
                lambda chunk_index, *_: chunk_index >= 0, reverse_chunk,
                (tf.constant(self.n_chunks - 1), carry, window, accumulators),
                parallel_iterations=1,
            )
        grad_v, grad_asc, grad_rise, grad_psc, grad_history, history_credit = carry
        grad_history = grad_history + history_credit
        initial_gradients = [None] * len(initial_state)
        initial_gradients[0] = tf.cast(grad_history, initial_state[0].dtype)
        initial_gradients[1], initial_gradients[3] = grad_v, grad_asc
        initial_gradients[4], initial_gradients[5] = grad_rise, grad_psc
        if step.has_input_history and initial_state[7].dtype.is_floating:
            initial_gradients[7] = tf.zeros_like(initial_state[7])
        owned = [
            (population["variable"], accumulator)
            for population, accumulator, keep in zip(populations, accumulators, accumulate) if keep
        ]
        variable_gradients = []
        for variable in variables:
            gradient = None
            for owner, accumulator in owned:
                if owner is variable or getattr(owner, "value", None) is variable:
                    gradient = accumulator
            if gradient is None:
                raise ValueError(
                    "Direct recorded adjoint only differentiates the recurrent and named-input "
                    f"CSR masters; unsupported trainable variable {getattr(variable, 'name', variable)}."
                )
            variable_gradients.append(gradient)
        return tuple(initial_gradients), variable_gradients

    # -- public ----------------------------------------------------------------------

    def __call__(self, inputs, initial_state, sampled_traces=()):
        sampled_traces = tuple(sampled_traces)
        for spec in sampled_traces:
            _window(spec.window, self.sequence_length)
        initial_state = tuple(initial_state)
        if inputs.shape[0] is None or inputs.shape[1] != self.sequence_length:
            raise ValueError(
                "Direct recorded adjoint requires a static batch and input length equal to sequence_length."
            )
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
                counts, traces, sequences, state, records = self._forward(
                    inputs, initial_state, sampled_traces, checks[0],
                )

            def grad(*upstream, variables=None):
                variables = tuple(variables or ())
                count_end = len(self.count_windows)
                trace_end = count_end + len(sampled_traces)
                initial_gradients, variable_gradients = self._backward(
                    inputs, initial_state, state, records, sampled_traces,
                    upstream[:count_end], upstream[count_end:trace_end],
                    upstream[trace_end], upstream[output_count:], variables,
                )
                arguments = (None,) + initial_gradients
                return (arguments, variable_gradients) if variables else arguments

            return counts + traces + sequences + state, grad

        flat = rollout(inputs, *initial_state)
        count_end = len(self.count_windows)
        trace_end = count_end + len(sampled_traces)
        return SpikeStatistics(
            tuple(flat[:count_end]), tuple(flat[count_end:trace_end]),
            tuple(flat[trace_end:output_count]), tuple(flat[output_count:]),
        )
