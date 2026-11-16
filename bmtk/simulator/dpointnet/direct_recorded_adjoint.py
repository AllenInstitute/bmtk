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

A :class:`SpikeExchange` attached to a named spike input lets that input carry
recurrent spikes of neurons simulated elsewhere (another process or device):
the runner asks the exchange for the input's frame before every step, giving it
the spikes this cell emitted on the previous step, and in the reverse loop hands
it the complete credit of each frame, receiving the credit of the cell's own
previous-step spikes in return. A network split by neurons across runners then
trains exactly like the unsplit network.

Structure adapted from Javier's ``v1_model_utils/recorded_adjoint.py`` reverse
step (credit: Javier); DPointNet keeps its NEST dynamics, event map, dtype
boundaries, independent per-source VJPs and canonical master ownership.
"""

import math

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
    NestDirectDispatch, advance_asc_block_record, advance_asc_block_record_available,
    advance_asc_record, direct_reverse_ops_available,
    direct_reverse_record, nest_record_forward_available, pack_step_record,
    record_counts, record_traces, shift_history_record, step_record_ops_available,
    unpack_step_record,
)
from .custom_ops.record_store_ops import (
    DEFAULT_PREFETCH, HostRecordStore, host_record_store_available, record_store_op_status,
)
from .spike_statistics import SpikeStatistics, _window

RECORD_FORMAT = "direct_step_record_v1"

# Record surfaces the direct route can keep in pinned host memory instead of
# device memory (``record_offload``): the per-step FP32 pre-reset membrane rows
# ("voltage") and the per-chunk records, i.e. the packed spike/refractory word
# frames and the chunk-start ASC snapshots ("history"). Offload moves bytes
# only; values, gradients and their arithmetic order are unchanged.
RECORD_OFFLOAD_CHOICES = ("voltage", "history")
# Device staging slots the forward writes before the oldest frame must have
# left the device: a few per-step membrane rows, three per-chunk word frames,
# two chunk-start ASC snapshots. Per-chunk frames are read one chunk at a time
# in reverse, so their stores prefetch one frame ahead (three resident).
VOLTAGE_STAGING_SLOTS = 8
HISTORY_STAGING_SLOTS = 3
ASC_STAGING_SLOTS = 2
CHUNK_PREFETCH = 1


def normalize_record_offload(value):
    """``record_offload`` request as a sorted tuple of surface names.

    Accepts ``None``/``False``/``"none"``/empty (no offload), ``True``/``"all"``
    (both surfaces), one name or an iterable of names from
    :data:`RECORD_OFFLOAD_CHOICES`.
    """
    if value is None or value is False or (isinstance(value, str) and value == "none"):
        return ()
    if value is True or (isinstance(value, str) and value == "all"):
        return tuple(RECORD_OFFLOAD_CHOICES)
    if isinstance(value, str):
        names = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        names = tuple(value)
        if not names:
            return ()
    else:
        names = (value,)
    unknown = [name for name in names if not isinstance(name, str) or name not in RECORD_OFFLOAD_CHOICES]
    if unknown:
        raise ValueError(
            f"record_offload accepts none/false, true/'all' or names from {RECORD_OFFLOAD_CHOICES}, "
            f"got {value!r}."
        )
    return tuple(name for name in RECORD_OFFLOAD_CHOICES if name in names)


class SpikeExchange:
    """Transport for a named spike input fed by recurrent spikes of other runners.

    The input keeps its own CSR projection, delays and (optionally trainable)
    weights; only the source of its frames changes. With ``x(t)`` the input
    frame projected at step ``t`` and ``z(t)`` this cell's spikes:

    * ``forward(time, spikes)`` runs before step ``time`` with ``spikes =
      z(time - 1)`` (``z(-1)`` is the newest slot of the initial spike
      history) and returns ``x(time)``, a ``[batch, width]`` binary frame.
    * ``reverse(time, credit)`` runs while step ``time`` is reversed, after
      every step that projected ``x(time)``; ``credit`` is the complete FP32
      cotangent ``[batch, width]`` of ``x(time)`` (silent sources included).
      It returns the cotangent ``[batch, neurons]`` of this cell's ``z(time -
      1)`` contributed by the other runners' use of it; ``time == 0`` returns
      the credit of the initial spike history.
    * ``begin()`` runs once before the forward loop and once before the
      reverse loop; it may return an op/tensor to order against, or ``None``.

    An exchanged input takes no columns in the runner's ``inputs``: the
    replay holds the remaining named inputs' columns, in cell order, so a
    partition never materializes a dense frame per remote source.

    Both hooks run exactly ``sequence_length`` times per rollout in time order
    (forward) and reverse time order (reverse), so collective transports stay
    matched across runners. Spikes are cast to the compute dtype; frames are
    stored bit-packed and must be 0/1. The spike-gradient scale of the
    recurrent population (``recurrent_spike_gradient_scale``) also scales the
    exchanged credit, as it would scale the same edges' local recurrence.
    Frames before step 0 come from the input's initial history state (also
    0/1); their credit is not returned, as for every named input.
    """

    def begin(self):
        return None

    def forward(self, time, spikes):
        raise NotImplementedError

    def reverse(self, time, credit):
        raise NotImplementedError


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


def direct_record_bytes(cell, batch, length, chunk_size, free_bytes=None, offload=()):
    """``(fits, reason)``: advisory estimate of the direct route's record payload.

    The payload is ``length * batch * neurons`` state-dtype membrane values
    plus ``2 * ceil(neurons / 32)`` int32 words per step and the per-chunk and
    per-block ASC snapshots. ``free_bytes`` defaults to the current device
    report. ``fits`` is ``None`` when the payload or the device memory is
    unknown, ``False`` when it exceeds ``DIRECT_RECORD_MEMORY_FRACTION`` of the
    free memory. The estimate never disables the route; callers warn.

    With ``offload`` surfaces (see :func:`normalize_record_offload`) the
    offloaded records count only their device staging and prefetch frames, and
    the reason names the pinned host bytes they take instead.
    """
    neurons = getattr(cell, "_n_neurons", None)
    if not neurons or not isinstance(batch, int) or batch <= 0 or not isinstance(length, int) or length <= 0:
        return None, "record payload unknown (static batch, sequence length and neuron count required)"
    offload = normalize_record_offload(offload)
    selective = getattr(cell, "state_precision", "compute") == "selective"
    state_bytes = tf.as_dtype(cell.state_dtype if selective else cell.compute_dtype).size
    words = 2 * (-(-neurons // 32))
    chunk_size = max(1, min(int(chunk_size), length))
    n_chunks = -(-length // chunk_size)
    asc_block = max(1, math.isqrt(chunk_size - 1) + 1)
    snapshots = n_chunks + asc_block + -(-chunk_size // asc_block) + 2
    membrane = length * batch * neurons * state_bytes
    history = (length + int(getattr(cell, "max_delay", 1))) * batch * words * 4
    host = 0
    if "voltage" in offload:
        host += membrane
        membrane = (VOLTAGE_STAGING_SLOTS + DEFAULT_PREFETCH + 2) * batch * neurons * state_bytes
    asc_snapshot = batch * 2 * neurons * state_bytes
    if "history" in offload:
        frame = chunk_size * batch * words * 4
        host += (n_chunks + 1) * frame + n_chunks * asc_snapshot
        history = (HISTORY_STAGING_SLOTS + 4) * frame
        snapshots += ASC_STAGING_SLOTS + CHUNK_PREFETCH + 2 - n_chunks
    payload = membrane + history + snapshots * asc_snapshot
    offloaded = f"; {host / 2**30:.2f} GiB pinned host memory for {', '.join(offload)} offload" if offload else ""
    if free_bytes is None:
        free_bytes = device_free_bytes()
        if free_bytes is None:
            return None, f"record payload {payload / 2**30:.2f} GiB; no queryable GPU memory report" + offloaded
    limit = DIRECT_RECORD_MEMORY_FRACTION * free_bytes
    reason = (
        f"record payload {payload / 2**30:.2f} GiB versus {limit / 2**30:.2f} GiB advisory "
        f"({DIRECT_RECORD_MEMORY_FRACTION:.0%} of {free_bytes / 2**30:.2f} GiB free)" + offloaded
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

    def __init__(self, cell, batch, initial_state, exchanged=()):
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
            "exchanged": False,
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
                "exchanged": name in exchanged,
            })
        unknown = set(exchanged) - set(cell.inputs)
        if unknown:
            raise ValueError(f"Exchanged inputs are not named inputs of the cell: {sorted(unknown)}")
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
        Exchanged populations' source cotangents carry the recurrent
        spike-gradient scale, as the recurrent population's do.

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
                source_grad, accumulated = csr_spike_ops.spike_grad_accumulate_batched(
                    csr_spike_ops._OPS.dpointnet_csr_spike_grad_accumulate_csr_master,
                    source, currents_grad, _metadata(connectivity), population["variable"],
                    self.basis, self.recurrent_scale, accumulator,
                    use_javier_batch32_backward=self.cell.use_javier_recurrent_vjp,
                    compute_spike_gradient=True, **attributes,
                )
                source_cotangents.append(source_grad)
                new_accumulators.append(accumulated)
                continue
            if credit:
                source_grad, weight_grad = csr_spike_ops.spike_grad_batched(
                    csr_spike_ops._OPS.dpointnet_csr_spike_grad_csr_master,
                    source, currents_grad, _metadata(connectivity), population["variable"],
                    self.basis,
                    self.recurrent_scale if population["exchanged"] else tf.ones((), self.spike_dtype),
                    use_packed_sm120_backward=False, write_csr_weight_gradient=True,
                    use_small_batch_backward=False, use_javier_batch32_backward=False,
                    **attributes,
                )
                source_cotangents.append(source_grad)
            else:
                source_cotangents.append(None)
                weight_grad = None if accumulator is None else csr_spike_ops.weight_grad_batched(
                    csr_spike_ops._OPS.dpointnet_csr_weight_grad,
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
                 differentiate_inputs=False, spike_exchanges=None, **kwargs):
        unsupported = direct_records_supported(cell)
        if unsupported:
            raise ValueError("Direct recorded adjoint requires: " + ", ".join(unsupported))
        if differentiate_inputs:
            raise ValueError("Direct recorded adjoint does not differentiate inputs.")
        self.record_offload = normalize_record_offload(kwargs.pop("record_offload", ()))
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
        if self.record_offload and not host_record_store_available():
            raise RuntimeError(
                "record_offload requires a GPU and the record store library: " + record_store_op_status()
            )
        # The staged (previous chunk, current chunk) word frames must hold every record
        # the reverse's delay-window shifts read, down to start - 1 - max_delay.
        if "history" in self.record_offload and int(cell.max_delay) >= self.chunk_size:
            raise ValueError(
                "History offload reads the previous chunk's word frame for delayed spikes, so the "
                f"recurrent delay ({int(cell.max_delay)} steps) must be smaller than chunk_size ({self.chunk_size})."
            )
        self._stores = {}
        self.n_chunks = -(-self.sequence_length // self.chunk_size)
        self.remainder_size = self.sequence_length % self.chunk_size
        # Exact prior-ASC values are re-walked per chunk in two passes: coarse
        # sub-snapshots every ``asc_block`` steps, then one block at a time, so
        # the resident buffers hold about asc_block + chunk_size / asc_block
        # tensors instead of chunk_size, for one extra walk of the chunk's
        # first blocks per chunk. The block near sqrt(chunk_size) minimizes
        # that sum (10 tensors for chunk 25 instead of 15 at chunk_size / 2).
        self.asc_block = max(1, math.isqrt(self.chunk_size - 1) + 1)
        self.differentiate_inputs = False
        self.count_windows = tuple(
            _window(window, self.sequence_length)
            for window in (((0, self.sequence_length),) if count_windows is None else count_windows)
        )
        if not self.count_windows:
            raise ValueError("Statistics require at least one count window.")
        self.n_sequence_outputs = 2
        self.spike_exchanges = dict(spike_exchanges or {})
        for name, exchange in self.spike_exchanges.items():
            net = cell.inputs.get(name)
            if net is None:
                raise ValueError(f"Spike exchange {name!r} is not a named input of the cell.")
            if net["input_type"] != "spikes":
                raise ValueError(f"Spike exchange {name!r} requires a 'spikes' input, not {net['input_type']!r}.")
            for method in ("begin", "forward", "reverse"):
                if not callable(getattr(exchange, method, None)):
                    raise TypeError(f"Spike exchange {name!r} has no callable {method}().")
        # Replay columns of the named inputs; exchanged inputs take none.
        self.replay_columns, offset = {}, 0
        for index, name in enumerate(cell.inputs):
            if name in self.spike_exchanges:
                continue
            width = int(cell.inputs_idx[index + 1]) - int(cell.inputs_idx[index])
            self.replay_columns[name] = (offset, offset + width)
            offset += width
        self.replay_width = offset

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
            start, stop = self.replay_columns[population["name"]]
            frame = tf.cast(inputs[:, time, start:stop], step.spike_dtype)
        # A tensor time index drops the static width the sweep validates.
        return tf.ensure_shape(frame, (step.batch, population["width"]))

    def _begin_exchanges(self, after=()):
        """Control inputs ordering every exchange's ``begin()`` after ``after`` and before a loop.

        ``begin`` usually has no data inputs; without ``after`` the reverse
        loop's call could be scheduled before the forward loop has finished.
        """
        started = []
        with tf.control_dependencies(list(after)):
            for exchange in self.spike_exchanges.values():
                value = exchange.begin()
                if value is not None:
                    started.append(value)
        return started

    @staticmethod
    def _frame_words(population):
        return 2 * ((population["width"] + 31) // 32)

    def _exchanged_source(self, step, population, frames, time):
        """Recorded exchanged frames ``x(time), x(time - 1), ...`` as the forward projected them."""
        history = population["history_steps"]
        parts = []
        for k in range(history + 1):
            words = tf.raw_ops.TensorListGetItem(
                input_handle=frames, index=time - k + history,
                element_shape=tf.constant([step.batch, self._frame_words(population)], tf.int32),
                element_dtype=tf.int32,
            )
            parts.append(unpack_step_record(
                words, population["width"], step.spike_dtype, step.refractory_dtype,
            )[0])
        return parts[0] if len(parts) == 1 else tf.concat(parts, axis=1)

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

    def _store(self, role, frames, frame_shape, dtype, staging_slots, prefetch=DEFAULT_PREFETCH):
        """Pinned-host record store of one role, created once per (shape, dtype)."""
        key = (role, int(frames), tuple(int(size) for size in frame_shape), tf.as_dtype(dtype).name)
        store = self._stores.get(key)
        if store is None:
            store = HostRecordStore(frames, frame_shape, dtype, staging_slots=staging_slots, prefetch=prefetch)
            self._stores[key] = store
        return store

    def close(self):
        """Release the pinned-host record stores (also released when the runner is collected)."""
        for store in self._stores.values():
            store.close()
        self._stores = {}

    def record_offload_report(self):
        """Pinned host bytes per offloaded surface of the stores created so far."""
        return {
            key[0]: {"frames": store.frames, "frame_bytes": store.frame_bytes, "host_bytes": store.host_bytes}
            for key, store in self._stores.items()
        }

    def _voltage_store(self, step, length):
        return self._store(
            "voltage", length, (step.batch, step.neurons), step.state_dtype, VOLTAGE_STAGING_SLOTS,
        )

    def _history_store(self, step):
        # Frame 0 holds the initial delay slots (in its last rows); frame c + 1 holds chunk c.
        return self._store(
            "history", self.n_chunks + 1, (self.chunk_size, step.batch, step.words), tf.float32,
            HISTORY_STAGING_SLOTS, prefetch=CHUNK_PREFETCH,
        )

    def _asc_store(self, step):
        """Chunk-start ASC snapshots (frame c = ASC entering chunk c), offloaded with the history."""
        return self._store(
            "asc", self.n_chunks, (step.batch, 2 * step.neurons), step.state_dtype, ASC_STAGING_SLOTS,
            prefetch=CHUNK_PREFETCH,
        )

    # -- forward driver ------------------------------------------------------------------

    def _forward(self, inputs, initial_state, sampled_traces, time_check):
        cell = self.cell
        batch, length = int(inputs.shape[0]), self.sequence_length
        with _rollout_context(cell):
            step = DirectRecordedStep(cell, batch, initial_state, exchanged=tuple(self.spike_exchanges))
            neurons = step.neurons
            tables = self._prepare_noise(step, inputs, initial_state)
            record_rollout_acceleration(getattr(cell, "acceleration_report", None), {
                "use_direct_state_rnn_loop": True,
                "use_prepared_cpu_poisson": bool(tables),
                "temporal_path": "direct_recorded_forward",
                "record_format": RECORD_FORMAT,
                **({"spike_exchanges": tuple(self.spike_exchanges)} if self.spike_exchanges else {}),
            })
            noise_steps = initial_state[6]
            offload_voltage = "voltage" in self.record_offload
            offload_history = "history" in self.record_offload
            # Per-step FP32 records as separate allocations land in oversized free
            # chunks (about 1.5x their size on the full network); a loop-carried
            # contiguous buffer makes Grappler's memory optimizer swap loop state to
            # the host. A persistent untracked scratch variable, written by row,
            # avoids both; it is sized once per (length, batch) and reused. With
            # voltage offload the variable is a small staging ring instead: the
            # kernel writes row ``time % slots`` and the store copies it to pinned
            # host memory on its own stream before the row is reused.
            if offload_voltage:
                voltage_store = self._voltage_store(step, length)
                scratch = self._scratch(
                    step, "direct_recorded_voltage_staging",
                    [VOLTAGE_STAGING_SLOTS, batch, neurons], step.state_dtype,
                )
            else:
                voltage_store = None
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
            history_store = words_staging = None
            history_token = tf.constant(0, tf.int32)
            trace_plan = self._trace_plan(sampled_traces, step)
            if offload_history:
                # Per-chunk word frames [chunk, batch, words] staged as float32 bit
                # views (TensorFlow's int32 GPU scatter/concat/slice kernels live in
                # host memory) and offloaded at every chunk end; statistics are then
                # accumulated per step instead of reduced from the stacked records.
                history_store = self._history_store(step)
                words_staging = self._scratch(
                    step, "direct_recorded_words_staging",
                    [HISTORY_STAGING_SLOTS * self.chunk_size, batch, step.words], tf.float32,
                )
                with step.device_scope():
                    rows = tf.range(self.chunk_size, dtype=tf.int64)
                    clear = tf.raw_ops.ResourceScatterUpdate(
                        resource=words_staging.handle, indices=rows,
                        updates=tf.zeros([self.chunk_size, batch, step.words], tf.float32),
                    )
                    # Frame 0 rows chunk_size - 1 - k hold initial slot k (time -1 - k).
                    writes = [clear]
                    for slot in range(step.delay):
                        initial_slot = tf.cast(
                            initial_state[0][:, slot * neurons:(slot + 1) * neurons], step.spike_dtype,
                        )
                        with tf.control_dependencies([op for op in writes if op is not None]):
                            writes.append(tf.raw_ops.ResourceScatterUpdate(
                                resource=words_staging.handle,
                                indices=tf.constant([self.chunk_size - 1 - slot], tf.int64),
                                updates=tf.bitcast(pack_step_record(initial_slot, no_refractory), tf.float32)[None],
                            ))
                    with tf.control_dependencies([op for op in writes if op is not None]):
                        history_token = history_store.offload(
                            words_staging, 0, 0, after=tf.identity(initial_state[1]),
                        )
            else:
                for slot in reversed(range(step.delay)):
                    initial_slot = tf.cast(initial_state[0][:, slot * neurons:(slot + 1) * neurons], step.spike_dtype)
                    words = tf.raw_ops.TensorListPushBack(
                        input_handle=words, tensor=pack_step_record(initial_slot, no_refractory),
                    )
            # Streamed statistics (history offload): counts and sampled traces are
            # reduced per chunk from the staged word frame by the same native
            # kernels the resident route applies to the stacked records. A window
            # whose bounds fall inside a chunk is accumulated per step instead.
            streamed_counts = tuple(
                tf.zeros([batch, neurons], tf.float32) for _ in self.count_windows
            ) if offload_history else ()
            streamed_traces = (
                tf.TensorArray(tf.float32, size=self.n_chunks, element_shape=(trace_plan["total"], self.chunk_size))
                if offload_history and trace_plan is not None else tf.zeros([0], tf.float32)
            )
            chunk_masks, stepwise_windows = [], []
            if offload_history:
                for index_w, (begin, end) in enumerate(self.count_windows):
                    aligned = begin % self.chunk_size == 0 and (end % self.chunk_size == 0 or end == length)
                    if aligned:
                        inside = [
                            float(c * self.chunk_size >= begin and min((c + 1) * self.chunk_size, length) <= end)
                            for c in range(self.n_chunks)
                        ]
                        with step.device_scope():
                            chunk_masks.append((index_w, tf.identity(tf.constant(inside, tf.float32))))
                    else:
                        stepwise_windows.append(index_w)

            def staging_rows(frame):
                """Device row indices of word frame ``frame`` in the staging variable."""
                with step.device_scope():
                    return tf.range(self.chunk_size, dtype=tf.int64) + tf.cast(
                        frame % HISTORY_STAGING_SLOTS, tf.int64) * self.chunk_size
            if offload_history:
                asc_store = self._asc_store(step)
                asc_snapshots = self._scratch(
                    step, "direct_recorded_asc_staging", [ASC_STAGING_SLOTS, batch, 2 * neurons],
                    step.state_dtype,
                )
            else:
                asc_store = None
                asc_snapshots = self._scratch(
                    step, "direct_recorded_asc_snapshot_scratch", [self.n_chunks, batch, 2 * neurons],
                    step.state_dtype,
                )
            penalties = tf.TensorArray(tf.float32, size=length, element_shape=(batch,))
            # Exchanged frames are recorded bit-packed, indexed by time + history:
            # the first ``history_steps`` entries hold the initial input history
            # (oldest first), as the word records hold the initial spike history.
            exchanged = [population["name"] for population in step.populations if population["exchanged"]]
            frames = []
            for population in step.populations[1:]:
                if not population["exchanged"]:
                    continue
                width, history = population["width"], population["history_steps"]
                no_blocking = tf.zeros([batch, width], step.refractory_dtype)
                record = tf.raw_ops.EmptyTensorList(
                    element_shape=tf.constant([batch, self._frame_words(population)], tf.int32),
                    max_num_elements=length + history, element_dtype=tf.int32,
                )
                start, _ = population["history"]
                for k in reversed(range(history)):
                    initial_frame = tf.cast(
                        initial_state[7][:, start + k * width:start + (k + 1) * width], step.spike_dtype,
                    )
                    record = tf.raw_ops.TensorListPushBack(
                        input_handle=record, tensor=pack_step_record(initial_frame, no_blocking),
                    )
                frames.append(record)
            frames = tuple(frames)

            def body(time, clock, state, voltages, words, penalties, frames, extra):
                counts, traces, token = extra
                sources = [tf.cast(state[0], step.spike_dtype)]
                previous_spikes = sources[0][:, :neurons]
                new_input_history, new_frames = [], list(frames)
                for population in step.populations[1:]:
                    if population["exchanged"]:
                        index = exchanged.index(population["name"])
                        source = tf.ensure_shape(tf.cast(
                            self.spike_exchanges[population["name"]].forward(time, previous_spikes),
                            step.spike_dtype,
                        ), (batch, population["width"]))
                        new_frames[index] = tf.raw_ops.TensorListPushBack(
                            input_handle=frames[index], tensor=pack_step_record(
                                source, tf.zeros([batch, population["width"]], step.refractory_dtype),
                            ),
                        )
                    else:
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
                row = time % VOLTAGE_STAGING_SLOTS if offload_voltage else time
                spikes, new_state, (pre_reset, record), penalty = step.forward(
                    sources, state, record_row=(scratch.handle, row) if in_place else None,
                )
                if step.has_input_history:
                    new_state += (tf.concat(new_input_history, axis=1),)
                written = None
                if buffered and not in_place:
                    written = tf.raw_ops.ResourceScatterUpdate(
                        resource=scratch.handle,
                        indices=(clock % VOLTAGE_STAGING_SLOTS if offload_voltage else clock)[None],
                        updates=pre_reset[None],
                    )
                elif not buffered:
                    voltages = tf.raw_ops.TensorListPushBack(input_handle=voltages, tensor=pre_reset)
                if offload_voltage:
                    # ``record`` comes from the kernel that wrote the staging row, so
                    # the copy is ordered after the write; the explicit dependency
                    # covers the separate scatter of the non-record forward.
                    with tf.control_dependencies([] if written is None else [written]):
                        token = voltage_store.offload(scratch, row, time, after=record)
                if offload_history:
                    # Frame c + 1 (chunk c) stages at slot (c + 1) % slots; row within
                    # the staging variable from the device clock (no index upload).
                    frame = clock // self.chunk_size + 1
                    staging_row = (frame % HISTORY_STAGING_SLOTS) * self.chunk_size + clock % self.chunk_size
                    tf.raw_ops.ResourceScatterUpdate(
                        resource=words_staging.handle, indices=staging_row[None],
                        updates=tf.bitcast(record, tf.float32)[None],
                    )
                    if stepwise_windows:
                        spikes32 = tf.cast(spikes, tf.float32)
                        counts = tuple(
                            count + spikes32 * tf.cast((clock >= begin) & (clock < end), tf.float32)
                            if index_w in stepwise_windows else count
                            for index_w, (count, (begin, end)) in enumerate(zip(counts, self.count_windows))
                        )
                else:
                    words = tf.raw_ops.TensorListPushBack(input_handle=words, tensor=record)
                penalties = penalties.write(time, penalty)
                with tf.control_dependencies([token]):
                    next_time = tf.identity(time + 1)
                return next_time, clock + 1, new_state, voltages, words, penalties, tuple(new_frames), (counts, traces, token)

            def chunk(index, clock, state, voltages, words, penalties, frames, extra):
                chunk_index = clock // self.chunk_size
                written = tf.raw_ops.ResourceScatterUpdate(
                    resource=asc_snapshots.handle,
                    indices=(chunk_index % ASC_STAGING_SLOTS if offload_history else chunk_index)[None],
                    updates=state[3][None],
                )
                if offload_history:
                    counts, traces, token = extra
                    with tf.control_dependencies([] if written is None else [written]):
                        token = asc_store.offload(
                            asc_snapshots, index % ASC_STAGING_SLOTS, index, after=tf.identity(token),
                        )
                    # Clear the chunk's staging slot so a short last chunk reduces
                    # zero rows, not an older frame's.
                    tf.raw_ops.ResourceScatterUpdate(
                        resource=words_staging.handle, indices=staging_rows(index + 1),
                        updates=tf.zeros([self.chunk_size, batch, step.words], tf.float32),
                    )
                    extra = (counts, traces, token)
                stop = tf.minimum((index + 1) * self.chunk_size, length)
                _, clock, state, voltages, words, penalties, frames, extra = tf.while_loop(
                    lambda time, *_: time < stop, body,
                    (index * self.chunk_size, clock, state, voltages, words, penalties, frames, extra),
                    parallel_iterations=1,
                )
                if offload_history:
                    # The inner loop's outputs follow every body operation, the row
                    # scatters included: reduce the chunk's statistics from its staged
                    # frame, then let the frame leave the device.
                    counts, traces, _ = extra
                    with tf.control_dependencies([state[1]]):
                        frame = tf.bitcast(tf.raw_ops.ResourceGather(
                            resource=words_staging.handle, indices=staging_rows(index + 1), dtype=tf.float32,
                        ), tf.int32)
                    counts = list(counts)
                    for index_w, mask in chunk_masks:
                        # (clock - 1) // chunk_size is this chunk's index on the device, also for a short last chunk.
                        counts[index_w] = counts[index_w] + tf.gather(mask, (clock - 1) // self.chunk_size) * record_counts(
                            frame, neurons, 0, self.chunk_size,
                        )
                    counts = tuple(counts)
                    if trace_plan is not None:
                        traces = traces.write(index, record_traces(
                            frame, trace_plan["indices"], trace_plan["mask32"], trace_plan["groups"],
                            neurons, trace_plan["total"], 0, self.chunk_size,
                        ))
                    token = history_store.offload(
                        words_staging, (index + 1) % HISTORY_STAGING_SLOTS, index + 1, after=frame,
                    )
                    extra = (counts, traces, token)
                return index + 1, clock, state, voltages, words, penalties, frames, extra

            with tf.control_dependencies([time_check, *self._begin_exchanges()]):
                state = tuple(tf.identity(value) for value in initial_state)
            # Device-resident int64 step clock beside the host loop counter: device
            # index consumers read it without a per-step host-to-device upload.
            with step.device_scope():
                clock = tf.identity(tf.zeros([], tf.int64))
            extra = (streamed_counts, streamed_traces, history_token)
            _, _, state, voltages, words, penalties, frames, extra = tf.while_loop(
                lambda index, *_: index < self.n_chunks, chunk,
                (tf.constant(0), clock, state, voltages, words, penalties, frames, extra),
                parallel_iterations=1,
            )
            streamed_counts, streamed_traces, history_token = extra
            with step.device_scope():
                if offload_history:
                    counts = streamed_counts
                    trace_outputs = ()
                    if trace_plan is not None:
                        # [chunks, rows, chunk_size] -> [rows, chunks * chunk_size]; the
                        # padded tail of a short last chunk is sliced off by the windows.
                        columns = tf.reshape(
                            tf.transpose(streamed_traces.stack(), [1, 0, 2]), [trace_plan["total"], -1],
                        )
                        offset = 0
                        for spec, extent in zip(sampled_traces, trace_plan["extents"]):
                            trace_outputs += (tf.cast(
                                columns[offset:offset + extent, spec.window[0]:spec.window[1]], step.spike_dtype,
                            ),)
                            offset += extent
                    # The reverse loop reads word frames from the store; a final
                    # state identity carries the ordering of the last chunk's offload.
                    with tf.control_dependencies([history_token]):
                        state = tuple(tf.identity(value) for value in state)
                else:
                    # Statistics come from the packed records after the loop: a stacked
                    # [time, batch, words] view costs one reduction per window instead of
                    # several small ops per step. The initial-history rows stay in the
                    # stack and the windows shift by ``delay`` instead: TensorFlow's int32
                    # StridedSlice is a host-memory kernel on GPUs, so slicing them off
                    # would round-trip the whole record through the host.
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
            # The rollout entry is rewritten by the reverse trace; the offload
            # surfaces and their pinned-host footprint keep their own entry.
            acceleration = getattr(cell, "acceleration_report", None)
            if isinstance(acceleration, dict):
                acceleration["record_offload"] = {
                    "surfaces": self.record_offload, "stores": self.record_offload_report(),
                }
        penalty_sequence = tf.transpose(penalties.stack())
        records = (
            voltage_store if offload_voltage else (scratch if buffered else voltages),
            history_store if offload_history else words,
            asc_store if offload_history else asc_snapshots, tables, noise_steps, buffered, frames,
        )
        return counts, trace_outputs, (penalty_sequence,), tuple(state), records

    # -- reverse driver ------------------------------------------------------------------

    def _backward(self, inputs, initial_state, final_state, records, sampled_traces,
                  count_gradients, trace_gradients, penalty_gradient, state_gradients, variables):
        cell = self.cell
        batch, length = int(inputs.shape[0]), self.sequence_length
        voltages, words, asc_snapshots, tables, noise_steps, buffered, frames = records
        voltage_store = voltages if isinstance(voltages, HostRecordStore) else None
        history_store = words if isinstance(words, HostRecordStore) else None
        asc_store = asc_snapshots if isinstance(asc_snapshots, HostRecordStore) else None
        with _rollout_context(cell):
            step = DirectRecordedStep(cell, batch, initial_state, exchanged=tuple(self.spike_exchanges))
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

            def voltage_record(time, words_at):
                """``(pre_reset, words, row)`` for ``DirectRecordedStep.reverse``."""
                if voltage_store is not None:
                    # Prefetched host frame; the upload of ``time - prefetch`` starts now.
                    return (voltage_store.take(time, time - DEFAULT_PREFETCH), words_at(time), 0)
                if buffered:
                    # The scratch variable itself: the native prelude reads row ``time``
                    # in place (a full read would copy the copy-on-read variable).
                    return (voltages, words_at(time), time)
                return (tf.raw_ops.TensorListGetItem(
                    input_handle=voltages, index=time,
                    element_shape=tf.constant([batch, neurons], tf.int32), element_dtype=T,
                ), words_at(time), 0)

            def resident_words_at(time):
                """Record of ``time``; ``-delay <= time < 0`` are the initial history slots."""
                return tf.raw_ops.TensorListGetItem(
                    input_handle=words, index=time + delay,
                    element_shape=tf.constant([batch, step.words], tf.int32), element_dtype=tf.int32,
                )

            def resident_block_words(begin, end):
                """Records of ``begin <= time < end`` as one ``[steps, batch, words]`` tensor."""
                return tf.raw_ops.TensorListGather(
                    input_handle=words, indices=tf.range(begin, end) + delay,
                    element_shape=tf.constant([batch, step.words], tf.int32), element_dtype=tf.int32,
                )

            resident_words_at.block = resident_block_words

            def chunk_words(chunk_index):
                """``words_at`` valid for times in ``[start - chunk_size, start + chunk_size)``.

                With history offload the chunk's frame and its predecessor are
                taken from the store (frame 0 holds the initial slots), so the
                previous chunk's delayed spikes are at hand without a conditional.
                """
                if history_store is None:
                    return resident_words_at
                start = chunk_index * self.chunk_size
                current = history_store.take(chunk_index + 1, chunk_index - 1)
                previous = history_store.take(chunk_index, -1)
                pair = tf.concat([previous, current], axis=0)

                def words_at(time):
                    return tf.bitcast(pair[time - start + self.chunk_size], tf.int32)

                def block_words(begin, end):
                    return tf.bitcast(
                        pair[begin - start + self.chunk_size:end - start + self.chunk_size], tf.int32,
                    )

                words_at.block = block_words
                return words_at

            populations = step.populations
            source_credit = tuple(population["recurrent"] or population["exchanged"] for population in populations)
            accumulate = tuple(population["trainable"] for population in populations)
            exchanged = [
                (index, population) for index, population in enumerate(populations) if population["exchanged"]
            ]
            exchanged_names = [population["name"] for _, population in exchanged]

            block = self.asc_block
            blocks_per_chunk = -(-self.chunk_size // block)

            asc_block_op = advance_asc_block_record_available()

            def walk_asc(begin, end, asc, size, stride, words_at):
                """``asc(begin + j * stride)`` for ``j = 0..ceil((end - begin) / stride)``.

                The value at ``end`` itself is stored last. With a rebuilt state
                library one native launch walks the block from the recorded words
                (``[slots, batch, 2 * neurons]`` tensor, read with ``asc_slot``);
                otherwise one advance per step fills a TensorArray of ``size``
                slots, every step writing slot ``ceil((time - begin) / stride)`` so
                the block-boundary step writes a slot last without a conditional.
                """
                if asc_block_op:
                    return advance_asc_block_record(asc, words_at.block(begin, end), step.asc_table, stride=stride)

                def body(time, asc, buffer):
                    index = (time - begin + stride - 1) // stride
                    return time + 1, step.advance_asc(asc, words_at(time)), buffer.write(index, asc)

                buffer = tf.TensorArray(T, size=size, element_shape=(batch, 2 * neurons))
                _, asc, buffer = tf.while_loop(
                    lambda time, *_: time < end, body, (begin, asc, buffer), parallel_iterations=1,
                )
                return buffer.write((end - begin + stride - 1) // stride, asc)

            def asc_slot(walked, index):
                return walked[index] if asc_block_op else walked.read(index)

            def chunk_snapshot(chunk_index):
                if asc_store is not None:
                    return asc_store.take(chunk_index, chunk_index - 1)
                row = tf.raw_ops.ResourceGather(
                    resource=asc_snapshots.handle, indices=chunk_index[None], dtype=T,
                )
                return tf.reshape(row, [batch, 2 * neurons])

            def reverse_step(time, carry, window, accumulators, asc_buffer, block_start, pending, words_at):
                sources = [window]
                for population, keep in zip(populations[1:], accumulate[1:]):
                    if population["exchanged"]:
                        sources.append(self._exchanged_source(
                            step, population, frames[exchanged_names.index(population["name"])], time,
                        ))
                    else:
                        sources.append(
                            self._reverse_source(step, population, inputs, time, tables, noise_steps, initial_state)
                            if keep else tf.zeros([batch, 0], S)
                        )
                carry, credit, accumulators = step.reverse(
                    voltage_record(time, words_at), asc_slot(asc_buffer, time - block_start), carry,
                    spike_base(time), penalty_gradient[:, time], sources, source_credit,
                    tuple(acc if keep else None for acc, keep in zip(accumulators, accumulate)),
                    trace=trace_at(time),
                )
                history_credit = credit[0]
                new_pending = []
                for (index, population), frame_credit in zip(exchanged, pending):
                    # Frame x(time) was projected by steps time..time+history; all are
                    # reversed now, so its credit is complete.
                    width = population["width"]
                    frame_credit = frame_credit + tf.cast(credit[index], tf.float32)
                    received = self.spike_exchanges[population["name"]].reverse(time, frame_credit[:, :width])
                    received = tf.ensure_shape(tf.cast(received, history_credit.dtype), (batch, neurons))
                    history_credit = history_credit + tf.pad(received, ((0, 0), (0, step.history_width - neurons)))
                    new_pending.append(tf.concat(
                        [frame_credit[:, width:], tf.zeros([batch, width], tf.float32)], axis=1,
                    ))
                carry = carry + (history_credit,)
                accumulators = tuple(
                    tf.zeros([0], tf.float32) if acc is None else acc for acc in accumulators
                )
                # At time 0 the shifted-in frame is never read; clamp it to the oldest record.
                window = step.shift_history(window, words_at(tf.maximum(time - 1 - delay, -delay)))
                return time - 1, carry, window, accumulators, asc_buffer, block_start, tuple(new_pending)

            def reverse_block(block_index, carry, window, accumulators, pending, coarse, chunk_start, chunk_stop,
                              words_at):
                block_start = chunk_start + block_index * block
                block_stop = tf.minimum(block_start + block, chunk_stop)
                # The last block's value at block_stop - 1 is written by the walk's
                # final write; the walk stops one step short of the block end.
                fine = walk_asc(block_start, block_stop - 1, asc_slot(coarse, block_index), block, 1, words_at)
                _, carry, window, accumulators, _, _, pending = tf.while_loop(
                    lambda time, *_: time >= block_start,
                    lambda *args: reverse_step(*args, words_at),
                    (block_stop - 1, carry, window, accumulators, fine, block_start, pending),
                    parallel_iterations=1,
                )
                return block_index - 1, carry, window, accumulators, pending, coarse, chunk_start, chunk_stop

            def reverse_chunk(chunk_index, carry, window, accumulators, pending):
                start = chunk_index * self.chunk_size
                stop = tf.minimum(start + self.chunk_size, length)
                last_block = (stop - start - 1) // block
                words_at = chunk_words(chunk_index)
                # Coarse sub-snapshots at every block start up to the last block only.
                coarse = walk_asc(
                    start, start + last_block * block, chunk_snapshot(chunk_index), blocks_per_chunk + 1, block,
                    words_at,
                )
                _, carry, window, accumulators, pending, _, _, _ = tf.while_loop(
                    lambda block_index, *_: block_index >= 0,
                    lambda *args: reverse_block(*args, words_at),
                    (last_block, carry, window, accumulators, pending, coarse, start, stop),
                    parallel_iterations=1,
                )
                return chunk_index - 1, carry, window, accumulators, pending

            def floating_gradient(gradient, value):
                return tf.zeros_like(value) if gradient is None else tf.cast(gradient, value.dtype)

            # Prime the voltage prefetch with the last frames the reverse reads first.
            primed = []
            if voltage_store is not None:
                for k in range(min(DEFAULT_PREFETCH, length)):
                    primed.append(voltage_store.prefetch(
                        length - 1 - k, after=primed[-1] if primed else tf.identity(final_state[1]),
                    ))
            with tf.control_dependencies([*self._begin_exchanges(after=final_state), *primed]):
                carry = tuple(tf.identity(value) for value in (
                    floating_gradient(state_gradients[1], final_state[1]),
                    floating_gradient(state_gradients[3], final_state[3]),
                    floating_gradient(state_gradients[4], final_state[4]),
                    floating_gradient(state_gradients[5], final_state[5]),
                    floating_gradient(state_gradients[0], final_state[0]),
                    tf.zeros_like(final_state[0]),
                ))
            # The delay window of the last step as one [batch, neurons * delay] tensor
            # (newest slot first), as the forward's history state holds it. Ordered
            # after ``carry`` so store reads cannot precede the forward's offloads.
            with tf.control_dependencies(carry):
                last_words_at = chunk_words(tf.constant(self.n_chunks - 1))
                window = tf.concat(
                    [step.unpack(last_words_at(tf.constant(length - 1 - k)))[0] for k in range(1, delay + 1)],
                    axis=1,
                )
            accumulators = tuple(
                tf.zeros_like(population["variable"]) if keep else tf.zeros([0], tf.float32)
                for population, keep in zip(populations, accumulate)
            )
            # Pending FP32 credit of exchanged frames x(time), ..., x(time - history). The
            # final input-history state holds x(L-1), ..., x(L-history): its cotangent seeds
            # the first frames (as the recurrent history cotangent seeds the carry), and
            # x(L-1-history) starts empty.
            history_gradient = state_gradients[7] if step.has_input_history else None
            seeds = []
            for _, population in exchanged:
                width, history = population["width"], population["history_steps"]
                with step.device_scope():
                    seed = tf.zeros([batch, width], tf.float32)
                    if history and history_gradient is not None and final_state[7].dtype.is_floating:
                        start, stop = population["history"]
                        seed = tf.concat([tf.cast(history_gradient[:, start:stop], tf.float32), seed], axis=1)
                    elif history:
                        seed = tf.zeros([batch, width * (history + 1)], tf.float32)
                seeds.append(seed)
            pending = tuple(seeds)
            _, carry, _, accumulators, _ = tf.while_loop(
                lambda chunk_index, *_: chunk_index >= 0, reverse_chunk,
                (tf.constant(self.n_chunks - 1), carry, window, accumulators, pending),
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
        if self.spike_exchanges and inputs.shape[2] != self.replay_width:
            raise ValueError(
                f"With spike exchanges the inputs hold only the {self.replay_width} replay columns of "
                f"the non-exchanged named inputs, got {inputs.shape[2]}."
            )
        output_count = len(self.count_windows) + len(sampled_traces) + 1
        checks = (
            tf.debugging.assert_equal(tf.shape(inputs)[1], self.sequence_length),
            tf.debugging.assert_equal(
                tf.reduce_all((initial_state[0] == 0) | (initial_state[0] == 1)), True,
                message="Recorded adjoints require binary initial spike history.",
            ),
        ) + tuple(
            tf.debugging.assert_equal(
                tf.reduce_all((history == 0) | (history == 1)), True,
                message=f"Spike exchange {name!r} requires a binary initial input history.",
            )
            for name, net in self.cell.inputs.items()
            if name in self.spike_exchanges and int(net["history_stop"]) > int(net["history_start"])
            # Only inputs with history reach here; a cell without any has no eighth state.
            for history in [initial_state[7][:, int(net["history_start"]):int(net["history_stop"])]]
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
