"""Fixed recurrent short-term plasticity ("release dynamics") for GLIF3Cell.

Each recurrent edge ``source -> target`` may carry fixed release dynamics chosen by
the (source class, target class) pair: the normalized expected-response model with
facilitation ``f`` and availability ``a``. Between steps ``f`` decays with
``facilitation_tau_ms`` and ``a`` recovers with ``depression_tau_ms``; at a
presynaptic arrival the edge transmits ``w * a * p / p0`` with
``p = p0 + (1 - p0) * f``, then ``f += (1 - f) * facilitation_amount`` and ``a`` is
depleted by ``p`` (``depression_amount == -1``) or by ``depression_amount``.
Backward credit is the event-gated surrogate extension of the binary forward:
``state' = state + event * (candidate - state)``, so credit reaches future release
state and silent connections.

Exact emission keying. Release state depends only on the presynaptic spike train
and the parameter set, and the resting state (``f = 0, a = 1``) is a fixed point of
recovery. The state of an edge at its arrival therefore equals the state of the key
``(source, parameter group)`` at emission, shifted by the edge's delay. Edges of one
key share one state; frames are delayed by each edge's own delay.

Correction input. Every recurrent edge stays in the standard recurrent projection
and recurrent master (constraints, regularizers, checkpoints and exports are
unchanged) and delivers ``w * s``. Release edges additionally appear in an internal
named input that delivers ``w * s * (e - 1)``, where ``e = a * p / p0``. The native
projection skips non-positive sources, so the correction is split into two
non-negative frames, ``s * max(e - 1, 0)`` on weight ``+w`` and ``s * max(1 - e, 0)``
on weight ``-w``, keyed ``(key, +)`` and ``(key, -)``; at most one is non-zero per
key and step. The internal input's weights are a shadow of the recurrent master
(refreshed after every update or assignment) and its weight gradient is folded into
the recurrent master's gradient, so the optimizer sees one weight per edge.

Paths. The ordinary cell path (warm-up, validation, inference) computes frames from
the cell's newest spike-history slot and carries ``(f, a)`` as two extra cell states;
its forward is exact, but a tape through it is not, because the named-input
projection gives no credit to its input columns. Training therefore requires the
direct recorded adjoint, which uses :class:`ReleaseExtension`, a graded spike
exchange that keeps ``(f, a)`` in device variables, records the presynaptic spikes
bit-packed and per-block state snapshots in the forward, and recomputes each reverse
chunk's states for the exact release VJP.
"""

from typing import NamedTuple

import numpy as np
import tensorflow as tf

from .cell_models.glif3_cell import GLIF3Cell
from .cell_models.nest_dynamics import time_steps

RELEASE_INPUT = "recurrent_release"
PARAMETER_FIELDS = (
    "base_release_probability", "facilitation_amount", "depression_amount",
    "depression_tau_ms", "facilitation_tau_ms",
)
_DEFAULT_MAX_DELAY = 5


class ReleaseParameters(NamedTuple):
    """Per-key constants (float32 tensors of shape ``[keys]``)."""
    p0: tf.Tensor
    increment: tf.Tensor
    depression: tf.Tensor
    depletion: tf.Tensor
    decay_f: tf.Tensor
    recover_a: tf.Tensor


def release_parameters(table, dt):
    """``ReleaseParameters`` from a ``[keys, 5]`` table in ``PARAMETER_FIELDS`` order."""
    table = np.asarray(table, np.float64)
    p0, increment, depression, tau_d, tau_f = table.T
    return ReleaseParameters(
        tf.constant(p0, tf.float32), tf.constant(increment, tf.float32),
        tf.constant(np.where(depression == -1, 0.0, depression), tf.float32),
        tf.constant(depression == -1, tf.bool),
        tf.constant(np.exp(-dt / tau_f), tf.float32),
        tf.constant(-np.expm1(-dt / tau_d), tf.float32),
    )


def release_transition(events, facilitation, availability, parameters):
    """One step: recovery over ``dt``, then the event-gated update.

    Returns ``(response, facilitation', availability')``; ``response`` is the
    normalized efficacy ``a * p / p0`` an arrival transmits at this step.
    """
    p = parameters
    f = facilitation * p.decay_f
    a = availability + (1.0 - availability) * p.recover_a
    probability = p.p0 + (1.0 - p.p0) * f
    response = a * probability / p.p0
    amount = tf.where(p.depletion, probability, p.depression)
    new_f = f + events * (1.0 - f) * p.increment
    new_a = a - events * a * amount
    return response, new_f, new_a


def correction_frames(events, response):
    """``[batch, 2 * keys]``: ``s * max(e - 1, 0)`` then ``s * max(1 - e, 0)``.

    ``tf.where`` keeps exactly one branch differentiable at ``e == 1`` (the resting
    efficacy), so the split carries the derivative of ``s * (e - 1)`` once.
    """
    up = response >= 1.0
    zero = tf.zeros_like(response)
    positive = tf.where(up, events * (response - 1.0), zero)
    negative = tf.where(up, zero, events * (1.0 - response))
    return tf.concat([positive, negative], axis=1)


def release_vjp(events, facilitation, availability, parameters, credit, grad_f, grad_a):
    """Exact VJP of ``(correction_frames, facilitation', availability')`` of one step.

    ``credit`` is the ``[batch, 2 * keys]`` cotangent of the frames and
    ``(grad_f, grad_a)`` those of the updated state. Returns ``(event_credit,
    facilitation_credit, availability_credit)`` for the step's inputs.
    """
    p = parameters
    keys = int(p.p0.shape[0])
    f = facilitation * p.decay_f
    a = availability + (1.0 - availability) * p.recover_a
    probability = p.p0 + (1.0 - p.p0) * f
    response = a * probability / p.p0
    up = response >= 1.0
    # Cotangent of the correction c = s * (e - 1): the positive frame carries +c,
    # the negative frame -c.
    grad_c = tf.where(up, credit[:, :keys], -credit[:, keys:])
    amount = tf.where(p.depletion, probability, p.depression)
    depletion = tf.cast(p.depletion, tf.float32)
    grad_response = grad_c * events
    event_credit = (
        grad_c * (response - 1.0) + grad_f * (1.0 - f) * p.increment - grad_a * a * amount
    )
    grad_f_post = (
        grad_f * (1.0 - events * p.increment)
        - grad_a * events * a * (1.0 - p.p0) * depletion
        + grad_response * a * (1.0 - p.p0) / p.p0
    )
    grad_a_post = grad_a * (1.0 - events * amount) + grad_response * probability / p.p0
    return event_credit, grad_f_post * p.decay_f, grad_a_post * (1.0 - p.recover_a)


# -- layout -----------------------------------------------------------------------------


class ReleaseLayout(NamedTuple):
    input_dict: dict
    key_source: np.ndarray       # [keys] presynaptic model neuron of each key
    key_parameters: np.ndarray   # [keys, 5]
    edge_recurrent: np.ndarray   # [2 * release edges] recurrent canonical edge of each input edge
    edge_sign: np.ndarray        # [2 * release edges] +1 (facilitating copy) or -1 (depressing copy)
    delay_limit_ms: float
    report: dict


def _load(value):
    if isinstance(value, (str, bytes)) or hasattr(value, "__fspath__"):
        return np.load(value)
    return np.asarray(value)


def _validate_parameters(table, supported):
    p0, increment, depression, tau_d, tau_f = np.asarray(table, np.float64)[supported].T
    finite = np.isfinite(np.asarray(table, np.float64)[supported]).all()
    valid = (
        finite and np.all((p0 > 0) & (p0 <= 1)) and np.all((increment >= 0) & (increment <= 1))
        and np.all((depression == -1) | ((depression >= 0) & (depression <= 1)))
        and np.all(tau_d > 0) and np.all(tau_f > 0)
    )
    if not valid:
        raise ValueError("Invalid release parameters: need 0 < p0 <= 1, 0 <= facilitation <= 1, "
                         "depression -1 or in [0, 1] and positive time constants.")


def build_release_layout(network, spec, dt=1.0, max_delay=_DEFAULT_MAX_DELAY):
    """Keys, parameters and the internal correction input for ``network``.

    ``spec`` holds ``node_classes`` (``[n_nodes]`` int, model neuron order, -1 for
    no release), ``pair_parameters`` (``[classes * classes, 5]``, row ``pre *
    classes + post``) and ``pair_supported`` (``[classes * classes]`` bool); arrays
    or ``.npy`` paths. Unsupported pairs and unclassed neurons stay static.
    """
    if not isinstance(spec, dict):
        raise ValueError("recurrent_release must be a dict with node_classes, pair_parameters, pair_supported.")
    unknown = set(spec) - {"node_classes", "pair_parameters", "pair_supported"}
    if unknown:
        raise ValueError(f"Unknown recurrent_release keys: {sorted(unknown)}")
    classes = _load(spec["node_classes"]).astype(np.int64)
    table = _load(spec["pair_parameters"]).astype(np.float64)
    supported = _load(spec["pair_supported"]).astype(bool)
    n_nodes = int(network["n_nodes"])
    if classes.shape != (n_nodes,):
        raise ValueError(f"node_classes must have shape ({n_nodes},), got {classes.shape}.")
    n_classes = int(round(np.sqrt(table.shape[0]))) if table.ndim == 2 else 0
    if table.ndim != 2 or table.shape != (n_classes * n_classes, 5) or supported.shape != (table.shape[0],):
        raise ValueError("pair_parameters must be [classes * classes, 5] and pair_supported [classes * classes].")
    if np.any(classes < -1) or np.any(classes >= n_classes):
        raise ValueError("node_classes must lie in [-1, classes).")
    node_types = np.asarray(network.get("node_type_ids", np.zeros(n_nodes, int)))
    for node_type in np.unique(node_types):
        if np.unique(classes[node_types == node_type]).size != 1:
            raise ValueError(f"node_classes differ within node type {int(node_type)}; check the neuron order.")
    _validate_parameters(table, supported)

    synapses = network["synapses"]
    indices = np.asarray(synapses["indices"], np.int64)
    weights = np.asarray(synapses["weights"])
    delays = np.asarray(synapses["delays"], np.float64)
    syn_ids = np.asarray(synapses["syn_ids"])
    pre, post = classes[indices[:, 1]], classes[indices[:, 0]]
    pair = np.maximum(pre, 0) * n_classes + np.maximum(post, 0)
    group = np.where((pre >= 0) & (post >= 0) & supported[pair], pair, -1)
    release = np.flatnonzero(group >= 0)
    if release.size == 0:
        raise ValueError("No recurrent edge has supported release parameters.")
    n_groups = n_classes * n_classes
    keys, inverse = np.unique(indices[release, 1] * n_groups + group[release], return_inverse=True)
    n_keys = keys.size
    key_source = (keys // n_groups).astype(np.int64)
    key_parameters = table[keys % n_groups].astype(np.float32)

    # Delays exactly as GLIF3Cell clips and rounds NEST recurrent delays.
    network_max_delay = float(np.max(delays, initial=dt))
    limit = network_max_delay if max_delay is None or max_delay <= 0 else min(network_max_delay, max_delay)
    steps = time_steps(np.clip(delays[release], dt, limit), dt)
    delays_ms = steps.astype(np.float64) * dt

    targets = indices[release, 0]
    input_dict = {
        "name": RELEASE_INPUT,
        "network_type": "input",
        "input_type": "spikes",
        "node_params": {},
        "n_inputs": 2 * n_keys,
        "indices": np.concatenate([
            np.column_stack((targets, inverse)), np.column_stack((targets, inverse + n_keys)),
        ]),
        "weights": np.concatenate([weights[release], -weights[release]]),
        "delays": np.concatenate([delays_ms, delays_ms]),
        "syn_ids": np.concatenate([syn_ids[release], syn_ids[release]]),
        # Pair metadata admits the input to the fused accumulate backward.
        "options": {"trainable": True, "compact_pairs": True},
    }
    report = {
        "recurrent_edges": int(indices.shape[0]),
        "release_edges": int(release.size),
        "release_fraction": float(release.size / max(indices.shape[0], 1)),
        "keys": int(n_keys),
        "supported_pairs": int(supported.sum()),
        "classes": n_classes,
    }
    return ReleaseLayout(
        input_dict, key_source, key_parameters,
        np.concatenate([release, release]).astype(np.int64),
        np.concatenate([np.ones(release.size), -np.ones(release.size)]).astype(np.float32),
        float(limit), report,
    )


# -- fused per-step math ------------------------------------------------------------------


class _ReleaseMath:
    """XLA-fused per-step release math shared by the ordinary path and the extension.

    Each function gathers the key events from ``[batch, neurons]`` spikes and runs the
    elementwise release math in one compiled cluster; both paths call the same
    compiled functions, so their arithmetic is identical.
    """

    def __init__(self, key_source, parameters, neurons):
        def events(spikes):
            return tf.gather(tf.cast(spikes, tf.float32), key_source, axis=1)

        @tf.function(jit_compile=True)
        def step(spikes, facilitation, availability):
            e = events(spikes)
            response, new_f, new_a = release_transition(e, facilitation, availability, parameters)
            return correction_frames(e, response), new_f, new_a

        @tf.function(jit_compile=True)
        def frames(spikes, facilitation, availability):
            e = events(spikes)
            response, _, _ = release_transition(e, facilitation, availability, parameters)
            return correction_frames(e, response)

        @tf.function(jit_compile=True)
        def transition(spikes, facilitation, availability):
            _, new_f, new_a = release_transition(events(spikes), facilitation, availability, parameters)
            return new_f, new_a

        @tf.function(jit_compile=True)
        def vjp(spikes, facilitation, availability, credit, grad_f, grad_a):
            event_credit, grad_f, grad_a = release_vjp(
                events(spikes), facilitation, availability, parameters, credit, grad_f, grad_a,
            )
            spike_credit = tf.math.unsorted_segment_sum(tf.transpose(event_credit), key_source, neurons)
            return tf.transpose(spike_credit), grad_f, grad_a

        self.step, self.frames, self.transition, self.vjp = step, frames, transition, vjp


# -- cell -------------------------------------------------------------------------------


class ReleaseGLIF3Cell(GLIF3Cell):
    """GLIF3Cell with fixed recurrent release dynamics (see the module docstring).

    Configure with ``rnn_cell_params.cell_model = "glif3_release"`` and
    ``rnn_cell_params.recurrent_release = {node_classes, pair_parameters,
    pair_supported}``. Requires NEST dynamics. Two extra states ``(facilitation,
    availability)`` of shape ``[batch, keys]`` (float32) follow the base states.

    Forward simulation is exact on every route. Gradients are exact only on the
    direct recorded adjoint: the generic named-input projection treats input
    columns as constants, so a tape through ``call`` drops the correction frames'
    credit to spikes and release states. ``TrainingEngine`` therefore refuses
    any other training route for this cell.
    """

    # The model treats this cell like GLIF3Cell for automatic acceleration and
    # admits it to the direct recorded adjoint through ``direct_extension``.
    standard_acceleration = True
    direct_extension_inputs = (RELEASE_INPUT,)
    direct_route_gradients_only = True

    @classmethod
    def extend_network_dicts(cls, network, inputs, cell_params, dt=1.0):
        """Append the correction input; called by the model before the cell is built."""
        if RELEASE_INPUT in inputs:
            raise ValueError(f"Input name {RELEASE_INPUT!r} is reserved for recurrent release.")
        spec = cell_params.get("recurrent_release")
        if spec is None:
            raise ValueError("cell_model 'glif3_release' requires rnn_cell_params.recurrent_release.")
        layout = build_release_layout(network, spec, dt, cell_params.get("max_delay", _DEFAULT_MAX_DELAY))
        cell_params["_release_layout"] = layout
        return {**inputs, RELEASE_INPUT: layout.input_dict}

    def __init__(self, glif_network, inputs, *args, recurrent_release=None, _release_layout=None, **kwargs):
        if kwargs.get("dynamics_mode", "nest") != "nest":
            raise ValueError("Recurrent release requires dynamics_mode='nest'.")
        if _release_layout is None:
            if recurrent_release is None:
                raise ValueError("ReleaseGLIF3Cell requires recurrent_release.")
            _release_layout = build_release_layout(
                glif_network, recurrent_release, kwargs.get("dt", 1.0), kwargs.get("max_delay", _DEFAULT_MAX_DELAY),
            )
            inputs = {**inputs, RELEASE_INPUT: _release_layout.input_dict}
        if list(inputs)[-1] != RELEASE_INPUT:
            raise ValueError(f"The {RELEASE_INPUT!r} input must be the cell's last named input.")
        self._release_layout = _release_layout
        super().__init__(glif_network, inputs, *args, **kwargs)
        layout = _release_layout
        if not np.isclose(self._delay_limit_ms, layout.delay_limit_ms):
            raise ValueError("Release delays were clipped with a different delay limit than the recurrence.")
        net = self.inputs[RELEASE_INPUT]
        self._release_keys = int(layout.key_source.size)
        self._release_base_states = len(super().state_size)
        self._release_width = 2 * self._release_keys
        self._release_columns = int(self.inputs_idx[-2])
        if int(self.inputs_idx[-1]) - self._release_columns != self._release_width:
            raise ValueError("Unexpected release input width.")
        device = getattr(self.recurrent_weight_values, "device", None) or None
        with tf.device(device):
            self._release_key_source = tf.identity(tf.constant(layout.key_source, tf.int64))
            self._release_parameters = ReleaseParameters(*[
                tf.identity(value) for value in release_parameters(layout.key_parameters, float(self._dt_value()))
            ])
        # Live (CSR-master) or canonical order of both masters.
        recurrent_layout = getattr(self, "recurrent_master_layout", None) if self._csr_master_weights else None
        input_layout = net.get("master_layout") if self._csr_master_weights else None
        n_input_edges = layout.edge_recurrent.size
        input_canonical = (
            np.asarray(input_layout.csr_to_canonical, np.int64) if input_layout is not None
            else np.arange(n_input_edges)
        )
        recurrent_live = layout.edge_recurrent[input_canonical]
        if recurrent_layout is not None:
            recurrent_live = np.asarray(recurrent_layout.canonical_to_csr, np.int64)[recurrent_live]
        self._release_n_recurrent = int(self.recurrent_weight_values.shape[0])
        with tf.device(device):
            self._release_tie_index = tf.identity(tf.constant(recurrent_live, tf.int64))
            self._release_tie_sign = tf.identity(tf.constant(layout.edge_sign[input_canonical], tf.float32))
        self._release_shadow = net["input_weight_values"]
        self._release_math = _ReleaseMath(self._release_key_source, self._release_parameters, self._n_neurons)
        self._release_extensions = {}
        self.release_report = dict(layout.report)
        self.refresh_recurrent_weight_shadow()

    def _dt_value(self):
        value = tf.get_static_value(self._dt) if tf.is_tensor(self._dt) else self._dt
        return float(np.asarray(value).reshape(-1)[0])

    # -- weights ------------------------------------------------------------------------

    def tie_release_weights(self):
        """Assign the correction input's weights from the recurrent master."""
        master = self.recurrent_weight_values
        values = tf.gather(tf.cast(master, tf.float32), self._release_tie_index) * self._release_tie_sign
        return self._release_shadow.assign(tf.cast(values, self._release_shadow.dtype))

    def refresh_recurrent_weight_shadow(self):
        if getattr(self, "_release_shadow", None) is not None:
            self.tie_release_weights()
        super().refresh_recurrent_weight_shadow()

    @staticmethod
    def _same(variable, owner):
        return variable is owner or getattr(variable, "value", None) is owner or getattr(owner, "value", None) is variable

    def fold_variable_gradients(self, variables, gradients):
        """Fold the correction input's weight gradient into the recurrent master's."""
        variables, gradients = list(variables), list(gradients)
        recurrent = shadow = None
        for index, variable in enumerate(variables):
            if self._same(variable, self.recurrent_weight_values):
                recurrent = index
            elif self._same(variable, self._release_shadow):
                shadow = index
        if shadow is None or gradients[shadow] is None:
            return gradients
        if recurrent is None:
            raise ValueError("Recurrent release requires trainable recurrent weights.")
        gradient = gradients[shadow]
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)
        folded = tf.math.unsorted_segment_sum(
            tf.cast(gradient, tf.float32) * self._release_tie_sign, self._release_tie_index,
            self._release_n_recurrent,
        )
        base = gradients[recurrent]
        if isinstance(base, tf.IndexedSlices):
            base = tf.convert_to_tensor(base)
        gradients[recurrent] = folded if base is None else base + tf.cast(folded, base.dtype)
        gradients[shadow] = tf.zeros_like(gradient)
        return gradients

    # -- states -------------------------------------------------------------------------

    @property
    def state_size(self):
        base = super().state_size
        if not hasattr(self, "_release_keys"):
            return base
        return tuple(base) + (self._release_keys, self._release_keys)

    def zero_state(self, batch_size, dtype, with_names=False):
        base = super().zero_state(batch_size, dtype, with_names=with_names)
        if not hasattr(self, "_release_keys"):
            return base
        extra = (
            tf.zeros((batch_size, self._release_keys), tf.float32),
            tf.ones((batch_size, self._release_keys), tf.float32),
        )
        if with_names:
            values, names = base
            return tuple(values) + extra, tuple(names) + ("release_facilitation0", "release_availability0")
        return tuple(base) + extra

    # -- ordinary path ------------------------------------------------------------------

    def release_step(self, newest_spikes, facilitation, availability):
        """``(frames, facilitation', availability')`` for spikes ``z(t - 1)`` ``[batch, neurons]``."""
        return self._release_math.step(newest_spikes, facilitation, availability)

    def _call_impl(self, inputs, states, **kwargs):
        n_base = self._release_base_states
        base = tuple(states[:n_base])
        facilitation, availability = states[n_base], states[n_base + 1]
        frames, new_f, new_a = self.release_step(base[0][:, :self._n_neurons], facilitation, availability)
        inputs = tf.cast(inputs, self.compute_dtype)
        if int(inputs.shape[-1]) != self._release_columns:
            raise ValueError(
                f"Expected {self._release_columns} external input columns, got {int(inputs.shape[-1])}."
            )
        inputs = tf.concat([inputs, tf.cast(frames, self.compute_dtype)], axis=-1)
        result = super()._call_impl(inputs, base, **kwargs)
        output, new_base = result[0], result[1]
        return (output, tuple(new_base) + (new_f, new_a)) + tuple(result[2:])

    # -- direct recorded adjoint --------------------------------------------------------

    def direct_extension(self, batch, sequence_length, chunk_size):
        key = (int(batch), int(sequence_length), int(chunk_size))
        if key not in self._release_extensions:
            self._release_extensions[key] = ReleaseExtension(self, *key)
        return self._release_extensions[key]


# -- direct recorded adjoint extension --------------------------------------------------


class ReleaseExtension:
    """Graded spike exchange and extra-state owner for the direct recorded adjoint.

    The runner calls ``start_forward(extra_states, initial_state)`` before its forward
    loop, ``forward(time, spikes)`` before every step (``spikes = z(time - 1)``) and
    ``final_states()`` after the loop; in the reverse it calls ``start_reverse()``,
    then per step ``replay(time)`` (the frames ``x(time), ..., x(time - history)`` as
    the forward projected them) and ``reverse(time, credit)``.

    Forward state lives in device variables. The presynaptic spikes of every step
    are recorded bit-packed; the state at the start of every reverse block (chunk
    start minus the input's delay history) is snapshotted. The reverse recomputes
    one block of states when it enters a chunk and applies :func:`release_vjp`
    step by step. Memory: about ``(n_chunks + chunk + history + 2) * 2 * batch *
    keys`` float32 values plus the packed spikes.
    """

    graded = True
    n_states = 2

    def __init__(self, cell, batch, sequence_length, chunk_size):
        from .custom_ops.glif_state_ops import pack_step_record, unpack_step_record

        self._pack, self._unpack = pack_step_record, unpack_step_record
        self.cell = cell
        self.batch, self.length, self.chunk = int(batch), int(sequence_length), int(chunk_size)
        self.keys = cell._release_keys
        self.width = 2 * self.keys
        self.neurons = int(cell._n_neurons)
        net = cell.inputs[RELEASE_INPUT]
        self.history_start, self.history_stop = int(net["history_start"]), int(net["history_stop"])
        self.history = (self.history_stop - self.history_start) // self.width
        if self.history >= self.chunk:
            raise ValueError("Recurrent release requires a delay history shorter than the checkpoint chunk.")
        self.n_chunks = -(-self.length // self.chunk)
        self.exchanges = {RELEASE_INPUT: self}
        starts = [max(c * self.chunk - self.history, 0) for c in range(self.n_chunks)]
        self.block_starts = starts
        snapshot_row = np.full(self.length, self.n_chunks, np.int64)
        for c, start in enumerate(starts):
            if snapshot_row[start] == self.n_chunks:
                snapshot_row[start] = c
        self.window_size = self.chunk + self.history
        self.words = 2 * ((self.neurons + 31) // 32)
        self.device = getattr(cell.recurrent_weight_values, "device", None) or None
        b, k = self.batch, self.keys
        # The runner may build the extension while tracing; its tables and variables
        # must live outside that function so every later trace can capture them.
        with tf.init_scope(), tf.device(self.device):
            self._snapshot_row = tf.identity(tf.constant(snapshot_row, tf.int64))
            self._block_start = tf.identity(tf.constant(np.asarray(starts, np.int64)))
            self._clock = tf.Variable(tf.zeros([], tf.int64), trainable=False, name="release_clock")
            self._f = tf.Variable(tf.zeros([b, k], tf.float32), trainable=False, name="release_f")
            self._a = tf.Variable(tf.ones([b, k], tf.float32), trainable=False, name="release_a")
            self._snapshots = tf.Variable(
                tf.zeros([self.n_chunks + 1, 2, b, k], tf.float32), trainable=False, name="release_snapshots",
            )
            # Packed int32 words stored as float32 bit views (int32 GPU scatters run on the host).
            self._spikes = tf.Variable(
                tf.zeros([self.length, b, self.words], tf.float32), trainable=False, name="release_spikes",
            )
            self._window = tf.Variable(
                tf.zeros([self.window_size, 2, b, k], tf.float32), trainable=False, name="release_window",
            )
            self._grad_f = tf.Variable(tf.zeros([b, k], tf.float32), trainable=False, name="release_grad_f")
            self._grad_a = tf.Variable(tf.zeros([b, k], tf.float32), trainable=False, name="release_grad_a")
            self._initial_history = tf.Variable(
                tf.zeros([b, self.history * self.width], tf.float32), trainable=False, name="release_history",
            )

    def _scope(self):
        return tf.device(self.device)

    # -- forward -----------------------------------------------------------------------

    def begin(self):
        return None

    def start_forward(self, extra_states, initial_state):
        facilitation, availability = extra_states
        with self._scope():
            ops = [
                self._clock.assign(tf.zeros([], tf.int64)),
                self._f.assign(tf.cast(facilitation, tf.float32)),
                self._a.assign(tf.cast(availability, tf.float32)),
            ]
            if self.history:
                ops.append(self._initial_history.assign(tf.cast(
                    initial_state[7][:, self.history_start:self.history_stop], tf.float32,
                )))
        return ops

    def forward(self, time, spikes):
        with self._scope():
            clock = self._clock.read_value()
            f, a = self._f.read_value(), self._a.read_value()
            snapshot = tf.raw_ops.ResourceScatterUpdate(
                resource=self._snapshots.handle, indices=tf.gather(self._snapshot_row, clock)[None],
                updates=tf.stack([f, a])[None],
            )
            words = self._pack(tf.cast(spikes, tf.float32), tf.zeros([self.batch, self.neurons], tf.int8))
            recorded = tf.raw_ops.ResourceScatterUpdate(
                resource=self._spikes.handle, indices=clock[None], updates=tf.bitcast(words, tf.float32)[None],
            )
            frames, new_f, new_a = self.cell._release_math.step(spikes, f, a)
            with tf.control_dependencies([snapshot, recorded]):
                updates = [self._f.assign(new_f), self._a.assign(new_a), self._clock.assign_add(1)]
            with tf.control_dependencies(updates):
                return tf.identity(frames)

    def final_states(self):
        with self._scope():
            return self._f.read_value(), self._a.read_value()

    # -- reverse -----------------------------------------------------------------------

    def start_reverse(self):
        with self._scope():
            return [
                self._grad_f.assign(tf.zeros_like(self._grad_f)),
                self._grad_a.assign(tf.zeros_like(self._grad_a)),
            ]

    def _recorded_spikes(self, time):
        words = tf.bitcast(tf.raw_ops.ResourceGather(
            resource=self._spikes.handle, indices=tf.cast(time, tf.int64)[None], dtype=tf.float32,
        )[0], tf.int32)
        return self._unpack(words, self.neurons, tf.float32, tf.int8)[0]

    def _recompute(self, chunk_index):
        """Store the states at the start of every step of chunk ``chunk_index``'s block."""
        start = tf.gather(self._block_start, chunk_index)
        stop = tf.minimum((tf.cast(chunk_index, tf.int64) + 1) * self.chunk, self.length)
        state = tf.raw_ops.ResourceGather(
            resource=self._snapshots.handle, indices=tf.cast(chunk_index, tf.int64)[None], dtype=tf.float32,
        )[0]

        def body(time, f, a):
            written = tf.raw_ops.ResourceScatterUpdate(
                resource=self._window.handle, indices=(time - start)[None], updates=tf.stack([f, a])[None],
            )
            with tf.control_dependencies([written]):
                new_f, new_a = self.cell._release_math.transition(self._recorded_spikes(time), f, a)
            return time + 1, new_f, new_a

        result = tf.while_loop(lambda time, *_: time < stop, body, (start, state[0], state[1]),
                               parallel_iterations=1)
        return result[0]

    def _state_at(self, time, start):
        state = tf.raw_ops.ResourceGather(
            resource=self._window.handle, indices=(tf.cast(time, tf.int64) - start)[None], dtype=tf.float32,
        )[0]
        return state[0], state[1]

    def _frame(self, time, start):
        f, a = self._state_at(time, start)
        return self.cell._release_math.frames(self._recorded_spikes(time), f, a)

    def replay(self, time):
        """``[batch, width * (history + 1)]``: ``x(time), x(time - 1), ..., x(time - history)``."""
        with self._scope():
            time = tf.cast(time, tf.int64)
            chunk_index = time // self.chunk
            last = tf.logical_or(time == self.length - 1, time % self.chunk == self.chunk - 1)
            recomputed = tf.cond(last, lambda: self._recompute(chunk_index), lambda: tf.zeros([], tf.int64))
            start = tf.gather(self._block_start, chunk_index)
            with tf.control_dependencies([recomputed]):
                frames = []
                for k in range(self.history + 1):
                    previous = time - k
                    if k == 0:
                        frames.append(self._frame(previous, start))
                        continue
                    slot = k - 1

                    def initial(slot=slot, previous=previous):
                        # Initial history holds x(-1), ..., x(-history), newest first.
                        index = -previous - 1
                        return tf.reshape(tf.gather(
                            tf.reshape(self._initial_history.read_value(), [self.batch, self.history, self.width]),
                            index, axis=1,
                        ), [self.batch, self.width])

                    frames.append(tf.cond(
                        previous >= 0, lambda previous=previous: self._frame(previous, start), initial,
                    ))
                return tf.concat(frames, axis=1)

    def reverse(self, time, credit):
        """Credit ``[batch, neurons]`` of ``z(time - 1)`` from the frame credit of ``x(time)``."""
        with self._scope():
            time = tf.cast(time, tf.int64)
            start = tf.gather(self._block_start, time // self.chunk)
            f, a = self._state_at(time, start)
            spike_credit, grad_f, grad_a = self.cell._release_math.vjp(
                self._recorded_spikes(time), f, a, tf.cast(credit, tf.float32),
                self._grad_f.read_value(), self._grad_a.read_value(),
            )
            updates = [self._grad_f.assign(grad_f), self._grad_a.assign(grad_a)]
            with tf.control_dependencies(updates):
                return tf.identity(spike_credit)
