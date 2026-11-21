"""Fixed recurrent release (short-term plasticity) on GLIF3Cell.

The reference is an independent per-edge, arrival-keyed implementation written with
plain TF ops (``ArrivalReferenceCell``), so its tape gradients include every credit
path. From rest with an empty spike buffer it is identical to the emission-keyed
release cell.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell, _rollout_constant
from bmtk.simulator.dpointnet.cell_models.nest_dynamics import time_steps
from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.direct_recorded_adjoint import (
    DirectRecordedAdjointRunner, direct_records_supported,
)
from bmtk.simulator.dpointnet.recurrent_release import (
    ReleaseGLIF3Cell, build_release_layout, correction_frames, release_parameters,
    release_transition, release_vjp,
)
from bmtk.simulator.dpointnet.spike_statistics import SpikeStatisticsRunner
from bmtk.simulator.dpointnet.training import TrainingEngine


@pytest.fixture(autouse=True)
def restore_policy():
    # Other modules can leave a mixed policy behind; tests here choose their precision.
    previous = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy("float32")
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def network_spec(seed=0, n=24, edges=160, poisson=True):
    rng = np.random.default_rng(seed)
    types = np.arange(n) % 3            # 0 exc, 1 pv-like, 2 unclassed
    pairs = set()
    while len(pairs) < edges:
        t, s = rng.integers(0, n, 2)
        if t != s:
            pairs.add((int(t), int(s)))
    indices = np.array(sorted(pairs), np.int64)
    sign = np.where(types[indices[:, 1]] == 0, 1.0, -1.0)
    network = {
        "n_nodes": n,
        "node_type_ids": types,
        "node_params": {
            "C_m": np.array([100.0, 60.0, 80.0]), "g": np.array([8.0, 6.0, 7.0]),
            "E_L": np.array([-70.0, -68.0, -69.0]), "V_th": np.array([-52.0, -50.0, -51.0]),
            "V_reset": np.array([-70.0, -69.0, -70.0]), "t_ref": np.array([2.0, 1.0, 2.0]),
            "k": np.array([[0.003, 0.1], [0.01, 0.05], [0.005, 0.02]]),
            "asc_amps": np.array([[-1.0, 2.0], [0.5, -0.5], [1.0, -1.0]]),
            "asc_r": np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        },
        "synapses": {
            "indices": indices.astype(np.uint32),
            "weights": sign * (20.0 + 30.0 * rng.random(edges)),
            "delays": rng.choice([1.0, 2.0, 3.0], edges),
            "dense_shape": (n, n),
            "syn_ids": rng.integers(0, 2, edges),
            "dynamics_params": {"basis_weights": [[1.0, 0.3, 0.1, 0.05], [0.2, 1.0, 0.4, 0.1]]},
        },
    }
    inputs = {
        "drive": {
            "n_inputs": n, "indices": np.column_stack((np.arange(n), np.arange(n))),
            "weights": np.full(n, 900.0), "delays": np.ones(n), "syn_ids": np.zeros(n, int),
            "input_type": "poisson_spikes_internal" if poisson else "spikes",
            "options": {"trainable": True, "firing_rate": 400.0},
        }
    }
    classes = np.array([0, 1, -1])[types]
    # rows pre * 2 + post: E->E depleting/facilitating, E->PV depressing, PV->E facilitating, PV->PV unsupported
    table = np.array([
        [0.3, 0.4, -1.0, 400.0, 60.0],
        [0.6, 0.0, 0.3, 200.0, 20.0],
        [0.2, 0.6, 0.1, 150.0, 300.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
    ])
    supported = np.array([True, True, True, False])
    spec = {"node_classes": classes, "pair_parameters": table, "pair_supported": supported}
    return network, inputs, spec


def cell_arguments(**options):
    arguments = dict(
        acceleration_profile=None, dynamics_mode="nest", state_precision="compute",
        hard_reset=True, hard_reset_gradient_mode="soft_surrogate",
        tau_basis=[2.0, 6.0, 10.0, 20.0], train_recurrent_per_type=False, batch_size=4,
        detach_reset=True, detach_asc_reset=False, pseudo_gauss=True, dampening_factor=0.5,
        gauss_std=0.28, return_voltage_sequences=True, track_voltage_penalty=True, max_delay=5,
    )
    arguments.update(options)
    return arguments


def make(release, spec_override=None, **options):
    network, inputs, spec = network_spec()
    if spec_override is not None:
        spec = spec_override(spec)
    args = cell_arguments(**options)
    if release:
        return ReleaseGLIF3Cell(copy.deepcopy(network), copy.deepcopy(inputs), recurrent_release=spec, **args)
    return GLIF3Cell(copy.deepcopy(network), copy.deepcopy(inputs), **args)


def rollout(cell, steps, batch=4, seed=3, initial=None):
    state = tuple(cell.zero_state(batch, cell.compute_dtype)) if initial is None else initial
    x = tf.zeros((batch, cell.inputs_idx[-2] if hasattr(cell, "_release_keys") else cell.inputs_idx[-1]), cell.compute_dtype)
    outputs = []
    for _ in range(steps):
        out, state = cell(x, state)
        outputs.append(out)
    return outputs, state




class ArrivalReferenceCell(GLIF3Cell):
    """Independent per-edge, arrival-keyed release reference (no keys, no +/- split, no tie)."""

    def __init__(self, network, inputs, *, edge_table, **kwargs):
        super().__init__(network, inputs, **kwargs)
        self._ref_gradient_scale = float(self.recurrent_spike_gradient_scale)
        synapses = network["synapses"]
        indices = np.asarray(synapses["indices"], np.int64)
        dt = 1.0
        steps = time_steps(np.clip(np.asarray(synapses["delays"], float), dt, self._delay_limit_ms), dt)
        self._ref_column = tf.constant(indices[:, 1] + self._n_neurons * (steps - 1), tf.int64)
        self._ref_target = tf.constant(indices[:, 0], tf.int64)
        self._ref_syn = tf.constant(np.asarray(synapses["syn_ids"]), tf.int64)
        self._ref_parameters = release_parameters(edge_table, dt)
        self._ref_edges = int(indices.shape[0])

    @property
    def state_size(self):
        base = super().state_size
        if not hasattr(self, "_ref_edges"):
            return base
        return tuple(base) + (self._ref_edges, self._ref_edges)

    def zero_state(self, batch_size, dtype, with_names=False):
        base = super().zero_state(batch_size, dtype, with_names=with_names)
        if not hasattr(self, "_ref_edges") or with_names:
            return base
        return tuple(base) + (tf.zeros((batch_size, self._ref_edges)), tf.ones((batch_size, self._ref_edges)))

    def _canonical_recurrent(self):
        layout = getattr(self, "recurrent_master_layout", None)
        if self._csr_master_weights and layout is not None:
            return tf.cast(layout.to_canonical(self.recurrent_weight_values), tf.float32)
        return tf.cast(self.recurrent_weight_values, tf.float32)

    def _call_impl(self, inputs, states, **kwargs):
        n = len(super().state_size)
        base, f, a = tuple(states[:n]), states[n], states[n + 1]
        events = tf.gather(tf.cast(base[0], tf.float32), self._ref_column, axis=1)
        scale = self._ref_gradient_scale

        @tf.custom_gradient
        def recurrent_credit(x):
            # The recurrent spike-gradient scale covers every recurrent path, release included.
            return tf.identity(x), lambda dy: dy * scale

        events = recurrent_credit(events)
        response, new_f, new_a = release_transition(events, f, a, self._ref_parameters)
        correction = events * (response - 1.0)
        basis = tf.cast(_rollout_constant(self, "synaptic_basis_weights"), tf.float32)
        per_edge = self._canonical_recurrent()[:, None] * tf.gather(basis, self._ref_syn)
        contribution = correction[:, :, None] * per_edge[None]
        batch = tf.shape(events)[0]
        current = tf.transpose(tf.math.unsorted_segment_sum(
            tf.transpose(contribution, [1, 0, 2]), self._ref_target, self._n_neurons), [1, 0, 2])
        current = tf.reshape(current, [batch, -1]) * tf.cast(_rollout_constant(self, "_lr_scale"), tf.float32)
        projected = self._project_step_currents(inputs, base, noise_seed=kwargs.get("noise_seed"))
        currents = projected[0] + tf.cast(current, projected[0].dtype)
        output, new_base = self.advance_with_currents(inputs, base, currents, noise_seed=kwargs.get("noise_seed"))
        return output, tuple(new_base) + (new_f, new_a)


def edge_table(network, spec):
    classes = np.asarray(spec["node_classes"])
    table, supported = np.asarray(spec["pair_parameters"]), np.asarray(spec["pair_supported"])
    c = int(round(np.sqrt(table.shape[0])))
    idx = np.asarray(network["synapses"]["indices"], np.int64)
    pre, post = classes[idx[:, 1]], classes[idx[:, 0]]
    pair = np.maximum(pre, 0) * c + np.maximum(post, 0)
    active = (pre >= 0) & (post >= 0) & supported[pair]
    rows = np.tile([1.0, 0.0, 0.0, 1.0, 1.0], (idx.shape[0], 1))
    rows[active] = table[pair[active]]
    return rows


def make_reference(**options):
    network, inputs, spec = network_spec()
    return ArrivalReferenceCell(copy.deepcopy(network), copy.deepcopy(inputs),
                                edge_table=edge_table(network, spec), **cell_arguments(**options))



def static_spec(spec):
    spec = dict(spec)
    table = np.array(spec["pair_parameters"], float)
    table[:] = [1.0, 0.0, 0.0, 1.0, 1.0]
    spec["pair_parameters"] = table
    return spec


DIRECT = dict(
    temporal_gradient_precision="compute", use_fused_cuda=True, use_fused_state=True,
    use_combined_current_sweep=True, use_fused_state_history=True, use_direct_state_rnn_loop=True,
    use_fused_nest_event_vjp=True, use_device_active_queue_forward=True, use_csr_master_weights=True,
    use_uint8_synapse_types=True, use_adaptive_routing=True, use_fused_recurrent_accumulation=True,
    use_direct_csr_recurrent_gradient=True, use_prepacked_nest_coefficients=True,
    use_native_voltage_penalty=True, return_voltage_sequences=False, use_pair_projection=True,
)


# -- CPU ---------------------------------------------------------------------------------


def _assert_outputs(ours, theirs, n_neurons):
    """Spikes exactly; continuous outputs to float32 accumulation order (keyed vs per-edge sums)."""
    assert sum(float(tf.reduce_sum(o[:, :n_neurons])) for o in theirs) > 0
    for a, b in zip(ours, theirs):
        a, b = a.numpy(), b.numpy()
        np.testing.assert_array_equal(a[:, :n_neurons], b[:, :n_neurons])
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)


def test_static_parameters_reproduce_glif3cell():
    plain_cell = make(False)
    plain, _ = rollout(plain_cell, 40)
    static, state = rollout(make(True, spec_override=static_spec), 40)
    _assert_outputs(static, plain, plain_cell._n_neurons)
    np.testing.assert_array_equal(state[-2].numpy(), 0.0)
    np.testing.assert_array_equal(state[-1].numpy(), 1.0)


def test_release_matches_arrival_reference_forward():
    cell, reference = make(True), make_reference()
    ours, _ = rollout(cell, 60)
    theirs, _ = rollout(reference, 60)
    _assert_outputs(ours, theirs, cell._n_neurons)


def test_release_vjp_matches_autodiff():
    rng = np.random.default_rng(0)
    table = np.array([[0.3, 0.4, -1.0, 400.0, 60.0], [0.6, 0.0, 0.3, 200.0, 20.0],
                      [0.2, 0.6, 0.1, 150.0, 300.0], [0.5, 0.2, -1.0, 50.0, 30.0]])
    parameters = release_parameters(table, 1.0)
    events = tf.constant((rng.random((3, 4)) < 0.5).astype(np.float32))
    f = tf.constant(rng.random((3, 4)).astype(np.float32) * 0.5)
    a = tf.constant(0.3 + 0.7 * rng.random((3, 4)).astype(np.float32))
    credit = tf.constant(rng.standard_normal((3, 8)).astype(np.float32))
    grad_f = tf.constant(rng.standard_normal((3, 4)).astype(np.float32))
    grad_a = tf.constant(rng.standard_normal((3, 4)).astype(np.float32))
    with tf.GradientTape() as tape:
        tape.watch([events, f, a])
        response, new_f, new_a = release_transition(events, f, a, parameters)
        frames = correction_frames(events, response)
        objective = (tf.reduce_sum(frames * credit) + tf.reduce_sum(new_f * grad_f)
                     + tf.reduce_sum(new_a * grad_a))
    expected = tape.gradient(objective, [events, f, a])
    actual = release_vjp(events, f, a, parameters, credit, grad_f, grad_a)
    for x, y in zip(expected, actual):
        np.testing.assert_allclose(y.numpy(), x.numpy(), rtol=1e-5, atol=1e-6)


def test_layout_keys_signs_and_validation():
    network, _, spec = network_spec()
    layout = build_release_layout(network, spec)
    classes = np.asarray(spec["node_classes"])
    indices = np.asarray(network["synapses"]["indices"], np.int64)
    pre, post = classes[indices[:, 1]], classes[indices[:, 0]]
    supported = (pre >= 0) & (post >= 0) & ~((pre == 1) & (post == 1))
    n_release = int(supported.sum())
    assert layout.edge_recurrent.size == 2 * n_release
    np.testing.assert_array_equal(layout.edge_sign, np.repeat([1.0, -1.0], n_release))
    np.testing.assert_array_equal(np.sort(layout.edge_recurrent[:n_release]), np.flatnonzero(supported))
    assert layout.key_source.size == len({(int(s), int(p) * 2 + int(q)) for s, p, q in
                                          zip(indices[supported, 1], pre[supported], post[supported])})
    mixed_classes = np.array(spec["node_classes"])
    mixed_classes[0] = 1 - mixed_classes[0]
    shuffled = dict(spec, node_classes=mixed_classes)
    with pytest.raises(ValueError, match="differ within node type"):
        build_release_layout(network, shuffled)
    bad = dict(spec, pair_parameters=np.where(np.arange(5) == 0, 0.0, spec["pair_parameters"]))
    with pytest.raises(ValueError, match="Invalid release parameters"):
        build_release_layout(network, bad)


def test_fold_moves_shadow_gradient_into_recurrent_master():
    cell = make(True)
    shadow = cell.inputs["recurrent_release"]["input_weight_values"]
    recurrent = cell.recurrent_weight_values
    np.testing.assert_allclose(
        shadow.numpy(),
        tf.gather(recurrent, cell._release_tie_index).numpy() * cell._release_tie_sign.numpy(),
    )
    shadow_grad = tf.constant(np.arange(1, shadow.shape[0] + 1, dtype=np.float32))
    recurrent_grad = tf.ones_like(recurrent)
    folded = cell.fold_variable_gradients([recurrent, shadow], [recurrent_grad, shadow_grad])
    expected = np.ones(recurrent.shape, np.float32)
    np.add.at(expected, cell._release_tie_index.numpy(), shadow_grad.numpy() * cell._release_tie_sign.numpy())
    np.testing.assert_allclose(folded[0].numpy(), expected)
    np.testing.assert_array_equal(folded[1].numpy(), 0.0)
    recurrent.assign(recurrent * 2.0)
    cell.refresh_recurrent_weight_shadow()
    np.testing.assert_allclose(
        shadow.numpy(),
        tf.gather(recurrent, cell._release_tie_index).numpy() * cell._release_tie_sign.numpy(),
    )


@pytest.mark.parametrize("adjoint", [False, "tape"])
def test_training_refuses_non_direct_routes(adjoint):
    cell = make(True)
    engine = SimpleNamespace(rnn=SimpleNamespace(_cell=cell), recorded_adjoint=adjoint)
    with pytest.raises(ValueError, match="only on the direct recorded adjoint"):
        TrainingEngine._apply_gradients(engine, [])


def test_optimizer_never_updates_the_tied_shadow():
    cell = make(True)
    shadow = cell.inputs["recurrent_release"]["input_weight_values"]
    recurrent, drive = cell.recurrent_weight_values, cell.inputs["drive"]["input_weight_values"]
    assert shadow.shape[0] not in (recurrent.shape[0], drive.shape[0])
    optimizer = tf.keras.optimizers.AdamW(learning_rate=0.1, weight_decay=0.5)
    engine = SimpleNamespace(
        rnn=SimpleNamespace(_cell=cell, model=SimpleNamespace(trainable_variables=[recurrent, shadow, drive])),
        recorded_adjoint="direct", optimizer=optimizer, learning_rule=SimpleNamespace(uses_bptt=True),
    )
    update = TrainingEngine._optimizer_update_variables(engine)
    assert len(update) == 2 and update[0] is recurrent and update[1] is drive
    shadow_before, recurrent_before = shadow.numpy().copy(), recurrent.numpy().copy()
    grads = [tf.ones_like(recurrent), tf.ones_like(shadow), tf.ones_like(drive)]
    TrainingEngine._apply_gradients(engine, grads)
    # Decoupled weight decay acts on every variable the optimizer owns; the shadow is not one.
    np.testing.assert_array_equal(shadow.numpy(), shadow_before)
    assert not np.array_equal(recurrent.numpy(), recurrent_before)
    assert all(tuple(v.shape) != tuple(shadow.shape) for v in optimizer.variables)
    cell.refresh_recurrent_weight_shadow()
    np.testing.assert_allclose(
        shadow.numpy(),
        tf.gather(recurrent, cell._release_tie_index).numpy() * cell._release_tie_sign.numpy(),
    )


# -- GPU: direct recorded adjoint -------------------------------------------------------


def _direct_and_reference(batch, length, chunk, mixed, javier=False, gradient_scale=1.0):
    tf.keras.mixed_precision.set_global_policy("mixed_float16" if mixed else "float32")
    precision = dict(state_precision="selective" if mixed else "compute", recurrent_spike_gradient_scale=gradient_scale)
    cell = make(True, batch_size=batch, **precision, **DIRECT,
                **({"use_javier_recurrent_vjp": True} if javier else {}))
    assert bool(cell.use_javier_recurrent_vjp) == javier
    # The fold must map through a non-identity CSR edge order.
    permutation = np.asarray(cell.recurrent_master_layout.canonical_to_csr)
    assert not np.array_equal(permutation, np.arange(permutation.size))
    reference = make_reference(batch_size=batch, return_voltage_sequences=False, **precision)
    assert not direct_records_supported(cell)
    # Pair metadata routes the correction input through the fused accumulate backward.
    assert int(cell.inputs["recurrent_release"]["fused_connectivity"]["n_pairs"]) > 0
    rng = np.random.default_rng(5)
    initial = list(cell.zero_state(batch, cell.compute_dtype))
    initial[1] = tf.cast(-60.0 + 8 * rng.random(initial[1].shape), initial[1].dtype)
    initial = tuple(initial)
    n_reference = len(reference.state_size) - 2
    reference_initial = (tuple(initial[:n_reference])
                         + tuple(reference.zero_state(batch, reference.compute_dtype))[n_reference:])
    reference.recurrent_weight_values.assign(tf.cast(
        cell.recurrent_master_layout.to_canonical(cell.recurrent_weight_values),
        reference.recurrent_weight_values.dtype))
    x = tf.keras.Input((None, 0), dtype=cell.compute_dtype)
    state_inputs = tuple(tf.keras.Input(v.shape[1:], dtype=v.dtype) for v in initial)
    out = ExplicitStateRNN(cell, return_sequences=True, return_state=True)(x, initial_state=state_inputs)
    model = tf.keras.Model((x, *state_inputs), (*out[0], *out[1:]))
    windows = ((0, length), (2, length - 1))
    runner = DirectRecordedAdjointRunner(model, cell, sequence_length=length, chunk_size=chunk,
                                         differentiate_inputs=False, count_windows=windows)
    weights = [tf.constant(rng.standard_normal((batch, cell._n_neurons)), tf.float32) for _ in windows]
    inputs = tf.zeros((batch, length, 0), cell.compute_dtype)

    @tf.function
    def direct():
        with tf.GradientTape() as tape:
            output = runner(inputs, initial)
            loss = tf.add_n([tf.reduce_sum(tf.cast(c, tf.float32) * w) for c, w in zip(output.counts, weights)])
        variables = model.trainable_variables
        grads = cell.fold_variable_gradients(variables, tape.gradient(loss, variables, unconnected_gradients="zero"))
        named = {v.name.split("/")[-1]: g for v, g in zip(variables, grads)}
        named["sparse_recurrent_weights"] = cell.recurrent_master_layout.to_canonical(
            named["sparse_recurrent_weights"])
        return output.counts, named

    def tape():
        with tf.GradientTape() as tape:
            state, spikes = reference_initial, []
            for _ in range(length):
                output, state = reference(tf.zeros((batch, 0), reference.compute_dtype), state)
                output = output[0] if isinstance(output, (tuple, list)) else output
                spikes.append(tf.cast(output, tf.float32)[:, :reference._n_neurons])
            z = tf.stack(spikes, 1)
            counts = [tf.reduce_sum(z[:, a:b], 1) for a, b in windows]
            loss = tf.add_n([tf.reduce_sum(c * w) for c, w in zip(counts, weights)])
        variables = reference.trainable_variables
        grads = tape.gradient(loss, variables, unconnected_gradients="zero")
        return counts, {v.name.split("/")[-1]: g for v, g in zip(variables, grads)}

    return direct(), tape()


@pytest.mark.parametrize("mixed,tolerance,batch,javier,gradient_scale", [
    (False, 1e-5, 8, False, 1.0), (True, 5e-3, 8, False, 1.0),
    # The training configuration: FP16 Javier batch-32 fused accumulate backward.
    (True, 5e-3, 32, True, 1.0),
    # A recurrent spike-gradient scale also scales the release-correction credit.
    (False, 1e-5, 8, False, 0.3),
])
def test_direct_route_gradients_match_arrival_reference(mixed, tolerance, batch, javier, gradient_scale):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    (counts, grads), (reference_counts, reference_grads) = _direct_and_reference(
        batch, 120, 20, mixed, javier, gradient_scale)
    assert float(tf.reduce_sum(reference_counts[0])) > 0
    np.testing.assert_array_equal(counts[1].numpy(), reference_counts[1].numpy())
    if not mixed:
        np.testing.assert_array_equal(counts[0].numpy(), reference_counts[0].numpy())
    for name in ("sparse_recurrent_weights", "drive_input_weights"):
        a, b = reference_grads[name].numpy(), grads[name].numpy()
        assert np.linalg.norm(a - b) <= tolerance * np.linalg.norm(a), name
    np.testing.assert_array_equal(grads["recurrent_release_input_weights"].numpy(), 0.0)


@pytest.mark.parametrize("spec_override", [None, static_spec])
def test_direct_forward_matches_ordinary_path(spec_override):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    tf.keras.mixed_precision.set_global_policy("float32")
    batch, length = 4, 60
    cell = make(True, spec_override=spec_override, batch_size=batch, **DIRECT)
    rng = np.random.default_rng(2)
    initial = list(cell.zero_state(batch, cell.compute_dtype))
    initial[0] = tf.cast(rng.random(initial[0].shape) < 0.2, initial[0].dtype)
    initial[1] = tf.cast(-60.0 + 4 * rng.random(initial[1].shape), initial[1].dtype)
    n_base = cell._release_base_states
    initial[n_base] = tf.constant(0.4 * rng.random(initial[n_base].shape), tf.float32)
    initial[n_base + 1] = tf.constant(0.5 + 0.5 * rng.random(initial[n_base + 1].shape), tf.float32)
    initial = tuple(initial)
    x = tf.keras.Input((None, 0), dtype=cell.compute_dtype)
    state_inputs = tuple(tf.keras.Input(v.shape[1:], dtype=v.dtype) for v in initial)
    out = ExplicitStateRNN(cell, return_sequences=True, return_state=True)(x, initial_state=state_inputs)
    model = tf.keras.Model((x, *state_inputs), (*out[0], *out[1:]))
    options = dict(sequence_length=length, chunk_size=15, differentiate_inputs=False,
                   count_windows=((0, length),))
    ordinary = SpikeStatisticsRunner(model, n_sequence_outputs=2, pack_spike_checkpoints=True, **options)
    direct = DirectRecordedAdjointRunner(model, cell, **options)
    inputs = tf.zeros((batch, length, 0), cell.compute_dtype)
    a, b = ordinary(inputs, initial), direct(inputs, initial)
    np.testing.assert_array_equal(a.counts[0].numpy(), b.counts[0].numpy())
    for x, y in zip(a.final_state, b.final_state):
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype.kind == "f":
            # Graded exchange vs ordinary projection: float32 accumulation order over 60 steps.
            np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(x, y)
