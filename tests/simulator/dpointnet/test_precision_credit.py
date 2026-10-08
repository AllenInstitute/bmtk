"""Independent event-Jacobian and mixed-state execution contracts."""

import itertools
import json
import copy
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    GLIF3Cell,
    spike_function,
    spike_gauss,
    straight_through_dampen,
)
from bmtk.simulator.dpointnet.cell_models.nest_dynamics import (
    active_update,
    spike_reset,
)
from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import (
    fused_glif_state_available,
    fused_spike_shift,
)
from bmtk.simulator.dpointnet.segmented_recompute import (
    FullBPTTGradientRunner,
    SegmentedRecomputeRunner,
)


@pytest.fixture(autouse=True)
def restore_policy():
    old = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(old)


def make_cell(
    mode="legacy",
    selective=True,
    fused=False,
    noise=False,
    two_inputs=False,
    return_spec=False,
    **options,
):
    tf.keras.mixed_precision.set_global_policy(
        "mixed_float16" if selective else "float32"
    )
    network = {
        "n_nodes": 2,
        "node_type_ids": np.array([1, 0]),
        "node_params": {
            "C_m": np.array([100.0, 120.0]),
            "g": np.array([10.0, 6.0]),
            "E_L": np.array([-70.0, -70.0]),
            "V_th": np.array([-50.0, -50.0]),
            "V_reset": np.array([-72.0, -71.0]),
            "t_ref": np.array([2.0, 3.0]),
            "k": np.array([[0.000123456789, 0.1], [0.003456789, 0.02]]),
            "asc_amps": np.array([[-1.0, 2.0], [3.0, -4.0]]),
            "asc_r": np.array([[0.8, 0.9], [1.1, 0.7]]),
        },
        "synapses": {
            "indices": np.array([[0, 1], [1, 0]], np.uint32),
            "weights": np.array([30.0, -20.0]),
            "delays": np.array([2.0, 1.0]),
            "dense_shape": (2, 2),
            "syn_ids": np.array([0, 0]),
            "dynamics_params": {"basis_weights": [[1.0, 0.3, 0.1, 0.05]]},
        },
    }
    inputs = {
        "drive": {
            "n_inputs": 2,
            "indices": np.array([[0, 0], [1, 1]]),
            "weights": np.array([1000.0, 1200.0]),
            "delays": np.array([1.0, 2.0]),
            "syn_ids": np.array([0, 0]),
            "input_type": "poisson_spikes_internal" if noise else "current",
            "options": {"trainable": True, "firing_rate": 500.0},
        }
    }
    if two_inputs:
        inputs["background"] = copy.deepcopy(inputs["drive"])
        inputs["background"]["weights"] *= 0.5
    arguments = dict(
        dynamics_mode=mode,
        state_precision="selective" if selective else "compute",
        hard_reset=False,
        tau_basis=[2.123456789, 6.0, 10.0, 20.0],
        train_recurrent_per_type=False,
        batch_size=2,
        use_fused_state=fused,
        detach_reset=False,
        detach_asc_reset=False,
        pseudo_gauss=True,
        dampening_factor=0.5,
        gauss_std=0.28,
        voltage_gradient_dampening=0.0,
        recurrent_dampening_factor=1.0,
        return_voltage_sequences=False,
        track_voltage_penalty=True,
    )
    arguments.update(options)
    if return_spec:
        return network, inputs, arguments
    return GLIF3Cell(network, inputs, **arguments)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_device_poisson_distribution_and_replay(mode):
    cell = make_cell(mode, noise=True, use_device_poisson=True)
    net = cell.inputs["drive"]

    @tf.function
    def sample(step):
        return cell.sample_noise_spikes(100000, tf.reshape(step, [1]), net)

    first = sample(tf.constant(0)).numpy()
    np.testing.assert_array_equal(first, sample(tf.constant(0)).numpy())
    assert not np.array_equal(first, sample(tf.constant(1)).numpy())
    assert first.dtype == np.int32
    assert abs(first.mean() - 0.5) < 0.01
    assert abs(first.var() - 0.5) < 0.015
    assert np.count_nonzero(first > 1) > 10000
    graph = sample.get_concrete_function(tf.constant(0)).graph.as_graph_def()
    assert not any("Poisson" in node.op for node in graph.node)


@pytest.mark.parametrize(
    "detach_reset,detach_asc", itertools.product([False, True], repeat=2)
)
@pytest.mark.parametrize("hard_reset", [False, True])
def test_nest_event_jacobian_and_forward_invariance(
    detach_reset, detach_asc, hard_reset
):
    voltage = tf.constant([[1.2, 0.8]], tf.float64)
    adaptation = tf.constant([[[0.7, -0.3], [-0.2, 0.4]]], tf.float64)
    event = tf.constant([[1.0, 0.0]], tf.float64)
    amps = tf.constant([[0.3, -0.8], [-0.4, 0.9]], tf.float64)
    rho = tf.constant([[0.6, 1.3], [0.9, 0.8]], tf.float64)

    def evaluate(dr, da):
        return spike_reset(
            voltage,
            tf.zeros((1, 2), tf.int8),
            adaptation,
            event,
            reset_voltage=tf.constant(-0.1, tf.float64),
            refractory_steps=tf.constant([2, 3], tf.int8),
            asc_amplitudes=amps,
            asc_refractory_decay=rho,
            hard_reset=hard_reset,
            detach_reset=dr,
            detach_asc_reset=da,
        )

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([voltage, adaptation, event])
        v, _, a = evaluate(detach_reset, detach_asc)
        loss_v, loss_a = tf.reduce_sum(v), tf.reduce_sum(a)
    zero = tf.UnconnectedGradients.ZERO
    gz_v = tape.gradient(loss_v, event, unconnected_gradients=zero)
    gz_a = tape.gradient(loss_a, event, unconnected_gradients=zero)
    expected_v = (-0.1 - voltage.numpy()) if hard_reset else np.full((1, 2), -1.1)
    expected_a = np.sum(amps.numpy() + (rho.numpy() - 1) * adaptation.numpy(), axis=-1)
    np.testing.assert_allclose(gz_v, expected_v * (not detach_reset), atol=1e-14)
    np.testing.assert_allclose(gz_a, expected_a * (not detach_asc), atol=1e-14)
    np.testing.assert_allclose(
        tape.gradient(loss_a, adaptation), [[[0.6, 1.3], [1.0, 1.0]]]
    )
    for actual, expected in zip(
        evaluate(detach_reset, detach_asc), evaluate(True, True)
    ):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "detach_reset,detach_asc", itertools.product([False, True], repeat=2)
)
@pytest.mark.parametrize("dampening", [0.0, 0.3, 1.0])
def test_legacy_analytic_local_jacobian(detach_reset, detach_asc, dampening):
    cell = make_cell(
        selective=False,
        detach_reset=detach_reset,
        detach_asc_reset=detach_asc,
        voltage_gradient_dampening=dampening,
    )
    z = tf.constant([[1.0, 0.0]])
    v = tf.constant([[0.8, -0.1]])
    a = tf.constant([[0.2, -0.1, 0.3, -0.4]])
    p = tf.constant([[0.1] * 8])
    with tf.GradientTape() as tape:
        tape.watch([z, v, a, p])
        outputs = cell._dense_update_impl(
            1, z, v, tf.zeros((1, 2), tf.int8), a, p, p, p
        )
        loss = tf.reduce_sum(outputs[0]) + tf.reduce_sum(outputs[2])
    gz, gv, ga, gp = tape.gradient(
        loss, [z, v, a, p], unconnected_gradients=tf.UnconnectedGradients.ZERO
    )
    np.testing.assert_allclose(
        gz,
        -(not detach_reset) + (not detach_asc) * np.sum(cell.asc_amps, axis=-1)[None],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        gv, cell.decay.numpy()[None] * (1 - dampening), atol=1e-7
    )
    np.testing.assert_allclose(
        ga,
        (cell.asc_decay + cell.current_factor[:, None]).numpy().reshape(1, -1),
        atol=1e-7,
    )
    np.testing.assert_allclose(
        gp, np.repeat(cell.current_factor.numpy(), 4)[None], atol=1e-7
    )


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_selective_footprint_coefficients_and_live_update(mode):
    cell = make_cell(mode)
    state = cell.zero_state(2, tf.float16)
    assert [x.dtype for x in cell.zero_state(2, tf.float32)] == [x.dtype for x in state]
    assert [x.dtype for x in state[:7]] == [
        tf.float16,
        tf.float32,
        tf.int8,
        tf.float32,
        tf.float16,
        tf.float16,
        tf.int32,
    ]
    assert (
        sum(int(np.prod(state[i].shape)) * state[i].dtype.size for i in (1, 3, 4, 5))
        // 4
        == 28
    )
    for name in (
        "syn_decay",
        "psc_initial",
        "asc_decay",
        "asc_amps",
        "decay",
        "current_factor",
    ):
        assert tf.as_dtype(getattr(cell, name).dtype) == tf.float32
    expected = np.exp(-np.array([[0.003456789, 0.02], [0.000123456789, 0.1]]))
    np.testing.assert_array_equal(cell.asc_decay, expected.astype(np.float32))
    assert cell.recurrent_weight_values.dtype == "float32"
    assert cell.recurrent_weight_values_compute.dtype == "float16"
    original = cell.asc_amps.numpy()
    cell.asc_amps.assign(original * 2)
    np.testing.assert_array_equal(cell.asc_amps, original * 2)


@pytest.mark.parametrize(
    "option,value",
    [
        ("state_precision", "mixed"),
        ("detach_reset", 1),
        ("detach_asc_reset", "false"),
        ("gauss_std", 0.0),
        ("gauss_std", float("nan")),
        ("dampening_factor", float("inf")),
        ("dampening_factor", -0.1),
    ],
)
def test_invalid_precision_credit_options(option, value):
    with pytest.raises(ValueError):
        make_cell(**{option: value})


def test_selective_rejects_non_mixed_policy():
    with pytest.raises(ValueError, match="mixed_float16"):
        make_cell(selective=False, state_precision="selective")


@pytest.mark.parametrize("index", [0, 1, 3, 4, 5])
def test_selective_cell_rejects_wrong_state_storage(index):
    cell = make_cell()
    state = list(cell.zero_state(2, tf.float16))
    wrong_dtype = tf.float16 if index in (1, 3) else tf.float32
    state[index] = tf.cast(state[index], wrong_dtype)
    with pytest.raises(ValueError, match="explicitly convert"):
        cell(tf.ones((2, 2), tf.float16), state)


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
def test_surrogate_values_and_causality(gaussian, dtype):
    u = tf.constant([[-2.0, -1.0, -0.28, 0.0, 0.28, 1.0, 2.0]], dtype)
    with tf.GradientTape() as tape:
        tape.watch(u)
        z = (
            spike_gauss(u, tf.cast(0.28, dtype), tf.cast(0.5, dtype))
            if gaussian
            else spike_function(u, tf.cast(0.5, dtype))
        )
    g = tape.gradient(z, u)
    reference = 0.5 * (
        np.exp(-((u.numpy().astype(float) / 0.28) ** 2))
        if gaussian
        else np.maximum(1 - np.abs(u.numpy()), 0)
    )
    np.testing.assert_allclose(
        g, reference, rtol=3e-3 if dtype == tf.float16 else 1e-6, atol=2e-7
    )
    np.testing.assert_array_equal(z, u.numpy() > 0)


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_fused_surrogate_mixed_dtype_and_refractory(gaussian, mixed):
    if not fused_glif_state_available():
        pytest.skip("GPU operator unavailable")
    dtype = tf.float32 if mixed else tf.float16
    u = tf.constant([[-2.0, -1.0, -0.28, 0.0, 0.28, 1.0, 2.0]], dtype)
    history = tf.zeros((1, 14), tf.float16)
    refractory = tf.constant([[False, False, False, False, False, True, True]])
    with tf.GradientTape() as tape:
        tape.watch([u, history])
        z, h = fused_spike_shift(
            u, refractory, history, 0.5, pseudo_gauss=gaussian, gauss_std=0.28
        )
        loss = tf.reduce_sum(z) + tf.reduce_sum(h)
    gu, gh = tape.gradient(loss, [u, history])
    reference = (
        np.exp(-((u.numpy().astype(float) / 0.28) ** 2))
        if gaussian
        else np.maximum(1 - np.abs(u.numpy()), 0)
    )
    reference[refractory.numpy()] = 0
    np.testing.assert_allclose(gu, reference, atol=3e-7, rtol=3e-3)
    assert z.dtype == h.dtype == history.dtype
    np.testing.assert_array_equal(gh, [[1.0] * 7 + [0.0] * 7])


def build_core(cell, length=31, batch=2):
    state = list(cell.zero_state(batch, cell.compute_dtype))
    width = 0 if cell.inputs["drive"]["input_type"] == "poisson_spikes_internal" else 2
    sequence = tf.keras.Input(shape=(None, width), dtype=cell.compute_dtype)
    states = [tf.keras.Input(shape=x.shape[1:], dtype=x.dtype) for x in state]
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    layer._autocast = False
    out = layer(sequence, initial_state=states)
    sequences = (
        list(out[0])
        if isinstance(out[0], (tuple, list))
        else [out[0][..., : cell._n_neurons], out[0][..., -1]]
    )
    core = tf.keras.Model([sequence, *states], [*sequences, *out[1:]])
    return core, state, tf.ones((batch, length, width), cell.compute_dtype)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("chunk", [1, 7, 25])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("device_poisson", [False, True])
def test_selective_exact_replay_and_poisson(mode, chunk, fused, device_poisson):
    if fused and not fused_glif_state_available():
        pytest.skip("GPU operator unavailable")
    cell = make_cell(mode, fused=fused, noise=True, use_device_poisson=device_poisson)
    core, state, sequence = build_core(cell)
    runner = SegmentedRecomputeRunner(core, 31, chunk, 2, pack_spike_checkpoints=True)
    floating = [x for x in state if x.dtype.is_floating]
    targets = [
        *floating,
        cell.recurrent_weight_values,
        cell.inputs["drive"]["input_weight_values"],
    ]

    def evaluate(replay):
        with tf.GradientTape() as tape:
            tape.watch(floating)
            out = runner(sequence, state) if replay else core([sequence, *state])
            loss = tf.reduce_mean(tf.cast(out[0], tf.float32)) + tf.reduce_mean(out[1])
        return out, tape.gradient(loss, targets)

    full, gf = tf.function(lambda: evaluate(False))()
    replay, gr = tf.function(lambda: evaluate(True))()
    for a, b in zip(replay, full):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    for a, b in zip(gr, gf):
        assert a is not None and b is not None
        np.testing.assert_allclose(a, b, rtol=4e-3, atol=5e-4)
    np.testing.assert_array_equal(full[8], [31, 31])
    assert full[0].dtype == tf.float16 and full[1].dtype == tf.float32
    assert int(cell.noise_stream) == 0


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_selective_prefix_and_loss_causality(mode):
    cell = make_cell(mode)
    core, state, sequence = build_core(cell, 9)
    changed = tf.concat([sequence[:, :4], sequence[:, 4:] * 3], axis=1)
    before = core([sequence, *state])
    after = core([changed, *state])
    for a, b in zip(before[:2], after[:2]):
        np.testing.assert_array_equal(a[:, :4], b[:, :4])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(sequence)
        out = core([sequence, *state])
        prefix_loss = tf.reduce_sum(out[1][:, :4])
        later_loss = tf.reduce_sum(out[1][:, -1])
    early = tape.gradient(prefix_loss, sequence)
    late = tape.gradient(later_loss, sequence)
    np.testing.assert_array_equal(early[:, 4:], 0)
    assert np.any(late[:, :4].numpy() != 0)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("precision", ["compute", "selective"])
@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("hard_reset", [False, True])
@pytest.mark.parametrize(
    "detach_reset,detach_asc", list(itertools.product([False, True], repeat=2))
)
def test_fused_cell_matches_fallback(
    mode, precision, gaussian, hard_reset, detach_reset, detach_asc
):
    if not fused_glif_state_available():
        pytest.skip("GPU operator unavailable")
    results = []
    for fused in (False, True):
        cell = make_cell(
            mode,
            fused=fused,
            state_precision=precision,
            pseudo_gauss=gaussian,
            detach_reset=detach_reset,
            detach_asc_reset=detach_asc,
            hard_reset=hard_reset,
        )
        core, state, sequence = build_core(cell, 13)
        state[1] = tf.constant([[1.1, 0.8], [0.4, 1.2]], state[1].dtype)
        state[3] = tf.constant([[0.1, -0.2, -0.4, 0.3]] * 2, state[3].dtype)
        floating = [x for x in state if x.dtype.is_floating]

        def evaluate():
            with tf.GradientTape() as tape:
                tape.watch(floating)
                out = core([sequence, *state])
                loss = tf.reduce_mean(tf.cast(out[0], tf.float32)) + tf.reduce_mean(
                    out[1]
                )
                loss += 0.1 * tf.reduce_mean(tf.cast(out[5], tf.float32))
            return out, tape.gradient(loss, floating)

        out, gradients = tf.function(evaluate)()
        results.append(([x.numpy() for x in out], [x.numpy() for x in gradients]))
    np.testing.assert_array_equal(results[0][0][0], results[1][0][0])
    tolerance = 7e-3 if precision == "compute" else 2e-3
    for expected, actual in zip(
        tf.nest.flatten(results[0]), tf.nest.flatten(results[1])
    ):
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=tolerance)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("unroll", [False, True])
@pytest.mark.parametrize("full_voltage", [False, True])
def test_mixed_rnn_direct_step_and_continuation(mode, unroll, full_voltage):
    cell = make_cell(mode, return_voltage_sequences=full_voltage)
    state = cell.zero_state(2, cell.compute_dtype)
    sequence = tf.ones((2, 8, 2), cell.compute_dtype)
    layer = ExplicitStateRNN(
        cell, return_sequences=True, return_state=True, unroll=unroll
    )
    whole = layer(sequence, initial_state=state)
    prefix = layer(sequence[:, :3], initial_state=state)
    suffix = layer(sequence[:, 3:], initial_state=prefix[1:])
    for actual, expected in zip(suffix[1:], whole[1:]):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    direct_state = state
    values = []
    for i in range(8):
        value, direct_state = cell(sequence[:, i], direct_state)
        values.append(value)
    direct = tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=1), *values)
    for expected, actual in zip(tf.nest.flatten(direct), tf.nest.flatten(whole[0])):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(whole[1:], direct_state):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


def test_selective_checkpoint_and_explicit_profile_conversion(tmp_path):
    cell = make_cell()
    state = cell.zero_state(2, cell.compute_dtype)
    state = (state[0], tf.ones_like(state[1]), *state[2:])
    saved = tf.train.Checkpoint(
        **{f"state_{i}": tf.Variable(x) for i, x in enumerate(state)}
    )
    checkpoint = saved.save(str(tmp_path / "mixed-state"))
    restored = tf.train.Checkpoint(
        **{f"state_{i}": tf.Variable(tf.zeros_like(x)) for i, x in enumerate(state)}
    )
    restored.restore(checkpoint).assert_consumed()
    for i, expected in enumerate(state):
        actual = getattr(restored, f"state_{i}")
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    old_state = tuple(
        tf.cast(x, tf.float16) if i in (1, 3) else x for i, x in enumerate(state)
    )
    layer = ExplicitStateRNN(cell, return_sequences=True)
    with pytest.raises(ValueError, match="explicitly convert"):
        layer(tf.ones((2, 3, 2), tf.float16), initial_state=old_state)
    converted = tuple(
        tf.cast(x, tf.float32) if i in (1, 3) else x for i, x in enumerate(old_state)
    )
    layer(tf.ones((2, 3, 2), tf.float16), initial_state=converted)


@pytest.mark.parametrize("gaussian", [False, True])
def test_options_json_roundtrip_and_compatibility_defaults(gaussian):
    options = dict(
        state_precision="selective",
        detach_reset=False,
        detach_asc_reset=True,
        pseudo_gauss=gaussian,
        dampening_factor=0.5,
        gauss_std=0.28,
    )
    restored = json.loads(json.dumps(options))
    cell = make_cell(**restored)
    assert cell.state_precision == restored["state_precision"]
    assert cell.detach_reset == restored["detach_reset"]
    assert cell.detach_asc_reset == restored["detach_asc_reset"]
    assert cell._pseudo_gauss == restored["pseudo_gauss"]


@pytest.mark.parametrize("retention", [0.0, 0.3, 1.0])
def test_recurrent_retention_scales_only_history_credit(retention):
    cell = make_cell(selective=False, recurrent_dampening_factor=retention)
    history = tf.ones((1, cell._n_neurons * cell.max_delay), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(history)
        current = cell.calculate_i_rec_with_custom_grad(history)
        loss = tf.reduce_sum(current)
    gh, gw = tape.gradient(loss, [history, cell.recurrent_weight_values])
    expected_history = np.zeros(history.shape, np.float32)
    expected_weight = []
    for index, weight, synapse in zip(
        cell.recurrent_indices.numpy(),
        cell.recurrent_weight_values_compute.numpy(),
        cell.syn_ids.numpy(),
    ):
        basis_sum = np.sum(cell.synaptic_basis_weights.numpy()[synapse])
        expected_history[0, index[1]] += retention * weight * basis_sum
        expected_weight.append(basis_sum)
    np.testing.assert_allclose(gh, expected_history, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(gw, expected_weight, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("chunk", [None, 7])
def test_selective_accelerated_canonical_update_and_all_shadows(mode, chunk):
    if not fused_glif_state_available():
        pytest.skip("GPU operator unavailable")
    gradients_by_order = []
    for direct in (False, True):
        cell = make_cell(
            mode,
            fused=True,
            noise=True,
            two_inputs=True,
            batch_size=32,
            use_fused_cuda=True,
            use_pair_projection=True,
            use_packed_sm120_backward=True,
            use_packed_sm120_external_backward=True,
            use_direct_csr_recurrent_gradient=direct,
            use_fused_current_accumulation=True,
        )
        core, state, sequence = build_core(cell, length=13, batch=32)
        transform = cell.restore_segmented_variable_gradients if direct else None
        if chunk:
            runner = SegmentedRecomputeRunner(
                core,
                13,
                chunk,
                2,
                pack_spike_checkpoints=True,
                variable_gradient_transform=transform,
            )
        elif direct:
            runner = FullBPTTGradientRunner(core, transform)
        else:
            runner = lambda inputs, initial: core([inputs, *initial])
        masters = [cell.recurrent_weight_values] + [
            inp["input_weight_values"] for inp in cell.inputs.values()
        ]

        @tf.function
        def evaluate():
            with tf.GradientTape() as tape:
                out = runner(sequence, state)
                loss = tf.reduce_mean(tf.cast(out[0], tf.float32)) + tf.reduce_mean(
                    out[1]
                )
            return tape.gradient(loss, masters)

        gradients = evaluate()
        assert all(g is not None and np.isfinite(g).all() for g in gradients)
        assert all(np.any(g.numpy() != 0) for g in gradients)
        gradients_by_order.append([g.numpy() for g in gradients])
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, global_clipnorm=1.0)
        optimizer.apply_gradients(zip(gradients, masters))
        cell.refresh_recurrent_weight_shadow()
        np.testing.assert_array_equal(
            cell.recurrent_weight_values_compute, tf.cast(masters[0], tf.float16)
        )
        assert masters[0][0] >= 0 and masters[0][1] <= 0
        for inp in cell.inputs.values():
            np.testing.assert_array_equal(
                inp["input_weight_values_compute"],
                tf.cast(inp["input_weight_values"], tf.float16),
            )
            assert np.all(inp["input_weight_values"].numpy() >= 0)
        cell.close_fused_cuda()
    for canonical, csr in zip(*gradients_by_order):
        np.testing.assert_allclose(csr, canonical, rtol=3e-3, atol=3e-5)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("state_input", [False, True])
@pytest.mark.parametrize("full_voltage", [False, True])
def test_rnn_model_selective_factory_and_extractor(mode, state_input, full_voltage):
    from bmtk.simulator.dpointnet.rnn_model import RNN

    network, inputs, options = make_cell(
        mode, return_spec=True, return_voltage_sequences=full_voltage
    )
    options.pop("train_recurrent_per_type")
    rnn = RNN(seq_len=9, batch_size=2, dtype="float16", cell_params=options)
    rnn._recurrent_networks["test"] = SimpleNamespace(
        to_dict=lambda: copy.deepcopy(network)
    )
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive",
        n_spiking_nodes=2,
        to_dict=lambda: copy.deepcopy(inputs["drive"]),
    )
    try:
        rnn.build(training=True, use_state_input=state_input)
        assert isinstance(rnn.rsnn_layer, ExplicitStateRNN)
        sequence = tf.ones((2, 9, 2), tf.float16)
        outputs = rnn.run_extractor(sequence, rnn.zero_state)
        assert outputs[1].dtype == tf.float16
        assert outputs[2].dtype == tf.float32
        assert outputs[4].dtype == tf.float32
        assert outputs[5].dtype == tf.float16
        assert outputs[6].dtype == tf.float16
        if not full_voltage:
            assert outputs[0][0].dtype == tf.float16
            assert outputs[0][1].dtype == tf.float32
    finally:
        rnn.cleanup()


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("profile", [None, "auto"])
@pytest.mark.parametrize("state_input", [False, True])
def test_legacy_compute_factory_honors_direct_loop(direct, profile, state_input):
    from bmtk.simulator.dpointnet.rnn_model import RNN

    network, inputs, options = make_cell(
        selective=False, return_spec=True, track_voltage_penalty=False,
        return_voltage_sequences=True, use_direct_state_rnn_loop=direct,
        acceleration_profile=profile,
    )
    options.pop("train_recurrent_per_type")
    rnn = RNN(seq_len=3, batch_size=2, cell_params=options)
    rnn._recurrent_networks["test"] = SimpleNamespace(
        to_dict=lambda: copy.deepcopy(network)
    )
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive", n_spiking_nodes=2,
        to_dict=lambda: copy.deepcopy(inputs["drive"]),
    )
    try:
        rnn.build(training=True, use_state_input=state_input)
        assert rnn.cell.dynamics_mode == "legacy"
        assert rnn.cell.state_precision == "compute"
        assert not rnn.cell._online_voltage_losses
        assert isinstance(rnn.rsnn_layer, ExplicitStateRNN) == direct
        sequence = tf.ones((2, 3, 2), tf.float32)
        outputs = rnn.rsnn_layer(sequence, initial_state=rnn.zero_state)
        state = rnn.zero_state
        steps = []
        for step in range(3):
            output, state = rnn.cell(sequence[:, step], state)
            steps.append(output)
        expected = tf.stack(steps, axis=1)
        np.testing.assert_allclose(outputs[0], expected, rtol=1e-6, atol=1e-6)
        for actual_state, expected_state in zip(outputs[1:], state):
            np.testing.assert_allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)
    finally:
        rnn.cleanup()


@pytest.mark.skipif(
    not fused_cuda_available(),
    reason="Actual compatible GPU operator required",
)
@pytest.mark.parametrize("trainable,per_type", [(False, False), (True, True), (True, False)])
def test_actual_auto_cell_preserves_recurrent_parameter_sharing(trainable, per_type):
    from bmtk.simulator.dpointnet.acceleration import resolve_acceleration_options

    network, inputs, options = make_cell(selective=False, return_spec=True)
    options.update(train_recurrent=trainable, train_recurrent_per_type=per_type)
    options, _ = resolve_acceleration_options(
        {"acceleration_profile": "auto", **options},
        compute_dtype=tf.float32, variable_dtype=tf.float32,
        batch_size=2, basis_width=4,
    )
    cell = GLIF3Cell(network, inputs, **options)
    try:
        assert cell._use_direct_csr_recurrent_gradient == (trainable and not per_type)
        assert cell.recurrent_weight_values.trainable == (trainable and not per_type)
        output, _ = cell(tf.ones((2, 2)), cell.zero_state(2, tf.float32))
        assert all(np.all(np.isfinite(value.numpy())) for value in tf.nest.flatten(output))
    finally:
        cell.close_fused_cuda()


@pytest.mark.skipif(not fused_cuda_available(), reason="Actual compatible GPU operator required")
def test_actual_legacy_compute_factory_executes_auto_weight_carrier(monkeypatch):
    from bmtk.simulator.dpointnet.acceleration import csr_spike_ops
    from bmtk.simulator.dpointnet.rnn_model import RNN

    if not csr_spike_ops._auto_native_architecture(csr_spike_ops._gpu_compute_architecture()):
        pytest.skip("Published automatic native policy requires SM75/SM86+")
    network, inputs, options = make_cell(
        selective=False, return_spec=True, track_voltage_penalty=False,
        return_voltage_sequences=True, use_direct_state_rnn_loop=True,
        acceleration_profile="auto",
    )
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    options.pop("train_recurrent_per_type")
    original = GLIF3Cell._call_impl
    observed_carriers = []

    def call_impl(self, *args, **kwargs):
        observed_carriers.append(kwargs.get("recurrent_weight_carrier") is not None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GLIF3Cell, "_call_impl", call_impl)
    rnn = RNN(seq_len=3, batch_size=2, dtype="float16", cell_params=options)
    rnn._recurrent_networks["test"] = SimpleNamespace(to_dict=lambda: copy.deepcopy(network))
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive", n_spiking_nodes=2, to_dict=lambda: copy.deepcopy(inputs["drive"]),
    )
    try:
        rnn.build(training=True)
        assert isinstance(rnn.rsnn_layer, ExplicitStateRNN)
        assert rnn.cell.state_precision == "compute"
        assert rnn.acceleration_report["selected"]["use_fused_recurrent_accumulation"] is True
        assert any(observed_carriers)
        with tf.GradientTape() as tape:
            output = rnn.rsnn_layer(tf.ones((2, 3, 2), tf.float16), initial_state=rnn.zero_state)
            loss = tf.reduce_sum(tf.cast(output[0], tf.float32))
        gradient = tape.gradient(loss, rnn.cell.recurrent_weight_values)
        assert gradient is not None
        assert np.all(np.isfinite(gradient.numpy()))
    finally:
        rnn.cleanup()


@pytest.mark.parametrize("hard_reset", [False, True])
def test_nest_continuous_jacobian_fixed_refractory(hard_reset):
    v = tf.constant([[0.2, 0.4]], tf.float64)
    a = tf.constant([[[0.1, -0.2], [0.3, -0.4]]], tf.float64)
    p = tf.constant([[[0.2] * 4, [0.1] * 4]], tf.float64)
    q = tf.constant([[[0.3] * 4, [-0.1] * 4]], tf.float64)
    decay = tf.constant([0.8, 0.9], tf.float64)
    factor = tf.constant([0.1, 0.2], tf.float64)
    beta = tf.constant([[0.99, 0.8], [0.95, 0.9]], tf.float64)
    mean = tf.constant([[0.995, 0.85], [0.97, 0.93]], tf.float64)
    pv = tf.constant([[0.1] * 4, [0.2] * 4], tf.float64)
    qv = tf.constant([[0.3] * 4, [0.4] * 4], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch([v, a, p, q])
        nv, _, na, _ = active_update(
            v,
            tf.constant([[0, 2]], tf.int8),
            a,
            p,
            q,
            decay=decay,
            current_factor=factor,
            asc_decay=beta,
            asc_mean=mean,
            psc_voltage=pv,
            rise_voltage=qv,
            reset_voltage=tf.constant(0.0, tf.float64),
            hard_reset=hard_reset,
        )
        loss = tf.reduce_sum(nv) + tf.reduce_sum(na)
    gv, ga, gp, gq = tape.gradient(loss, [v, a, p, q])
    gate = np.array([1.0, 0.0 if hard_reset else 1.0])
    np.testing.assert_allclose(gv, (decay.numpy() * gate)[None], atol=1e-15)
    expected_a = (
        np.array([[0.99, 0.8], [1.0, 1.0]])
        + (factor.numpy() * gate)[:, None] * mean.numpy()
    )
    np.testing.assert_allclose(ga, expected_a[None], atol=1e-15)
    np.testing.assert_allclose(gp, (pv.numpy() * gate[:, None])[None], atol=1e-15)
    np.testing.assert_allclose(gq, (qv.numpy() * gate[:, None])[None], atol=1e-15)


def test_fused_state_rejects_stale_precision_abi(monkeypatch):
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops

    monkeypatch.setattr(
        glif_state_ops,
        "_OPS",
        SimpleNamespace(dpointnet_spike_shift_backward=lambda: None),
    )
    monkeypatch.setattr(glif_state_ops, "_glif_gpu_compatibility_error", lambda: None)
    assert not glif_state_ops.fused_glif_state_available()
