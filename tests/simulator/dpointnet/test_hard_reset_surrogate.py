from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet import acceleration
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
from bmtk.simulator.dpointnet.cell_models.nest_dynamics import (
    active_update, spike_reset, validate_hard_reset_gradient_mode,
)
from bmtk.simulator.dpointnet.rnn_model import RNN
from test_acceleration import hardware
from test_nest_dynamics import make_network_inputs


@pytest.fixture(autouse=True)
def restore_precision_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("traced", [False, True])
def test_refractory_forward_exact_and_surrogate_candidate_derivatives(dtype, traced):
    def evaluate(voltage, current, mode):
        with tf.GradientTape() as tape:
            tape.watch((voltage, current))
            output = active_update(
                voltage, tf.constant([[0, 2]], tf.int16),
                tf.zeros((1, 2, 2), dtype), tf.zeros((1, 2, 1), dtype),
                tf.zeros((1, 2, 1), dtype),
                decay=tf.constant([0.75, 0.5], dtype),
                current_factor=tf.constant([0.25, 0.125], dtype),
                asc_decay=tf.ones((2, 2), dtype),
                asc_mean=tf.ones((2, 2), dtype),
                psc_voltage=tf.zeros((2, 1), dtype),
                rise_voltage=tf.zeros((2, 1), dtype),
                reset_voltage=tf.constant(0.0, dtype),
                hard_reset=True, direct_current=current,
                hard_reset_gradient_mode=mode,
            )
            loss = tf.reduce_sum(output[0])
        return output, tape.gradient(loss, (voltage, current))

    call = tf.function(evaluate) if traced else evaluate
    voltage = tf.constant([[0.5, 0.75]], dtype)
    current = tf.constant([[1.0, 2.0]], dtype)
    exact, exact_grads = call(voltage, current, "exact")
    surrogate, surrogate_grads = call(voltage, current, "soft_surrogate")
    for left, right in zip(exact, surrogate):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    np.testing.assert_array_equal(surrogate[0].numpy(), [[0.625, 0.0]])
    np.testing.assert_array_equal(exact_grads[0].numpy(), [[0.75, 0.0]])
    np.testing.assert_array_equal(exact_grads[1].numpy(), [[0.25, 0.0]])
    np.testing.assert_array_equal(surrogate_grads[0].numpy(), [[0.75, 0.5]])
    np.testing.assert_array_equal(surrogate_grads[1].numpy(), [[0.25, 0.125]])


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("detach_asc", [False, True])
def test_spike_reset_exact_forward_and_independent_surrogate_jacobian(
    dtype, detach_reset, detach_asc
):
    def evaluate(mode):
        voltage = tf.constant([[1.5, 0.5]], dtype)
        event = tf.constant([[1.0, 0.0]], dtype)
        adaptation = tf.constant([[[0.25, 0.5], [0.25, 0.5]]], dtype)
        with tf.GradientTape() as tape:
            tape.watch((voltage, event, adaptation))
            outputs = spike_reset(
                voltage, tf.zeros((1, 2), tf.int16), adaptation, event,
                reset_voltage=tf.constant(0.0, dtype),
                refractory_steps=tf.constant([2, 3], tf.int16),
                asc_amplitudes=tf.constant([[0.5, -0.25], [0.5, -0.25]], dtype),
                asc_refractory_decay=tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype),
                hard_reset=True, detach_reset=detach_reset, detach_asc_reset=detach_asc,
                hard_reset_gradient_mode=mode,
            )
            loss = tf.reduce_sum(outputs[0]) + tf.reduce_sum(outputs[2])
        return outputs, tape.gradient(
            loss, (voltage, event, adaptation),
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )

    exact, _ = evaluate("exact")
    surrogate, gradients = evaluate("soft_surrogate")
    for left, right in zip(exact, surrogate):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    np.testing.assert_array_equal(surrogate[0].numpy(), [[0.0, 0.5]])
    np.testing.assert_array_equal(gradients[0].numpy(), [[1.0, 1.0]])
    event_gradient = (0.0 if detach_reset else -1.0) + (0.0 if detach_asc else -0.125)
    np.testing.assert_array_equal(gradients[1].numpy(), [[event_gradient, event_gradient]])
    np.testing.assert_array_equal(gradients[2].numpy(), [[[0.5, 0.5], [1.0, 1.0]]])


@pytest.mark.parametrize("mode,hard,dynamics", [
    ("invalid", True, "nest"), (None, True, "nest"),
    ("soft_surrogate", False, "nest"), ("soft_surrogate", None, "nest"),
    ("soft_surrogate", True, "legacy"),
])
def test_invalid_gradient_mode_rejected(mode, hard, dynamics):
    with pytest.raises(ValueError, match="hard_reset_gradient_mode"):
        validate_hard_reset_gradient_mode(mode, hard_reset=hard, dynamics_mode=dynamics)


def make_surrogate_cell(mode="soft_surrogate", policy="float32", **kwargs):
    tf.keras.mixed_precision.set_global_policy(policy)
    network, inputs = make_network_inputs()
    return GLIF3Cell(
        network, inputs, tau_basis=[2.0], hard_reset=True,
        hard_reset_gradient_mode=mode, detach_reset=False, detach_asc_reset=False,
        use_fused_cuda=False, acceleration_profile=None,
        return_voltage_sequences=True, track_voltage_penalty=False,
        spike_surrogate_gain=0.3,
        **kwargs,
    )


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
def test_cell_rollout_keeps_all_hard_forward_states_and_restores_voltage_credit(policy):
    try:
        exact = make_surrogate_cell("exact", policy)
        surrogate = make_surrogate_cell("soft_surrogate", policy)
        exact.build((1, 1))
        surrogate.build((1, 1))
        states = [
            list(exact.zero_state(1, exact.compute_dtype)),
            list(surrogate.zero_state(1, surrogate.compute_dtype)),
        ]
        # Force a spike at the first step, then test credit through its refractory clamp.
        for state in states:
            state[1] = tf.ones_like(state[1]) * 1.5
        initial_voltages = [state[1] for state in states]
        gradients = []
        trajectories = []
        for cell, state, voltage in zip((exact, surrogate), states, initial_voltages):
            with tf.GradientTape() as tape:
                tape.watch(voltage)
                trace = []
                for _ in range(5):
                    output, state = cell(tf.zeros((1, 1), cell.compute_dtype), state)
                    trace.extend(tf.nest.flatten((output, state)))
                loss = tf.reduce_sum(state[1])
            gradients.append(tape.gradient(loss, voltage))
            trajectories.append(trace)
        for left, right in zip(*trajectories):
            np.testing.assert_array_equal(left.numpy(), right.numpy())
        np.testing.assert_array_equal(gradients[0].numpy(), [[0.0]])
        assert np.isfinite(gradients[1].numpy()).all()
        assert np.abs(gradients[1].numpy()).max() > 0
    finally:
        tf.keras.mixed_precision.set_global_policy("float32")


def test_training_requires_matching_explicit_surrogate_on_built_model():
    params = {"dynamics_mode": "nest", "hard_reset": True, "hard_reset_gradient_mode": "soft_surrogate"}
    rnn = RNN(cell_params=params)
    original = dict(rnn.cell_params)
    assert rnn._resolve_cell_params(training=True) == original
    assert rnn.cell_params == original
    rnn._model_built = True
    rnn._cell = SimpleNamespace(_hard_reset=True, hard_reset_gradient_mode="soft_surrogate")
    assert rnn._resolve_cell_params(training=True) == original
    rnn._cell.hard_reset_gradient_mode = "exact"
    with pytest.raises(ValueError, match="already built with hard reset"):
        rnn._resolve_cell_params(training=True)
    rnn._cell._hard_reset = False
    with pytest.raises(ValueError, match="already built with soft reset"):
        rnn._resolve_cell_params(training=True)


@pytest.mark.parametrize("profile", ["auto", None])
def test_surrogate_uses_ordinary_acceleration_eligibility(hardware, profile):
    def resolve(mode):
        return acceleration.resolve_acceleration_options(
            {"acceleration_profile": profile, "hard_reset_gradient_mode": mode},
            compute_dtype=tf.float16, variable_dtype=tf.float32,
            batch_size=32, basis_width=4,
        )

    exact, exact_report = resolve("exact")
    surrogate, surrogate_report = resolve("soft_surrogate")
    assert exact.pop("hard_reset_gradient_mode") == "exact"
    assert surrogate.pop("hard_reset_gradient_mode") == "soft_surrogate"
    assert exact == surrogate
    if profile is not None:
        assert exact_report["selected"] == surrogate_report["selected"]
        assert exact_report["reasons"] == surrogate_report["reasons"]


@pytest.mark.parametrize("profile", ["auto", None])
@pytest.mark.parametrize("name", ["use_fused_state", "use_direct_state_rnn_loop", "use_native_voltage_penalty"])
def test_explicit_native_acceleration_retains_constructor_validation(hardware, profile, name):
    options, _ = acceleration.resolve_acceleration_options(
        {"acceleration_profile": profile, "hard_reset_gradient_mode": "soft_surrogate", name: True},
        compute_dtype=tf.float32, variable_dtype=tf.float32, batch_size=1, basis_width=4,
    )
    assert options[name] is True


@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("chunk", [1, 3])
def test_checkpointing_preserves_hard_surrogate_outputs_and_gradients(policy, chunk):
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    cell = make_surrogate_cell(policy=policy)
    initial = list(cell.zero_state(1, cell.compute_dtype))
    initial[1] = tf.ones_like(initial[1]) * 1.5
    sequence = tf.keras.Input(batch_shape=(1, None, 1), dtype=cell.compute_dtype)
    state_inputs = [
        tf.keras.Input(batch_shape=state.shape, dtype=state.dtype)
        for state in initial
    ]
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    outputs = layer(sequence, initial_state=state_inputs)
    core = tf.keras.Model([sequence, *state_inputs], outputs)
    runner = SegmentedRecomputeRunner(
        core, sequence_length=6, chunk_size=chunk,
        n_sequence_outputs=1, differentiate_inputs=True,
    )
    x = tf.ones((1, 6, 1), cell.compute_dtype)

    def evaluate(checkpointed):
        with tf.GradientTape() as tape:
            tape.watch((x, initial[1]))
            result = runner(x, initial) if checkpointed else core([x, *initial])
            loss = tf.reduce_sum(tf.cast(result[0], tf.float32))
            loss += tf.reduce_sum(tf.cast(result[2], tf.float32))
        gradients = tape.gradient(
            loss, [x, initial[1], *core.trainable_variables],
            unconnected_gradients=tf.UnconnectedGradients.ZERO,
        )
        return result, gradients

    full, full_gradients = evaluate(False)
    replay, replay_gradients = evaluate(True)
    assert np.abs(full_gradients[1].numpy()).max() > 0
    assert np.abs(full_gradients[-1].numpy()).max() > 0
    for left, right in zip(full, replay):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    for index, (left, right) in enumerate(zip(full_gradients, replay_gradients)):
        assert left is not None and right is not None, (
            index, [variable.name for variable in core.trainable_variables]
        )
        assert np.isfinite(right.numpy()).all()
        np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("traced", [False, True])
def test_fp32_temporal_surrogate_keeps_hard_forward_and_restores_voltage_credit(traced):
    from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner

    results = []
    gradients = []
    for mode in ("exact", "soft_surrogate"):
        cell = make_surrogate_cell(
            mode, "mixed_float16", temporal_gradient_precision="float32"
        )
        runner = TemporalAdjointRunner(cell, chunk_size=2)
        state = list(cell.zero_state(1, cell.compute_dtype))
        state[1] = tf.ones_like(state[1]) * 1.5

        def evaluate(voltage):
            with tf.GradientTape() as tape:
                tape.watch(voltage)
                initial = [state[0], voltage, *state[2:]]
                output, final = runner(tf.zeros((1, 5, 1), tf.float32), initial)
                loss = tf.reduce_sum(final[1])
            return output, final, tape.gradient(loss, voltage)

        output, final, gradient = (tf.function(evaluate) if traced else evaluate)(state[1])
        results.append(tf.nest.flatten((output, final)))
        gradients.append(gradient)
    for left, right in zip(*results):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    np.testing.assert_array_equal(gradients[0].numpy(), [[0.0]])
    assert np.isfinite(gradients[1].numpy()).all()
    assert np.abs(gradients[1].numpy()).max() > 0
