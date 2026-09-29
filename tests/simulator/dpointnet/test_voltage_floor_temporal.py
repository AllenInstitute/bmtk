"""Combined native online loss and selective FP32 temporal-adjoint contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import VoltageRateFloor
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner
from test_temporal_adjoint import long_credit_cell, primal_trajectory, independent_adjoint
from test_voltage_rate_floor import make_rnn
from test_recurrent_accumulation import gpu, make_fused_cell
from test_precision_credit import make_cell, build_core
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import fused_glif_state_available


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
@pytest.mark.parametrize("chunk", [1, 5])
@pytest.mark.parametrize("graph", [False, True])
def test_online_floor_matches_independent_temporal_vjp(mode, replay_mode, chunk, graph):
    loss = VoltageRateFloor(SimpleNamespace(dt=1.0), cost=2.0)
    cell = long_credit_cell(mode, replay_mode=replay_mode, online_voltage_losses=[loss])
    cell._return_voltage_sequences = False
    loss.build(cell)
    loss.initialize_rates([0.0, 0.05])
    state = cell.zero_state(1, cell.compute_dtype)
    x = tf.ones((1, 12, 2), tf.float32) * 0.01
    runner = TemporalAdjointRunner(cell, chunk_size=chunk)

    def evaluate():
        with tf.GradientTape() as tape:
            tape.watch(x)
            outputs, final = runner(x, state)
            value = loss(outputs[0], final)
        dx, dw = tape.gradient(value, [x, cell.inputs["probe"]["input_weight_values"]])
        return value, outputs, final, dx, dw

    value, outputs, final, dx, dw = (tf.function(evaluate) if graph else evaluate)()
    trajectory = primal_trajectory(cell, x, state)
    expected_dx, _, expected_dw = independent_adjoint(
        cell, trajectory, x, end=12, voltage_floor=loss
    )
    voltage = np.asarray(trajectory[1])[1:, 0]
    assert np.max(voltage) < 0.1
    assert not np.any(outputs[0])
    expected_value = np.mean(
        2.0 * np.maximum(0.9 - voltage, 0)**2 * np.asarray(loss.gate)
    )
    np.testing.assert_allclose(value, expected_value, rtol=2e-6)
    np.testing.assert_allclose(dx[0], expected_dx, rtol=3e-5, atol=1e-8)
    np.testing.assert_allclose(dw, expected_dw[1], rtol=3e-5, atol=1e-8)
    assert np.all(np.asarray(dx[0, :-3]) < 0)
    assert np.all(np.asarray(dw) < 0)
    assert outputs[0].dtype == tf.float16
    assert outputs[1].shape == (1, 12)
    assert final[-1].dtype == final[1].dtype == final[3].dtype == tf.float32
    assert final[-1].shape == (1, 1)
    assert loss.accepted_updates.numpy() == 0
    for actual, expected in zip(final, trajectory):
        np.testing.assert_array_equal(actual, expected[-1])


def test_programmatic_identical_configs_share_history():
    rnn = SimpleNamespace(dt=1.0)
    first = VoltageRateFloor(rnn)
    same = VoltageRateFloor(rnn, cost=1, target=0.9, floor_hz=0.1, ema_decay=0.95)
    different = VoltageRateFloor(rnn, cost=2)
    assert same is first
    assert different is not first
    assert rnn._online_voltage_losses == [first, different]


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_native_engine_selective_bootstrap_and_validation(mode, replay_mode, tmp_path):
    rnn, engine, loss = make_rnn(
        mode=mode, policy="float16", selective=True, replay_mode=replay_mode,
    )
    x = tf.ones((2, 6, 1), tf.float16)
    state = rnn.cell.zero_state(2, tf.float16)
    for accepted in (1, 2):
        result = engine._train_step_single([x], {}, state)
        assert loss.accepted_updates.numpy() == accepted
        assert (result["test"]["floor"].numpy() == 0) == (accepted == 1)
    previous = [np.array(v) for v in (loss.rate_ema_hz, loss.gate, loss.accepted_updates)]
    engine._validation_step([x], [{}], state, "single")
    for actual, expected in zip((loss.rate_ema_hz, loss.gate, loss.accepted_updates), previous):
        np.testing.assert_array_equal(actual, expected)
    checkpoint = tf.train.Checkpoint(model=rnn.model, optimizer=engine.optimizer)
    path = checkpoint.save(str(tmp_path / "selective"))
    seed = np.array(rnn.cell.noise_seed)
    rnn.cell.advance_noise_seed()
    loss.initialize_rates([10.0])
    checkpoint.restore(path).assert_consumed()
    np.testing.assert_array_equal(rnn.cell.noise_seed, seed)
    np.testing.assert_array_equal(loss.gate, previous[1])
    rnn.cleanup()


@gpu
@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
@pytest.mark.parametrize("fused_state", [False, True])
def test_native_floor_same_cache_accumulation_and_canonical_gradients(mode, replay_mode, fused_state):
    loss = VoltageRateFloor(SimpleNamespace(dt=1.0))
    cell = make_fused_cell(
        mode, replay_mode, fused_state=fused_state, online_voltage_losses=[loss],
        return_voltage_sequences=False, pseudo_gauss=True,
        detach_reset=False, detach_asc_reset=False,
    )
    loss.build(cell)
    loss.initialize_rates([0.0, 0.0])
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    initial = list(cell.zero_state(32, cell.compute_dtype))
    initial[0] = tf.ones_like(initial[0])
    initial = tuple(initial)
    x = tf.zeros((32, 11, 0), tf.float32)
    masters = tuple(
        value.value if not callable(getattr(value, "value", None)) else value
        for value in cell.trainable_variables
    )

    @tf.function
    def evaluate():
        cache = runner._forward(x, initial)
        final_gradients = [None] * len(initial)
        final_gradients[-1] = tf.ones_like(cache[1][-1]) / (32 * 11)
        def reverse(fused):
            cell.use_fused_recurrent_accumulation = fused
            return runner._backward(
                x, initial, cache, (None, None), final_gradients, masters, capture=True
            )
        expected = reverse(False)
        actual = reverse(True)
        return cache[1][-1], expected, actual

    try:
        penalty, expected, actual = evaluate()
        assert penalty.dtype == tf.float32
        assert np.any(np.asarray(penalty) > 0)
        for observed, reference in zip(tf.nest.flatten(actual), tf.nest.flatten(expected)):
            np.testing.assert_allclose(observed, reference, rtol=4e-5, atol=1e-7)
        assert np.any(np.asarray(actual[2][0]) != 0)
        assert all(value.dtype == tf.float32 for value in actual[2])
        assert loss.accepted_updates.numpy() == 0
    finally:
        cell.close_fused_cuda()


@pytest.mark.skipif(not fused_glif_state_available(), reason="Fused state CUDA unavailable")
@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("detach", [False, True])
@pytest.mark.parametrize("hard_reset", [False, True])
def test_fused_native_pre_reset_credit_matches_tensorflow(selective, gaussian, detach, hard_reset):
    results = []
    for fused in (False, True):
        loss = VoltageRateFloor(SimpleNamespace(dt=1.0))
        cell = make_cell(
            "nest", selective=selective, fused=fused, online_voltage_losses=[loss],
            pseudo_gauss=gaussian, hard_reset=hard_reset,
            detach_reset=detach, detach_asc_reset=detach,
        )
        loss.build(cell)
        loss.initialize_rates([0.0, 0.05])
        core, state, sequence = build_core(cell, length=13)
        state[1] = tf.constant([[1.1, 0.8], [0.4, 1.2]], state[1].dtype)
        state[3] = tf.constant([[0.1, -0.2, -0.4, 0.3]] * 2, state[3].dtype)
        floating = [value for value in state if value.dtype.is_floating]

        @tf.function
        def evaluate():
            with tf.GradientTape() as tape:
                tape.watch(floating)
                output = core([sequence, *state])
                value = loss(output[0], output[2:]) + 0.1 * tf.reduce_mean(output[5])
            return output, tape.gradient(value, floating)

        results.append(evaluate())
    for observed, expected in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0])):
        np.testing.assert_allclose(observed, expected, rtol=3e-3, atol=2e-3)
