import copy
import gc
import time

import numpy as np
import pytest
from scipy.linalg import expm

tf = pytest.importorskip("tensorflow")
from bmtk.simulator.dpointnet.cell_models.nest_dynamics import (
    integration_coefficients,
    active_update,
    spike_reset,
    time_steps,
)


def make_cell(
    dt=1.0,
    delay=1.0,
    hard_reset=True,
    mode="nest",
    internal_noise=False,
    use_fused_cuda=False,
):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    nodes = {
        "C_m": np.array([100.0]),
        "g": np.array([10.0]),
        "E_L": np.array([-70.0]),
        "V_m": np.array([-70.0]),
        "V_th": np.array([-50.0]),
        "V_reset": np.array([-70.0]),
        "t_ref": np.array([2.0]),
        "k": np.array([[0.01, 0.1]]),
        "asc_amps": np.array([[0.0, 0.0]]),
    }
    network = {
        "n_nodes": 1,
        "node_type_ids": np.array([0]),
        "node_params": nodes,
        "synapses": {
            "indices": np.array([[0, 0]], dtype=np.uint32),
            "weights": np.array([0.0]),
            "delays": np.array([2.0]),
            "dense_shape": (1, 1),
            "syn_ids": np.array([0]),
            "dynamics_params": {"basis_weights": [[1.0]]},
        },
    }
    inputs = {
        "drive": {
            "n_inputs": 1,
            "indices": np.array([[0, 0]]),
            "weights": np.array([10.0]),
            "delays": np.array([delay]),
            "syn_ids": np.array([0]),
            "input_type": "spikes",
            "options": {"trainable": True},
        }
    }
    if internal_noise:
        inputs["drive"]["input_type"] = "poisson_spikes_internal"
        inputs["drive"]["options"]["firing_rate"] = 250.0
    cell_options = {}
    if mode is not None:
        cell_options["dynamics_mode"] = mode
    return GLIF3Cell(
        network,
        inputs,
        dt=dt,
        tau_basis=[2.0],
        hard_reset=hard_reset,
        train_recurrent_per_type=False,
        use_fused_cuda=use_fused_cuda,
        **cell_options,
    )


def test_default_dynamics_mode_preserves_legacy_behavior():
    cell = make_cell(mode=None, hard_reset=None)

    assert cell.dynamics_mode == "legacy"
    assert cell._hard_reset is False
    assert len(cell.zero_state(1, tf.float32)) == 7


def test_full_cell_delays_and_input_gradient():
    immediate = make_cell(delay=1.0)
    delayed = make_cell(delay=3.0)

    def run(cell):
        state = cell.zero_state(1, tf.float32)
        voltages = []
        for step in range(8):
            output, state = cell(tf.constant([[1.0 if step == 0 else 0.0]]), state)
            voltages.append(output[0, 1])
        return tf.stack(voltages), state

    with tf.GradientTape() as tape:
        values, state = run(delayed)
        loss = tf.reduce_sum(values)
    gradient = tape.gradient(loss, delayed.inputs["drive"]["input_weight_values"])
    baseline, _ = run(immediate)
    np.testing.assert_allclose(values[2:], baseline[:-2], atol=1e-8)
    assert len(state) == 8 and state[6].numpy()[0] == 8
    assert np.isfinite(gradient.numpy()).all() and np.any(gradient.numpy() != 0)


def test_submillisecond_recurrent_delay_buffer_has_step_units():
    cell = make_cell(dt=0.25, delay=0.5)
    assert cell.max_delay == 8
    assert tuple(cell.recurrent_dense_shape) == (1, 8)
    assert cell.recurrent_indices.numpy()[0, 1] == 7


def test_refractory_quantization_uses_nest_ticks_then_ceiling():
    np.testing.assert_array_equal(
        time_steps([1.65, 3.25, 3.2500000000000004], 1.0), [2, 4, 4]
    )
    np.testing.assert_array_equal(
        time_steps([1.65, 3.25, 3.2500000000000004], 0.25), [7, 13, 13]
    )


def test_cell_rnn_chunk_continuation_and_soft_reset_training():
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    cell = make_cell(delay=3.0, hard_reset=False)
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    layer._autocast = False
    values = tf.constant([[[2.0], [0.0], [1.0], [0.0], [0.0], [3.0], [0.0], [0.0]]])
    initial = cell.zero_state(1, tf.float32)
    with tf.GradientTape() as tape:
        full = layer(values, initial_state=initial)
        loss = tf.reduce_sum(full[0][..., 1])
    gradient = tape.gradient(loss, cell.inputs["drive"]["input_weight_values"])
    assert (
        gradient is not None and np.isfinite(gradient).all() and np.any(gradient != 0)
    )
    first = layer(values[:, :4], initial_state=initial)
    second = layer(values[:, 4:], initial_state=first[1:])
    np.testing.assert_allclose(
        full[0], tf.concat([first[0], second[0]], axis=1), atol=1e-7
    )
    for expected, actual in zip(full[1:], second[1:]):
        np.testing.assert_allclose(expected, actual, atol=1e-7)


def test_fused_legacy_state_cannot_be_selected_for_nest_mode():
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell

    with pytest.raises(ValueError, match="legacy fused state"):
        GLIF3Cell({}, {}, dynamics_mode="nest", use_fused_state=True)


@pytest.mark.parametrize("dt", [1.0, 0.25])
@pytest.mark.parametrize("policy", ["float32", "mixed_float16"])
@pytest.mark.parametrize("hard_reset,batch_size", [(True, 1), (False, 5)])
@pytest.mark.parametrize("drive_scale", [0.05, 1.0])
def test_nest_cuda_matches_cpu_and_gpu_fallback(
    dt, policy, hard_reset, batch_size, drive_scale, record_property
):
    from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Fused CUDA operators unavailable")
    network = {
        "n_nodes": 3,
        "node_type_ids": np.array([0, 1, 0]),
        "node_params": {
            "C_m": np.array([100.0, 80.0]),
            "g": np.array([10.0, 8.0]),
            "E_L": np.array([-70.0, -65.0]),
            "V_m": np.array([-69.0, -64.0]),
            "V_th": np.array([-50.0, -45.0]),
            "V_reset": np.array([-68.0, -64.0]),
            "t_ref": np.array([2.0, 1.65]),
            "k": np.array([[0.01, 0.1], [0.02, 0.15]]),
            "asc_amps": np.array([[-10.0, -50.0], [-20.0, -40.0]]),
            "asc_init": np.array([[-1.0, -2.0], [-2.0, -1.0]]),
        },
        "synapses": {
            "indices": np.array([[0, 0], [0, 1], [1, 0], [2, 1], [2, 2]]),
            "weights": np.array([80.0, -100.0, 40.0, -60.0, 50.0]),
            "delays": np.array([1.0, 2.0, 3.0, 1.0, 2.0]),
            "dense_shape": (3, 3),
            "syn_ids": np.array([0, 1, 0, 1, 0]),
            "dynamics_params": {
                "basis_weights": [[1.0, 0.3, 0.1, 0.05], [0.1, 1.0, 0.2, 0.1]]
            },
        },
    }
    inputs = {
        "drive": {
            "n_inputs": 4,
            "indices": np.array(
                [[0, 0], [0, 1], [1, 1], [2, 0], [2, 2]], dtype=np.uint32
            ),
            "weights": np.array([1200.0, 800.0, 1000.0, 900.0, 700.0]),
            "delays": np.array([1.0, 3.0, 2.0, 1.0, 3.0]),
            "syn_ids": np.array([0, 1, 0, 1, 0]),
            "input_type": "spikes",
            "options": {"trainable": True},
        }
    }
    steps = int(40 / dt)
    inputs["drive"]["weights"] *= drive_scale
    values = np.zeros((batch_size, steps, 4), np.float32)
    values[:, (np.array([0.0, 5.0, 13.0, 25.0]) / dt).astype(int), :3] = [1.0, 2.0, 1.0]
    values *= np.arange(1, batch_size + 1, dtype=np.float32)[:, None, None]
    old_policy = tf.keras.mixed_precision.global_policy()
    results = {}
    tolerance = 2e-2 if policy == "mixed_float16" else 1e-5
    try:
        tf.keras.mixed_precision.set_global_policy(policy)
        for label, device, fused in (
            ("cpu", "/CPU:0", False),
            ("gpu_fallback", "/GPU:0", False),
            ("cuda", "/GPU:0", True),
        ):
            started = time.perf_counter()
            with tf.device(device):
                cell = GLIF3Cell(
                    copy.deepcopy(network),
                    copy.deepcopy(inputs),
                    dt=dt,
                    tau_basis=[2.0, 6.0, 10.0, 20.0],
                    dynamics_mode="nest",
                    hard_reset=hard_reset,
                    use_fused_cuda=fused,
                    use_fused_state=False,
                    train_recurrent_per_type=False,
                    batch_size=batch_size,
                )
                assert cell._use_fused_cuda is fused
                layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
                layer._autocast = False
                sequence = tf.constant(values, dtype=cell.compute_dtype)
                initial = list(cell.zero_state(batch_size, cell.compute_dtype))
                initial[6] = tf.fill((batch_size,), 4095)

                @tf.function(
                    input_signature=[
                        tf.TensorSpec((batch_size, None, 4), sequence.dtype),
                        [tf.TensorSpec(value.shape, value.dtype) for value in initial],
                    ]
                )
                def rollout(sequence, initial):
                    return layer(sequence, initial_state=initial)

                with tf.GradientTape() as tape:
                    full = rollout(sequence, initial)
                    loss = tf.reduce_mean(tf.cast(full[0][..., 3:], tf.float32))
                gradients = tape.gradient(
                    loss,
                    (
                        cell.recurrent_weight_values,
                        cell.inputs["drive"]["input_weight_values"],
                    ),
                )
                split = int(7 / dt)
                first = rollout(sequence[:, :split], initial)
                second = rollout(sequence[:, split:], list(first[1:]))
                np.testing.assert_allclose(
                    full[0],
                    tf.concat([first[0], second[0]], axis=1),
                    rtol=tolerance,
                    atol=tolerance,
                )
                for expected, actual in zip(full[1:], second[1:]):
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                for gradient in gradients:
                    assert gradient is not None
                    assert np.isfinite(gradient).all() and np.any(gradient != 0)
                results[label] = (
                    [value.numpy() for value in full],
                    [value.numpy() for value in gradients],
                )
                spike_count = np.count_nonzero(full[0][..., :3])
                record_property(f"{label}_spikes", int(spike_count))
                if drive_scale == 1.0:
                    assert spike_count > 0
                np.testing.assert_array_equal(
                    full[7], np.full(batch_size, 4095 + steps)
                )
            record_property(f"{label}_seconds", time.perf_counter() - started)
            cell.close_fused_cuda()
            del rollout, layer, cell
            gc.collect()
        expected_outputs, expected_gradients = results["cpu"]
        for label in ("gpu_fallback", "cuda"):
            outputs, gradients = results[label]
            voltage_delta = np.abs(
                outputs[0][..., 3:].astype(np.float64) - expected_outputs[0][..., 3:]
            )
            record_property(
                f"{label}_voltage_max_abs_mv", float(np.max(voltage_delta) * 20)
            )
            record_property(
                f"{label}_voltage_max_scaled_error",
                float(
                    np.max(
                        voltage_delta
                        / np.maximum(1, np.abs(expected_outputs[0][..., 3:]))
                    )
                ),
            )
            spike_differences = []
            for sample in range(batch_size):
                for neuron in range(3):
                    actual_steps = np.flatnonzero(outputs[0][sample, :, neuron])
                    expected_steps = np.flatnonzero(
                        expected_outputs[0][sample, :, neuron]
                    )
                    np.testing.assert_equal(actual_steps.size, expected_steps.size)
                    spike_differences.extend(np.abs(actual_steps - expected_steps))
            max_spike_error = max(spike_differences, default=0)
            assert max_spike_error <= 1
            spikes_match = max_spike_error == 0
            record_property(f"{label}_spike_times_exact", bool(spikes_match))
            record_property(
                f"{label}_max_spike_time_error_ms", float(max_spike_error * dt)
            )
            record_property(
                f"{label}_jittered_spikes", int(np.count_nonzero(spike_differences))
            )
            record_property(f"{label}_continuous_parity_checked", bool(spikes_match))
            for index, (actual, expected) in enumerate(zip(outputs, expected_outputs)):
                record_property(
                    f"{label}_state_{index}_max_abs",
                    float(np.max(np.abs(actual.astype(np.float64) - expected))),
                )
                if index in (7, 8) or (spikes_match and index in (1, 3)):
                    np.testing.assert_array_equal(actual, expected)
                elif spikes_match:
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
            for index, (actual, expected) in enumerate(
                zip(gradients, expected_gradients)
            ):
                record_property(
                    f"{label}_gradient_{index}_max_abs",
                    float(np.max(np.abs(actual - expected))),
                )
                if spikes_match:
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
    finally:
        tf.keras.mixed_precision.set_global_policy(old_policy)


def test_nest_internal_noise_matches_cuda_and_cpu():
    from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available

    if not fused_cuda_available():
        pytest.skip("Fused CUDA operators unavailable")
    results = []
    for device, fused in (("/CPU:0", False), ("/GPU:0", True)):
        with tf.device(device):
            cell = make_cell(delay=3.0, internal_noise=True, use_fused_cuda=fused)
            state = list(cell.zero_state(5, tf.float32))
            state[6] = tf.fill((5,), 4095)
            outputs = []
            histories = []
            for step in range(16):
                output, state = cell(tf.zeros((5, 0)), state)
                outputs.append(output.numpy())
                histories.append(state[-1].numpy())
            results.append((np.array(outputs), np.array(histories), state))
            cell.close_fused_cuda()
    expected, actual = results
    assert np.count_nonzero(expected[1]) > 0
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2][6], expected[2][6])
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-5)
    for expected_state, actual_state in zip(expected[2], actual[2]):
        np.testing.assert_allclose(actual_state, expected_state, rtol=1e-5, atol=1e-5)


def test_internal_noise_uses_original_population_size_and_delayed_history():
    cell = make_cell(delay=3.0, internal_noise=True)
    initial = cell.zero_state(2, tf.float32)
    expected = cell.sample_noise_spikes(2, initial[6], cell.inputs["drive"])
    assert expected.shape == (2, 1)
    _, state = cell(tf.zeros((2, 0)), initial)
    np.testing.assert_array_equal(state[7][:, :1], expected)
    np.testing.assert_array_equal(state[7][:, 1:], [[0.0], [0.0]])
    np.testing.assert_array_equal(state[6], [1, 1])


def test_cached_legacy_state_cannot_silently_drop_input_history(tmp_path):
    from types import SimpleNamespace
    from bmtk.simulator.dpointnet.state_modules.cached_states import CachedInitState

    cell = make_cell(delay=3.0)
    complete, names = cell.zero_state(1, tf.float32, with_names=True)
    filename = tmp_path / "state.npz"
    np.savez(filename, **dict(zip(names, [value.numpy() for value in complete])))
    rnn = SimpleNamespace(cell=cell, batch_size=1, dtype=tf.float32)
    cached = CachedInitState(str(filename), file_type="npz", rnn=rnn)
    state = cached.get_state()
    assert len(state) == 8
    cached._load_fn = lambda *_: complete[:7]
    with pytest.raises(ValueError, match="missing external delay history"):
        cached.get_state()


@pytest.mark.parametrize("return_sequences", [True, False])
def test_symbolic_rnn_preserves_scalar_noise_and_following_history(return_sequences):
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN

    cell = make_cell(delay=3.0)
    initial, names = cell.zero_state(1, tf.float32, with_names=True)
    values = tf.keras.Input(shape=(None, 1))
    state_inputs = [
        tf.keras.Input(shape=state.shape[1:], dtype=state.dtype, name=name)
        for state, name in zip(initial, names)
    ]
    layer = ExplicitStateRNN(cell, return_sequences=return_sequences, return_state=True)
    layer._autocast = False
    outputs = layer(values, initial_state=state_inputs)
    model = tf.keras.Model([values, *state_inputs], outputs)
    assert len(model.outputs) == 9
    first = model([tf.ones((1, 4, 1)), *initial])
    assert first[7].dtype == tf.int32 and first[8].shape == (1, 2)
    np.testing.assert_array_equal(first[7], [4])
    np.testing.assert_array_equal(first[8], [[1.0, 1.0]])
    late_initial = list(initial)
    late_initial[6] = tf.constant([4095], tf.int32)
    late = model([tf.ones((1, 4, 1)), *late_initial])
    np.testing.assert_array_equal(late[7], [4099])
    second = model([tf.zeros((1, 4, 1)), *first[1:]])
    full = model(
        [tf.concat([tf.ones((1, 4, 1)), tf.zeros((1, 4, 1))], axis=1), *initial]
    )
    for expected, actual in zip(full[1:], second[1:]):
        np.testing.assert_allclose(expected, actual, atol=1e-7)


@pytest.mark.parametrize("dt", [1.0, 0.25])
@pytest.mark.parametrize("tau", [2.0, 10.0, 10.000001, 30.0])
def test_exact_synaptic_integral_matches_matrix_exponential(dt, tau):
    decay, current_factor, current, rise = integration_coefficients(
        dt, [100.0], [10.0], [tau]
    )
    generator = np.array([[-0.1, 0.01, 0], [0, -1 / tau, 1], [0, 0, -1 / tau]])
    expected = expm(dt * generator)
    np.testing.assert_allclose(
        [decay[0], current[0, 0], rise[0, 0]], expected[0], rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(current_factor, [(1 - np.exp(-dt / 10)) / 10])


def test_nest_refractory_holds_adaptation_and_blocks_last_refractory_step():
    voltage = tf.constant([[0.7]])
    adaptation = tf.constant([[[-0.1, -0.2]]])
    result = active_update(
        voltage,
        tf.constant([[1]]),
        adaptation,
        tf.zeros((1, 1, 1)),
        tf.zeros((1, 1, 1)),
        decay=0.9,
        current_factor=0.1,
        asc_decay=0.8,
        asc_mean=0.9,
        psc_voltage=1.0,
        rise_voltage=1.0,
        reset_voltage=0.0,
        hard_reset=True,
    )
    assert result[0].numpy()[0, 0] == 0
    assert result[1].numpy()[0, 0] == 0
    assert not result[3].numpy()[0, 0]
    np.testing.assert_array_equal(result[2], adaptation)


def test_reset_applies_refractory_decay_only_to_old_adaptation():
    voltage, remaining, adaptation = spike_reset(
        tf.constant([[1.2]]),
        tf.constant([[0]]),
        tf.constant([[[-2.0, -4.0]]]),
        tf.constant([[1.0]]),
        reset_voltage=0.0,
        refractory_steps=3,
        asc_amplitudes=tf.constant([-1.0, -2.0]),
        asc_refractory_decay=0.5,
        hard_reset=True,
    )
    np.testing.assert_allclose(adaptation, [[[-2.0, -4.0]]])
    assert voltage.numpy()[0, 0] == 0
    assert remaining.numpy()[0, 0] == 3
