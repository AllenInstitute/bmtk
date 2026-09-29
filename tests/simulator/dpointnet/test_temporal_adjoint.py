import numpy as np
import pytest
import copy
import json
from types import SimpleNamespace

tf = pytest.importorskip("tensorflow")

from test_precision_credit import make_cell
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner
from bmtk.simulator.dpointnet.cell_models.glif3_cell import GLIF3Cell, quantized_fp32
from bmtk.simulator.dpointnet.custom_ops.glif_state_ops import (
    fused_glif_state_available,
)
from bmtk.simulator.dpointnet.optimizers import create_optimizer


@pytest.fixture(autouse=True)
def restore_policy():
    policy = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(policy)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("chunk", [1, 3, 9])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_fp32_runner_values_and_gradients(mode, chunk, graph, replay_mode):
    cell = make_cell(
        mode, temporal_gradient_precision="float32", return_voltage_sequences=True,
        current_replay_mode=replay_mode,
    )
    runner = TemporalAdjointRunner(cell, chunk_size=chunk)
    state = cell.zero_state(2, cell.compute_dtype)
    x = tf.ones((2, 9, 2), tf.float32) * 0.7
    with pytest.raises(ValueError, match="TemporalAdjointRunner"):
        cell.call(x[:, 0], state)

    def evaluate():
        with tf.GradientTape() as tape:
            tape.watch(x)
            output, final = runner(x, state)
            loss = tf.reduce_sum(tf.square(output[:, -1, 2:] - 0.75))
        dx, dw = tape.gradient(loss, [x, cell.inputs["drive"]["input_weight_values"]])
        diagnostic = runner.differentiate(
            x,
            state,
            lambda output, final: tf.reduce_sum(tf.square(output[:, -1, 2:] - 0.75)),
            probe_steps=(0, 3, 8),
        )
        return output, final, dx, dw, diagnostic

    output, final, dx, dw, diagnostic = (tf.function(evaluate) if graph else evaluate)()
    assert dx.dtype == tf.float32 and dw.dtype == tf.float32
    assert np.any(dx.numpy() != 0)
    np.testing.assert_allclose(dx, diagnostic["input_gradients"], rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        dw, diagnostic["variable_gradients"][-1], rtol=1e-6, atol=1e-7
    )
    if graph:
        expected_outputs, expected_state = tf.function(lambda: runner._loop(x, state))()
        expected = expected_outputs[0][:, -1]
    else:
        expected_state = state
        for index in range(9):
            expected, expected_state = cell._call_impl(x[:, index], expected_state)
    np.testing.assert_array_equal(output[:, -1], expected)
    for actual, expected in zip(final, expected_state):
        assert actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)
    assert all(value.dtype == tf.float32 for value in diagnostic["state_cotangents"])


def long_credit_cell(mode, fused=False, direct=False, replay_mode="record", **options):
    """The independently reported 600-ms plateau fixture, with trainable masters."""
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    network = {
        "n_nodes": 2,
        "node_type_ids": np.array([0, 1]),
        "node_params": {
            "C_m": np.array([100.0, 120.0]),
            "g": np.array([10.0, 10.0]),
            "E_L": np.array([-70.0, -70.0]),
            "V_m": np.array([-70.0, -70.0]),
            "V_th": np.array([-50.0, -50.0]),
            "V_reset": np.array([-70.0, -70.0]),
            "t_ref": np.array([2.0, 2.0]),
            "k": np.array([[0.002, 0.02], [0.004, 0.04]]),
            "asc_amps": np.array([[-2.0, -1.0], [-1.0, -0.5]]),
        },
        "synapses": {
            "indices": np.array([[0, 1], [1, 0]], np.uint32),
            "weights": np.array([-2.0, 2.0]),
            "delays": np.array([2.0, 2.0]),
            "dense_shape": (2, 2),
            "syn_ids": np.array([0, 0]),
            "dynamics_params": {"basis_weights": [[1.0, 0.3, 0.1, 0.05]]},
        },
    }
    inputs = {
        "probe": {
            "n_inputs": 2,
            "indices": np.array([[0, 0], [1, 1]]),
            "weights": np.array([300.0, 300.0]),
            "delays": np.array([1.0, 1.0]),
            "syn_ids": np.array([0, 0]),
            "input_type": "current",
            "options": {"trainable": True},
        }
    }
    return GLIF3Cell(
        network,
        inputs,
        tau_basis=[2.0, 4.0, 8.0, 16.0],
        hard_reset=False,
        dynamics_mode=mode,
        state_precision="selective",
        temporal_gradient_precision="float32",
        current_replay_mode=replay_mode,
        detach_reset=True,
        detach_asc_reset=False,
        pseudo_gauss=True,
        dampening_factor=0.5,
        gauss_std=0.28,
        recurrent_dampening_factor=1.0,
        voltage_gradient_dampening=0.0,
        train_recurrent=True,
        train_recurrent_per_type=False,
        use_fused_cuda=fused,
        use_fused_state=fused,
        use_direct_csr_recurrent_gradient=direct,
        use_packed_sm120_backward=False,
        use_packed_sm120_external_backward=False,
        return_voltage_sequences=True,
        batch_size=1,
        **options,
    )


def primal_trajectory(cell, inputs, initial):
    arrays = tuple(
        tf.TensorArray(x.dtype, size=inputs.shape[1] + 1).write(0, x) for x in initial
    )

    def step(index, state, arrays):
        _, state = cell._call_impl(inputs[:, index], state)
        return (
            index + 1,
            state,
            tuple(a.write(index + 1, x) for a, x in zip(arrays, state)),
        )

    _, _, arrays = tf.while_loop(
        lambda index, *_: index < inputs.shape[1],
        step,
        (0, initial, arrays),
        parallel_iterations=1,
    )
    return tuple(array.stack() for array in arrays)


def independent_adjoint(cell, trajectory, inputs, end=600, voltage_floor=None):
    """Float64 analytic VJP along the actual quantized forward trajectory."""
    old = [np.asarray(value)[:, 0].astype(np.float64) for value in trajectory]
    value = lambda name: np.asarray(getattr(cell, name)).astype(np.float64)
    n = cell._n_neurons
    decay, factor = value("decay"), value("current_factor")
    syn, initial = value("syn_decay").reshape(n, 4), value("psc_initial").reshape(n, 4)
    beta, amps = value("asc_decay").reshape(n, 2), value("asc_amps").reshape(n, 2)
    basis = value("synaptic_basis_weights")
    recurrent = np.asarray(cell.recurrent_indices)
    w_rec = value("recurrent_weight_values_compute")
    rec_types = np.asarray(cell.syn_ids)
    net = cell.inputs["probe"]
    incoming = np.asarray(net["input_indices"])
    w_input = np.asarray(net["input_weight_values_compute"]).astype(np.float64)
    input_types = np.asarray(net["input_syn_ids"])
    history = np.zeros(old[0].shape[1], np.float64)
    v = np.zeros(n, np.float64)
    a = np.zeros((n, 2), np.float64)
    q = np.zeros((n, 4), np.float64)
    p = np.zeros((n, 4), np.float64)
    dx = np.zeros((inputs.shape[1], n), np.float64)
    dw_rec, dw_input = np.zeros_like(w_rec), np.zeros_like(w_input)
    saved = {}
    for step in range(end - 1, -1, -1):
        if step == end - 1 and voltage_floor is None:
            v += old[1][step + 1] - 0.75
        fired = old[0][step + 1, :n]
        if voltage_floor is not None:
            before = old[1][step + 1]
            active_floor = old[2][step + 1] <= 0
            if cell.dynamics_mode == "nest":
                before = before + fired * (1 - value("v_reset"))
                active_floor = old[2][step] <= 0
            v += (
                -2 * float(voltage_floor.cost)
                * np.maximum(float(voltage_floor.target) - before, 0)
                * np.asarray(voltage_floor.gate) * active_floor / (end * n)
            )
        event = history[:n].copy()
        new_history = np.concatenate([history[n:], np.zeros(n)])
        if cell.dynamics_mode == "legacy":
            u = (old[1][step + 1].astype(np.float32) - np.float32(1)).astype(np.float64)
            active = old[2][step + 1] <= 0
            dv = (
                v
                + event
                * float(cell._dampening_factor)
                * np.exp(-u * u / float(cell._gauss_std) ** 2)
                * active
            )
            new_history[:n] += np.sum(a * amps, axis=-1)
            next_a = a * beta + dv[:, None] * factor[:, None]
            next_p = p * syn + dv[:, None] * factor[:, None]
            next_q = q * syn + p * float(cell._dt) * syn
        else:
            active = old[2][step] <= 0
            rho, mean = value("asc_refractory_decay"), value("asc_mean")
            a_minus = old[3][step].reshape(n, 2) * np.where(active[:, None], beta, 1)
            sensitivity = amps + (rho - 1) * a_minus
            event += np.sum(a * sensitivity, axis=-1)
            before = (old[1][step + 1] + fired * (1 - value("v_reset"))).astype(
                np.float32
            )
            u = (before - np.float32(1)).astype(np.float64)
            dv = (
                v
                + event
                * float(cell._dampening_factor)
                * np.exp(-u * u / float(cell._gauss_std) ** 2)
                * active
            )
            next_a = (
                a
                * np.where(fired[:, None] > 0, rho, 1)
                * np.where(active[:, None], beta, 1)
            )
            next_a += dv[:, None] * factor[:, None] * mean
            next_p = p * syn + dv[:, None] * value("psc_voltage")
            next_q = (
                q * syn
                + p * float(cell._dt) * syn
                + dv[:, None] * value("rise_voltage")
            )
        current = q * initial * float(cell._lr_scale)
        for edge, ((post, pre), weight, kind) in enumerate(
            zip(recurrent, w_rec, rec_types)
        ):
            sensitivity = current[post] @ basis[kind]
            new_history[pre] += sensitivity * weight * float(cell._recurrent_dampening)
            dw_rec[edge] += sensitivity * old[0][step, pre]
        quantized_input = (
            np.asarray(inputs)[0, step].astype(np.float16).astype(np.float64)
        )
        for edge, ((post, pre), weight, kind) in enumerate(
            zip(incoming, w_input, input_types)
        ):
            sensitivity = current[post] @ basis[kind]
            dx[step, pre] += sensitivity * weight
            dw_input[edge] += sensitivity * quantized_input[pre]
        history, v, a, q, p = new_history, dv * decay, next_a, next_q, next_p
        if step in (100, 350, 500):
            saved[step] = (
                history.copy(),
                v.copy(),
                a.reshape(-1).copy(),
                q.reshape(-1).copy(),
                p.reshape(-1).copy(),
            )
    return dx, saved, (dw_rec, dw_input)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("chunk", [17, 25, 620])
@pytest.mark.parametrize("execution", ["tf", "cuda", "cuda_csr"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_600ms_quantized_forward_fp32_adjoint_oracle(mode, chunk, execution, replay_mode):
    if execution != "tf" and not fused_glif_state_available():
        pytest.skip("CUDA state operators unavailable")
    cell = long_credit_cell(
        mode, fused=execution != "tf", direct=execution == "cuda_csr", replay_mode=replay_mode
    )
    runner = TemporalAdjointRunner(cell, chunk_size=chunk, pack_spike_checkpoints=True)
    state = cell.zero_state(1, cell.compute_dtype)
    values = (
        np.random.default_rng(813).uniform(0.65, 0.85, (620, 1, 2)).astype(np.float32)
    )
    x = tf.constant(values.transpose(1, 0, 2))
    trajectory = tf.function(lambda: primal_trajectory(cell, x, state))()
    reference, states, weights = independent_adjoint(cell, trajectory, x)
    result = tf.function(
        lambda: runner.differentiate(
            x,
            state,
            lambda output, final: tf.reduce_mean(tf.square(output[:, 599, 2:] - 0.75)),
            probe_steps=(100, 350, 500),
        )
    )()
    np.testing.assert_allclose(
        result["input_gradients"][0], reference, rtol=1e-4, atol=1e-15
    )
    np.testing.assert_array_equal(result["input_gradients"][:, 600:], 0)
    for j, step in enumerate((100, 350, 500)):
        for actual, expected in zip(result["state_cotangents"], states[step]):
            np.testing.assert_allclose(actual[j, 0], expected, rtol=1e-4, atol=1e-25)
        lag = 600 - step
        print(
            mode,
            "lag",
            lag,
            "input_norm",
            np.linalg.norm(result["input_gradients"][0, step]),
            "reference",
            np.linalg.norm(reference[step]),
            "PSC_norm",
            np.linalg.norm(result["state_cotangents"][4][j, 0]),
            "reference",
            np.linalg.norm(states[step][4]),
        )
        metrics = {
            "mode": mode,
            "chunk_size": chunk,
            "execution": execution,
            "lag_ms": lag,
        }
        for label, actual, expected in (
            ("input", result["input_gradients"][0, step], reference[step]),
            ("psc", result["state_cotangents"][4][j, 0], states[step][4]),
        ):
            actual = np.asarray(actual, dtype=np.float64)
            norm = float(np.linalg.norm(expected))
            error = float(np.linalg.norm(actual - expected))
            metrics[label] = {
                "norm": float(np.linalg.norm(actual)),
                "reference_norm": norm,
                "absolute_error": error,
                "relative_error": error / norm,
            }
        print("ADJOINT_METRICS " + json.dumps(metrics, sort_keys=True))
    for actual, expected in zip(result["variable_gradients"], weights):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-9)
    masters = [
        cell.recurrent_weight_values,
        cell.inputs["probe"]["input_weight_values"],
    ]
    reference_weights = [
        tf.Variable(np.asarray(value), constraint=value.constraint) for value in masters
    ]
    optimizer = create_optimizer("exp_adam", 0.001, {"global_clipnorm": 1.0})
    reference_optimizer = create_optimizer("exp_adam", 0.001, {"global_clipnorm": 1.0})
    optimizer.apply_gradients(zip(result["variable_gradients"], masters))
    reference_optimizer.apply_gradients(
        zip([tf.constant(value, tf.float32) for value in weights], reference_weights)
    )
    cell.refresh_recurrent_weight_shadow()
    for actual, expected in zip(masters, reference_weights):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(
        cell.recurrent_weight_values_compute, tf.cast(masters[0], tf.float16)
    )
    np.testing.assert_array_equal(
        cell.inputs["probe"]["input_weight_values_compute"],
        tf.cast(masters[1], tf.float16),
    )
    cell.close_fused_cuda()


class DecayCell:
    temporal_gradient_precision = "float32"
    _temporal_continuous_inputs = False
    _use_direct_csr_recurrent_gradient = False
    compute_dtype = "float16"
    output_size = 4
    inputs = {}

    def __init__(self):
        self.decay = tf.constant(
            np.exp(-1 / np.array([16.0, 32.0, 64.0, 128.0])), tf.float32
        )
        self.recurrent_weight_values = tf.Variable([0.0], trainable=False)

    def validate_state_precision(self, states):
        assert states[0].dtype == tf.float16

    def _call_impl(self, inputs, state, adjoint_replay=False):
        value = tf.cast(state[0], tf.float32) * self.decay
        value = quantized_fp32(value) if adjoint_replay else tf.cast(value, tf.float16)
        return tf.cast(value, tf.float32), (value,)


@pytest.mark.parametrize("chunk", [1, 25, 73, 600])
def test_analytic_decay_carrier_does_not_stall_with_quantized_primals(chunk):
    cell = DecayCell()
    state = (tf.ones((1, 4), tf.float16),)
    inputs = tf.zeros((1, 600, 0), tf.float32)
    runner = TemporalAdjointRunner(cell, chunk_size=chunk)
    result = tf.function(
        lambda: runner.differentiate(
            inputs,
            state,
            lambda output, final: tf.reduce_sum(output[:, -1]),
            probe_steps=(100, 350, 500),
        )
    )()
    for position, lag in enumerate((500, 250, 100)):
        expected = cell.decay.numpy().astype(np.float64) ** lag
        actual = result["state_cotangents"][0][position, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=0)
    assert result["final_state"][0].dtype == tf.float16
    assert result["state_cotangents"][0].dtype == tf.float32
    # Its forward state is intentionally quantized; removing that floor would
    # silently change the physical trajectory rather than repair the VJP.
    assert result["final_state"][0][0, 0] == tf.cast(8 * 2**-24, tf.float16)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("state_input", [False, True])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_rnn_factory_temporal_route_state_and_checkpoint_plumbing(
    mode, state_input, compact, replay_mode
):
    from bmtk.simulator.dpointnet.rnn_model import RNN

    network, inputs, options = make_cell(
        mode,
        return_spec=True,
        temporal_gradient_precision="float32",
        temporal_checkpoint_chunk_size=3,
        return_voltage_sequences=not compact,
        current_replay_mode=replay_mode,
    )
    options.pop("train_recurrent_per_type")
    rnn = RNN(seq_len=9, batch_size=2, dtype="float16", cell_params=options)
    rnn._recurrent_networks["test"] = SimpleNamespace(
        to_dict=lambda: copy.deepcopy(network)
    )
    rnn._input_networks["drive"] = SimpleNamespace(
        name="drive", n_spiking_nodes=2, to_dict=lambda: copy.deepcopy(inputs["drive"])
    )
    engine = rnn.set_training(
        rnn=rnn,
        n_epochs=1,
        steps_per_epoch=1,
        gradient_checkpointing=True,
        gradient_checkpoint_chunk_size=3,
    )
    engine.add_parameters("test", batch_size=2, seq_len=9)
    try:
        rnn.build(training=True, use_state_input=state_input)
        assert rnn._cell.current_replay_mode == replay_mode
        initial = list(rnn.zero_state)
        if state_input:
            initial[1] = tf.ones_like(initial[1]) * 0.25
            initial[6] = tf.fill((2,), 131)
        x = tf.ones((2, 9, 2), tf.float32) * 0.7
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = rnn.run_extractor(x, initial)
            loss = tf.reduce_sum(tf.cast(out[0][0], tf.float32)) + tf.reduce_sum(
                out[0][1]
            )
        grad = tape.gradient(loss, x)
        assert grad.dtype == tf.float32 and np.any(grad != 0)
        state_only = rnn.state_only_model(rnn.model_inputs(x, initial))
        for actual, expected in zip(state_only, out[1:]):
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)
        engine.prepare_gradient_checkpointing()
        assert engine._extractor_forward is None
        with pytest.raises(ValueError, match="float32 model inputs"):
            rnn.model_inputs(tf.cast(x, tf.float16), initial)
    finally:
        rnn.cleanup()


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("direct", [False, True])
def test_temporal_poisson_replay_and_canonical_updates(mode, direct):
    if not fused_glif_state_available():
        pytest.skip("CUDA state operators unavailable")
    cell = make_cell(
        mode,
        temporal_gradient_precision="float32",
        fused=True,
        noise=True,
        two_inputs=True,
        batch_size=32,
        use_fused_cuda=True,
        use_pair_projection=True,
        use_fused_current_accumulation=True,
        use_direct_csr_recurrent_gradient=direct,
        use_packed_sm120_backward=False,
        use_packed_sm120_external_backward=False,
    )
    initial = list(cell.zero_state(32, cell.compute_dtype))
    initial[6] = tf.fill((32,), 4095)
    x = tf.zeros((32, 31, 0), tf.float32)
    results = []
    for chunk in (1, 7, 31):
        runner = TemporalAdjointRunner(
            cell, chunk_size=chunk, pack_spike_checkpoints=True
        )
        results.append(
            tf.function(
                lambda: runner.differentiate(
                    x,
                    initial,
                    lambda output, state: tf.reduce_mean(output[0])
                    + tf.reduce_mean(output[1]),
                )
            )()
        )
    for result in results[1:]:
        for actual, expected in zip(
            tf.nest.flatten(result["final_state"]),
            tf.nest.flatten(results[0]["final_state"]),
        ):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(
            result["variable_gradients"], results[0]["variable_gradients"]
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(results[0]["final_state"][6], np.full(32, 4095 + 31))
    assert int(cell.noise_stream) == 0
    masters = [cell.recurrent_weight_values] + [
        net["input_weight_values"] for net in cell.inputs.values()
    ]
    optimizer = create_optimizer("sgd", 0.01, {"global_clipnorm": 1.0})
    optimizer.apply_gradients(zip(results[0]["variable_gradients"], masters))
    cell.refresh_recurrent_weight_shadow()
    np.testing.assert_array_equal(
        cell.recurrent_weight_values_compute, tf.cast(masters[0], tf.float16)
    )
    rec_order = np.argsort(cell.recurrent_indices.numpy()[:, 1], kind="stable")
    np.testing.assert_array_equal(
        cell.recurrent_csr_weight_values_compute,
        tf.cast(tf.gather(masters[0], rec_order), tf.float16),
    )
    for net in cell.inputs.values():
        np.testing.assert_array_equal(
            net["input_weight_values_compute"],
            tf.cast(net["input_weight_values"], tf.float16),
        )
        order = np.argsort(net["input_indices"].numpy()[:, 1], kind="stable")
        np.testing.assert_array_equal(
            net["csr_weight_values_compute"],
            tf.cast(tf.gather(net["input_weight_values"], order), tf.float16),
        )
    cell.close_fused_cuda()


@pytest.mark.parametrize(
    "options",
    [
        {"temporal_gradient_precision": "half"},
        {"temporal_gradient_precision": "float32", "state_precision": "compute"},
        {"temporal_gradient_precision": "float32", "use_packed_sm120_backward": True},
        {"temporal_checkpoint_chunk_size": 0},
        {"temporal_pack_spike_checkpoints": "false"},
    ],
)
def test_temporal_options_fail_explicitly(options):
    with pytest.raises(ValueError):
        make_cell(**options)


def test_temporal_input_boundary_and_memory_layout():
    cell = make_cell(temporal_gradient_precision="float32")
    runner = TemporalAdjointRunner(cell)
    state = cell.zero_state(2, cell.compute_dtype)
    with pytest.raises(ValueError, match="float32 inputs"):
        runner(tf.zeros((2, 3, 2), tf.float16), state)
    primal = sum(np.prod(state[i].shape) * state[i].dtype.size for i in (1, 3, 4, 5))
    replay = sum(np.prod(state[i].shape) * 4 for i in (1, 3, 4, 5))
    assert primal // 4 == 28 and replay // 4 == 44


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_temporal_causality_and_outer_wrapper_rejection(mode):
    from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
    from bmtk.simulator.dpointnet.segmented_recompute import (
        SegmentedRecomputeRunner,
        FullBPTTGradientRunner,
    )

    cell = make_cell(
        mode, temporal_gradient_precision="float32", return_voltage_sequences=True
    )
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    state = cell.zero_state(2, cell.compute_dtype)
    x = tf.ones((2, 11, 2), tf.float32) * 0.7
    changed = tf.concat([x[:, :7], x[:, 7:] * 2], axis=1)
    output, _ = runner(x, state)
    perturbed, _ = runner(changed, state)
    np.testing.assert_array_equal(output[:, :7], perturbed[:, :7])
    with tf.GradientTape() as tape:
        tape.watch(x)
        sequence, _ = runner(x, state)
        loss = tf.reduce_sum(sequence[:, :7, 2:])
    grad = tape.gradient(loss, x)
    np.testing.assert_array_equal(grad[:, 7:], 0)
    symbolic = tf.keras.Input(shape=(11, 2), dtype=tf.float32)
    initial = [
        tf.keras.Input(shape=value.shape[1:], dtype=value.dtype) for value in state
    ]
    layer = ExplicitStateRNN(cell, return_sequences=True, return_state=True)
    outputs = layer(symbolic, initial_state=initial)
    model = tf.keras.Model([symbolic, *initial], list(outputs))
    with pytest.raises(ValueError, match="own their reverse"):
        SegmentedRecomputeRunner(model, 11, 3, 1)
    with pytest.raises(ValueError, match="own their reverse"):
        FullBPTTGradientRunner(model, lambda variables, gradients: gradients)
    with pytest.raises(ValueError, match="non-unrolled"):
        ExplicitStateRNN(cell, unroll=True)


def test_half_initial_state_boundary_is_distinct_from_internal_cotangent():
    cell = DecayCell()
    runner = TemporalAdjointRunner(cell, chunk_size=25)
    state = (tf.ones((1, 4), tf.float16),)
    x = tf.zeros((1, 600, 0), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(state)
        output, _ = runner(x, state)
        loss = tf.reduce_sum(output[:, -1])
    external = tape.gradient(loss, state)[0]
    internal = runner.differentiate(
        x, state, lambda output, final: tf.reduce_sum(output[:, -1])
    )["initial_state_gradients"][0]
    assert external.dtype == tf.float16 and external[0, 0] == 0
    assert internal.dtype == tf.float32 and internal[0, 0] > 0


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_temporal_replay_snapshots_each_rollout_noise_stream(mode, fused, replay_mode):
    if fused and not fused_glif_state_available():
        pytest.skip("CUDA state operators unavailable")

    def evaluate(joint):
        cell = make_cell(
            mode,
            temporal_gradient_precision="float32",
            noise=True,
            fused=fused,
            use_fused_cuda=fused,
            current_replay_mode=replay_mode,
        )
        runner = TemporalAdjointRunner(cell, chunk_size=3)
        state = cell.zero_state(2, cell.compute_dtype)
        x = tf.zeros((2, 13, 0), tf.float32)
        masters = [
            cell.recurrent_weight_values,
            cell.inputs["drive"]["input_weight_values"],
        ]

        def loss():
            outputs, _ = runner(x, state)
            return tf.reduce_mean(tf.cast(outputs[0], tf.float32)) + tf.reduce_mean(
                outputs[1]
            )

        @tf.function
        def run():
            if joint:
                with tf.GradientTape() as tape:
                    first = loss()
                    cell.advance_noise_seed()
                    second = loss()
                    total = first + second
                return tape.gradient(total, masters)
            with tf.GradientTape() as tape:
                first = loss()
            first_gradients = tape.gradient(first, masters)
            cell.advance_noise_seed()
            with tf.GradientTape() as tape:
                second = loss()
            second_gradients = tape.gradient(second, masters)
            return [a + b for a, b in zip(first_gradients, second_gradients)]

        gradients = run()
        assert int(cell.noise_stream) == 1
        cell.close_fused_cuda()
        return gradients

    for actual, expected in zip(evaluate(True), evaluate(False)):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
