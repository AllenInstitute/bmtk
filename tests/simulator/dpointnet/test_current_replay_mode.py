"""Replay policy is independent of primal and temporal precision."""

import json
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from test_precision_credit import make_cell
from test_temporal_current_tape import variables
from bmtk.simulator.dpointnet import temporal_adjoint
from bmtk.simulator.dpointnet.temporal_adjoint import TemporalAdjointRunner


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def cache_replay_error_metrics(runner, inputs, initial_state, cache):
    """Copyable GPU-harness diagnostic; keep outside the native timed update."""
    output_count = len(cache[0])
    width = output_count + len(initial_state)

    def compare(actual, expected):
        a, b = tf.cast(actual, tf.float64), tf.cast(expected, tf.float64)
        tf.debugging.assert_all_finite(a, "Nonfinite replay values")
        tf.debugging.assert_all_finite(b, "Nonfinite original values")
        return (
            tf.reduce_sum(tf.cast(tf.not_equal(a, b), tf.int64)),
            tf.reduce_max(tf.abs(a - b)),
        )

    def chunk(index, counts, errors, spike_count):
        out, state, original_out, original_state = runner._replay_cached_chunk(
            inputs, initial_state, cache, index
        )
        metrics = [
            compare(a, b)
            for a, b in zip(out + state, original_out + original_state)
        ]
        spikes, original_spikes = out[0], original_out[0]
        if not isinstance(runner.cell.output_size, tuple):
            spikes = spikes[..., :runner.cell._n_neurons]
            original_spikes = original_spikes[..., :runner.cell._n_neurons]
        return (
            index + 1,
            counts + tf.stack([m[0] for m in metrics]),
            tf.maximum(errors, tf.stack([m[1] for m in metrics])),
            spike_count + compare(spikes, original_spikes)[0],
        )

    count, mismatches, errors, spike_count = tf.while_loop(
        lambda index, *_: index < tf.size(cache[2]) - 1,
        chunk,
        (tf.constant(0), tf.zeros([width], tf.int64),
         tf.zeros([width], tf.float64), tf.constant(0, tf.int64)),
        parallel_iterations=1,
    )
    return {
        "chunks": count,
        "spike_mismatch_count": spike_count,
        "output_mismatch_count": mismatches[:output_count],
        "output_max_abs_error": errors[:output_count],
        "state_boundary_mismatch_count": mismatches[output_count:],
        "state_boundary_max_abs_error": errors[output_count:],
    }


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("compact", [False, True])
def test_cached_chunk_diagnostic_contract(mode, replay_mode, packed, graph, compact):
    cell = make_cell(
        mode, noise=True, temporal_gradient_precision="float32",
        current_replay_mode=replay_mode,
        return_voltage_sequences=not compact,
    )
    runner = TemporalAdjointRunner(cell, chunk_size=3, pack_spike_checkpoints=packed)
    initial = cell.zero_state(2, cell.compute_dtype)
    inputs = tf.zeros((2, 7, 0), tf.float32)

    def run():
        cache = runner._forward(inputs, initial, probe_steps=(1, 5))
        assert len(cache) == 7
        assert (cache[5] is not None) == (replay_mode == "record")
        cell.advance_noise_seed()
        for v in cell.trainable_variables:
            v.assign(v * 1.7)
        cell.refresh_recurrent_weight_shadow()
        return cache_replay_error_metrics(runner, inputs, initial, cache)

    metrics = (tf.function(run) if graph else run)()
    assert int(metrics["chunks"]) == 5
    assert metrics["state_boundary_max_abs_error"].shape == (len(initial),)
    for name, value in metrics.items():
        if name != "chunks":
            np.testing.assert_array_equal(value, 0)


def test_cached_chunk_diagnostic_rejects_manual_tape_toggle():
    cell = make_cell(temporal_gradient_precision="float32")
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    initial = cell.zero_state(2, cell.compute_dtype)
    inputs = tf.ones((2, 3, 2), tf.float32) * 0.7
    cache = runner._forward(inputs, initial)
    runner.record_currents = False
    with pytest.raises(ValueError, match="do not toggle record_currents"):
        runner._replay_cached_chunk(inputs, initial, cache, 0)


def test_cached_chunk_metric_counts_and_max_errors(monkeypatch):
    cell = make_cell(temporal_gradient_precision="float32")
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    initial = cell.zero_state(2, cell.compute_dtype)
    inputs = tf.ones((2, 7, 2), tf.float32) * 0.7
    cache = runner._forward(inputs, initial)
    replay_chunk = runner._replay_cached_chunk

    def perturbed(*args):
        out, state, original_out, original_state = replay_chunk(*args)
        out = (tf.cast(original_out[0], tf.float64) + [1.0, 0.0],) + out[1:]
        state = (
            state[0], tf.cast(original_state[1], tf.float64) + 0.25,
        ) + state[2:]
        return out, state, original_out, original_state

    monkeypatch.setattr(runner, "_replay_cached_chunk", perturbed)
    metrics = cache_replay_error_metrics(runner, inputs, initial, cache)
    assert int(metrics["spike_mismatch_count"]) == 14
    np.testing.assert_array_equal(metrics["output_mismatch_count"], [14, 0])
    np.testing.assert_array_equal(metrics["output_max_abs_error"], [1.0, 0.0])
    np.testing.assert_array_equal(
        metrics["state_boundary_mismatch_count"], [0, 12, 0, 0, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        metrics["state_boundary_max_abs_error"], [0, 0.25, 0, 0, 0, 0, 0]
    )


@pytest.mark.parametrize("value", [True, False, "auto", "off", "fp16", 1])
def test_unknown_current_replay_modes_rejected(value):
    with pytest.raises(ValueError, match="current_replay_mode"):
        make_cell(temporal_gradient_precision="float32", current_replay_mode=value)


def test_default_and_incompatible_policy():
    for temporal in ("compute", "float32"):
        cell = make_cell(temporal_gradient_precision=temporal)
        assert cell.current_replay_mode == ("record" if temporal == "float32" else None)
        assert cell.current_replay_mode_requested is None
        if temporal == "float32":
            assert TemporalAdjointRunner(cell).record_currents
    with pytest.warns(RuntimeWarning, match="not bitwise deterministic"):
        cell = make_cell(
            temporal_gradient_precision="float32", current_replay_mode="recompute"
        )
    assert cell.state_precision == "selective"
    assert cell.temporal_gradient_precision == "float32"
    cell.current_replay_mode = "invalid"
    with pytest.raises(ValueError, match="current_replay_mode"):
        TemporalAdjointRunner(cell)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("selective", [False, True])
def test_ordinary_compute_default_is_inactive_and_preserves_values(mode, selective):
    cells = [
        make_cell(mode, selective=selective, **options)
        for options in ({}, {"current_replay_mode": None})
    ]
    outputs = []
    for cell in cells:
        assert cell.current_replay_mode is None
        assert cell.current_replay_mode_requested is None
        assert cell.temporal_gradient_precision == "compute"
        state = cell.zero_state(2, cell.compute_dtype)
        outputs.append(cell.call(tf.ones((2, 2), cell.compute_dtype) * 0.7, state))
    for a, b in zip(tf.nest.flatten(outputs[0]), tf.nest.flatten(outputs[1])):
        np.testing.assert_array_equal(a, b)
    for replay_mode in ("record", "recompute"):
        with pytest.raises(ValueError, match="requires.*temporal_gradient_precision"):
            make_cell(mode, selective=selective, current_replay_mode=replay_mode)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_fp32_carry_default_and_null_resolve_to_record(mode):
    cells = [
        make_cell(mode, temporal_gradient_precision="float32", **options)
        for options in ({}, {"current_replay_mode": None}, {"current_replay_mode": "record"})
    ]
    results = []
    for cell in cells:
        assert cell.current_replay_mode == "record"
        runner = TemporalAdjointRunner(cell, chunk_size=3)
        assert runner.record_currents
        state = cell.zero_state(2, cell.compute_dtype)
        x = tf.ones((2, 7, 2), tf.float32) * 0.7
        results.append(runner.differentiate(
            x, state, lambda output, final: tf.reduce_sum(final[1] ** 2)
        ))
    assert cells[-1].current_replay_mode_requested == "record"
    for actual in results[1:]:
        for a, b in zip(tf.nest.flatten(actual), tf.nest.flatten(results[0])):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("replay_mode", [None, "record", "recompute"])
def test_json_config_routes_replay_mode_unchanged(replay_mode, monkeypatch):
    from bmtk.simulator.dpointnet.rnn_model import RNN, NetworkAdaptor, SimulationConfig

    options = {
        "cell_model": "GLIF3Cell",
        "state_precision": "selective",
        "temporal_gradient_precision": "float32",
        "current_replay_mode": replay_mode,
    }
    config = SimulationConfig(json.loads(json.dumps({
        "run": {"seq_len": 7, "batch_size": 2, "dtype": "float16"},
        "rnn_cell_params": options,
        "networks": {},
        "inputs": {},
    })))
    monkeypatch.setattr(NetworkAdaptor, "from_dict", lambda _: ([], []))
    rnn = RNN.from_config(config)
    assert rnn.cell_params == {
        **{k: v for k, v in options.items() if k != "cell_model"},
        "acceleration_profile": "auto",
    }
    assert config["rnn_cell_params"] == options


@pytest.mark.parametrize("mode", ["legacy", "nest"])
def test_recompute_allocates_no_host_current_tape(mode, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("recompute must not allocate a current tape")

    monkeypatch.setattr(temporal_adjoint, "_HostCurrentTape", forbidden)
    cell = make_cell(
        mode, temporal_gradient_precision="float32", current_replay_mode="recompute"
    )
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    state = cell.zero_state(2, cell.compute_dtype)
    x = tf.ones((2, 7, 2), tf.float32) * 0.7
    assert not runner.record_currents
    cache = runner._forward(x, state)
    assert cache[5] is None and cache[6] is not None

    @tf.function
    def run():
        return runner.differentiate(
            x, state, lambda output, final: tf.reduce_sum(final[1] ** 2)
        )

    result = run()
    assert np.any(result["input_gradients"].numpy())
    assert all(x.dtype == tf.float32 for x in result["state_cotangents"])
    graph = run.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [n for f in graph.library.function for n in f.node_def]
    assert not any(
        "CurrentTape" in n.op or "HashTable" in n.op or "original_current_tape_copy" in n.name
        for n in nodes
    )


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("noise", [False, True])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_snapshot_replay_bitwise_and_weight_mutation(mode, noise, replay_mode):
    cell = make_cell(
        mode, noise=noise, two_inputs=True,
        temporal_gradient_precision="float32", current_replay_mode=replay_mode,
    )
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    state = cell.zero_state(2, cell.compute_dtype)
    x = tf.ones((2, 7, 0 if noise else 4), tf.float32) * 0.7319
    cache = runner._forward(x, state)
    sequences, _, boundaries, saved, seed, currents, values = cache
    assert values[0].dtype == tf.float16
    assert all(v.dtype == tf.float16 for v in values[1])
    if not noise:
        original, _ = cell._project_step_currents(
            x[:, 0], state, seed, projection_values=values
        )
        float32_projection, _ = cell._adjoint_currents(
            x[:, 0],
            tuple(tf.cast(v, tf.float32) if v.dtype.is_floating else v for v in state),
            projection_context=cell._prepare_adjoint_projection_context(values),
            noise_seed=seed,
        )
        assert np.any(original.numpy() != tf.cast(float32_projection, tf.float16).numpy())

    def replay():
        differences = []
        for i in range(3):
            primal = tuple(a.read(i) for a in saved)
            start, stop = int(boundaries[i]), int(boundaries[i + 1])
            # Reprojection is the original FP16 operation, not FP32 then cast.
            original, _ = cell._project_step_currents(
                x[:, start], primal, seed, projection_values=values
            )
            assert original.dtype == tf.float16
            output, final = runner._loop(
                x[:, start:stop],
                tuple(tf.cast(v, tf.float32) if v.dtype.is_floating else v for v in primal),
                replay=True,
                recorded_currents=runner._read_current_chunk(currents, i),
                projection_context=cell._prepare_adjoint_projection_context(values),
                projection_values=values,
                noise_seed=seed,
            )
            expected = tuple(v[:, start:stop] for v in sequences) + tuple(
                a.read(i + 1) for a in saved
            )
            for a, b in zip(output + tuple(final), expected):
                np.testing.assert_array_equal(tf.cast(a, b.dtype), b)
            differences.append(original)
        return differences

    final_gradients = [None] * len(state)
    final_gradients[1] = 2 * cache[1][1]

    def reverse():
        result = runner._backward(
            x, state, cache, (None, None), final_gradients, variables(cell), capture=True
        )
        assert all(v.dtype == tf.float32 for v in result[3])
        return result[0], *result[2]

    expected, expected_currents = reverse(), replay()
    for v in cell.trainable_variables:
        v.assign(v * 1.7)
    cell.refresh_recurrent_weight_shadow()
    cell.synaptic_basis_weights = cell.synaptic_basis_weights * tf.constant(0.5, tf.float16)
    cell.advance_noise_seed()
    # Another rollout cannot overwrite the first invocation's snapshots.
    runner._forward(x, state)
    shadows = [cell.recurrent_weight_values_compute.numpy().copy()] + [
        net["input_weight_values_compute"].numpy().copy() for net in cell.inputs.values()
    ]
    for a, b in zip(reverse(), expected):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(replay(), expected_currents):
        np.testing.assert_array_equal(a, b)
    for actual, expected in zip(
        [cell.recurrent_weight_values_compute] +
        [net["input_weight_values_compute"] for net in cell.inputs.values()], shadows
    ):
        np.testing.assert_array_equal(actual, expected)
    if noise:
        draws = cell.sample_noise_spikes(2, state[6], cell.inputs["drive"], seed)
        assert draws.dtype == tf.int32
        counts = [
            cell.sample_noise_spikes(2, tf.fill((2,), i), cell.inputs["drive"], seed)
            for i in range(7)
        ]
        assert np.max(np.stack(counts)) > 1


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
@pytest.mark.parametrize("noise", [False, True])
def test_graph_multiple_forwards_snapshot_weights_basis_and_rng(mode, replay_mode, noise):
    def evaluate(joint):
        cell = make_cell(
            mode, noise=noise, temporal_gradient_precision="float32",
            current_replay_mode=replay_mode,
        )
        cell.synaptic_basis_weights = tf.Variable(
            cell.synaptic_basis_weights, trainable=False
        )
        runner = TemporalAdjointRunner(cell, chunk_size=3)
        state = cell.zero_state(2, cell.compute_dtype)
        x = tf.ones((2, 7, 0 if noise else 2), tf.float32) * 0.7319
        masters = variables(cell)

        def loss():
            _, final = runner(x, state)
            return tf.reduce_sum(final[1] ** 2)

        def mutate():
            for v in masters:
                v.assign(v * 1.7)
            cell.refresh_recurrent_weight_shadow()
            cell.synaptic_basis_weights.assign(cell.synaptic_basis_weights * 0.5)
            cell.advance_noise_seed()

        @tf.function
        def run():
            if joint:
                with tf.GradientTape() as tape:
                    first = loss()
                    mutate()
                    second = loss()
                    total = first + second
                gradients = tape.gradient(total, masters)
            else:
                with tf.GradientTape() as tape:
                    first = loss()
                before = tape.gradient(first, masters)
                mutate()
                with tf.GradientTape() as tape:
                    second = loss()
                after = tape.gradient(second, masters)
                gradients = [a + b for a, b in zip(before, after)]
            return first, second, gradients

        result = run()
        assert int(cell.noise_stream) == 1
        np.testing.assert_array_equal(
            cell.recurrent_weight_values_compute,
            tf.cast(cell.recurrent_weight_values, tf.float16),
        )
        return result

    for a, b in zip(tf.nest.flatten(evaluate(True)), tf.nest.flatten(evaluate(False))):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("mode", ["legacy", "nest"])
@pytest.mark.parametrize("replay_mode", ["record", "recompute"])
def test_changing_projection_mismatch_is_expected_only_for_recompute(mode, replay_mode):
    cell = make_cell(
        mode, temporal_gradient_precision="float32", current_replay_mode=replay_mode
    )
    project = cell._project_step_currents
    calls = tf.Variable(0, dtype=tf.int32, trainable=False)

    def changing(*args, **kwargs):
        current, history = project(*args, **kwargs)
        return current + tf.cast(calls.assign_add(1), current.dtype) * 0.125, history

    cell._project_step_currents = changing
    runner = TemporalAdjointRunner(cell, chunk_size=3)
    state = cell.zero_state(2, cell.compute_dtype)
    x = tf.ones((2, 3, 2), tf.float32) * 0.7
    cache = runner._forward(x, state)
    _, final = runner._loop(
        x, tuple(tf.cast(v, tf.float32) if v.dtype.is_floating else v for v in state),
        replay=True, noise_seed=cache[4],
        recorded_currents=runner._read_current_chunk(cache[5], 0),
        projection_context=cell._prepare_adjoint_projection_context(cache[6]),
        projection_values=cache[6],
    )
    differs = any(
        np.any(tf.cast(a, b.dtype).numpy() != b.numpy()) for a, b in zip(final, cache[1])
    )
    assert differs == (replay_mode == "recompute")
    assert int(calls) == (6 if replay_mode == "recompute" else 3)
    metrics = cache_replay_error_metrics(runner, x, state, cache)
    assert bool(tf.reduce_any(metrics["state_boundary_mismatch_count"] > 0)) == differs
    assert bool(tf.reduce_any(metrics["state_boundary_max_abs_error"] > 0)) == differs
