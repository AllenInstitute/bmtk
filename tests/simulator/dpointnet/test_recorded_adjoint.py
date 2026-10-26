import numpy as np
import pytest
import tensorflow as tf
from functools import wraps
from types import SimpleNamespace

from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.recorded_adjoint import RecordedAdjointRunner
from bmtk.simulator.dpointnet.spike_statistics import SpikeStatisticsRunner, SampledSpikeTraces
from test_precision_credit import make_cell


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


@pytest.mark.parametrize("mode,hard", [("legacy", False), ("nest", False)])
def test_recorded_runner_rejects_unsupported_dynamics(mode, hard):
    cell = make_cell(mode, hard_reset=hard)
    try:
        with pytest.raises(ValueError, match="Recorded adjoint requires"):
            RecordedAdjointRunner(None, cell, 7, 3)
    finally:
        cell.close_fused_cuda()


def test_recorded_runner_rejects_unknown_cell_before_accessing_glif_fields():
    with pytest.raises(ValueError, match="unmodified GLIF3Cell"):
        RecordedAdjointRunner(None, SimpleNamespace(), 7, 3)


@pytest.mark.parametrize("approach", ["single", "series", "series_accumulate", "parallel"])
@pytest.mark.parametrize("mixed", [False, True])
def test_native_training_engine_recorded_updates_match_checkpointed(approach, mixed):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.training import TrainingEngine
    from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate

    results = []
    for recorded in (False, True):
        tf.keras.utils.set_random_seed(17)
        cell = make_cell(
            "nest", selective=mixed, fused=True, noise=True, two_inputs=True,
            batch_size=32, hard_reset=True, hard_reset_gradient_mode="soft_surrogate",
            use_fused_cuda=True, use_combined_current_sweep=True,
            use_fused_state_history=True, use_direct_state_rnn_loop=True,
            use_csr_master_weights=True, use_uint8_synapse_types=True,
            use_adaptive_routing=True, use_scalar_input_projection=True,
        )
        try:
            batch = 32 if approach == "single" else 16
            states = list(cell.zero_state(batch, cell.compute_dtype))
            states[1] = tf.tile(tf.cast([[0.9, 1.1]], states[1].dtype), [batch, 1])
            x = tf.keras.Input((None, 0), dtype=cell.compute_dtype)
            state_inputs = tuple(tf.keras.Input(value.shape[1:], dtype=value.dtype) for value in states)
            output = ExplicitStateRNN(cell, return_sequences=True, return_state=True)(
                x, initial_state=state_inputs,
            )
            model = tf.keras.Model((x, *state_inputs), (*output[0], *output[1:]))
            rnn = SimpleNamespace(
                cell=cell, extractor_model=model, model=model, dt=1.0, seq_len=7,
                strategy=tf.distribute.get_strategy(),
                _unpack_extractor_output=lambda flat: ((flat[0], flat[1]), *flat[2:]),
            )
            engine = TrainingEngine(
                rnn, 1, 1, training_approach=approach,
                gradient_checkpoint_chunk_size=3, pack_spike_checkpoints=True,
                stream_spike_statistics=True, recorded_adjoint=recorded,
            )
            for index in range(1 if approach == "single" else 2):
                parameter = engine.add_parameters(f"p{index}", batch_size=batch, seq_len=7)
                parameter.add_loss_function(
                    "rate", TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True),
                )
            engine._inputs_sig_factory = SimpleNamespace(
                build=lambda ys: ys, lu_tables=[{} for _ in engine.parameters],
            )
            engine.set_optimizer(tf.keras.optimizers.Adam(0.001, epsilon=1e-11))
            engine.prepare_gradient_checkpointing()
            assert isinstance(engine._extractor_forward, RecordedAdjointRunner) is recorded
            initial_weights = tuple(value.numpy().copy() for value in model.trainable_variables)
            values = tf.zeros((batch, 7, 0), cell.compute_dtype)
            if approach == "single":
                losses, gradients, _ = engine._single_gradients((values,), {}, tuple(states))
                assert all(np.any(gradient.numpy() != 0) for gradient in gradients)
                engine._apply_gradients(gradients)
            elif approach == "series":
                losses = engine._train_step_series((values, values), ({}, {}), tuple(states))
            elif approach == "series_accumulate":
                losses = engine._distributed_train_step_series_accumulate(
                    (values, values), ({}, {}), tuple(states),
                )
            else:
                parallel_states = tuple(tf.concat((value, value), axis=0) for value in states)
                losses = engine._train_step_parallel(
                    (values, values), ({}, {}), parallel_states,
                )
            engine.learning_rule.on_global_step_end()
            cell.refresh_recurrent_weight_shadow()
            for before, after in zip(initial_weights, model.trainable_variables):
                assert np.any(before != after.numpy())
            results.append((
                losses, tuple(tf.identity(value) for value in model.trainable_variables),
                tf.identity(engine.optimizer.iterations),
            ))
        finally:
            cell.close_fused_cuda()
    for expected, actual in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1])):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("neurons", [1, 65])
@pytest.mark.parametrize("delay", [1, 2, 8])
@pytest.mark.parametrize("length", [1, 5, 31])
def test_delayed_history_rematerialization_is_bitwise(dtype, neurons, delay, length):
    from bmtk.simulator.dpointnet.recorded_adjoint import (
        _pack_record_flags, _restore_record_flags,
    )
    rng = np.random.default_rng(17)
    initial = tf.constant(rng.integers(0, 2, (2, delay * neurons)), dtype)
    events = tf.constant(rng.integers(0, 2, (length, 2, 2 * neurons)), dtype)
    history = initial
    expected = []
    for index in range(length):
        expected.append(tf.concat((history, events[index]), axis=1))
        history = tf.concat((events[index, :, neurons:], history[:, :-neurons]), axis=1)
    observed = _restore_record_flags(initial, _pack_record_flags(events), neurons, length)
    np.testing.assert_array_equal(observed.numpy(), tf.stack(expected).numpy())


@pytest.mark.parametrize("selective,state_precision", [(True, "selective"), (False, "compute"), (True, "compute")])
@pytest.mark.parametrize("time_major", [False, True])
def test_asc_rematerialization_is_bitwise_native(selective, state_precision, time_major):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.recorded_adjoint import _restore_asc
    cell = make_cell(
        "nest", selective=selective, state_precision=state_precision, fused=True, noise=True,
        hard_reset=True, temporal_gradient_precision="compute",
        use_fused_cuda=True, use_combined_current_sweep=True,
        use_fused_state_history=True, use_direct_state_rnn_loop=True,
    )
    try:
        initial = list(cell.zero_state(2, cell.compute_dtype))
        initial[1] = tf.cast([[0.9, 1.1], [1.05, 0.8]], initial[1].dtype)
        initial[3] = tf.cast([[0.1, -0.2, 0.3, 0.4]] * 2, initial[3].dtype)
        state = tuple(initial)
        expected, flags = [], []
        for _ in range(31):
            expected.append(state[3])
            old_refractory = tf.cast(state[2] > 0, cell.compute_dtype)
            output, next_state = cell._call_impl(tf.zeros((2, 0), cell.compute_dtype), state)
            flags.append(tf.concat((old_refractory, output[0]), axis=1))
            state = next_state
        flags = tf.stack(flags, axis=1)
        assert np.any(flags[:, :, 2:].numpy())
        observed = _restore_asc(
            initial[3], tf.transpose(flags, (1, 0, 2)) if time_major else flags,
            cell.asc_decay, cell.asc_amps,
            cell.asc_refractory_decay, cell._n_neurons,
            time_major=time_major,
        )
        np.testing.assert_array_equal(
            observed.numpy(), tf.stack(expected, axis=0 if time_major else 1).numpy(),
        )
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("mode", ["exact", "soft_surrogate"])
@pytest.mark.parametrize("event_vjp", [False, True])
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("batch,csr,scalar", [
    (2, False, False), (32, False, False), (32, True, False), (32, True, True),
])
def test_recorded_matches_checkpointed_values_state_and_named_master_vjps(
    selective, mode, event_vjp, graph, batch, csr, scalar,
):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    cell = make_cell(
        "nest", selective=selective, fused=True, noise=True, two_inputs=True,
        hard_reset=True, hard_reset_gradient_mode=mode,
        temporal_gradient_precision="compute", use_fused_cuda=True,
        use_combined_current_sweep=True, use_fused_state_history=True,
        use_direct_state_rnn_loop=True, use_fused_nest_event_vjp=event_vjp,
        use_device_active_queue_forward=True,
        batch_size=batch, use_csr_master_weights=csr,
        use_uint8_synapse_types=scalar, use_adaptive_routing=scalar,
        use_scalar_input_projection=scalar,
    )
    try:
        initial = list(cell.zero_state(batch, cell.compute_dtype))
        initial[0] = tf.tile(tf.cast([[0, 1, 1, 0], [1, 0, 0, 1]], initial[0].dtype), [batch // 2, 1])
        initial[1] = tf.tile(tf.cast([[0.9, 1.1], [1.05, 0.8]], initial[1].dtype), [batch // 2, 1])
        initial[2] = tf.tile(tf.cast([[0, 1], [1, 0]], initial[2].dtype), [batch // 2, 1])
        initial[3] = tf.cast([[0.1, -0.2, 0.3, 0.4]] * batch, initial[3].dtype)
        initial = tuple(initial)
        x = tf.keras.Input((None, 0), dtype=cell.compute_dtype)
        state_inputs = tuple(tf.keras.Input(value.shape[1:], dtype=value.dtype) for value in initial)
        out = ExplicitStateRNN(cell, return_sequences=True, return_state=True)(
            x, initial_state=state_inputs,
        )
        model = tf.keras.Model((x, *state_inputs), (*out[0], *out[1:]))
        options = dict(
            sequence_length=7, chunk_size=3, differentiate_inputs=False,
            count_windows=((0, 7), (1, 6)), pack_spike_checkpoints=True,
            variable_gradient_transform=(
                cell.restore_segmented_variable_gradients
                if cell._use_direct_csr_recurrent_gradient and not cell._csr_master_weights
                else None
            ),
        )
        reference = SpikeStatisticsRunner(model, n_sequence_outputs=2, **options)
        actual = RecordedAdjointRunner(model, cell, **options)
        traces = (SampledSpikeTraces(
            tf.constant([[0, 0], [1, 0]], tf.int64), tf.ones((2, 1, 2), cell.compute_dtype),
            (1, 7), (0, 2),
        ),)
        inputs = tf.zeros((batch, 7, 0), cell.compute_dtype)
        floating = tuple(value for value in initial if value.dtype.is_floating)

        def evaluate():
            comparisons = []
            for runner in (reference, actual):
                with tf.GradientTape() as tape:
                    tape.watch(floating)
                    output = runner(inputs, initial, sampled_traces=traces)
                    values = output.counts + output.traces + output.sequences + tuple(
                        value for value in output.final_state if value.dtype.is_floating
                    )
                    loss = tf.add_n([
                        tf.reduce_sum(tf.square(tf.cast(value, tf.float32))) * (index + 1)
                        for index, value in enumerate(values)
                    ])
                gradients = tape.gradient(
                    loss, (*floating, *model.trainable_variables),
                    unconnected_gradients=tf.UnconnectedGradients.ZERO,
                )
                comparisons.append((values, gradients))
            return comparisons

        expected, observed = tf.function(evaluate)() if graph else evaluate()
        assert cell.native_execution_options["use_scalar_input_projection"] is scalar
        if scalar:
            paths = cell.acceleration_report["input_paths"]
            assert sum(path["scalar_projection"] for path in paths.values()) == 1
        for target, value in zip(tf.nest.flatten(expected), tf.nest.flatten(observed)):
            np.testing.assert_allclose(value, target, rtol=2e-5, atol=2e-5)
        assert all(np.any(value.numpy() != 0) for value in observed[1][-3:])
    finally:
        cell.close_fused_cuda()


def test_recorded_reverse_constructs_no_state_or_projection_forward(monkeypatch):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.cell_models import glif3_cell
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops
    from bmtk.simulator.dpointnet.recorded_adjoint import _RecordedSequence

    cell = make_cell(
        "nest", selective=False, fused=True, noise=True, two_inputs=True,
        hard_reset=True, temporal_gradient_precision="compute",
        use_fused_cuda=True, use_combined_current_sweep=True,
        use_fused_state_history=True, use_direct_state_rnn_loop=True,
    )
    calls = {"state": 0, "projection": 0}
    native = glif_state_ops._OPS

    class CountingOperations:
        def __getattr__(self, name):
            operation = getattr(native, name)
            if not name.startswith("dpointnet_nest_state") or "_forward" not in name:
                return operation

            @wraps(operation)
            def counted(*args, **kwargs):
                calls["state"] += 1
                return operation(*args, **kwargs)

            return counted

    combined = glif3_cell.fused_combined_currents

    def counted_projection(*args, **kwargs):
        calls["projection"] += 1
        return combined(*args, **kwargs)

    try:
        layer = _RecordedSequence(cell)
        initial = cell.zero_state(2, cell.compute_dtype)
        inputs = tf.zeros((2, 7, 0), cell.compute_dtype)
        forward = layer.run(inputs, initial)
        assert forward[4].shape == (7, 2, (2 * cell._n_neurons + 62) // 63)
        retained = tuple(forward[2:6])
        monkeypatch.setattr(glif_state_ops, "_OPS", CountingOperations())
        monkeypatch.setattr(glif3_cell, "fused_combined_currents", counted_projection)

        @tf.function
        def reverse():
            return layer.chunk_vjp(
                inputs, initial, retained,
                (tf.ones_like(forward[0]), tf.ones_like(forward[1])),
                tuple(tf.zeros_like(value) if value.dtype.is_floating else None for value in initial),
                tuple(value if isinstance(value, tf.Variable) else value.value for value in cell.trainable_variables),
            )[-len(cell.trainable_variables):]

        gradients = reverse()
        assert calls == {"state": 0, "projection": 0}
        assert all(value is not None for value in gradients)
        assert float(tf.linalg.global_norm(gradients).numpy()) > 0
    finally:
        cell.close_fused_cuda()
