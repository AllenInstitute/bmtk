from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate, LowRateFloor
from bmtk.simulator.dpointnet.loss_functions import loss_utils
from bmtk.simulator.dpointnet.loss_functions.synchronization_loss import (
    SynchronizationLoss, _gather_population_traces,
)
from bmtk.simulator.dpointnet.spike_summary import SpikeSummary
from bmtk.simulator.dpointnet.training import TrainingEngine


@pytest.mark.parametrize("graph", [False, True])
def test_summary_loss_values_vjps_trimming_and_batch_slices(graph):
    values = tf.reshape(tf.cast(tf.range(84) % 3, tf.float32), (4, 7, 3))
    windows = ((0, 7), (2, 6))
    rnn = SimpleNamespace(dt=1.0, seq_len=7)
    target = TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True)
    floor = LowRateFloor(rnn, floor_hz=400)

    def evaluate():
        results = []
        for summarized in (False, True):
            with tf.GradientTape() as tape:
                tape.watch(values)
                spikes = SpikeSummary(
                    tuple(tf.reduce_sum(values[:, start:stop], axis=1) for start, stop in windows),
                    windows, 7,
                ) if summarized else values
                trimmed = loss_utils.spike_trimming(spikes, 2, 1)
                loss = target(spikes) + floor(spikes)
                loss += tf.reduce_sum(tf.square(loss_utils.temporal_sum(trimmed)))
                loss += tf.reduce_sum(loss_utils.temporal_sum(spikes[1:3]))
                mean = TrainingEngine._mean_spikes(spikes)
                shape = TrainingEngine._spike_shape(spikes)
            results.append((loss, mean, shape, tape.gradient(loss, values)))
        return results

    reference, actual = tf.function(evaluate)() if graph else evaluate()
    for expected, observed in zip(tf.nest.flatten(reference), tf.nest.flatten(actual)):
        np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("graph", [False, True])
def test_summary_synchronization_matches_original_plan_value_and_vjp(monkeypatch, graph):
    names = np.array(["e0"] * 20)
    rnn = SimpleNamespace(recurrent_network={"n_nodes": 20, "node_params": {"pop_name": names}})
    monkeypatch.setattr(loss_utils, "get_pop_names", lambda network, data_dir="": names)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: np.ones((3, 20), np.float32))
    dense = SynchronizationLoss(rnn, t_start=2, t_end=20, n_samples=5, seed=31)
    streamed = SynchronizationLoss(rnn, t_start=2, t_end=20, n_samples=5, seed=31)
    values = tf.constant(np.random.default_rng(31).uniform(size=(2, 20, 20)), tf.float32)

    def evaluate():
        with tf.GradientTape() as tape:
            tape.watch(values)
            reference = dense(values)
        reference_gradient = tape.gradient(reference, values)
        with tf.GradientTape() as tape:
            tape.watch(values)
            plan = streamed.prepare_sampled_statistics(2, values.dtype, 20)
            sampled = _gather_population_traces(
                values, plan.sample_ids, plan.neuron_mask, 2, 20,
                plan.neuron_mask.shape[1], plan.neuron_mask.shape[2],
            )[:, plan.window[0]:plan.window[1]]
            summary = SpikeSummary(
                (tf.reduce_sum(values, axis=1),), ((0, 20),), 20, (sampled,),
                ((streamed.name, 0, 2),),
            )
            actual = streamed(summary)
        return reference, actual, reference_gradient, tape.gradient(actual, values)

    expected, actual, expected_gradient, gradient = tf.function(evaluate)() if graph else evaluate()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-6)
    assert int(dense._seed_stream.numpy()) == int(streamed._seed_stream.numpy()) == 1


def test_summary_rejects_missing_windows_and_partial_trace_groups():
    summary = SpikeSummary(
        (tf.ones((3, 4)),), ((0, 7),), 7, (tf.ones((2, 5)),), (("sync", 1, 3),),
    )
    assert summary[1:3].trace_for_loss("sync").shape == (2, 5)
    with pytest.raises(ValueError, match="exactly one complete batch trace"):
        summary[:2].trace_for_loss("sync")
    with pytest.raises(ValueError, match="requested window"):
        summary.trim(1, 1)
    with tf.GradientTape() as tape:
        tape.watch(summary.counts)
        loss = tf.reduce_sum(summary.stop_gradient().temporal_sum())
    assert tape.gradient(loss, summary.counts) == (None,)


@pytest.mark.parametrize("name", ["stream_spike_statistics", "recorded_adjoint"])
@pytest.mark.parametrize("value", ["yes", 1, None])
def test_statistics_and_recording_options_require_booleans(name, value):
    with pytest.raises(ValueError, match=name):
        TrainingEngine(SimpleNamespace(), 1, 1, **{name: value})


def test_statistics_and_recording_options_default_to_auto():
    engine = TrainingEngine(SimpleNamespace(), 1, 1)
    assert engine.stream_spike_statistics_request == "auto"
    assert engine.recorded_adjoint_request == "auto"
    # Unresolved automatic requests read as the dense route until preparation.
    assert engine.stream_spike_statistics is False and engine.recorded_adjoint is False
    explicit = TrainingEngine(SimpleNamespace(), 1, 1, stream_spike_statistics=True, recorded_adjoint="direct")
    assert explicit.stream_spike_statistics is True and explicit.recorded_adjoint == "direct"


@pytest.mark.parametrize("value", [True, "direct"])
def test_recorded_option_requires_statistics(value):
    with pytest.raises(ValueError, match="requires stream_spike_statistics"):
        TrainingEngine(SimpleNamespace(), 1, 1, recorded_adjoint=value, stream_spike_statistics=False)


def _training_fixture(streamed, approach):
    inputs = tf.keras.Input((None, 3))
    initial = tf.keras.Input((3,))
    sequence, final = tf.keras.layers.RNN(
        tf.keras.layers.SimpleRNNCell(
            3, use_bias=False, kernel_initializer=tf.keras.initializers.Constant(0.3),
            recurrent_initializer=tf.keras.initializers.Constant(0.2),
        ), return_sequences=True, return_state=True,
    )(inputs, initial_state=(initial,))
    model = tf.keras.Model((inputs, initial), (sequence, sequence * 0.5, final))
    seed = tf.Variable(0, dtype=tf.int64)
    cell = SimpleNamespace(
        temporal_gradient_precision="compute",
        advance_noise_seed=lambda: seed.assign_add(1),
        reset_voltage_penalty_state=lambda state: state,
        refresh_recurrent_weight_shadow=lambda: None,
    )
    rnn = SimpleNamespace(
        cell=cell, extractor_model=model, model=model, dt=1.0, seq_len=7,
        strategy=tf.distribute.get_strategy(),
        _unpack_extractor_output=lambda flat: ((flat[0], flat[1]), *flat[2:]),
    )
    engine = TrainingEngine(
        rnn, 1, 1, training_approach=approach, gradient_checkpoint_chunk_size=3,
        pack_spike_checkpoints=False, stream_spike_statistics=streamed,
    )
    for index in range(1 if approach == "single" else 2):
        parameter = engine.add_parameters(f"p{index}", batch_size=2, seq_len=7)
        parameter.add_loss_function(
            "rate", TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True)
        )
        parameter.add_loss_function("floor", LowRateFloor(rnn, floor_hz=2))
    engine._inputs_sig_factory = SimpleNamespace(
        build=lambda ys: ys, lu_tables=[{} for _ in engine.parameters],
    )
    engine.set_optimizer(tf.keras.optimizers.SGD(1e-6))
    engine.prepare_gradient_checkpointing()
    return engine, model, seed


@pytest.mark.parametrize("approach", ["single", "series", "series_accumulate", "parallel"])
def test_training_engine_streaming_preserves_losses_gradients_and_updates(approach):
    values = tf.reshape(tf.cast(tf.range(42) % 3, tf.float32), (2, 7, 3)) * 0.1
    initial = tf.zeros((2, 3))
    outcomes = []
    for streamed in (False, True):
        engine, model, seed = _training_fixture(streamed, approach)
        if approach == "single":
            losses, gradients, _ = engine._single_gradients((values,), {}, (initial,))
            engine._apply_gradients(gradients)
        elif approach == "series":
            losses = engine._train_step_series((values, values * 0.5), ({}, {}), (initial,))
        elif approach == "series_accumulate":
            losses = engine._distributed_train_step_series_accumulate(
                (values, values * 0.5), ({}, {}), (initial,),
            )
        else:
            losses = engine._train_step_parallel(
                (values, values * 0.5), ({}, {}), (tf.zeros((4, 3)),),
            )
        outcomes.append((
            losses, tuple(tf.identity(value) for value in model.trainable_variables),
            seed.read_value(), tf.identity(engine.optimizer.iterations),
        ))
    for expected, actual in zip(tf.nest.flatten(outcomes[0]), tf.nest.flatten(outcomes[1])):
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)
