import numpy as np
import pytest
import tensorflow as tf
from types import SimpleNamespace

from bmtk.simulator.dpointnet.cell_models.state_rnn import ExplicitStateRNN
from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.direct_recorded_adjoint import (
    DirectRecordedAdjointRunner, DirectRecordedStep, direct_records_supported,
)
from bmtk.simulator.dpointnet.recorded_adjoint import RecordedAdjointRunner
from bmtk.simulator.dpointnet.spike_statistics import SpikeStatisticsRunner, SampledSpikeTraces
from test_precision_credit import make_cell


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


DIRECT_OPTIONS = dict(
    fused=True, noise=True, two_inputs=True, hard_reset=True,
    temporal_gradient_precision="compute", use_fused_cuda=True,
    use_combined_current_sweep=True, use_fused_state_history=True,
    use_direct_state_rnn_loop=True, use_fused_nest_event_vjp=True,
    use_device_active_queue_forward=True, use_csr_master_weights=True,
    use_uint8_synapse_types=True, use_adaptive_routing=True,
    use_scalar_input_projection=True, use_fused_recurrent_accumulation=True,
    use_direct_csr_recurrent_gradient=True, use_pair_projection=True,
    use_prepacked_nest_coefficients=True, use_native_voltage_penalty=True,
)


def _initial_state(cell, batch):
    initial = list(cell.zero_state(batch, cell.compute_dtype))
    initial[0] = tf.tile(tf.cast([[0, 1, 1, 0], [1, 0, 0, 1]], initial[0].dtype), [batch // 2, 1])
    initial[1] = tf.tile(tf.cast([[0.9, 1.1], [1.05, 0.8]], initial[1].dtype), [batch // 2, 1])
    initial[2] = tf.tile(tf.cast([[0, 1], [1, 0]], initial[2].dtype), [batch // 2, 1])
    initial[3] = tf.cast([[0.1, -0.2, 0.3, 0.4]] * batch, initial[3].dtype)
    return tuple(initial)


def _model(cell, initial):
    x = tf.keras.Input((None, 0), dtype=cell.compute_dtype)
    state_inputs = tuple(tf.keras.Input(value.shape[1:], dtype=value.dtype) for value in initial)
    out = ExplicitStateRNN(cell, return_sequences=True, return_state=True)(x, initial_state=state_inputs)
    return tf.keras.Model((x, *state_inputs), (*out[0], *out[1:]))


def test_direct_runner_rejects_unknown_cell_before_accessing_glif_fields():
    with pytest.raises(ValueError, match="unmodified GLIF3Cell"):
        DirectRecordedAdjointRunner(None, SimpleNamespace(), 7, 3)


@pytest.mark.parametrize("mode,hard", [("legacy", False), ("nest", False)])
def test_direct_runner_rejects_unsupported_dynamics(mode, hard):
    cell = make_cell(mode, hard_reset=hard)
    try:
        unsupported = direct_records_supported(cell)
        assert "NEST hard forward" in unsupported
        with pytest.raises(ValueError, match="Direct recorded adjoint requires"):
            DirectRecordedAdjointRunner(None, cell, 7, 3)
    finally:
        cell.close_fused_cuda()


def test_direct_runner_rejects_input_differentiation_and_transforms():
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        assert direct_records_supported(cell) == ()
        with pytest.raises(ValueError, match="does not differentiate inputs"):
            DirectRecordedAdjointRunner(None, cell, 7, 3, differentiate_inputs=True)
        with pytest.raises(ValueError, match="without a transform"):
            DirectRecordedAdjointRunner(None, cell, 7, 3, variable_gradient_transform=lambda v, g: g)
        with pytest.raises(ValueError, match="chunk_size"):
            DirectRecordedAdjointRunner(None, cell, 7, 9)
    finally:
        cell.close_fused_cuda()


@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("refractory", [tf.int8, tf.int16])
@pytest.mark.parametrize("neurons", [1, 31, 32, 33, 1000])
def test_step_record_pack_unpack_roundtrip(dtype, refractory, neurons):
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops as ops

    rng = np.random.default_rng(17)
    spikes = tf.cast(rng.random((3, neurons)) < 0.3, dtype)
    counters = tf.cast(rng.integers(0, 3, (3, neurons)), refractory)
    words = ops.pack_step_record(spikes, counters)
    assert words.shape == (3, 2 * ((neurons + 31) // 32))
    unpacked, blocked = ops.unpack_step_record(words, neurons, dtype, refractory)
    np.testing.assert_array_equal(unpacked.numpy(), spikes.numpy())
    np.testing.assert_array_equal(blocked.numpy(), (counters.numpy() > 0).astype(refractory.as_numpy_dtype))


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("mode", ["exact", "soft_surrogate"])
@pytest.mark.parametrize("batch", [4, 32])
@pytest.mark.parametrize("delay_windows", [((0, 7),), ((0, 7), (1, 6))])
def test_direct_matches_reference_values_state_and_master_vjps(selective, mode, batch, delay_windows):
    """Forward values and state/history cotangents are bitwise those of the
    checkpointed reference; master gradients differ only by FP32 accumulation
    order (per-step native accumulation versus per-chunk tape sums)."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    cell = make_cell("nest", selective=selective, batch_size=batch, hard_reset_gradient_mode=mode, **DIRECT_OPTIONS)
    try:
        initial = _initial_state(cell, batch)
        model = _model(cell, initial)
        options = dict(sequence_length=7, chunk_size=3, differentiate_inputs=False, count_windows=delay_windows)
        reference = SpikeStatisticsRunner(model, n_sequence_outputs=2, pack_spike_checkpoints=True, **options)
        recorded = RecordedAdjointRunner(model, cell, pack_spike_checkpoints=True, **options)
        direct = DirectRecordedAdjointRunner(model, cell, **options)
        traces = (
            SampledSpikeTraces(tf.constant([[0, 0], [1, 0]], tf.int64), tf.ones((2, 1, 2), cell.compute_dtype), (1, 7), (0, 2)),
            SampledSpikeTraces(tf.constant([[1, 1]], tf.int64), tf.constant([[[1, 0]]], cell.compute_dtype), (0, 4), (2, 3)),
        )
        inputs = tf.zeros((batch, 7, 0), cell.compute_dtype)
        floating = tuple(value for value in initial if value.dtype.is_floating)

        @tf.function
        def evaluate():
            comparisons = []
            for runner in (reference, recorded, direct):
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

        expected, chunked, observed = evaluate()
        n_values = len(expected[0])
        for index, (target, value) in enumerate(zip(tf.nest.flatten(expected), tf.nest.flatten(observed))):
            if index >= n_values + len(floating):
                # Masters: per-step FP32 accumulation order differs from the chunked tape sums.
                np.testing.assert_allclose(value, target, rtol=1e-6, atol=0)
            else:
                np.testing.assert_array_equal(value.numpy(), target.numpy())
        for target, value in zip(tf.nest.flatten(chunked), tf.nest.flatten(observed)):
            np.testing.assert_allclose(value, target, rtol=1e-6, atol=0)
        assert all(np.any(value.numpy() != 0) for value in observed[1][-3:])
        assert cell.acceleration_report["rollout_selected"]["temporal_path"] == "direct_recorded_vjp"
    finally:
        cell.close_fused_cuda()


def test_direct_step_primitives_expose_per_population_source_credit():
    """The step primitive returns dense per-population source cotangents on request."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.cell_models.state_rnn import _rollout_context

    batch = 32
    cell = make_cell("nest", selective=True, batch_size=batch, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        initial = _initial_state(cell, batch)

        @tf.function
        def run():
            with _rollout_context(cell):
                step = DirectRecordedStep(cell, batch, initial)
                sources = [tf.cast(initial[0], cell.compute_dtype)] + [
                    tf.ones((batch, population["width"] * (population["history_steps"] + 1)), cell.compute_dtype)
                    for population in step.populations[1:]
                ]
                spikes, state, record, penalty = step.forward(sources, initial)
                carry = tuple(tf.ones_like(value) for value in (state[1], state[3], state[4], state[5], state[0]))
                previous, credit, accumulators = step.reverse(
                    record, initial[3], carry, tf.ones_like(spikes), tf.ones([batch], tf.float32),
                    sources, (True, True, False), (None, tf.zeros_like(step.populations[1]["variable"]), None),
                )
            return spikes, penalty, previous, credit, accumulators

        spikes, penalty, previous, credit, accumulators = run()
        assert spikes.shape == (batch, 2) and penalty.shape == (batch,)
        assert [None if value is None else tuple(value.shape) for value in credit] == [
            (batch, 4), (batch, 4), None,
        ]
        assert accumulators[0].shape == (2,) and accumulators[1].shape == (2,) and accumulators[2] is None
        assert all(np.all(np.isfinite(value.numpy())) for value in previous)
        assert np.any(credit[1].numpy() != 0)
    finally:
        cell.close_fused_cuda()


def _record_forward_inputs(rng, batch, neurons, delay, state_dtype, spike_dtype, refractory_dtype, types):
    cast = lambda array, dtype: tf.constant(array.astype(dtype.as_numpy_dtype))
    return dict(
        v=cast(rng.uniform(0, 4.0, (batch, neurons)), state_dtype),
        r=cast(rng.integers(0, 3, (batch, neurons)), refractory_dtype),
        asc=cast(rng.normal(0, 0.1, (batch, 2 * neurons)), state_dtype),
        rise=cast(rng.normal(0, 0.1, (batch, 4 * neurons)), spike_dtype),
        psc=cast(rng.normal(0, 0.1, (batch, 4 * neurons)), spike_dtype),
        currents=cast(rng.normal(0, 0.3, (batch, 4 * neurons)), spike_dtype),
        history=cast(rng.integers(0, 2, (batch, delay * neurons)), spike_dtype),
        type_coefficients=cast(rng.uniform(0, 1, (types, 28)), state_dtype),
        type_indices=tf.constant(rng.integers(0, types, (neurons,)).astype(np.int64)),
        t_ref=cast(np.full((neurons,), 2), refractory_dtype),
        dt=tf.constant(1.0, state_dtype), v_th=tf.constant(1.0, state_dtype),
    )


@pytest.mark.parametrize("state_dtype,spike_dtype", [(tf.float32, tf.float16), (tf.float32, tf.float32), (tf.float16, tf.float16)])
@pytest.mark.parametrize("refractory_dtype", [tf.int8, tf.int16])
@pytest.mark.parametrize("neurons,delay", [(33, 1), (70, 3), (1024, 2)])
@pytest.mark.parametrize("variant", ["plain", "type_indexed", "device_dispatch"])
def test_record_forward_matches_history_forward_and_pack(state_dtype, spike_dtype, refractory_dtype, neurons, delay, variant):
    """The record-writing NEST forwards equal the plain history forward (states,
    spikes, history), ``pack_step_record`` (words) and write the pre-reset
    membrane into the requested scratch row, bitwise."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops as ops

    assert ops.nest_record_forward_available()
    rng = np.random.default_rng(23)
    batch, rows, row = 3, 5, 2
    i = _record_forward_inputs(rng, batch, neurons, delay, state_dtype, spike_dtype, refractory_dtype, 7)
    coefficients = tf.gather(i["type_coefficients"], i["type_indices"])
    with tf.device("/GPU:0"):
        buffer = tf.Variable(tf.zeros([rows, batch, neurons], state_dtype), trainable=False)
    common = (i["v"], i["r"], i["asc"], i["rise"], i["psc"], i["currents"])
    tail = (i["t_ref"], i["dt"], i["v_th"], i["history"])
    if variant == "plain":
        expected = ops._OPS.dpointnet_nest_state_history_forward(
            *common, coefficients, *tail, hard_reset=True, emit_pre_reset_voltage=True)
        observed = ops._OPS.dpointnet_nest_state_history_forward_record(
            *common, coefficients, *tail, buffer.handle, tf.constant(row, tf.int32), hard_reset=True)
    else:
        dispatch = {} if variant == "type_indexed" else dict(
            fallback_coefficients=coefficients, type_identity=tf.constant(True))
        suffix = variant
        expected = getattr(ops._OPS, "dpointnet_nest_state_history_forward_" + suffix)(
            *common, i["type_coefficients"], i["type_indices"], *tail, hard_reset=True,
            emit_pre_reset_voltage=True, **dispatch)
        observed = getattr(ops._OPS, "dpointnet_nest_state_history_forward_record_" + suffix)(
            *common, i["type_coefficients"], i["type_indices"], *tail, buffer.handle,
            tf.constant(row, tf.int32), hard_reset=True, **dispatch)
    words = ops.pack_step_record(expected[6], i["r"])
    assert len(observed) == 8
    for target, value in zip(expected[1:8], observed[:7]):
        np.testing.assert_array_equal(value.numpy(), target.numpy())
    np.testing.assert_array_equal(observed[7].numpy(), words.numpy())
    assert np.any(words.numpy()[:, : (neurons + 31) // 32] != 0)
    stored = buffer.numpy()
    np.testing.assert_array_equal(stored[row], expected[0].numpy())
    assert not np.any(stored[[k for k in range(rows) if k != row]])


def test_record_forward_rejects_bad_buffers_and_rows():
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.custom_ops import glif_state_ops as ops

    rng = np.random.default_rng(5)
    neurons = 40
    i = _record_forward_inputs(rng, 2, neurons, 1, tf.float32, tf.float16, tf.int16, 3)
    args = (i["v"], i["r"], i["asc"], i["rise"], i["psc"], i["currents"], i["type_coefficients"],
            i["type_indices"], i["t_ref"], i["dt"], i["v_th"], i["history"])
    with tf.device("/GPU:0"):
        good = tf.Variable(tf.zeros([2, 2, neurons], tf.float32), trainable=False)
        wrong_dtype = tf.Variable(tf.zeros([2, 2, neurons], tf.float16), trainable=False)
        wrong_shape = tf.Variable(tf.zeros([2, 3, neurons], tf.float32), trainable=False)
    call = ops._OPS.dpointnet_nest_state_history_forward_record_type_indexed
    call(*args, good.handle, tf.constant(1, tf.int32), hard_reset=True)
    for buffer, row, message in (
        (good, 2, "0 <= row < time"), (good, -1, "0 <= row < time"),
        (wrong_shape, 0, "shape \\[time, batch, neurons\\]"), (wrong_dtype, 0, "dtype must match"),
    ):
        with pytest.raises(tf.errors.InvalidArgumentError, match=message):
            call(*args, buffer.handle, tf.constant(row, tf.int32), hard_reset=True)


def test_direct_step_forward_record_row_matches_forward():
    """``forward(record_row=...)`` yields the same step as ``forward`` with the
    pre-reset membrane landing in the named scratch row."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.cell_models.state_rnn import _rollout_context

    batch = 32
    cell = make_cell("nest", selective=True, batch_size=batch, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        initial = _initial_state(cell, batch)
        with tf.device("/GPU:0"):
            buffer = tf.Variable(tf.zeros([4, batch, 2], tf.float32), trainable=False)

        @tf.function
        def run():
            with _rollout_context(cell):
                step = DirectRecordedStep(cell, batch, initial)
                sources = [tf.cast(initial[0], cell.compute_dtype)] + [
                    tf.ones((batch, population["width"] * (population["history_steps"] + 1)), cell.compute_dtype)
                    for population in step.populations[1:]
                ]
                spikes, state, (pre_reset, words), penalty = step.forward(sources, initial)
                spikes2, state2, (none, words2), penalty2 = step.forward(
                    sources, initial, record_row=(buffer.handle, tf.constant(3, tf.int32)))
                assert none is None
            return (spikes, *state, words, penalty), (spikes2, *state2, words2, penalty2), pre_reset

        expected, observed, pre_reset = run()
        for target, value in zip(expected, observed):
            np.testing.assert_array_equal(value.numpy(), target.numpy())
        np.testing.assert_array_equal(buffer.numpy()[3], pre_reset.numpy())
        assert not np.any(buffer.numpy()[:3])
    finally:
        cell.close_fused_cuda()


def test_training_engine_direct_route_matches_recorded():
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.training import TrainingEngine
    from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate

    results = []
    for recorded in (True, "direct"):
        tf.keras.utils.set_random_seed(17)
        cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
        try:
            states = list(cell.zero_state(32, cell.compute_dtype))
            states[1] = tf.tile(tf.cast([[0.9, 1.1]], states[1].dtype), [32, 1])
            model = _model(cell, tuple(states))
            rnn = SimpleNamespace(
                cell=cell, extractor_model=model, model=model, dt=1.0, seq_len=7,
                strategy=tf.distribute.get_strategy(),
                _unpack_extractor_output=lambda flat: ((flat[0], flat[1]), *flat[2:]),
            )
            engine = TrainingEngine(
                rnn, 1, 1, training_approach="single", gradient_checkpoint_chunk_size=3,
                stream_spike_statistics=True, recorded_adjoint=recorded,
            )
            parameter = engine.add_parameters("p0", batch_size=32, seq_len=7)
            parameter.add_loss_function("rate", TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True))
            engine._inputs_sig_factory = SimpleNamespace(build=lambda ys: ys, lu_tables=[{}])
            engine.set_optimizer(tf.keras.optimizers.Adam(0.001, epsilon=1e-11))
            engine.prepare_gradient_checkpointing()
            assert isinstance(engine._extractor_forward, DirectRecordedAdjointRunner) is (recorded == "direct")
            values = tf.zeros((32, 7, 0), cell.compute_dtype)
            # The native carrier VJP needs a traced graph, as the engine's compiled
            # train steps provide; the eager helper is wrapped here for the same reason.
            losses, gradients, _ = tf.function(
                lambda: engine._single_gradients((values,), {}, tuple(states))
            )()
            assert all(np.any(gradient.numpy() != 0) for gradient in gradients)
            engine._apply_gradients(gradients)
            engine.learning_rule.on_global_step_end()
            cell.refresh_recurrent_weight_shadow()
            results.append((losses, tuple(tf.identity(value) for value in model.trainable_variables)))
        finally:
            cell.close_fused_cuda()
    for expected, actual in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1])):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_training_engine_rejects_invalid_recorded_adjoint_option():
    from bmtk.simulator.dpointnet.training import TrainingEngine

    with pytest.raises((ValueError, TypeError)):
        TrainingEngine(SimpleNamespace(cell=None), 1, 1, stream_spike_statistics=True, recorded_adjoint="indirect")


def _engine_for(cell, states, model, losses, **engine_options):
    from bmtk.simulator.dpointnet.training import TrainingEngine

    rnn = SimpleNamespace(
        cell=cell, extractor_model=model, model=model, dt=1.0, seq_len=7, _cell=cell,
        strategy=tf.distribute.get_strategy(),
        _unpack_extractor_output=lambda flat: ((flat[0], flat[1]), *flat[2:]),
    )
    engine = TrainingEngine(rnn, 1, 1, training_approach="single", gradient_checkpoint_chunk_size=3, **engine_options)
    parameter = engine.add_parameters("p0", batch_size=32, seq_len=7)
    for name, loss in losses(rnn).items():
        parameter.add_loss_function(name, loss)
    engine._inputs_sig_factory = SimpleNamespace(build=lambda ys: ys, lu_tables=[{}])
    engine.set_optimizer(tf.keras.optimizers.Adam(0.001, epsilon=1e-11))
    return engine


def _rate_losses(rnn):
    from bmtk.simulator.dpointnet.loss_functions import TargetFiringRate
    return {"rate": TargetFiringRate(rnn, firing_rate=5, compiled_spike_counts=True)}


def test_training_engine_auto_defaults_select_direct_and_match_explicit():
    """Default ``"auto"`` requests resolve to the direct route on an eligible
    GLIF3 cell and produce the same losses/updates as the explicit request."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    results = []
    for options in ({}, {"stream_spike_statistics": True, "recorded_adjoint": "direct"}):
        tf.keras.utils.set_random_seed(17)
        cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
        try:
            states = list(cell.zero_state(32, cell.compute_dtype))
            states[1] = tf.tile(tf.cast([[0.9, 1.1]], states[1].dtype), [32, 1])
            model = _model(cell, tuple(states))
            engine = _engine_for(cell, states, model, _rate_losses, **options)
            engine.prepare_gradient_checkpointing()
            assert engine.stream_spike_statistics is True and engine.recorded_adjoint == "direct"
            assert isinstance(engine._extractor_forward, DirectRecordedAdjointRunner)
            report = engine.training_route_report
            assert report["selected"] == {"stream_spike_statistics": True, "recorded_adjoint": "direct"}
            if not options:
                assert report["requested"] == {"stream_spike_statistics": "auto", "recorded_adjoint": "auto"}
                assert "record payload" in report["reasons"]["recorded_adjoint"]
                assert cell.acceleration_report["training_route"] is report
            else:
                assert report["reasons"]["recorded_adjoint"] == "explicit request"
            values = tf.zeros((32, 7, 0), cell.compute_dtype)
            losses, gradients, _ = tf.function(lambda: engine._single_gradients((values,), {}, tuple(states)))()
            engine._apply_gradients(gradients)
            results.append((losses, tuple(tf.identity(v) for v in model.trainable_variables)))
        finally:
            cell.close_fused_cuda()
    for expected, actual in zip(tf.nest.flatten(results[1]), tf.nest.flatten(results[0])):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


def test_training_engine_auto_warns_when_records_may_not_fit(monkeypatch, caplog):
    """A record payload above the advisory share still selects the direct route and warns."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    import logging
    from bmtk.simulator.dpointnet import direct_recorded_adjoint as module

    monkeypatch.setattr(module, "device_free_bytes", lambda: 1024)
    cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        states = list(cell.zero_state(32, cell.compute_dtype))
        model = _model(cell, tuple(states))
        engine = _engine_for(cell, states, model, _rate_losses)
        with caplog.at_level(logging.WARNING):
            engine.prepare_gradient_checkpointing()
        assert engine.stream_spike_statistics is True and engine.recorded_adjoint == "direct"
        assert isinstance(engine._extractor_forward, DirectRecordedAdjointRunner)
        report = engine.training_route_report
        assert report["record_payload"]["fits"] is False
        assert "record payload" in report["reasons"]["recorded_adjoint"]
        assert any("may not fit the device" in record.getMessage() for record in caplog.records)
    finally:
        cell.close_fused_cuda()


def test_training_engine_auto_keeps_dense_route_for_unsupported_losses():
    """An unsupported loss keeps the dense segmented route under auto and still
    raises under an explicit streaming request."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner

    class CustomLoss:
        def __call__(self, spikes, **kwargs):
            return tf.reduce_mean(tf.cast(spikes, tf.float32))

    def losses(rnn):
        return {**_rate_losses(rnn), "custom": CustomLoss()}

    cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        states = list(cell.zero_state(32, cell.compute_dtype))
        model = _model(cell, tuple(states))
        engine = _engine_for(cell, states, model, losses)
        engine.prepare_gradient_checkpointing()
        assert engine.stream_spike_statistics is False and engine.recorded_adjoint is False
        assert isinstance(engine._extractor_forward, SegmentedRecomputeRunner)
        assert "CustomLoss" in engine.training_route_report["reasons"]["stream_spike_statistics"]
        assert engine.training_route_report["reasons"]["recorded_adjoint"] == "recorded routes require streamed statistics"
        explicit = _engine_for(cell, states, model, losses, stream_spike_statistics=True)
        with pytest.raises(ValueError, match="does not support CustomLoss"):
            explicit.prepare_gradient_checkpointing()
    finally:
        cell.close_fused_cuda()


def test_training_engine_auto_keeps_streaming_without_direct_eligibility():
    """A legacy-dynamics cell streams statistics but cannot record: auto picks
    checkpointed streaming and names the unmet direct prerequisite."""
    if not fused_cuda_available():
        pytest.skip("Authorized GPU and rebuilt CUDA operators required.")
    from bmtk.simulator.dpointnet.spike_statistics import SpikeStatisticsRunner

    cell = make_cell("legacy", selective=True, batch_size=32, fused=True, noise=True, two_inputs=True, use_fused_cuda=True)
    try:
        states = list(cell.zero_state(32, cell.compute_dtype))
        model = _model(cell, tuple(states))
        engine = _engine_for(cell, states, model, _rate_losses)
        engine.prepare_gradient_checkpointing()
        assert engine.stream_spike_statistics is True and engine.recorded_adjoint is False
        assert isinstance(engine._extractor_forward, SpikeStatisticsRunner)
        assert "NEST hard forward" in engine.training_route_report["reasons"]["recorded_adjoint"]
    finally:
        cell.close_fused_cuda()


def test_direct_record_bytes_admission():
    from bmtk.simulator.dpointnet.direct_recorded_adjoint import direct_record_bytes

    cell = SimpleNamespace(_n_neurons=1000, state_precision="selective", state_dtype=tf.float32,
                           compute_dtype=tf.float16, max_delay=2)
    fits, reason = direct_record_bytes(cell, 4, 100, 25, free_bytes=10 * 2**20)
    assert fits is True and "record payload" in reason and "60%" in reason
    fits, _ = direct_record_bytes(cell, 4, 100, 25, free_bytes=2 * 2**20)
    assert fits is False
    assert direct_record_bytes(cell, None, 100, 25, free_bytes=2**30) == (
        None, "record payload unknown (static batch, sequence length and neuron count required)")
    assert direct_record_bytes(SimpleNamespace(_n_neurons=0), 4, 100, 25, free_bytes=2**30)[0] is None
