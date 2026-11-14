"""Pinned-host offload of the direct recorded adjoint's records.

The store moves bytes only, so every offload combination must reproduce the
resident direct route: forward values and state cotangents bitwise, masters
within the FP32 accumulation-order tolerance the resident route itself shows
run to run (native atomic sums).
"""

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.custom_ops import fused_cuda_available
from bmtk.simulator.dpointnet.custom_ops.record_store_ops import (
    HostRecordStore, host_record_store_available,
)
from bmtk.simulator.dpointnet.direct_recorded_adjoint import (
    DirectRecordedAdjointRunner, RECORD_OFFLOAD_CHOICES, direct_record_bytes, normalize_record_offload,
)
from bmtk.simulator.dpointnet.rnn_model import training_route_options
from bmtk.simulator.dpointnet.spike_statistics import SampledSpikeTraces
from test_direct_recorded_adjoint import DIRECT_OPTIONS, _engine_for, _initial_state, _model, _rate_losses
from test_precision_credit import make_cell

OFFLOADS = (("voltage",), ("history",), ("voltage", "history"))


@pytest.fixture(autouse=True)
def restore_policy():
    previous = tf.keras.mixed_precision.global_policy()
    yield
    tf.keras.mixed_precision.set_global_policy(previous)


def _gpu_required():
    if not fused_cuda_available() or not host_record_store_available():
        pytest.skip("Authorized GPU, rebuilt CUDA operators and the record store library required.")


@pytest.mark.parametrize("value,expected", [
    (None, ()), (False, ()), ("none", ()), ((), ()), ([], ()),
    (True, RECORD_OFFLOAD_CHOICES), ("all", RECORD_OFFLOAD_CHOICES),
    ("voltage", ("voltage",)), (["history"], ("history",)),
    (("history", "voltage"), ("voltage", "history")),
])
def test_normalize_record_offload(value, expected):
    assert normalize_record_offload(value) == expected


@pytest.mark.parametrize("value", ["voltages", ["voltage", "words"], 3, [None]])
def test_normalize_record_offload_rejects_unknown_surfaces(value):
    with pytest.raises((ValueError, TypeError)):
        normalize_record_offload(value)


def test_training_route_options_pass_only_named_keys():
    assert training_route_options({"n_epochs": 1}) == {}
    assert training_route_options({
        "recorded_adjoint": False, "record_offload": ["voltage"], "stream_spike_statistics": "auto", "x": 1,
    }) == {"recorded_adjoint": False, "record_offload": ["voltage"], "stream_spike_statistics": "auto"}


def test_direct_record_bytes_accounts_for_offload():
    from types import SimpleNamespace

    cell = SimpleNamespace(_n_neurons=200000, state_precision="selective", state_dtype=tf.float32,
                           compute_dtype=tf.float16, max_delay=2)
    free = 100 * 2**30
    fits, reason = direct_record_bytes(cell, 32, 500, 25, free_bytes=free)
    _, voltage_reason = direct_record_bytes(cell, 32, 500, 25, free_bytes=free, offload="voltage")
    _, both_reason = direct_record_bytes(cell, 32, 500, 25, free_bytes=free, offload=True)
    payload = float(reason.split()[2])
    voltage_payload = float(voltage_reason.split()[2])
    both_payload = float(both_reason.split()[2])
    assert fits is True and payload > voltage_payload > both_payload > 0
    assert "pinned host memory for voltage offload" in voltage_reason
    assert "pinned host memory for voltage, history offload" in both_reason
    assert "pinned" not in reason


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
def test_host_record_store_round_trips_frames_in_reverse(dtype):
    """Offloaded frames return bitwise, in reverse order with prefetch, across
    repeated rollouts and repeated takes; never-offloaded frames raise."""
    _gpu_required()
    frames, batch, width, slots, prefetch = 37, 4, 1000, 8, 3
    rng = np.random.default_rng(0)
    values = rng.standard_normal((frames, batch, width))
    values = values.astype(dtype.as_numpy_dtype)
    store = HostRecordStore(frames, (batch, width), dtype, prefetch=prefetch, staging_slots=slots)
    with tf.device("/GPU:0"):
        staging = tf.Variable(tf.zeros([slots, batch, width], dtype), trainable=False)
        data = tf.constant(values)

    @tf.function
    def forward():
        def body(time, token):
            with tf.control_dependencies([tf.raw_ops.ResourceScatterUpdate(
                resource=staging.handle, indices=tf.cast(time % slots, tf.int32)[None], updates=data[time][None],
            )]):
                token = store.offload(staging, time % slots, time, after=token)
            return time + 1, token

        return tf.while_loop(lambda t, *_: t < frames, body, (tf.constant(0), tf.constant(0)), parallel_iterations=1)[1]

    @tf.function
    def backward():
        primed = [store.prefetch(frames - 1, after=tf.constant(0))]
        for k in range(1, prefetch):
            primed.append(store.prefetch(frames - 1 - k, after=primed[-1]))
        out = tf.TensorArray(dtype, size=frames, element_shape=(batch, width))

        def body(time, out):
            with tf.control_dependencies(primed):
                frame = store.take(time, time - prefetch)
            return time - 1, out.write(time, frame)

        _, out = tf.while_loop(lambda t, *_: t >= 0, body, (tf.constant(frames - 1), out), parallel_iterations=1)
        return out.stack(), store.stats()

    for _ in range(3):
        forward()
        got, stats = backward()
        np.testing.assert_array_equal(got.numpy(), values)
        assert stats.numpy()[0] == frames and stats.numpy()[1] == batch * width * dtype.size
        assert stats.numpy()[2] == frames * batch * width * dtype.size and stats.numpy()[3] == frames
        assert stats.numpy()[4] <= prefetch + 2
    np.testing.assert_array_equal(store.take(5).numpy(), values[5])
    np.testing.assert_array_equal(store.take(5).numpy(), values[5])
    empty = HostRecordStore(3, (batch, width), dtype, name="dpointnet_test_empty_store_" + dtype.name)
    with pytest.raises(tf.errors.FailedPreconditionError, match="never offloaded"):
        empty.take(1)
    with pytest.raises(tf.errors.InvalidArgumentError, match="outside"):
        store.take(frames)


def _compare(runners, cell, initial, model, length, traces):
    inputs = tf.zeros((int(initial[0].shape[0]), length, 0), cell.compute_dtype)
    floating = tuple(value for value in initial if value.dtype.is_floating)

    @tf.function
    def evaluate():
        comparisons = []
        for runner in runners:
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
                loss, (*floating, *model.trainable_variables), unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            comparisons.append((values, gradients))
        return comparisons

    results = evaluate()
    n_exact = len(results[0][0]) + len(floating)
    return results, n_exact


@pytest.mark.parametrize("selective", [False, True])
@pytest.mark.parametrize("mode", ["exact", "soft_surrogate"])
@pytest.mark.parametrize("batch,windows,length,chunk", [
    (4, ((0, 7),), 7, 3), (32, ((0, 7), (1, 6)), 7, 3), (4, ((0, 13), (2, 9)), 13, 5), (4, ((0, 12),), 12, 4),
])
def test_offload_matches_resident_direct_route(selective, mode, batch, windows, length, chunk):
    """Every offload combination reproduces the resident direct route: values and
    state cotangents bitwise, masters within the resident route's own run-to-run
    FP32 atomic-accumulation spread."""
    _gpu_required()
    cell = make_cell("nest", selective=selective, batch_size=batch, hard_reset_gradient_mode=mode, **DIRECT_OPTIONS)
    try:
        initial = _initial_state(cell, batch)
        model = _model(cell, initial)
        options = dict(sequence_length=length, chunk_size=chunk, differentiate_inputs=False, count_windows=windows)
        runners = [DirectRecordedAdjointRunner(model, cell, record_offload=offload, **options)
                   for offload in ((),) + OFFLOADS]
        traces = (
            SampledSpikeTraces(tf.constant([[0, 0], [1, 0]], tf.int64), tf.ones((2, 1, 2), cell.compute_dtype),
                               (1, length), (0, 2)),
            SampledSpikeTraces(tf.constant([[1, 1]], tf.int64), tf.constant([[[1, 0]]], cell.compute_dtype),
                               (0, 4), (2, 3)),
        )
        results, n_exact = _compare(runners, cell, initial, model, length, traces)
        reference = tf.nest.flatten(results[0])
        for offload, observed in zip(OFFLOADS, results[1:]):
            for index, (expected, value) in enumerate(zip(reference, tf.nest.flatten(observed))):
                if index < n_exact:
                    np.testing.assert_array_equal(value.numpy(), expected.numpy(), err_msg=f"{offload} tensor {index}")
                else:
                    np.testing.assert_allclose(value, expected, rtol=1e-6, atol=0, err_msg=f"{offload} master {index}")
        assert all(np.any(value.numpy() != 0) for value in results[-1][1][-3:])
        report = cell.acceleration_report["record_offload"]
        assert report["surfaces"] == ("voltage", "history")
        stores = report["stores"]
        assert stores["voltage"]["frames"] == length and stores["history"]["frames"] == -(-length // chunk) + 1
        assert stores["asc"]["frames"] == -(-length // chunk)
        assert stores["asc"]["frame_bytes"] == batch * 2 * cell._n_neurons * tf.as_dtype(
            cell.state_dtype if selective else cell.compute_dtype).size
        state_bytes = tf.as_dtype(cell.state_dtype if selective else cell.compute_dtype).size
        assert stores["voltage"]["host_bytes"] == length * batch * cell._n_neurons * state_bytes
    finally:
        cell.close_fused_cuda()


def test_offload_runner_validation():
    _gpu_required()
    cell = make_cell("nest", selective=True, batch_size=4, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        initial = _initial_state(cell, 4)
        model = _model(cell, initial)
        assert cell.max_delay > 1
        with pytest.raises(ValueError, match="must not exceed chunk_size"):
            DirectRecordedAdjointRunner(model, cell, sequence_length=7, chunk_size=1, record_offload="history")
        runner = DirectRecordedAdjointRunner(model, cell, sequence_length=7, chunk_size=1, record_offload="voltage")
        assert runner.record_offload == ("voltage",)
        with pytest.raises(ValueError, match="record_offload accepts"):
            DirectRecordedAdjointRunner(model, cell, sequence_length=7, chunk_size=3, record_offload="words")
    finally:
        cell.close_fused_cuda()


def test_training_engine_record_offload_requires_direct_route():
    """An offload request never falls back silently: explicit non-direct requests
    fail at construction, an automatic non-direct resolution at preparation."""
    _gpu_required()
    from bmtk.simulator.dpointnet.training import TrainingEngine
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="requires the direct recorded adjoint"):
        TrainingEngine(SimpleNamespace(cell=None), 1, 1, stream_spike_statistics=True, recorded_adjoint=True,
                       record_offload="voltage")
    with pytest.raises(ValueError, match="requires the direct recorded adjoint"):
        TrainingEngine(SimpleNamespace(cell=None), 1, 1, recorded_adjoint=False, record_offload=["history"])

    class CustomLoss:
        def __call__(self, spikes, **kwargs):
            return tf.reduce_mean(tf.cast(spikes, tf.float32))

    cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
    try:
        states = list(cell.zero_state(32, cell.compute_dtype))
        model = _model(cell, tuple(states))
        engine = _engine_for(cell, states, model, lambda rnn: {**_rate_losses(rnn), "custom": CustomLoss()},
                             record_offload="all")
        with pytest.raises(ValueError, match="resolved recorded_adjoint=False"):
            engine.prepare_gradient_checkpointing()
    finally:
        cell.close_fused_cuda()


def test_training_engine_record_offload_matches_resident_update():
    _gpu_required()
    results = []
    for offload in ((), ("voltage", "history")):
        tf.keras.utils.set_random_seed(17)
        cell = make_cell("nest", selective=True, batch_size=32, hard_reset_gradient_mode="soft_surrogate", **DIRECT_OPTIONS)
        try:
            states = list(cell.zero_state(32, cell.compute_dtype))
            states[1] = tf.tile(tf.cast([[0.9, 1.1]], states[1].dtype), [32, 1])
            model = _model(cell, tuple(states))
            engine = _engine_for(cell, states, model, _rate_losses, record_offload=offload)
            engine.prepare_gradient_checkpointing()
            assert engine.recorded_adjoint == "direct"
            assert engine.training_route_report["record_offload"] == offload
            assert engine._extractor_forward.record_offload == offload
            values = tf.zeros((32, 7, 0), cell.compute_dtype)
            losses, gradients, _ = tf.function(lambda: engine._single_gradients((values,), {}, tuple(states)))()
            assert all(np.any(gradient.numpy() != 0) for gradient in gradients)
            engine._apply_gradients(gradients)
            results.append((losses, tuple(tf.identity(v) for v in model.trainable_variables)))
        finally:
            cell.close_fused_cuda()
    for expected, actual in zip(tf.nest.flatten(results[0]), tf.nest.flatten(results[1])):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=0)
