import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.segmented_recompute import SegmentedRecomputeRunner
from bmtk.simulator.dpointnet.spike_statistics import (
    SampledSpikeTraces, SpikeStatisticsRunner,
)
from bmtk.simulator.dpointnet.loss_functions.synchronization_loss import (
    _gather_population_traces,
)


class _Recurrence(tf.keras.layers.Layer):
    def build(self, shape):
        self.weight = self.add_weight(shape=(3,), initializer="ones")

    def call(self, values):
        inputs, initial = values
        states = tf.scan(
            lambda state, value: state * 0.5 + value * self.weight,
            tf.transpose(inputs, (1, 0, 2)), initializer=initial,
        )
        spikes = tf.transpose(states, (1, 0, 2))
        return spikes, tf.reduce_sum(tf.square(spikes), axis=-1), states[-1]


def _model():
    inputs = tf.keras.Input((None, 3))
    initial = tf.keras.Input((3,))
    return tf.keras.Model((inputs, initial), _Recurrence()((inputs, initial)))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 7])
@pytest.mark.parametrize("with_traces", [False, True])
def test_statistics_values_and_all_vjps_match_dense_rollout(graph, chunk, with_traces):
    model = _model()
    kwargs = dict(sequence_length=7, chunk_size=chunk, n_sequence_outputs=2,
                  differentiate_inputs=True)
    dense = SegmentedRecomputeRunner(model, **kwargs)
    windows = ((0, 7), (2, 6))
    streamed = SpikeStatisticsRunner(model, count_windows=windows, **kwargs)
    traces = (SampledSpikeTraces(
        tf.constant([[0, 0, 1, 2], [2, 1, 1, 0]], tf.int64),
        tf.constant([[[1, 1], [1, 0]], [[1, 1], [0, 1]]], tf.float32),
        (1, 6), (1, 3),
    ),) if with_traces else ()
    inputs = tf.reshape(tf.cast(tf.range(63) % 5, tf.float32), (3, 7, 3)) * 0.25
    initial = tf.ones((3, 3), tf.float32)
    targets = (inputs, initial, *model.trainable_variables)

    def evaluate():
        comparisons = []
        for summary in (False, True):
            with tf.GradientTape() as tape:
                tape.watch((inputs, initial))
                if summary:
                    output = streamed(inputs, (initial,), sampled_traces=traces)
                    values = output.counts + output.traces + output.sequences + output.final_state
                else:
                    spikes, penalty, final = dense(inputs, (initial,))
                    counts = tuple(
                        tf.reduce_sum(tf.cast(spikes[:, start:stop], tf.float32), axis=1)
                        for start, stop in windows
                    )
                    sampled = tuple(
                        _gather_population_traces(
                            spikes[spec.batch_range[0]:spec.batch_range[1]],
                            spec.sample_ids, spec.neuron_mask, 2, 7, 2, 2,
                        )[:, spec.window[0]:spec.window[1]] for spec in traces
                    )
                    values = counts + sampled + (penalty, final)
                loss = tf.add_n([
                    tf.reduce_sum(tf.square(value) * (index + 1))
                    for index, value in enumerate(values)
                ])
            comparisons.append((values, tape.gradient(loss, targets)))
        return comparisons

    reference, actual = tf.function(evaluate)() if graph else evaluate()
    for expected, observed in zip(tf.nest.flatten(reference), tf.nest.flatten(actual)):
        np.testing.assert_allclose(observed.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("window", [(-1, 3), (3, 3), (1, 8), (0, True), (0, 2.5)])
def test_invalid_count_windows_raise(window):
    with pytest.raises(ValueError, match="Statistics window"):
        SpikeStatisticsRunner(
            _model(), 7, 3, 2, count_windows=(window,),
        )


def test_wrong_sequence_length_and_trace_indices_raise():
    runner = SpikeStatisticsRunner(_model(), 7, 3, 2)
    with pytest.raises(tf.errors.InvalidArgumentError, match="Statistics input length"):
        runner(tf.zeros((2, 6, 3)), (tf.zeros((2, 3)),))
    bad = SampledSpikeTraces(
        tf.constant([[4], [0]], tf.int64), tf.ones((2, 1, 1)), (0, 7), (0, 2)
    )
    with pytest.raises(tf.errors.InvalidArgumentError):
        runner(tf.zeros((2, 7, 3)), (tf.zeros((2, 3)),), sampled_traces=(bad,))


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("chunk", [1, 3, 7])
def test_projected_traces_and_vjps_match_dense_projection(graph, chunk):
    from bmtk.simulator.dpointnet.spike_statistics import ProjectedSpikeTraces

    model = _model()
    kwargs = dict(sequence_length=7, chunk_size=chunk, n_sequence_outputs=2,
                  differentiate_inputs=True)
    dense = SegmentedRecomputeRunner(model, **kwargs)
    streamed = SpikeStatisticsRunner(model, count_windows=((0, 7),), **kwargs)
    projection = tf.constant([[0.5, -1.0], [2.0, 0.25], [-0.75, 1.5]], tf.float32)
    spec = ProjectedSpikeTraces(projection, (2, 7), (1, 3))
    inputs = tf.reshape(tf.cast(tf.range(63) % 5, tf.float32), (3, 7, 3)) * 0.25
    initial = tf.ones((3, 3), tf.float32)
    targets = (inputs, initial, *model.trainable_variables)
    weights = tf.reshape(tf.range(1, 21, dtype=tf.float32), (4, 5)) / 7.0

    def evaluate():
        results = []
        for summary in (False, True):
            with tf.GradientTape() as tape:
                tape.watch((inputs, initial))
                if summary:
                    trace = streamed(inputs, (initial,), sampled_traces=(spec,)).traces[0]
                else:
                    spikes, _, _ = dense(inputs, (initial,))
                    trace = tf.reshape(
                        tf.einsum("btn,nf->bft", spikes[1:3], projection), (4, 7)
                    )[:, 2:7]
                loss = tf.reduce_sum(tf.square(trace) * weights)
            results.append((trace, tape.gradient(loss, targets)))
        return results

    reference, actual = tf.function(evaluate)() if graph else evaluate()
    for expected, observed in zip(tf.nest.flatten(reference), tf.nest.flatten(actual)):
        np.testing.assert_allclose(observed.numpy(), expected.numpy(), rtol=2e-6, atol=2e-6)


def test_projected_trace_projection_shape_is_validated():
    from bmtk.simulator.dpointnet.spike_statistics import ProjectedSpikeTraces

    runner = SpikeStatisticsRunner(_model(), 7, 3, 2)
    bad = ProjectedSpikeTraces(tf.ones((4, 2), tf.float32), (0, 7), (0, 2))
    with pytest.raises(ValueError, match="Projected traces"):
        runner(tf.zeros((2, 7, 3)), (tf.zeros((2, 3)),), sampled_traces=(bad,))


def _tf32_round(value):
    """Round-to-nearest TF32 emulation (10 explicit mantissa bits), as tensor cores do."""
    bits = tf.bitcast(value, tf.int32)
    rounded = tf.bitwise.bitwise_and(bits + 0x1000, tf.constant(-8192, tf.int32))
    return tf.bitcast(rounded, tf.float32)


def _tf32_matmul(a, b, transpose_b=False):
    return tf.matmul(_tf32_round(a), _tf32_round(b), transpose_b=transpose_b)


def test_tf32_parts_are_exact_and_representable():
    from bmtk.simulator.dpointnet.spike_statistics import _tf32_part

    value = tf.constant(np.random.default_rng(1).normal(size=1000) * 10.0 ** np.arange(-3, 3).repeat(167)[:1000], tf.float32)
    part = _tf32_part(value)
    assert np.all(tf.bitcast(part, tf.int32).numpy() & 0x1FFF == 0)
    np.testing.assert_array_equal((part + (value - part)).numpy(), value.numpy())
    np.testing.assert_array_equal(_tf32_round(part).numpy(), part.numpy())


@pytest.mark.parametrize("transpose_b", [False, True])
def test_strict_fp32_matmul_recovers_fp32_accuracy_under_tf32(transpose_b):
    """Emulated TF32 loses ~1e-3 relative accuracy; the split products do not."""
    from bmtk.simulator.dpointnet.spike_statistics import strict_fp32_matmul

    rng = np.random.default_rng(7)
    spikes = rng.integers(0, 3, (40, 300)).astype(np.float32)
    projection = rng.normal(size=(300, 5)).astype(np.float32)
    columns = rng.normal(size=(40, 5)).astype(np.float32)
    exact = spikes.astype(np.float64) @ projection.astype(np.float64)
    split = strict_fp32_matmul(tf.constant(spikes), tf.constant(projection), exact_a=True,
                               force_split=True, _matmul=_tf32_matmul).numpy()
    plain = _tf32_matmul(tf.constant(spikes), tf.constant(projection)).numpy()
    scale = np.abs(exact).max()
    assert np.abs(plain - exact).max() / scale > 1e-5
    assert np.abs(split - exact).max() / scale < 2e-6
    # General operands (the spike-credit VJP): a1 b1 + a1 b2 + a2 b1.
    credit_exact = columns.astype(np.float64) @ projection.T.astype(np.float64)
    credit = strict_fp32_matmul(tf.constant(columns), tf.constant(projection if transpose_b else projection.T.copy()),
                                transpose_b=transpose_b, force_split=True, _matmul=_tf32_matmul).numpy()
    assert np.abs(credit - credit_exact).max() / np.abs(credit_exact).max() < 2e-6


def test_strict_fp32_matmul_gradients_are_fp32_accurate():
    from bmtk.simulator.dpointnet.spike_statistics import strict_fp32_matmul

    rng = np.random.default_rng(3)
    spikes = tf.constant(rng.integers(0, 2, (16, 200)).astype(np.float32))
    projection = tf.constant(rng.normal(size=(200, 4)).astype(np.float32))
    upstream = rng.normal(size=(16, 4)).astype(np.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch((spikes, projection))
        value = strict_fp32_matmul(spikes, projection, exact_a=True, force_split=True, _matmul=_tf32_matmul)
        loss = tf.reduce_sum(value * upstream)
    grad_spikes, grad_projection = tape.gradient(loss, (spikes, projection))
    expect_spikes = upstream.astype(np.float64) @ projection.numpy().astype(np.float64).T
    expect_projection = spikes.numpy().astype(np.float64).T @ upstream.astype(np.float64)
    for got, want in ((grad_spikes, expect_spikes), (grad_projection, expect_projection)):
        assert np.abs(got.numpy() - want).max() / np.abs(want).max() < 2e-6


def test_strict_fp32_matmul_is_a_single_matmul_without_tf32(monkeypatch):
    from bmtk.simulator.dpointnet import spike_statistics

    monkeypatch.setattr(spike_statistics, "_tf32_may_apply", lambda: False)
    calls = []
    def counting(a, b, transpose_b=False):
        calls.append(1)
        return tf.matmul(a, b, transpose_b=transpose_b)
    a, b = tf.ones((3, 4)), tf.ones((4, 2))
    np.testing.assert_array_equal(
        spike_statistics.strict_fp32_matmul(a, b, exact_a=True, _matmul=counting).numpy(), tf.matmul(a, b).numpy(),
    )
    assert calls == [1]
