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
