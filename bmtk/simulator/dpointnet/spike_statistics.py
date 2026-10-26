"""Differentiable rollout statistics without a full spike-sequence allocation."""

from typing import NamedTuple

import tensorflow as tf
from tensorflow.python.eager import record

from .segmented_recompute import SegmentedRecomputeRunner
from .loss_functions.synchronization_loss import _gather_population_traces


class SampledSpikeTraces(NamedTuple):
    sample_ids: tf.Tensor
    neuron_mask: tf.Tensor
    window: tuple
    batch_range: tuple


class SpikeStatistics(NamedTuple):
    counts: tuple
    traces: tuple
    sequences: tuple
    final_state: tuple


def _window(value, length):
    if (
        not isinstance(value, (tuple, list)) or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
        or not 0 <= value[0] < value[1] <= length
    ):
        raise ValueError(f"Statistics window must satisfy 0 <= start < stop <= {length}.")
    return tuple(value)


class _StatisticsCollector:
    def __init__(self, runner, sampled_traces):
        self.runner = runner
        self.sampled_traces = tuple(sampled_traces)
        for spec in self.sampled_traces:
            _window(spec.window, runner.sequence_length)
            if (
                not isinstance(spec.batch_range, (tuple, list))
                or len(spec.batch_range) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in spec.batch_range)
                or not 0 <= spec.batch_range[0] < spec.batch_range[1]
            ):
                raise ValueError("Sampled traces require a nonempty static batch_range.")
            rows = spec.batch_range[1] - spec.batch_range[0]
            if (
                spec.sample_ids.shape.rank != 2
                or spec.neuron_mask.shape.rank != 3
                or spec.sample_ids.shape[0] != rows
                or spec.neuron_mask.shape[0] != rows
                or not spec.neuron_mask.shape.is_fully_defined()
                or spec.sample_ids.shape[1] != (
                    spec.neuron_mask.shape[1] * spec.neuron_mask.shape[2]
                )
                or not spec.sample_ids.dtype.is_integer
            ):
                raise ValueError("Sampled trace indices/mask must have matching static batch/sample shapes.")

    def initialize(self, inputs):
        width = self.runner.sequence_specs[0].shape[-1]
        for spec in self.sampled_traces:
            if inputs.shape[0] is None or spec.batch_range[1] > inputs.shape[0]:
                raise ValueError("Sampled batch_range exceeds the static rollout batch.")
        counts = tuple(
            tf.zeros([tf.shape(inputs)[0], width], tf.float32)
            for _ in self.runner.count_windows
        )
        traces = tuple(
            tf.TensorArray(
                tf.as_dtype(self.runner.sequence_specs[0].dtype), self.runner.n_chunks,
                infer_shape=self.runner.remainder_size == 0,
            )
            for _ in self.sampled_traces
        )
        sequences = tuple(
            tf.TensorArray(
                spec.dtype, self.runner.n_chunks,
                infer_shape=self.runner.remainder_size == 0,
            )
            for spec in self.runner.sequence_specs[1:]
        )
        return counts, traces, sequences

    def reduce(self, spikes, start, length):
        steps = tf.range(length, dtype=tf.int64) + tf.cast(start, tf.int64)
        counts = tuple(
            tf.reduce_sum(
                tf.cast(spikes, tf.float32)
                * tf.cast((steps >= begin) & (steps < end), tf.float32)[None, :, None],
                axis=1,
            )
            for begin, end in self.runner.count_windows
        )
        traces = []
        for spec in self.sampled_traces:
            begin, end = spec.batch_range
            mask = spec.neuron_mask
            ids = spec.sample_ids
            checks = (
                tf.debugging.assert_greater_equal(ids, tf.cast(0, ids.dtype)),
                tf.debugging.assert_less(
                    ids, tf.cast(self.runner.sequence_specs[0].shape[-1], ids.dtype)
                ),
            )
            with tf.control_dependencies(checks):
                traces.append(_gather_population_traces(
                    spikes[begin:end], ids, tf.cast(mask, spikes.dtype), end - begin,
                    length, mask.shape[1], mask.shape[2],
                ))
        return counts, tuple(traces)

    def append(self, carry, index, sequences):
        old_counts, trace_arrays, sequence_arrays = carry
        length = int(sequences[0].shape[1])
        counts, traces = self.reduce(sequences[0], index * self.runner.chunk_size, length)
        return (
            tuple(old + new for old, new in zip(old_counts, counts)),
            tuple(
                array.write(index, tf.transpose(value))
                for array, value in zip(trace_arrays, traces)
            ),
            tuple(
                array.write(index, value)
                for array, value in zip(sequence_arrays, sequences[1:])
            ),
        )

    def finish(self, carry, inputs):
        counts, traces, sequences = carry
        return (
            counts
            + tuple(
                tf.transpose(array.concat())[..., spec.window[0]:spec.window[1]]
                for array, spec in zip(traces, self.sampled_traces)
            )
            + tuple(
                self.runner._assemble_chunks(array, spec, inputs)
                for array, spec in zip(sequences, self.runner.sequence_specs[1:])
            )
        )

    def gradient_fn(self, upstream, inputs):
        count_count = len(self.runner.count_windows)
        trace_count = len(self.sampled_traces)
        count_gradients = upstream[:count_count]
        trace_gradients = tuple(
            tf.pad(value, ((0, 0), (spec.window[0], self.runner.sequence_length - spec.window[1])))
            for value, spec in zip(
                upstream[count_count:count_count + trace_count], self.sampled_traces
            )
        )
        other_gradients = upstream[count_count + trace_count:]
        width = self.runner.sequence_specs[0].shape[-1]
        dtype = tf.as_dtype(self.runner.sequence_specs[0].dtype)

        def gradients(start, length):
            spikes = tf.zeros([tf.shape(inputs)[0], length, width], dtype)
            with tf.GradientTape() as tape:
                tape.watch(spikes)
                counts, traces = self.reduce(spikes, start, length)
            spike_gradient = tape.gradient(
                counts + traces, spikes,
                output_gradients=count_gradients + tuple(
                    value[:, start:start + length] for value in trace_gradients
                ),
                unconnected_gradients=tf.UnconnectedGradients.ZERO,
            )
            return (spike_gradient,) + tuple(
                value[:, start:start + length] for value in other_gradients
            )

        return gradients


class SpikeStatisticsRunner(SegmentedRecomputeRunner):
    """Checkpoint exact BPTT while retaining counts, sampled traces and other outputs.

    The first core-model sequence must be full-population floating-point spikes.
    Other sequences and final states retain the ordinary runner's semantics.
    """

    def __init__(self, *args, count_windows=None, **kwargs):
        super().__init__(*args, **kwargs)
        spec = self.sequence_specs[0]
        if len(spec.shape) != 3 or spec.shape[-1] is None or not tf.as_dtype(spec.dtype).is_floating:
            raise ValueError("Statistics require a static-width rank-three floating spike sequence.")
        self.count_windows = tuple(
            _window(window, self.sequence_length)
            for window in (
                ((0, self.sequence_length),) if count_windows is None else count_windows
            )
        )
        if not self.count_windows:
            raise ValueError("Statistics require at least one count window.")

    def __call__(self, inputs, initial_state, sampled_traces=()):
        collector = _StatisticsCollector(self, sampled_traces)
        output_count = (
            len(self.count_windows) + len(sampled_traces) + self.n_sequence_outputs - 1
        )
        time_check = tf.debugging.assert_equal(
            tf.shape(inputs)[1], self.sequence_length,
            message="Statistics input length does not match sequence_length.",
        )

        @tf.custom_gradient
        def rollout(inputs, *initial_state):
            with record.stop_recording():
                statistics, state, boundaries = self._forward(
                    inputs, initial_state, time_check, collector=collector
                )

            def grad(*upstream, variables=None):
                variables = tuple(variables or ())
                input_gradient, state_gradients, weight_gradients = self._backward(
                    inputs, boundaries, state, (), upstream[output_count:], variables,
                    sequence_gradient_fn=collector.gradient_fn(upstream[:output_count], inputs),
                )
                argument_gradients = (input_gradient,) + state_gradients
                return (
                    (argument_gradients, list(weight_gradients))
                    if variables else argument_gradients
                )

            return statistics + state, grad

        flat = rollout(inputs, *tuple(initial_state))
        split = len(self.count_windows)
        trace_end = split + len(sampled_traces)
        return SpikeStatistics(
            tuple(flat[:split]), tuple(flat[split:trace_end]),
            tuple(flat[trace_end:output_count]), tuple(flat[output_count:]),
        )
