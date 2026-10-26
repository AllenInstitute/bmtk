"""Typed, window-aware input to losses that only consume spike statistics."""

from typing import Tuple

import tensorflow as tf


class SpikeSummary(tf.experimental.ExtensionType):
    counts: Tuple[tf.Tensor, ...]
    windows: Tuple[Tuple[int, int], ...]
    sequence_length: int
    traces: Tuple[tf.Tensor, ...] = ()
    trace_keys: Tuple[Tuple[str, int, int], ...] = ()
    active_window: Tuple[int, int] = (0, 0)

    def __validate__(self):
        if not self.counts or len(self.counts) != len(self.windows):
            raise ValueError("Spike summaries require one count tensor per window.")
        if len(set(self.windows)) != len(self.windows) or any(
            not 0 <= start < stop <= self.sequence_length for start, stop in self.windows
        ):
            raise ValueError("Spike summary windows must be unique, nonempty and within the sequence.")
        if self.window not in self.windows:
            raise ValueError("Spike summaries require counts for their active window.")
        if any(value.shape.rank != 2 or value.dtype != tf.float32 for value in self.counts):
            raise ValueError("Spike summary counts must be rank-two FP32 tensors.")
        if len(self.traces) != len(self.trace_keys):
            raise ValueError("Spike summary traces require matching sampling keys.")

    @property
    def window(self):
        return self.active_window if self.active_window != (0, 0) else (0, self.sequence_length)

    @property
    def shape(self):
        return tf.TensorShape((
            self.counts[0].shape[0], self.window[1] - self.window[0],
            self.counts[0].shape[-1],
        ))

    def temporal_sum(self, dtype=tf.float32):
        if self.window not in self.windows:
            raise ValueError(f"Spike statistics do not contain the requested window {self.window}.")
        return tf.cast(self.counts[self.windows.index(self.window)], dtype)

    def trim(self, pre=0, post=0):
        start, stop = self.window
        window = (start + (pre or 0), stop - (post or 0))
        if window not in self.windows:
            raise ValueError(f"Spike statistics do not contain the requested window {window}.")
        return SpikeSummary(
            self.counts, self.windows, self.sequence_length, self.traces,
            self.trace_keys, window,
        )

    def trace_for_loss(self, key):
        matches = [
            value for value, (name, start, stop) in zip(self.traces, self.trace_keys)
            if name == key and start == 0 and stop == self.shape[0]
        ]
        if len(matches) != 1:
            raise ValueError(f"Spike statistics need exactly one complete batch trace for loss {key}.")
        return matches[0]

    def stop_gradient(self):
        return SpikeSummary(
            tuple(tf.stop_gradient(value) for value in self.counts),
            self.windows, self.sequence_length,
            tuple(tf.stop_gradient(value) for value in self.traces),
            self.trace_keys, self.active_window,
        )

    def __getitem__(self, batch_slice):
        if not isinstance(batch_slice, slice) or batch_slice.step is not None:
            raise ValueError("Spike summaries support contiguous batch slicing only.")
        size = self.shape[0]
        if size is None:
            raise ValueError("Spike summary slicing needs a static batch size.")
        start, stop, _ = batch_slice.indices(size)
        selected = [
            (value, (name, first - start, last - start))
            for value, (name, first, last) in zip(self.traces, self.trace_keys)
            if first >= start and last <= stop
        ]
        return SpikeSummary(
            tuple(value[start:stop] for value in self.counts), self.windows,
            self.sequence_length, tuple(value for value, _ in selected),
            tuple(key for _, key in selected), self.active_window,
        )
