"""Rate-distribution target for neuron groups defined by node attributes.

Neurons are grouped by the values of node attributes (for example ``area`` and
``cell_class`` columns of the node-types table). For each group the smoothed
empirical CDF of ``log(1 + rate_Hz)`` on a fixed grid,

    F_g(t_k) = mean_{i in g} sigmoid((t_k - log1p(rate_i)) / bandwidth),

is compared with a target CDF, and the loss is

    rate_cost * sum_g weight_g * mean_k (F_g(t_k) - target_g(t_k)) ** 2.

The per-group sums are additive over neurons, and both the forward value and its
custom gradient are evaluated in bounded chunks of neurons, so the memory does
not grow with ``n_neurons * grid`` (whole-cortex networks have ~10M neurons).
Neurons whose attribute values match no target group are not constrained. Several
groups may share attribute values (for example two observations of one cell class);
they then have the same neurons and their own target CDFs and weights.

The target file is JSON::

    {"group_by": ["area", "cell_class"],
     "grid_log1p_hz": [...], "bandwidth_log1p_hz": 0.15,
     "groups": [{"values": ["left:VISp", "E"], "target_cdf": [...], "weight": 0.01,
                 "model_neurons": 1234}, ...]}

``weight`` values must be positive and sum to 1; ``model_neurons``, when given,
must equal the number of matching neurons in the network.
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..spike_summary import SpikeSummary
from . import loss_utils
from .streaming import StreamingSpikeLoss


def recurrent_network_adaptor(rnn):
    networks = list(getattr(rnn, "_recurrent_networks", {}).values())
    if len(networks) != 1 or not hasattr(networks[0], "node_attribute_values"):
        raise ValueError("Grouped rate targets require one SONATA recurrent network.")
    return networks[0]


def load_rate_cdf_targets(path):
    targets = json.loads(Path(path).read_text())
    group_by = list(targets.get("group_by", ()))
    grid = np.asarray(targets["grid_log1p_hz"], dtype=np.float64)
    bandwidth = float(targets["bandwidth_log1p_hz"])
    groups = targets["groups"]
    if not group_by or not groups:
        raise ValueError("Rate CDF targets need group_by columns and at least one group.")
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("grid_log1p_hz must be a finite, strictly increasing list.")
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth_log1p_hz must be positive.")
    keys, cdfs, weights, sizes = [], [], [], []
    for group in groups:
        values = tuple(group["values"])
        cdf = np.asarray(group["target_cdf"], dtype=np.float64)
        if len(values) != len(group_by):
            raise ValueError(f"Rate target group {values!r} must give one value per group_by column.")
        if (cdf.shape != grid.shape or not np.all(np.isfinite(cdf)) or np.any(cdf < 0) or np.any(cdf > 1)
                or np.any(np.diff(cdf) < 0)):
            raise ValueError(f"Target CDF of group {values!r} must be finite and nondecreasing in [0, 1] on the grid.")
        keys.append(values)
        cdfs.append(cdf)
        weights.append(float(group["weight"]))
        sizes.append(group.get("model_neurons"))
    weights = np.asarray(weights)
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Rate target group weights must be positive and sum to 1.")
    return group_by, grid, bandwidth, keys, np.stack(cdfs), weights, sizes


def group_membership(attributes, group_by, keys):
    """Distinct-key index of every neuron (``-1`` for no target group) and key sizes.

    Distinct keys are numbered in order of first appearance in ``keys``.
    """
    import pandas as pd

    codes, uniques = [], []
    for column in group_by:
        code, unique = pd.factorize(np.asarray(attributes[column]).astype(str))
        codes.append(code.astype(np.int64))
        uniques.append(unique)
    combined = np.zeros(len(codes[0]), dtype=np.int64)
    for code, unique in zip(codes, uniques):
        combined = combined * len(unique) + code
    present, inverse = np.unique(combined, return_inverse=True)
    lookup = {}
    for key in keys:
        lookup.setdefault(tuple(str(v) for v in key), len(lookup))
    label_of_present = np.empty(len(present), dtype=np.int32)
    for row, value in enumerate(present):
        parts = []
        for unique in reversed(uniques):
            value, code = divmod(int(value), len(unique))
            parts.append(str(unique[code]))
        label_of_present[row] = lookup.get(tuple(reversed(parts)), -1)
    labels = label_of_present[np.asarray(inverse).reshape(-1)]
    sizes = np.bincount(labels[labels >= 0], minlength=len(lookup))
    return labels, sizes


def smooth_cdf_group_sums(values, segments, n_groups, grid, bandwidth, chunk_size):
    """``[n_groups, len(grid)]`` sums of ``sigmoid((grid - x) / bandwidth)`` per group, chunked.

    ``segments`` holds each value's group (``n_groups`` for none). The custom gradient
    recomputes each chunk's features, so no ``[n_values, len(grid)]`` tensor is kept.
    """
    grid = tf.constant(grid, tf.float32)
    bandwidth = tf.constant(bandwidth, tf.float32)
    size = int(values.shape[0])
    sink = n_groups + 1

    def features(x, start):
        stop = tf.minimum(start + chunk_size, size)
        return stop, tf.sigmoid((grid[None, :] - x[start:stop, None]) / bandwidth), segments[start:stop]

    @tf.custom_gradient
    def evaluate(x):
        def body(start, total):
            stop, feature, segment = features(x, start)
            return stop, total + tf.math.unsorted_segment_sum(feature, segment, sink)

        _, total = tf.while_loop(
            lambda start, _: start < size, body,
            (tf.constant(0), tf.zeros([sink, grid.shape[0]], tf.float32)), parallel_iterations=1,
        )

        def gradient(upstream):
            padded = tf.concat([upstream, tf.zeros([1, grid.shape[0]], upstream.dtype)], axis=0)

            def reverse(start, parts):
                stop, feature, segment = features(x, start)
                credit = tf.gather(padded, segment)
                derivative = -tf.reduce_sum(credit * feature * (1.0 - feature), axis=1) / bandwidth
                return stop, parts.write(start // chunk_size, derivative)

            parts = tf.TensorArray(tf.float32, size=(size + chunk_size - 1) // chunk_size,
                                   infer_shape=False, element_shape=(None,))
            _, parts = tf.while_loop(
                lambda start, _: start < size, reverse, (tf.constant(0), parts), parallel_iterations=1,
            )
            return tf.reshape(parts.concat(), [size]) if size else tf.zeros_like(x)

        return total[:n_groups], gradient

    return evaluate(values)


class GroupedRateCDFTarget(StreamingSpikeLoss):
    """Weighted squared distance between smoothed group rate CDFs and target CDFs."""

    supports_neuron_partition = True

    def __init__(self, rnn, target_file, rate_cost=1.0, pre_delay=0, post_delay=0,
                 chunk_size=1 << 16, dtype=tf.float32, **kwargs):
        if int(chunk_size) < 1:
            raise ValueError("chunk_size must be positive.")
        self._rnn = rnn
        self._rate_cost = float(rate_cost)
        self._pre_delay = int(pre_delay or 0)
        self._post_delay = int(post_delay or 0)
        self._dt = float(rnn.dt)
        self._dtype = dtype
        self._chunk_size = int(chunk_size)
        group_by, grid, bandwidth, keys, cdfs, weights, declared = load_rate_cdf_targets(target_file)
        attributes = recurrent_network_adaptor(rnn).node_attribute_values(group_by)
        labels, key_sizes = group_membership(attributes, group_by, keys)
        distinct = {}
        for key in keys:
            distinct.setdefault(tuple(str(v) for v in key), len(distinct))
        group_key = np.array([distinct[tuple(str(v) for v in key)] for key in keys], dtype=np.int32)
        sizes = key_sizes[group_key]
        for key, size, expected in zip(keys, sizes, declared):
            if size == 0:
                raise ValueError(f"No neuron matches rate target group {key!r}.")
            if expected is not None and int(expected) != int(size):
                raise ValueError(f"Rate target group {key!r} declares {expected} neurons; network has {size}.")
        self.group_by, self.keys = group_by, keys
        self.n_groups = len(keys)
        self._n_keys = len(distinct)
        self._group_key = tf.constant(group_key)
        self._grid, self._bandwidth = grid.astype(np.float32), bandwidth
        self._segments = tf.constant(np.where(labels >= 0, labels, self._n_keys).astype(np.int32))
        self._sizes = tf.constant(sizes.astype(np.float32))
        self._target = tf.constant(cdfs.astype(np.float32))
        self._weights = tf.constant(weights.astype(np.float32))
        self.group_sizes = sizes

    @staticmethod
    def module():
        return "GroupedRateCDFTarget"

    def group_losses(self, rates_hz):
        """Per-group mean squared CDF distance for rates (Hz) of every neuron."""
        rates_hz = tf.cast(rates_hz, tf.float32)
        sums = smooth_cdf_group_sums(
            tf.math.log1p(rates_hz), self._segments, self._n_keys, self._grid, self._bandwidth,
            self._chunk_size,
        )
        sums = tf.gather(sums, self._group_key)
        return tf.reduce_mean(tf.square(sums / self._sizes[:, None] - self._target), axis=1)

    def __call__(self, spikes, trim=True, **kwargs):
        spikes = loss_utils.spike_trimming(
            spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim,
        )
        counts = loss_utils.temporal_sum(spikes, dtype=tf.float32)
        steps = spikes.shape[1] if isinstance(spikes, SpikeSummary) else tf.shape(spikes)[1]
        seconds = tf.cast(steps, tf.float32) * self._dt / 1000.0
        rates = tf.reduce_mean(counts, axis=0) / seconds
        loss = tf.reduce_sum(self.group_losses(rates) * self._weights)
        return tf.cast(loss * self._rate_cost, self._dtype)
