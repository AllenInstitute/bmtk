import json
from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import GroupedRateCDFTarget
from bmtk.simulator.dpointnet.loss_functions.grouped_rate_cdf import (
    group_membership, load_rate_cdf_targets, smooth_cdf_group_sums,
)

GRID = np.linspace(0.0, 4.0, 9)
BANDWIDTH = 0.15


def _dense_sums(x, segments, n_groups):
    feature = tf.sigmoid((tf.constant(GRID, tf.float32)[None, :] - x[:, None]) / BANDWIDTH)
    return tf.math.unsorted_segment_sum(feature, segments, n_groups + 1)[:n_groups]


@pytest.mark.parametrize("chunk", [1, 7, 64, 1000])
def test_chunked_sums_and_gradient_match_dense(chunk):
    rng = np.random.default_rng(0)
    x = tf.constant(rng.uniform(0, 4, 50).astype(np.float32))
    segments = tf.constant(rng.integers(0, 4, 50).astype(np.int32))  # 3 = no group
    upstream = tf.constant(rng.normal(size=(3, len(GRID))).astype(np.float32))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        chunked = smooth_cdf_group_sums(x, segments, 3, GRID, BANDWIDTH, chunk)
        dense = _dense_sums(x, segments, 3)
        a, b = tf.reduce_sum(chunked * upstream), tf.reduce_sum(dense * upstream)
    np.testing.assert_allclose(chunked.numpy(), dense.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tape.gradient(a, x).numpy(), tape.gradient(b, x).numpy(), rtol=1e-4, atol=1e-5)
    ungrouped = segments.numpy() == 3
    assert np.all(tape.gradient(a, x).numpy()[ungrouped] == 0)


def test_group_membership_maps_attribute_combinations():
    attributes = {"area": np.array(["A", "B", "A", "C", "B"]), "cls": np.array(["E", "E", "PV", "E", "E"])}
    labels, sizes = group_membership(attributes, ["area", "cls"], [("B", "E"), ("A", "PV"), ("A", "E")])
    np.testing.assert_array_equal(labels, [2, 0, 1, -1, 0])
    np.testing.assert_array_equal(sizes, [2, 1, 1])


def _targets(path, groups, **extra):
    data = {"group_by": ["area"], "grid_log1p_hz": GRID.tolist(), "bandwidth_log1p_hz": BANDWIDTH,
            "groups": groups, **extra}
    path.write_text(json.dumps(data))
    return str(path)


def _cdf(seed):
    return np.sort(np.random.default_rng(seed).uniform(size=len(GRID))).tolist()


@pytest.mark.parametrize("groups, message", [
    ([{"values": ["A"], "target_cdf": _cdf(1), "weight": 0.5}], "sum to 1"),
    ([{"values": ["A"], "target_cdf": _cdf(1)[::-1], "weight": 1.0}], "nondecreasing"),
    ([{"values": ["A", "E"], "target_cdf": _cdf(1), "weight": 1.0}], "one value per group_by"),
])
def test_target_file_validation(tmp_path, groups, message):
    with pytest.raises(ValueError, match=message):
        load_rate_cdf_targets(_targets(tmp_path / "t.json", groups))


class _Adaptor:
    def __init__(self, areas):
        self.areas = np.asarray(areas)

    def node_attribute_values(self, columns):
        return {"area": self.areas}


def _loss(tmp_path, areas, groups, **kwargs):
    rnn = SimpleNamespace(dt=1.0, _recurrent_networks={"cells": _Adaptor(areas)})
    return GroupedRateCDFTarget(rnn, _targets(tmp_path / "t.json", groups), **kwargs)


def test_loss_matches_dense_reference_with_burn_in(tmp_path):
    rng = np.random.default_rng(3)
    areas = rng.choice(["A", "B", "Z"], 40)
    groups = [{"values": ["A"], "target_cdf": _cdf(4), "weight": 0.25},
              {"values": ["B"], "target_cdf": _cdf(5), "weight": 0.75}]
    loss = _loss(tmp_path, areas, groups, pre_delay=20, rate_cost=2.0, chunk_size=6)
    spikes = tf.constant((rng.uniform(size=(2, 120, 40)) < 0.03).astype(np.float32))
    rates = spikes[:, 20:].numpy().sum(axis=1).mean(axis=0) / 0.1
    expected = 0.0
    for group in groups:
        mask = areas == group["values"][0]
        cdf = (1 / (1 + np.exp(-(GRID[None, :] - np.log1p(rates[mask])[:, None]) / BANDWIDTH))).mean(axis=0)
        expected += group["weight"] * np.mean((cdf - np.asarray(group["target_cdf"])) ** 2)
    np.testing.assert_allclose(float(loss(spikes)), 2.0 * expected, rtol=1e-5)
    np.testing.assert_array_equal(loss.group_sizes, [np.sum(areas == "A"), np.sum(areas == "B")])


def test_groups_sharing_attribute_values(tmp_path):
    rng = np.random.default_rng(6)
    areas = rng.choice(["A", "B"], 30)
    groups = [{"values": ["A"], "target_cdf": _cdf(7), "weight": 0.5},
              {"values": ["B"], "target_cdf": _cdf(8), "weight": 0.2},
              {"values": ["A"], "target_cdf": _cdf(9), "weight": 0.3}]
    loss = _loss(tmp_path, areas, groups)
    rates = tf.constant(rng.uniform(0, 20, 30).astype(np.float32))
    per_group = loss.group_losses(rates).numpy()
    x = np.log1p(rates.numpy())
    for group, value in zip(groups, per_group):
        mask = areas == group["values"][0]
        cdf = (1 / (1 + np.exp(-(GRID[None, :] - x[mask][:, None]) / BANDWIDTH))).mean(axis=0)
        np.testing.assert_allclose(value, np.mean((cdf - np.asarray(group["target_cdf"])) ** 2), rtol=1e-5)
    np.testing.assert_array_equal(loss.group_sizes, [np.sum(areas == "A"), np.sum(areas == "B"), np.sum(areas == "A")])


def test_declared_group_size_must_match(tmp_path):
    groups = [{"values": ["A"], "target_cdf": _cdf(4), "weight": 1.0, "model_neurons": 3}]
    with pytest.raises(ValueError, match="declares 3 neurons"):
        _loss(tmp_path, ["A", "A", "B"], groups)
    with pytest.raises(ValueError, match="No neuron matches"):
        _loss(tmp_path, ["B"], [{"values": ["A"], "target_cdf": _cdf(4), "weight": 1.0}])
