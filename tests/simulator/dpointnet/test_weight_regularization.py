"""Tests for the EMD recurrent-weight regularizer (loss_functions.weight_regularization)."""
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import weight_regularization as wr
from bmtk.simulator.dpointnet.loss_functions import EMDWeightRegularization


def test_build_group_order():
    group_ids = np.array([2, 0, 1, 0, 2, 0], dtype=np.int64)
    order, row_splits = wr._build_group_order(group_ids, 3)
    # row_splits are cumulative counts per group (0:3, 1:1, 2:2).
    assert row_splits.tolist() == [0, 3, 4, 6]
    # each segment must reference only members of its group, stably ordered.
    assert sorted(order[0:3].tolist()) == [1, 3, 5]  # group 0
    assert order[3:4].tolist() == [2]                # group 1
    assert sorted(order[4:6].tolist()) == [0, 4]     # group 2


def test_sort_initial_values_by_group():
    vals = np.array([5.0, 9.0, 7.0, 1.0, 6.0, 3.0], dtype=np.float32)
    group_ids = np.array([2, 0, 1, 0, 2, 0], dtype=np.int64)
    order, row_splits = wr._build_group_order(group_ids, 3)
    out = wr._sort_initial_values_by_group(vals, order, row_splits)
    # group 0 members {9,1,3} sorted -> [1,3,9]; group 1 {7}; group 2 {5,6} -> [5,6]
    assert out[0:3].tolist() == [1.0, 3.0, 9.0]
    assert out[3:4].tolist() == [7.0]
    assert out[4:6].tolist() == [5.0, 6.0]


class _FakeCell:
    def __init__(self, weights):
        self.recurrent_weight_values = tf.Variable(weights, dtype=tf.float32)


class _FakeRNN:
    """Minimal stand-in exposing the attributes EMDWeightRegularization reads."""
    def __init__(self, weights, group_ids):
        self.cell = _FakeCell(weights)
        n = len(weights)
        self.recurrent_network = {
            'n_nodes': n,
            'synapses': {'indices': np.zeros((n, 2), dtype=np.int64)},
        }
        self._group_ids = np.asarray(group_ids, dtype=np.int64)


@pytest.fixture
def patched_conn_types(monkeypatch):
    # Bypass file-backed pop_name lookup; feed connection-type ids directly.
    monkeypatch.setattr(
        wr, "_connection_type_ids",
        lambda network, data_dir='': _CURRENT_RNN._group_ids,
    )


# module-level handle so the monkeypatched lambda can reach the active rnn
_CURRENT_RNN = None


def _make_reg(weights, group_ids):
    global _CURRENT_RNN
    rnn = _FakeRNN(np.asarray(weights, dtype=np.float32), group_ids)
    _CURRENT_RNN = rnn
    return rnn, EMDWeightRegularization(rnn=rnn, cost=1.0)


def test_emd_zero_at_init(patched_conn_types):
    rnn, reg = _make_reg([1.0, -2.0, 3.0, -4.0], [0, 0, 1, 1])
    assert float(reg().numpy()) == pytest.approx(0.0, abs=1e-7)


def test_emd_finite_and_positive_after_perturbation(patched_conn_types):
    rnn, reg = _make_reg([1.0, -2.0, 3.0, -4.0], [0, 0, 1, 1])
    rnn.cell.recurrent_weight_values.assign([2.0, -2.0, 3.0, -4.0])
    loss = float(reg().numpy())
    assert np.isfinite(loss)
    # group 0: |sort([2,-2]) - sort([1,-2])| -> mean(|−2−−2|,|2−1|)=0.5; group1: 0. cost=1 -> mean(0.5,0)=0.25
    assert loss == pytest.approx(0.25, abs=1e-6)


def test_emd_invariant_to_within_group_permutation(patched_conn_types):
    # EMD compares sorted distributions, so reordering weights within a group changes nothing.
    rnn, reg = _make_reg([1.0, 5.0, 3.0, 9.0], [0, 0, 0, 0])
    rnn.cell.recurrent_weight_values.assign([5.0, 1.0, 9.0, 3.0])
    assert float(reg().numpy()) == pytest.approx(0.0, abs=1e-6)


def test_emd_gradient_flows(patched_conn_types):
    rnn, reg = _make_reg([1.0, -2.0, 3.0, -4.0], [0, 0, 1, 1])
    w = rnn.cell.recurrent_weight_values
    w.assign([2.5, -2.0, 3.0, -4.0])
    with tf.GradientTape() as tape:
        loss = reg()
    grad = tape.gradient(loss, w)
    assert grad is not None
    assert float(tf.reduce_sum(tf.abs(grad)).numpy()) > 0


def test_emd_matches_groupwise_reference_value_and_gradient(patched_conn_types):
    initial = np.array([1.0, -2.0, 4.0, 3.0, -1.0, 2.0], dtype=np.float32)
    group_ids = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    rnn, reg = _make_reg(initial, group_ids)
    weights = rnn.cell.recurrent_weight_values
    weights.assign([1.5, -2.5, 5.0, 2.0, -0.5, 2.5])

    with tf.GradientTape() as actual_tape:
        actual = reg()
    actual_gradient = actual_tape.gradient(actual, weights)

    with tf.GradientTape() as reference_tape:
        losses = []
        for group_id in np.unique(group_ids):
            indices = np.flatnonzero(group_ids == group_id)
            current = tf.sort(tf.gather(weights, indices))
            baseline = tf.constant(np.sort(initial[indices]), tf.float32)
            losses.append(tf.reduce_mean(tf.abs(current - baseline)))
        reference = tf.reduce_mean(tf.stack(losses))
    reference_gradient = reference_tape.gradient(reference, weights)

    np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        tf.convert_to_tensor(actual_gradient),
        tf.convert_to_tensor(reference_gradient),
        rtol=1e-7,
        atol=1e-7,
    )


def test_emd_empty_network_returns_connected_zero(patched_conn_types):
    rnn, reg = _make_reg([], [])
    weights = rnn.cell.recurrent_weight_values

    with tf.GradientTape() as tape:
        loss = reg()
    gradient = tape.gradient(loss, weights)

    assert float(loss.numpy()) == 0.0
    assert gradient is not None
    assert gradient.shape == weights.shape
