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


def _make_reg(weights, group_ids, cost=1.0, **kwargs):
    global _CURRENT_RNN
    rnn = _FakeRNN(np.asarray(weights, dtype=np.float32), group_ids)
    _CURRENT_RNN = rnn
    return rnn, EMDWeightRegularization(rnn=rnn, cost=cost, **kwargs)


def _fp64_groupwise_reference(initial, current, group_ids):
    initial = np.asarray(initial, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    group_ids = np.asarray(group_ids, dtype=np.int64)
    losses = []
    grad = np.zeros_like(current)
    for group_id in np.unique(group_ids):
        indices = np.flatnonzero(group_ids == group_id)
        order = indices[np.argsort(current[indices], kind="stable")]
        deviation = current[order] - np.sort(initial[indices])
        losses.append(np.mean(np.abs(deviation)))
        grad[order] = np.sign(deviation) / (len(indices) * len(np.unique(group_ids)))
    return np.mean(losses), grad


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


def test_graph_dedup_preserves_independent_series_tapes(patched_conn_types):
    rnn, reg = _make_reg([1.0, 2.0], [0, 0], deduplicate_within_graph=True)
    weights = rnn.cell.recurrent_weight_values
    weights.assign([2.0, 3.0])

    @tf.function
    def series_updates():
        values = []
        for _ in range(2):
            with tf.GradientTape() as tape:
                loss = reg()
            gradient = tape.gradient(loss, weights)
            assert gradient is not None
            values.append(loss)
            weights.assign_sub(0.2 * tf.convert_to_tensor(gradient))
        return tf.stack(values), tf.identity(weights)

    values, final_weights = series_updates()
    np.testing.assert_allclose(values, [1.0, 0.9], rtol=1e-6)
    np.testing.assert_allclose(final_weights, [1.8, 2.8], rtol=1e-6)


def test_scoped_graph_dedup_preserves_series_updates(patched_conn_types):
    rnn, reg = _make_reg([1.0, 2.0], [0, 0], deduplicate_within_graph=True)
    weights = rnn.cell.recurrent_weight_values
    weights.assign([2.0, 3.0])

    @tf.function
    def series_updates():
        values = []
        for _ in range(2):
            with tf.GradientTape() as tape, wr.weight_regularization_scope():
                first = reg()
                second = reg()
                assert first is second
                loss = (first + second) / 2.0
            gradient = tape.gradient(loss, weights)
            assert gradient is not None
            values.append(loss)
            weights.assign_sub(0.2 * tf.convert_to_tensor(gradient))
        return tf.stack(values), tf.identity(weights)

    values, final_weights = series_updates()
    np.testing.assert_allclose(values, [1.0, 0.9], rtol=1e-6)
    np.testing.assert_allclose(final_weights, [1.8, 2.8], rtol=1e-6)
    assert not hasattr(reg, "_graph_cache")


def test_scope_restores_outer_cache_and_clears_on_exception(patched_conn_types):
    _, reg = _make_reg([1.0, 2.0], [0, 0], deduplicate_within_graph=True)
    caches = []

    @tf.function
    def nested_scopes():
        with wr.weight_regularization_scope():
            outer = wr._weight_regularization_scope.get()
            caches.append(outer)
            first = reg()
            assert len(outer) == 1
            with pytest.raises(RuntimeError, match="scope interrupted"):
                with wr.weight_regularization_scope():
                    inner = wr._weight_regularization_scope.get()
                    caches.append(inner)
                    second = reg()
                    assert first is not second
                    assert len(inner) == 1
                    raise RuntimeError("scope interrupted")
            assert not inner
            assert wr._weight_regularization_scope.get() is outer
            assert reg() is first
        assert not outer
        assert wr._weight_regularization_scope.get() is None
        return first

    assert float(nested_scopes().numpy()) == 0.0
    assert all(not cache for cache in caches)
    assert wr._weight_regularization_scope.get() is None


def test_repeated_tracing_does_not_retain_scope_entries(patched_conn_types):
    rnn, reg = _make_reg([1.0, 2.0], [0, 0], deduplicate_within_graph=True)
    rnn.cell.recurrent_weight_values.assign([2.0, 3.0])
    caches = []
    for index in range(8):
        @tf.function
        def evaluate():
            with wr.weight_regularization_scope():
                cache = wr._weight_regularization_scope.get()
                caches.append(cache)
                first = reg()
                assert reg() is first
                assert len(cache) == 1
            assert not cache
            return first

        assert float(evaluate().numpy()) == pytest.approx(1.0)
        assert len(caches) >= index + 1
        assert all(not cache for cache in caches)
        assert not hasattr(reg, "_graph_cache")
        assert wr._weight_regularization_scope.get() is None


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
@pytest.mark.parametrize(
    "initial,current,group_ids",
    [
        ([1.0, -2.0, 4.0, 3.0, -1.0, 2.0, 7.0],
         [1.25, -2.5, 5.0, 2.0, -0.5, 2.5, 6.5], [0, 0, 1, 1, 0, 1, 2]),
        ([1.0, 3.0, -4.0, -2.0, 5.0, 7.0],
         [2.0, 2.0, -3.0, -3.0, 6.0, 6.0], [1, 1, 0, 0, 2, 2]),
        ([1.0, 1.0, -2.0, -2.0, 4.0, 4.0],
         [1.0, 1.0, -2.0, -2.0, 4.0, 4.0], [0, 0, 1, 1, 2, 2]),
    ],
    ids=["unequal-groups", "nonzero-ties", "zero-ties"],
)
def test_emd_matches_groupwise_reference_value_and_gradient(
    patched_conn_types, initial, current, group_ids, dtype
):
    initial = np.array(initial, dtype=np.float32)
    group_ids = np.array(group_ids, dtype=np.int64)
    cost = 1.75
    rnn, reg = _make_reg(initial, group_ids, cost=cost, dtype=dtype)
    weights = rnn.cell.recurrent_weight_values
    weights.assign(current)

    with tf.GradientTape() as actual_tape:
        actual = reg()
    actual_gradient = actual_tape.gradient(actual, weights)

    with tf.GradientTape() as reference_tape:
        losses = []
        for group_id in np.unique(group_ids):
            indices = np.flatnonzero(group_ids == group_id)
            sorted_current = tf.sort(tf.cast(tf.gather(weights, indices), dtype))
            baseline = tf.constant(np.sort(initial[indices]), dtype)
            losses.append(tf.reduce_mean(tf.abs(sorted_current - baseline)))
        reference = cost * tf.reduce_mean(tf.stack(losses))
    reference_gradient = reference_tape.gradient(reference, weights)

    np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        tf.convert_to_tensor(actual_gradient),
        tf.convert_to_tensor(reference_gradient),
        rtol=1e-7,
        atol=1e-7,
    )
    ref_value, ref_grad = _fp64_groupwise_reference(initial, current, group_ids)
    np.testing.assert_allclose(actual, cost * ref_value, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        tf.convert_to_tensor(actual_gradient), cost * ref_grad, rtol=1e-7, atol=1e-7
    )


@pytest.mark.parametrize("option", ["use_grouped_custom_gradient", "use_javier_grouped_emd"])
def test_emd_rejects_retired_implementation_options(patched_conn_types, option):
    with pytest.raises(TypeError, match=option):
        _make_reg([1.0], [0], **{option: True})


def test_emd_accepts_loss_registry_metadata(patched_conn_types):
    _, reg = _make_reg([1.0], [0], module="EMDWeightRegularization", enabled=True)
    assert float(reg().numpy()) == 0.0


def test_emd_empty_network_returns_connected_zero(patched_conn_types):
    rnn, reg = _make_reg([], [])
    weights = rnn.cell.recurrent_weight_values

    with tf.GradientTape() as tape:
        loss = reg()
    gradient = tape.gradient(loss, weights)

    assert float(loss.numpy()) == 0.0
    assert gradient is not None
    assert gradient.shape == weights.shape


def test_emd_compact_permutation_storage(patched_conn_types):
    _, reg = _make_reg([1.0, 2.0, 3.0], [1, 0, 1])
    assert reg._group_order.dtype == tf.uint32
    assert not reg._group_order.trainable
    np.testing.assert_array_equal(reg._group_order, [1, 0, 2])


def test_emd_restores_original_int32_checkpoint_schema(patched_conn_types, tmp_path):
    rnn, reg = _make_reg([1.0, -2.0, 3.0, -4.0], [1, 0, 1, 0])
    weights = rnn.cell.recurrent_weight_values
    legacy_order = tf.Variable(reg._group_order.numpy(), dtype=tf.int32, trainable=False)
    legacy_initial = tf.Variable(reg._sorted_initial_values.numpy(), trainable=False)
    path = tf.train.Checkpoint(
        order=legacy_order, initial=legacy_initial, weights=weights,
    ).save(str(tmp_path / "original-schema"))
    checkpoint_dtypes = tf.train.load_checkpoint(path).get_variable_to_dtype_map()
    assert checkpoint_dtypes["order/.ATTRIBUTES/VARIABLE_VALUE"] == tf.int32
    weights.assign([2.0, -3.0, 4.0, -5.0])

    def evaluate():
        with tf.GradientTape() as tape:
            loss = reg()
        return loss.numpy(), tape.gradient(loss, weights).numpy()

    expected = evaluate()
    reg._group_order.assign(tf.zeros_like(reg._group_order))
    reg._sorted_initial_values.assign(tf.ones_like(reg._sorted_initial_values) * 100)
    weights.assign(tf.ones_like(weights) * 200)
    tf.train.Checkpoint(
        order=reg._group_order, initial=reg._sorted_initial_values, weights=weights,
    ).restore(path).assert_consumed()
    np.testing.assert_array_equal(reg._group_order, legacy_order)
    np.testing.assert_array_equal(reg._sorted_initial_values, legacy_initial)
    assert float(reg().numpy()) == 0.0
    weights.assign([2.0, -3.0, 4.0, -5.0])
    for actual, reference in zip(evaluate(), expected):
        np.testing.assert_array_equal(actual, reference)
    new_path = tf.train.Checkpoint(
        order=reg._group_order, initial=reg._sorted_initial_values, weights=weights,
    ).save(str(tmp_path / "new-schema"))
    assert dict(tf.train.list_variables(new_path)) == dict(tf.train.list_variables(path))
    legacy_order.assign(tf.zeros_like(legacy_order))
    tf.train.Checkpoint(
        order=legacy_order, initial=legacy_initial, weights=weights,
    ).restore(new_path).assert_consumed()
    np.testing.assert_array_equal(legacy_order, reg._group_order)


@pytest.mark.parametrize(
    "length,index_dtype",
    [(2, tf.int32), (None, tf.int64), (2**31 - 1, tf.int32), (2**31, tf.int64)],
)
def test_emd_index_column_stays_compact_when_size_is_known(length, index_dtype):
    @tf.function(input_signature=[tf.TensorSpec([length], tf.uint32)])
    def evaluate(order):
        return wr._grouped_emd_custom_gradient(
            tf.constant([2.0, 1.0]), order, tf.constant([0.0, 1.0]), ((0, 2),), (2,)
        )

    concrete = evaluate.get_concrete_function()
    casts = [
        op for op in concrete.graph.get_operations()
        if op.type == "Cast" and op.inputs[0].dtype == tf.uint32
    ]
    assert len(casts) == 1
    assert casts[0].outputs[0].dtype == index_dtype
    if length in (2, None):
        assert float(evaluate(tf.constant([0, 1], tf.uint32)).numpy()) == 1.0


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_emd_stable_ties_match_original_sort_bitwise(patched_conn_types, dtype):
    rnn, reg = _make_reg(
        [-4.0, 2.0, -1.0, 4.0, 0.0, 3.0, -2.0, 5.0],
        [0, 1, 0, 1, 0, 1, 0, 1],
        cost=1.75,
        dtype=dtype,
    )
    weights = rnn.cell.recurrent_weight_values
    weights.assign([0.0, 2.0, -0.0, 2.0, 0.0, 2.0, 0.0, 2.0])

    @tf.function
    def compare():
        with tf.GradientTape(persistent=True) as tape:
            actual = reg()
            values = tf.cast(weights, dtype)
            grouped = tf.gather(values, tf.cast(reg._group_order, tf.int32))
            losses = []
            for start, end in reg._group_slices:
                segment = grouped[start:end]
                sorted_segment = tf.gather(segment, tf.argsort(segment, stable=True))
                reference_segment = reg._sorted_initial_values[start:end]
                losses.append(tf.reduce_mean(tf.abs(sorted_segment - reference_segment)))
            reference = tf.reduce_mean(tf.stack(losses)) * tf.cast(1.75, dtype)
        return (
            actual, reference,
            tape.gradient(actual, weights),
            tf.convert_to_tensor(tape.gradient(reference, weights)),
        )

    actual, reference, actual_gradient, reference_gradient = compare()
    np.testing.assert_array_equal(actual, reference)
    np.testing.assert_array_equal(actual_gradient, reference_gradient)


@pytest.mark.skipif(not tf.config.list_physical_devices("GPU"), reason="GPU required")
def test_emd_permutation_and_gradient_reside_on_gpu(patched_conn_types):
    with tf.device("/GPU:0"):
        rnn, reg = _make_reg([1.0, 2.0, 3.0, 4.0], [1, 0, 1, 0])
        weights = rnn.cell.recurrent_weight_values
        weights.assign([2.0, 3.0, 4.0, 5.0])
        with tf.GradientTape() as tape:
            loss = reg()
        gradient = tape.gradient(loss, weights)
    assert reg._group_order.device.endswith("GPU:0")
    assert reg._group_order.read_value().device.endswith("GPU:0")
    assert gradient.device.endswith("GPU:0")
    np.testing.assert_array_equal(gradient, np.full(4, 0.25, dtype=np.float32))
