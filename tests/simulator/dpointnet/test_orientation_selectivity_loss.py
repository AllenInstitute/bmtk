import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import loss_utils
from bmtk.simulator.dpointnet.loss_functions.orientation_selectivity_loss import (
    OrientationSelectivityLoss,
)


def _make_loss(method):
    loss = object.__new__(OrientationSelectivityLoss)
    loss._dtype = tf.float32
    loss._pre_delay = 1
    loss._post_delay = 1
    loss._method = method
    loss._subtraction_ratio = 0.4
    loss._osi_cost = 1.7
    loss._tf_pi = tf.constant(np.pi, tf.float32)
    loss._core_mask = tf.constant([True, False, True, True])
    loss._tuning_angles = tf.constant([10.0, 70.0, 150.0])
    loss._use_ema_normalizer = False
    loss._annulus_crowd_osi = None
    loss._min_rates_threshold = tf.constant(0.0005, tf.float32)
    loss.node_type_ids = tf.constant([0, 1, 0], tf.int32)
    loss._n_node_types = 2
    loss.osi_target_values = tf.constant([0.2, -0.1], tf.float32)
    loss.dsi_target_values = tf.constant([0.1, 0.3], tf.float32)
    loss.cell_type_count = tf.constant([2.0, 1.0], tf.float32)
    return loss


def _direct_reference(loss, spikes, angle):
    rates = tf.reduce_mean(tf.cast(spikes[:, 1:-1], tf.float32), axis=1)
    rates = tf.boolean_mask(rates, loss._core_mask, axis=1)
    delta_angle = angle[:, None] - loss._tuning_angles[None, :]

    if loss._method == "crowd_spikes":
        delta_angle = tf.where(delta_angle > 90.0, delta_angle - 180.0, delta_angle)
        delta_angle = tf.where(delta_angle < -90.0, delta_angle + 180.0, delta_angle)
        mean_angle = rates * delta_angle
        expected_sum_angle = tf.reduce_mean(rates) * 45.0
        return (
            tf.reduce_mean(tf.abs(mean_angle))
            - expected_sum_angle * loss._subtraction_ratio
        ) * loss._osi_cost

    radians_delta_angle = delta_angle * (loss._tf_pi / 180.0)
    batch_size = tf.shape(rates)[0]
    batch_offsets = tf.range(batch_size, dtype=tf.int32) * loss._n_node_types
    segment_ids = loss.node_type_ids[None, :] + batch_offsets[:, None]
    num_segments = batch_size * loss._n_node_types

    def segment_mean(values):
        means = tf.math.unsorted_segment_mean(
            tf.reshape(values, [-1]), tf.reshape(segment_ids, [-1]), num_segments
        )
        return tf.reshape(means, [batch_size, loss._n_node_types])

    denominator = tf.maximum(segment_mean(rates), 0.0005)
    osi = tf.reduce_mean(
        segment_mean(rates * tf.math.cos(2.0 * radians_delta_angle)) / denominator,
        axis=0,
    )
    dsi = tf.reduce_mean(
        segment_mean(rates * tf.math.cos(radians_delta_angle)) / denominator,
        axis=0,
    )
    squared_error = tf.square(osi - loss.osi_target_values) + tf.square(
        dsi - loss.dsi_target_values
    )
    return (
        tf.reduce_sum(squared_error * loss.cell_type_count)
        / tf.reduce_sum(loss.cell_type_count)
        * loss._osi_cost
    )


@pytest.mark.parametrize("method", ["crowd_spikes", "crowd_osi"])
@pytest.mark.parametrize("force_chunks", [False, True])
def test_orientation_loss_matches_direct_value_and_gradient(
    method, force_chunks, monkeypatch
):
    if force_chunks:
        temporal_sum = loss_utils.temporal_sum

        def forced_temporal_sum(values, dtype=tf.float32, **_kwargs):
            return temporal_sum(
                values,
                dtype=dtype,
                chunk_size=2,
                full_tensor_element_limit=1,
            )

        monkeypatch.setattr(loss_utils, "temporal_sum", forced_temporal_sum)

    values = (
        np.random.default_rng(41).uniform(0.0, 1.0, size=(2, 7, 4)).astype(np.float32)
    )
    angle = tf.constant([30.0, 80.0], tf.float32)
    loss = _make_loss(method)

    actual_spikes = tf.Variable(values)
    with tf.GradientTape() as tape:
        actual = loss(actual_spikes, y={"orientation": angle})
    actual_gradient = tape.gradient(actual, actual_spikes)

    reference_spikes = tf.Variable(values)
    with tf.GradientTape() as tape:
        reference = _direct_reference(loss, reference_spikes, angle)
    reference_gradient = tape.gradient(reference, reference_spikes)

    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_gradient, reference_gradient, rtol=1e-6, atol=1e-6
    )


def test_orientation_temporal_reduction_traces_int32_unsafe_shape():
    shape = (32, 500, 203_816)
    assert np.prod(shape, dtype=np.int64) > np.iinfo(np.int32).max

    @tf.function(input_signature=[tf.TensorSpec(shape, tf.float16)], autograph=False)
    def reduce_time(values):
        return loss_utils.temporal_sum(values)

    concrete = reduce_time.get_concrete_function()
    assert concrete.output_shapes == tf.TensorShape([32, 203_816])
    reduce_ops = [op for op in concrete.graph.get_operations() if op.type == "Sum"]
    assert len(reduce_ops) == 20
