import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import loss_utils
from bmtk.simulator.dpointnet.loss_functions.spike_rate_distribution_target import (
    SpikeRateDistributionTarget,
)


def test_temporal_sum_chunking_matches_direct_value_and_gradient():
    values = tf.Variable(np.arange(2 * 7 * 3, dtype=np.float32).reshape(2, 7, 3))
    upstream = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))

    with tf.GradientTape() as tape:
        reduced = loss_utils.temporal_sum(
            values,
            chunk_size=3,
            full_tensor_element_limit=1,
        )
        loss = tf.reduce_sum(reduced * upstream)
    gradient = tape.gradient(loss, values)

    np.testing.assert_allclose(reduced.numpy(), tf.reduce_sum(values, axis=1))
    np.testing.assert_allclose(
        gradient.numpy(),
        tf.broadcast_to(upstream[:, None, :], values.shape),
    )


def test_temporal_mean_chunking_matches_direct_mean():
    values = tf.reshape(tf.range(30, dtype=tf.float32), (2, 5, 3))

    result = loss_utils.temporal_mean(
        values,
        chunk_size=2,
        full_tensor_element_limit=1,
    )

    np.testing.assert_allclose(result.numpy(), tf.reduce_mean(values).numpy())


def test_spike_rate_distribution_matches_direct_mean_value_and_gradient():
    target = object.__new__(SpikeRateDistributionTarget)
    target._pre_delay = 0
    target._post_delay = 0
    target._dtype = tf.float32
    target._rate_cost = 0.7
    target._annulus_target_rates = None
    target._target_rates = {
        "population": {
            "neuron_ids": tf.constant([0, 1, 2], dtype=tf.int32),
            "sorted_target_rates": tf.constant([0.1, 0.3, 0.6], tf.float32),
        }
    }
    spikes = tf.Variable(
        np.random.default_rng(13).uniform(size=(2, 7, 3)).astype(np.float32)
    )

    with tf.GradientTape() as actual_tape:
        actual = target(spikes)
    actual_gradient = actual_tape.gradient(actual, spikes)

    with tf.GradientTape() as reference_tape:
        rates = tf.reduce_mean(spikes, axis=(0, 1))
        reference = 0.7 * loss_utils.compute_spike_rate_target_loss(
            rates, target._target_rates, dtype=tf.float32
        )
    reference_gradient = reference_tape.gradient(reference, spikes)

    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_gradient, reference_gradient, rtol=1e-6, atol=1e-6
    )
