from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.cell_models.glif3_cell import (
    voltage_penalty_mean_step,
)
from bmtk.simulator.dpointnet.loss_functions.voltage_regularization import (
    VoltageRegularization,
)


@pytest.mark.parametrize("penalty_mode", ["range", "threshold"])
def test_online_voltage_accumulator_matches_sequence_value_and_gradient(
    penalty_mode,
):
    rnn = SimpleNamespace(
        recurrent_network={"n_nodes": 4},
        seq_len=7,
        cell=SimpleNamespace(
            _track_voltage_penalty=True,
            _voltage_penalty_mode=penalty_mode,
        ),
    )
    offline_loss = VoltageRegularization(
        rnn, voltage_cost=0.7, penalty_mode=penalty_mode
    )
    online_loss = VoltageRegularization(
        rnn, voltage_cost=0.7, penalty_mode=penalty_mode, online=True
    )
    values = (
        np.random.default_rng(19)
        .uniform(low=-0.4, high=1.6, size=(3, 7, 4))
        .astype(np.float32)
    )

    offline_voltages = tf.Variable(values)
    with tf.GradientTape() as tape:
        offline_value = offline_loss(offline_voltages)
    offline_gradient = tape.gradient(offline_value, offline_voltages)

    online_voltages = tf.Variable(values)
    with tf.GradientTape() as tape:
        accumulator = tf.stack(
            [
                voltage_penalty_mean_step(online_voltages[:, timestep], 4, penalty_mode)
                for timestep in range(7)
            ],
            axis=1,
        )
        online_value = online_loss(accumulator)
    online_gradient = tape.gradient(online_value, online_voltages)

    np.testing.assert_allclose(online_value, offline_value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(online_gradient, offline_gradient, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("penalty_mode", ["range", "threshold"])
def test_online_voltage_accumulator_is_fp32_for_extreme_float16_values(
    penalty_mode,
):
    values = np.array(
        [
            [[-400.0, -1.0, 0.5, 400.0], [300.0, 1.0, -300.0, 0.0]],
            [[-350.0, 2.0, 1.0, 350.0], [275.0, -2.0, -275.0, 0.5]],
        ],
        dtype=np.float16,
    )

    voltages = tf.Variable(values)
    with tf.GradientTape() as tape:
        accumulator = tf.stack(
            [
                voltage_penalty_mean_step(
                    voltages[:, timestep], values.shape[2], penalty_mode
                )
                for timestep in range(values.shape[1])
            ],
            axis=1,
        )
        actual = tf.reduce_mean(accumulator)
    actual_gradient = tape.gradient(actual, voltages)

    reference_voltages = tf.Variable(values)
    with tf.GradientTape() as tape:
        fp32_voltages = tf.cast(reference_voltages, tf.float32)
        if penalty_mode == "range":
            penalty = tf.square(tf.nn.relu(tf.abs(fp32_voltages - 0.5) - 0.5))
        else:
            penalty = tf.square(fp32_voltages - 1.0)
        reference = tf.reduce_mean(penalty)
    reference_gradient = tape.gradient(reference, reference_voltages)

    assert accumulator.dtype == tf.float32
    assert np.isfinite(actual.numpy())
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_gradient, reference_gradient, rtol=1e-3, atol=1e-3
    )
