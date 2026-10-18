from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bmtk.simulator.dpointnet.loss_functions.target_firing_rates import TargetFiringRate


@pytest.mark.parametrize("units", ["Hz", "spikes_per_step"])
@pytest.mark.parametrize("steps", [1, 25, 500])
@pytest.mark.parametrize("dtype", [tf.float16, tf.float32])
@pytest.mark.parametrize("activity", ["silent", "binary", "counts"])
def test_compiled_rate_counts_match_fp32_values_and_scaled_vjps(units, steps, dtype, activity):
    rng = np.random.default_rng(17)
    values = rng.integers(
        0, 5 if activity == "counts" else 2, (2, steps, 19)
    ).astype(dtype.as_numpy_dtype)
    if activity == "silent":
        values.fill(0)
    rnn = SimpleNamespace(dt=1.0, seq_len=steps)
    reference = TargetFiringRate(rnn, 5.0 if units == "Hz" else 0.005, rate_units=units)
    compiled = TargetFiringRate(
        rnn, reference.target_fr, rate_units=units, compiled_spike_counts=True
    )
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    with tf.device(device):
        spikes = tf.constant(values)
        loss_scale = 0.001 if units == "Hz" else 32768.0

        @tf.function
        def evaluate():
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(spikes)
                expected = reference(tf.cast(spikes, tf.float32)) * loss_scale
                actual = compiled(spikes) * loss_scale
            return actual, expected, tape.gradient(actual, spikes), tape.gradient(expected, spikes)

        actual, expected, actual_grad, expected_grad = evaluate()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_grad, expected_grad)
    assert actual.dtype == tf.float32
    assert actual_grad.dtype == dtype
    assert np.isfinite(actual_grad.numpy()).all()
    assert np.any(actual_grad.numpy())


@pytest.mark.parametrize("value", [1, "true", None])
def test_compiled_rate_counts_reject_nonboolean_options(value):
    with pytest.raises((TypeError, ValueError), match="compiled_spike_counts"):
        TargetFiringRate(SimpleNamespace(dt=1.0, seq_len=500), 5.0, compiled_spike_counts=value)


def test_uncompiled_default_preserves_original_dtype_and_value():
    rnn = SimpleNamespace(dt=1.0, seq_len=25)
    spikes = tf.ones((2, 25, 19), tf.float16)
    default = TargetFiringRate(rnn, 5.0)(spikes)
    explicit = TargetFiringRate(rnn, 5.0, compiled_spike_counts=False)(spikes)
    np.testing.assert_array_equal(default, explicit)
    assert default.dtype == tf.float16


def test_compilation_errors_are_not_silently_replaced(monkeypatch):
    from bmtk.simulator.dpointnet.loss_functions import target_firing_rates

    def fail(spikes):
        raise RuntimeError("explicit compilation failure")

    monkeypatch.setattr(target_firing_rates, "_compiled_fp32_spike_counts", fail)
    objective = TargetFiringRate(
        SimpleNamespace(dt=1.0, seq_len=25), 5.0, compiled_spike_counts=True
    )
    with pytest.raises(RuntimeError, match="explicit compilation failure"):
        objective(tf.ones((2, 25, 19), tf.float16))
