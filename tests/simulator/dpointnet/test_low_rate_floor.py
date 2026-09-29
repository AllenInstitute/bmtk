from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import LossModules
from bmtk.simulator.dpointnet.loss_functions.low_rate_floor import LowRateFloor
from bmtk.simulator.dpointnet.loss_functions import loss_utils


@pytest.mark.parametrize("dtype", [tf.float32, tf.float16])
@pytest.mark.parametrize("dt", [1.0, 2.0])
def test_low_rate_floor_value_and_gradient(dtype, dt):
    loss_class = LossModules().get_module("LowRateFloor")
    loss = loss_class(SimpleNamespace(dt=dt), floor_hz=250.0, cost=3.0)
    spikes = tf.Variable(np.tile([0.0, 0.25, 1.0], (2, 4, 1)), dtype=dtype)
    with tf.GradientTape() as tape:
        actual = loss(spikes)
    gradient = tape.gradient(actual, spikes)
    rates = np.array([0.0, 0.25, 1.0]) * 1000.0 / dt
    deficits = np.maximum(250.0 - rates, 0.0)
    expected = 3.0 * np.mean(deficits**2)
    expected_gradient = -2 * deficits * 1000.0 / (dt * 2 * 4)
    assert actual.dtype == tf.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_allclose(
        gradient, np.broadcast_to(expected_gradient, spikes.shape), rtol=1e-3
    )


def test_low_rate_floor_selected_neurons_and_empty_selection():
    loss_class = LossModules().get_module("LowRateFloor")
    spikes = tf.Variable(tf.zeros((2, 5, 3)))
    loss = loss_class(SimpleNamespace(dt=1.0), neuron_ids=[0, 2], cost=10.0)
    with tf.GradientTape() as tape:
        actual = loss(spikes)
    gradient = tape.gradient(actual, spikes)
    np.testing.assert_allclose(actual, 0.1, rtol=1e-6)
    np.testing.assert_array_equal(gradient[:, :, 1], 0.0)
    assert np.all(tf.gather(gradient, [0, 2], axis=2).numpy() < 0)
    empty = loss_class(SimpleNamespace(dt=1.0), neuron_ids=[])
    assert float(empty(spikes)) == 0.0


@pytest.mark.parametrize("compiled", [False, True])
def test_low_rate_floor_graph_pools_samples_before_penalty(compiled):
    loss = LowRateFloor(SimpleNamespace(dt=1.0), floor_hz=100.0)

    @tf.function(
        input_signature=[tf.TensorSpec([None, 5, 3], tf.float32)],
        jit_compile=compiled,
    )
    def evaluate(spikes):
        with tf.GradientTape() as tape:
            tape.watch(spikes)
            value = loss(spikes)
        return value, tape.gradient(value, spikes)

    spikes = np.zeros((2, 5, 3), dtype=np.float32)
    spikes[0, 0, :] = 1.0
    value, gradient = evaluate(spikes)
    assert float(value) == 0.0
    np.testing.assert_array_equal(gradient, 0.0)
    silent, silent_gradient = evaluate(np.zeros((1, 5, 3), dtype=np.float32))
    assert float(silent) == 10000.0
    assert np.all(silent_gradient.numpy() < 0)


def test_low_rate_floor_core_cell_type_selection(monkeypatch):
    network = object()
    mask = [True, False, True, True]

    def resolve(actual_network, core_mask, radius, data_dir):
        assert actual_network is network
        assert core_mask is None and radius == 200.0 and data_dir == "fixture"
        return mask

    def populations(actual_network, data_dir, core_mask):
        assert actual_network is network and data_dir == "fixture" and core_mask == mask
        return {"L4 Exc": [0], "L5 Exc": [3], "L4 PV": [2]}

    monkeypatch.setattr(loss_utils, "resolve_core_mask", resolve)
    monkeypatch.setattr(loss_utils, "get_population_neuron_ids", populations)
    loss = LowRateFloor(
        SimpleNamespace(dt=1.0, recurrent_network=network),
        cell_types=["L4 Exc", "L5 Exc"],
        core_radius=200.0,
        data_dir="fixture",
        cost=10.0,
    )
    np.testing.assert_array_equal(loss.ids, [0, 3])
    np.testing.assert_allclose(loss(tf.zeros((2, 5, 4))), 0.1, rtol=1e-6)


def test_low_rate_floor_explicit_core_mask_needs_no_population_file():
    loss = LowRateFloor(
        SimpleNamespace(dt=1.0, recurrent_network={}), core_mask=[False, True, True]
    )
    np.testing.assert_array_equal(loss.ids, [1, 2])


@pytest.mark.parametrize("setting", [{"cost": 0}, {"floor_hz": 0}])
def test_zero_cost_or_floor_disables_regularizer(setting):
    loss = LowRateFloor(SimpleNamespace(dt=1.0), **setting)
    spikes = tf.Variable(tf.zeros((2, 5, 3)))
    with tf.GradientTape() as tape:
        value = loss(spikes)
    assert float(value) == 0.0
    np.testing.assert_array_equal(tape.gradient(value, spikes), 0.0)


@pytest.mark.parametrize(
    "options",
    [
        {"cost": -1},
        {"cost": float("nan")},
        {"floor_hz": -0.1},
        {"floor_hz": float("inf")},
        {"neuron_ids": [-1]},
        {"neuron_ids": [1, 1]},
        {"neuron_ids": [0.5]},
        {"neuron_ids": [[1]]},
        {"neuron_ids": [0], "core_mask": [True]},
        {"cell_types": ["unknown"]},
        {"core_radius": -1},
    ],
)
def test_invalid_low_rate_floor_configuration(options):
    with pytest.raises(ValueError):
        LowRateFloor(SimpleNamespace(dt=1.0), **options)


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan")])
def test_invalid_low_rate_floor_dt(dt):
    with pytest.raises(ValueError, match="dt"):
        LowRateFloor(SimpleNamespace(dt=dt))
