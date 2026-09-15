from types import SimpleNamespace

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.loss_functions import loss_utils
from bmtk.simulator.dpointnet.loss_functions.synchronization_loss import (
    SynchronizationLoss,
    fano_sampling_plan,
)


def _build_dummy_rnn():
    return SimpleNamespace(
        recurrent_network={
            "n_nodes": 4,
            "node_params": {
                "pop_name": ["e0", "i0", "e1", "i1"],
            },
        }
    )


def test_synchronization_loss_uses_millisecond_window(monkeypatch):
    loaded_paths = []

    monkeypatch.setattr(
        loss_utils,
        "get_pop_names",
        lambda network, data_dir="": np.array(["e0", "i0", "e1", "i1"]),
    )
    monkeypatch.setattr("os.path.exists", lambda path: True)

    def fake_load(path, allow_pickle=True):
        loaded_paths.append(path)
        return np.ones((3, 20), dtype=np.float32)

    monkeypatch.setattr(np, "load", fake_load)

    loss = SynchronizationLoss(
        _build_dummy_rnn(),
        t_start=200,
        t_end=500,
        neuropixels_data_dir="Synchronization_data",
    )

    assert loss._t_start_idx == 200
    assert loss._t_end_idx == 500
    assert loaded_paths == [
        "Synchronization_data/Fano_factor_v1/v1_fano_running_300ms_evoked.npy"
    ]


def test_synchronization_loss_rejects_second_style_window(monkeypatch):
    monkeypatch.setattr(
        loss_utils, "get_pop_names", lambda network: np.array(["e0", "i0", "e1", "i1"])
    )
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        np, "load", lambda path, allow_pickle=True: np.ones((3, 20), dtype=np.float32)
    )

    with pytest.raises(ValueError, match="multiply by 1000.*t_start=200, t_end=500"):
        SynchronizationLoss(_build_dummy_rnn(), t_start=0.2, t_end=0.5)


def test_fano_sampling_plan_is_deterministic_and_balances_trials():
    plan = fano_sampling_plan(n_pool=40, n_samples=5, n_trials=3, seed=17)
    repeated = fano_sampling_plan(n_pool=40, n_samples=5, n_trials=3, seed=17)

    assert plan["per_trial"] == 2
    assert plan["n_effective"] == 6
    assert plan["positions"].shape == (3, 2 * plan["max_count"])
    assert plan["neuron_mask"].shape == (3, 2, plan["max_count"])
    assert np.all((15 <= plan["counts"]) & (plan["counts"] <= 40))
    for key in ("counts", "offsets", "positions", "neuron_mask"):
        np.testing.assert_array_equal(plan[key], repeated[key])


def test_fano_sampling_plan_rejects_too_few_excitatory_neurons():
    with pytest.raises(ValueError, match="at least 15 neurons.*got 14"):
        fano_sampling_plan(n_pool=14, n_samples=5, n_trials=1)


def test_synchronization_loss_tracks_original_core_excitatory_ids(monkeypatch):
    pop_names = np.array(["e0" if index % 2 == 0 else "i0" for index in range(40)])
    core_mask = np.zeros(40, dtype=bool)
    core_mask[[0, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38]] = True
    expected = np.flatnonzero(core_mask)
    rnn = SimpleNamespace(
        recurrent_network={
            "n_nodes": 40,
            "node_params": {"pop_name": pop_names},
        }
    )
    monkeypatch.setattr(
        loss_utils, "get_pop_names", lambda network, data_dir="": pop_names
    )
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        np,
        "load",
        lambda path, allow_pickle=True: np.ones((3, 20), dtype=np.float32),
    )

    first = SynchronizationLoss(rnn, core_mask=core_mask, seed=29)
    second = SynchronizationLoss(rnn, core_mask=core_mask, seed=29)

    np.testing.assert_array_equal(first._core_e_indices_np, expected)
    first_seed = first._next_seed_pair().numpy()
    second_seed = second._next_seed_pair().numpy()
    np.testing.assert_array_equal(first_seed, second_seed)
    assert not np.array_equal(first_seed, first._next_seed_pair().numpy())


def test_synchronization_loss_batched_sampling_has_finite_gradient(monkeypatch):
    pop_names = np.array(["e0"] * 20)
    rnn = SimpleNamespace(
        recurrent_network={
            "n_nodes": 20,
            "node_params": {"pop_name": pop_names},
        }
    )
    monkeypatch.setattr(
        loss_utils, "get_pop_names", lambda network, data_dir="": pop_names
    )
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(
        np,
        "load",
        lambda path, allow_pickle=True: np.ones((3, 20), dtype=np.float32),
    )
    loss = SynchronizationLoss(
        rnn,
        t_start=0,
        t_end=20,
        n_samples=5,
        seed=31,
    )
    spikes = tf.Variable(
        np.random.default_rng(31).uniform(size=(2, 20, 20)).astype(np.float32)
    )

    with tf.GradientTape() as tape:
        value = loss(spikes)
    gradient = tape.gradient(value, spikes)

    assert np.isfinite(value.numpy())
    assert gradient is not None
    assert gradient.shape == spikes.shape
    assert np.all(np.isfinite(gradient.numpy()))
    assert loss._plan_cache[2]["n_effective"] == 6


def test_synchronization_loss_matches_loop_reference_value_and_gradient(monkeypatch):
    pop_names = np.array(["e0"] * 40)
    rnn = SimpleNamespace(
        recurrent_network={
            "n_nodes": 40,
            "node_params": {"pop_name": pop_names},
        }
    )
    experimental_fanos = np.linspace(0.5, 1.5, 60, dtype=np.float32).reshape(3, 20)
    monkeypatch.setattr(
        loss_utils, "get_pop_names", lambda network, data_dir="": pop_names
    )
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr(np, "load", lambda path, allow_pickle=True: experimental_fanos)
    actual_loss = SynchronizationLoss(rnn, t_start=0, t_end=20, n_samples=5, seed=37)
    reference_loss = SynchronizationLoss(rnn, t_start=0, t_end=20, n_samples=5, seed=37)
    spikes = tf.Variable(
        np.random.default_rng(41).uniform(size=(2, 20, 40)).astype(np.float32)
    )

    with tf.GradientTape() as actual_tape:
        actual_value = actual_loss(spikes)
    actual_gradient = actual_tape.gradient(actual_value, spikes)

    with tf.GradientTape() as reference_tape:
        plan = reference_loss._plan(2)
        shuffled_pool = reference_loss._draw_pool(
            reference_loss._next_seed_pair(), plan["n_epochs"]
        )
        positions = plan["positions"].reshape(2, plan["per_trial"], plan["max_count"])
        samples = []
        for trial in range(2):
            for sample in range(plan["per_trial"]):
                valid = plan["neuron_mask"][trial, sample].astype(bool)
                neuron_ids = tf.gather(shuffled_pool, positions[trial, sample, valid])
                samples.append(
                    tf.reduce_sum(tf.gather(spikes[trial], neuron_ids, axis=1), axis=1)
                )
        selected_spikes = tf.stack(samples)
        reference_fanos = []
        for bin_size in reference_loss._bin_sizes_ms:
            counts = tf.nn.conv1d(
                selected_spikes[..., None],
                tf.ones((bin_size, 1, 1), dtype=tf.float32),
                stride=bin_size,
                padding="VALID",
            )[..., 0]
            mean_count = tf.maximum(
                tf.reduce_mean(counts, axis=1), tf.constant(1e-7, tf.float32)
            )
            reference_fanos.append(
                tf.reduce_mean(tf.math.reduce_variance(counts, axis=1) / mean_count)
            )
        reference_fanos = tf.stack(reference_fanos)
        reference_value = reference_loss._sync_cost * tf.reduce_mean(
            tf.square(reference_loss.experimental_fanos_mean - reference_fanos)
        )
    reference_gradient = reference_tape.gradient(reference_value, spikes)

    np.testing.assert_allclose(actual_value, reference_value, rtol=1e-4, atol=1e-5)
    gradient_max_absolute_error = np.max(
        np.abs(actual_gradient.numpy() - reference_gradient.numpy())
    )
    assert gradient_max_absolute_error < 1e-4
