from types import SimpleNamespace

import h5py
import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.input_modules.spikes_files_generator import SpikesFilesGenerator


def _spikes_file(path, node_ids, timestamps):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("spikes/virt")
        group.attrs["sorting"] = "by_id"
        group.create_dataset("node_ids", data=np.asarray(node_ids, np.uint64))
        group.create_dataset("timestamps", data=np.asarray(timestamps, np.float64))
    return str(path)


def _module(path, **kwargs):
    network = SimpleNamespace(name="virt", population_name="virt", n_nodes=3, options={})
    return SpikesFilesGenerator(
        rnn=SimpleNamespace(), name="spikes", input_network=network, spikes_paths=path, **kwargs
    )


def _first_table(module, seq_len=4):
    table, _ = next(iter(module.create_generator(seq_len, dt=1.0)))
    return table.numpy()


@pytest.fixture
def spikes_path(tmp_path):
    # Node 1 spikes twice in step 2; node 0 once in step 0; node 2 three times in step 3.
    return _spikes_file(tmp_path / "spikes.h5", [0, 1, 1, 2, 2, 2], [0.2, 2.1, 2.9, 3.0, 3.4, 3.99])


def test_spikes_files_default_keeps_one_spike_per_step(spikes_path):
    expected = np.zeros((4, 3), np.float32)
    expected[0, 0] = expected[2, 1] = expected[3, 2] = 1.0
    np.testing.assert_array_equal(_first_table(_module(spikes_path)), expected)


def test_spikes_files_count_spikes_keeps_multiplicity(spikes_path):
    expected = np.zeros((4, 3), np.float32)
    expected[0, 0], expected[2, 1], expected[3, 2] = 1.0, 2.0, 3.0
    np.testing.assert_array_equal(_first_table(_module(spikes_path, count_spikes=True)), expected)


@pytest.mark.parametrize("option", ["background_noise_rate", "dropout_prob", "jitter_var"])
def test_spikes_files_count_spikes_rejects_binary_transforms(spikes_path, option):
    with pytest.raises(ValueError, match="count_spikes"):
        _module(spikes_path, count_spikes=True, **{option: 0.1})


def test_spikes_files_load_without_world_barrier(spikes_path, monkeypatch):
    # Under neuron partitioning only rank 0 draws inputs; a world barrier would deadlock.
    import bmtk.utils.reports.spike_trains.spike_train_readers as readers

    def barrier():
        raise AssertionError("spike input loading must not enter a world barrier")

    monkeypatch.setattr(readers, "comm_barrier", barrier)
    expected = np.zeros((4, 3), np.float32)
    expected[0, 0] = expected[2, 1] = expected[3, 2] = 1.0
    np.testing.assert_array_equal(_first_table(_module(spikes_path)), expected)
