import h5py
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet.id_maps import TFIDMap
from bmtk.simulator.dpointnet.network_adaptor import compact_delays
from bmtk.simulator.dpointnet.results import PartitionedInferenceResults


@pytest.mark.parametrize("values, dtype", [
    ([1, 2, 13, 255], np.uint8),
    ([0.5, 2.25, 300.0], np.float16),
    ([0.1, 2.0], np.float32),
])
def test_compact_delays_are_exact(values, dtype):
    compact = compact_delays(np.array(values, np.float32))
    assert compact.dtype == dtype
    np.testing.assert_array_equal(compact.astype(np.float32), np.array(values, np.float32))


@pytest.fixture
def id_map(monkeypatch):
    monkeypatch.setattr(TFIDMap, "_tf_id_map_instance", None)
    monkeypatch.setattr(TFIDMap, "_initialized", False)
    value = TFIDMap()
    value.add_bmtk_ids("cells", np.array([7, 3, 5, 0]), "recurrent")
    return value


def test_partitioned_inference_results_write_sonata_spikes(id_map, tmp_path):
    results = PartitionedInferenceResults(
        seq_len=10, dt=1.0, batch_size=1,
        batch_num=np.array([0, 0, 0]), steps=np.array([4, 1, 4]), tf_ids=np.array([2, 0, 1]),
        n_neurons=4, time_offset_steps=1,
    )
    assert results.spikes.mean_firing_rate() == pytest.approx(3 / 4 / 0.01)
    results.save_results(output_dir=str(tmp_path), spikes_file=str(tmp_path / "spikes.h5"))
    with h5py.File(tmp_path / "spikes.h5") as handle:
        group = handle["spikes/cells"]
        np.testing.assert_array_equal(group["node_ids"][()], [7, 3, 5])
        np.testing.assert_array_equal(group["timestamps"][()], [2.0, 5.0, 5.0])
    with pytest.raises(ValueError, match="spikes only"):
        results.save_results(output_dir=str(tmp_path), voltages_file="v.h5")
