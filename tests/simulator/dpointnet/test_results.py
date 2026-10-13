import h5py
import numpy as np
import pandas as pd
import pytest
import shutil

tf = pytest.importorskip("tensorflow")

from bmtk.simulator.dpointnet import results


@pytest.fixture
def extractor_results(monkeypatch):
    mapping = pd.DataFrame({
        "node_id": [7, 3, 19],
        "population": ["first", "second", "first"],
    })
    monkeypatch.setattr(results.TFIDMap, "tf2bmtk_id_map", lambda self: mapping)
    spikes = np.zeros((2, 4, 3), dtype=np.float32)
    spikes[0, 1, 0] = spikes[1, 2, 1] = spikes[1, 3, 2] = 1
    voltages = tf.constant(np.arange(24).reshape(2, 4, 3), dtype=tf.float16)
    return results.RNNExtractorResults(
        seq_len=4, dt=0.5, batch_size=2,
        extractor_results=((spikes, voltages), ()), time_offset_steps=8,
    )


def assert_datasets_equal(actual_path, reference_path, compression):
    with h5py.File(actual_path) as actual, h5py.File(reference_path) as reference:
        datasets = []
        actual.visititems(
            lambda name, item: datasets.append(name) if isinstance(item, h5py.Dataset) else None
        )
        assert datasets
        for name in datasets:
            np.testing.assert_array_equal(actual[name][...], reference[name][...])
            assert actual[name].dtype == reference[name].dtype
            assert dict(actual[name].attrs) == dict(reference[name].attrs)
            expected = None if compression in (None, "none") else "lzf" if compression == "lzf" else "gzip"
            assert actual[name].compression == expected
            if isinstance(compression, int):
                assert actual[name].compression_opts == compression


@pytest.mark.parametrize("compression", ["default", "gzip", "lzf", None, "none", 1, 9])
@pytest.mark.parametrize("split_batches", [False, True])
@pytest.mark.parametrize("kind", ["spikes", "voltages"])
def test_result_exports_compression_preserves_data(
    tmp_path, extractor_results, compression, split_batches, kind
):
    output = getattr(extractor_results, kind)
    path = tmp_path / "nested" / f"{kind}.h5"
    reference = tmp_path / f"{kind}-reference.h5"
    options = {} if compression == "default" else {"compression": compression}
    output.to_sonata(path, split_batches=split_batches, **options)
    output.to_sonata(reference, split_batches=split_batches, compression=None)
    paths = (
        [(path.with_name(f"{kind}.batch_{batch}.h5"),
          reference.with_name(f"{kind}-reference.batch_{batch}.h5")) for batch in range(2)]
        if split_batches else [(path, reference)]
    )
    for actual_path, reference_path in paths:
        assert_datasets_equal(actual_path, reference_path, compression)
    output.to_sonata(path, split_batches=split_batches, overwrite=True, compression="lzf")
    for actual_path, reference_path in paths:
        assert_datasets_equal(actual_path, reference_path, "lzf")


@pytest.mark.parametrize("compression", ["default", "lzf", None, "none", 2])
def test_save_results_passes_compression_to_both_exports(tmp_path, extractor_results, compression):
    options = {} if compression == "default" else {"compression": compression}
    extractor_results.save_results(
        spikes_file=str(tmp_path / "spikes.h5"),
        voltages_file=str(tmp_path / "voltages.h5"),
        **options,
    )
    for kind in ("spikes", "voltages"):
        reference = tmp_path / f"{kind}-reference.h5"
        getattr(extractor_results, kind).to_sonata(reference, compression=None)
        assert_datasets_equal(tmp_path / f"{kind}.h5", reference, compression)


@pytest.mark.parametrize("kind", ["spikes", "voltages"])
def test_single_batch_result_compression(tmp_path, extractor_results, kind):
    extractor_results.batch_size = 1
    spikes, voltages = extractor_results._extractor_results[0]
    extractor_results._extractor_results = ((spikes[:1], voltages[:1]), ())
    path = tmp_path / f"{kind}.h5"
    getattr(extractor_results, kind).to_sonata(path)
    with h5py.File(path) as handle:
        if kind == "spikes":
            group = handle["spikes/first"]
            assert "batch_num" not in group
            assert group["timestamps"].compression == "gzip"
        else:
            assert handle["report/first/data"].compression == "gzip"


@pytest.mark.parametrize("kind", ["spikes", "voltages"])
def test_invalid_result_compression_does_not_create_output(tmp_path, extractor_results, kind):
    path = tmp_path / "invalid" / f"{kind}.h5"
    with pytest.raises(ValueError, match="SONATA compression"):
        getattr(extractor_results, kind).to_sonata(path, compression=True)
    assert not path.parent.exists()


@pytest.mark.parametrize("kind", ["spikes", "voltages"])
def test_result_overwrite_false_keeps_existing_compression(tmp_path, extractor_results, kind):
    path = tmp_path / f"{kind}.h5"
    output = getattr(extractor_results, kind)
    output.to_sonata(path)
    reference = tmp_path / f"{kind}-reference.h5"
    shutil.copyfile(path, reference)
    output.to_sonata(path, overwrite=False, compression=None)
    assert_datasets_equal(path, reference, "gzip")


def test_save_results_rejects_invalid_compression_before_writing(tmp_path, extractor_results):
    path = tmp_path / "spikes.h5"
    with pytest.raises(ValueError, match="SONATA compression"):
        extractor_results.save_results(spikes_file=str(path), compression="invalid")
    assert not path.exists()


def test_export_preserves_timestamps_node_ids_voltage_values_and_mapping(tmp_path, extractor_results):
    extractor_results.spikes.to_sonata(tmp_path / "spikes.h5")
    extractor_results.voltages.to_sonata(tmp_path / "voltages.h5")
    with h5py.File(tmp_path / "spikes.h5") as handle:
        np.testing.assert_array_equal(handle["spikes/first/node_ids"], [7, 19])
        np.testing.assert_array_equal(handle["spikes/first/timestamps"], [4.5, 5.5])
        np.testing.assert_array_equal(handle["spikes/first/batch_num"], [0, 1])
        assert handle["spikes/first/timestamps"].attrs["units"] == "ms"
    with h5py.File(tmp_path / "voltages.h5") as handle:
        group = handle["report/first"]
        np.testing.assert_array_equal(group["mapping/node_id"], [7, 19])
        np.testing.assert_array_equal(group["mapping/time"], [4.0, 6.0, 0.5])
        expected = np.arange(24).reshape(2, 4, 3)
        np.testing.assert_array_equal(group["data/batch_0"], expected[0][:, [0, 2]])
        np.testing.assert_array_equal(group["data/batch_1"], expected[1][:, [0, 2]])


def test_default_compression_reduces_repetitive_voltage_file_size(tmp_path, extractor_results):
    extractor_results.batch_size = 1
    extractor_results.seq_len = 4096
    output = results._VoltageResults(extractor_results, tf.zeros((1, 4096, 3)))
    compressed = tmp_path / "compressed.h5"
    uncompressed = tmp_path / "uncompressed.h5"
    output.to_sonata(compressed)
    output.to_sonata(uncompressed, compression=None)
    assert_datasets_equal(compressed, uncompressed, "gzip")
    assert compressed.stat().st_size < uncompressed.stat().st_size


def test_empty_result_exports(tmp_path, extractor_results):
    extractor_results.seq_len = 0
    spikes, voltages = extractor_results._extractor_results[0]
    extractor_results._extractor_results = ((spikes[:, :0], voltages[:, :0]), ())
    extractor_results.spikes.to_sonata(tmp_path / "spikes.h5")
    extractor_results.voltages.to_sonata(tmp_path / "voltages.h5")
    with h5py.File(tmp_path / "spikes.h5") as handle:
        assert "spikes" not in handle
    with h5py.File(tmp_path / "voltages.h5") as handle:
        dataset = handle["report/first/data/batch_0"]
        assert dataset.shape == (0, 2)
        assert dataset.compression == "gzip"
