from pathlib import Path
import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

import bmtk.simulator.dpointnet.input_modules.lgn_generator as lgn_generator


@pytest.mark.parametrize(
    "orientation,phase,regular",
    [(None, None, False), (None, 17, True), ([0, 90], 31, False)],
)
def test_parameter_stream_preserves_existing_seed_schedule(orientation, phase, regular):
    dataset = lgn_generator.create_drifting_grating_parameters(
        71, orientation=orientation, phase=phase, regular=regular
    )
    base_seed = lgn_generator._stateless_seed_pair(71, salt=1001)
    for index, (actual_theta, actual_phase, actual_seed) in enumerate(dataset.take(4)):
        seed = lgn_generator._fold_in_seed(base_seed, index)
        theta_seed = lgn_generator._fold_in_seed(seed, 0)
        phase_seed = lgn_generator._fold_in_seed(seed, 1)
        expected_seed = lgn_generator._fold_in_seed(seed, 2)
        theta = (
            orientation[index % len(orientation)]
            if orientation is not None
            else (
                index * 45
                if regular
                else lgn_generator._tensorflow_uniform_scalar(
                    0, 360, tf.float32, theta_seed
                )
            )
        )
        expected_phase = (
            phase
            if phase is not None
            else lgn_generator._tensorflow_uniform_scalar(
                0, 360, tf.float32, phase_seed
            )
        )
        np.testing.assert_array_equal(actual_theta, [theta])
        np.testing.assert_array_equal(actual_phase, expected_phase)
        np.testing.assert_array_equal(actual_seed, expected_seed)


@pytest.mark.parametrize("prefetch", [False, True])
def test_device_batch_matches_seeded_host_generator(prefetch):
    from bmtk.simulator.dpointnet.data_iterator import DataIterator

    class FakeLGN:
        n_nodes = 4

        @staticmethod
        def spatial_response(videos, bmtk_compat):
            response = tf.reduce_mean(videos, axis=(1, 2, 3))[:, None]
            return tf.tile(response, (1, 4)), tf.zeros((tf.shape(videos)[0], 0))

        @staticmethod
        def firing_rates_from_spatial(dominant, non_dominant):
            return 100.0 + dominant * 20.0

    strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
    options = dict(
        row_size=5,
        col_size=7,
        orientation=[0, 45, 90],
        phase=21,
        pre_delay=2,
        post_delay=1,
        rotation="ccw",
        billeh_phase=True,
    )
    mod = object.__new__(lgn_generator.LGNGenerator)
    mod.rnn = SimpleNamespace(default_seed=None, strategy=strategy)
    mod.population_name = "lgn"
    mod.stimulus_opts = dict(options, seed=71)
    mod.lgn = FakeLGN()
    mod.use_device_generation = True
    actual = DataIterator([mod], 3, 8, ["lgn"], prefetch_device_inputs=prefetch)
    expected = iter(
        lgn_generator.create_drifting_gratings_generator(
            FakeLGN(), seq_len=8, seed=71, **options
        ).batch(3)
    )
    try:
        for _ in range(3):
            spikes, signatures = actual.next_spikes()
            expected_values = next(expected)
            for value, reference in zip(
                tf.nest.flatten((spikes, signatures[0])),
                tf.nest.flatten(expected_values),
            ):
                np.testing.assert_array_equal(value, reference)
        with pytest.raises(ValueError, match="requires eager iterator fetching"):
            tf.function(actual.next_spikes).get_concrete_function()
    finally:
        actual.close()
    assert actual._prefetch_executor is None


def test_two_replica_pipeline_keeps_batches_and_constants_local():
    script = r"""
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(30)
from types import SimpleNamespace
import numpy as np
import tensorflow as tf
physical = tf.config.list_physical_devices('CPU')[0]
tf.config.set_logical_device_configuration(physical, [tf.config.LogicalDeviceConfiguration(), tf.config.LogicalDeviceConfiguration()])
from bmtk.simulator.dpointnet.input_modules.lgn_generator import LGNGenerator, create_drifting_gratings_generator
from bmtk.simulator.dpointnet.data_iterator import DataIterator
class FakeLGN:
    n_nodes = 4
    def __init__(self):
        self.rate = tf.constant(100.0)
    def spatial_response(self, movie, compatible):
        return tf.ones((tf.shape(movie)[0], 4)) * self.rate, tf.zeros((tf.shape(movie)[0], 0))
    def firing_rates_from_spatial(self, dominant, nondominant):
        return dominant
strategy = tf.distribute.MirroredStrategy(['/CPU:0', '/CPU:1'])
mod = object.__new__(LGNGenerator)
mod.rnn = SimpleNamespace(default_seed=None, strategy=strategy)
mod.population_name = 'lgn'
mod.stimulus_opts = dict(row_size=5, col_size=7, seed=71, regular=True, phase=21)
mod.lgn = FakeLGN()
mod.use_device_generation = True
iterator = DataIterator([mod], 3, 8, ['lgn'])
reference = list(create_drifting_gratings_generator(mod.lgn, 8, **mod.stimulus_opts).batch(3).take(2))
try:
    for step in range(2):
        spikes, targets = iterator.next_spikes()
        expected, expected_targets = reference[step]
        for replica, local in enumerate(strategy.experimental_local_results(spikes)):
            assert local.device.endswith('CPU:' + str(replica)), local.device
            np.testing.assert_array_equal(local, expected)
            theta = strategy.experimental_local_results(targets[0]['orientation'])[replica]
            np.testing.assert_array_equal(theta, expected_targets['orientation'])
        reduced = strategy.run(lambda value: tf.reduce_sum(tf.cast(value, tf.int32)), args=(spikes,))
        assert len(strategy.experimental_local_results(reduced)) == 2
    for replica, (generator, _) in enumerate(iterator.data_itrs[0].generators):
        assert generator.lgn.rate.device.endswith('CPU:' + str(replica))
finally:
    local_iterator = iterator.data_itrs[0]
    iterator.close()
assert iterator._prefetch_executor is None
assert local_iterator._executor is None
faulthandler.cancel_dump_traceback_later()
print('Two replica seeded batches and constant placement passed')
"""
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="2",
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(str(error.stdout) + str(error.stderr))
    assert result.returncode == 0, result.stdout + result.stderr


class _DummyLGN:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


def _build_generator(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(lgn_generator, "LGN", _DummyLGN)
    input_network = SimpleNamespace(
        name="lgn",
        n_nodes=696,
        _cache_file=str(tmp_path / "network_cache" / "lgn_network.pkl"),
    )
    return lgn_generator.LGNGenerator(
        rnn=SimpleNamespace(),
        name="lgn_inputs",
        input_network=input_network,
        stimulus_type="gray_screen",
        stimulus_options={"row_size": 80, "col_size": 120, "contrast": 0.0},
        **kwargs,
    )


def test_lgn_generator_uses_network_cache_dir_by_default(tmp_path, monkeypatch):
    _build_generator(tmp_path, monkeypatch)

    kwargs = _DummyLGN.last_kwargs
    cache_dir = tmp_path / "network_cache" / "lgn_cache"

    assert Path(kwargs["spon_frs_path"]).parent == cache_dir
    assert Path(kwargs["temp_krns_path"]).parent == cache_dir
    assert Path(kwargs["spatial_krns_path"]).parent == cache_dir
    assert Path(kwargs["spon_frs_path"]).name == "lgn_696_80x120.spontaneous.pkl"
    assert Path(kwargs["temp_krns_path"]).name == "lgn_696_80x120.temporal.pkl"
    assert Path(kwargs["spatial_krns_path"]).name == "lgn_696_80x120.spatial.pkl"


def test_lgn_generator_honors_custom_cache_file_and_overwrite(tmp_path, monkeypatch):
    cache_file = tmp_path / "custom_cache" / "lgn_inputs.pkl"
    prefix = cache_file.with_suffix("")
    stale_files = [
        prefix.parent / f"{prefix.name}.spontaneous.pkl",
        prefix.parent / f"{prefix.name}.temporal.pkl",
        prefix.parent / f"{prefix.name}.spatial.pkl",
    ]
    for file_path in stale_files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("stale")

    _build_generator(
        tmp_path,
        monkeypatch,
        cache_file=str(cache_file),
        cache_overwrite=True,
    )

    kwargs = _DummyLGN.last_kwargs
    assert kwargs["spon_frs_path"] == str(stale_files[0])
    assert kwargs["temp_krns_path"] == str(stale_files[1])
    assert kwargs["spatial_krns_path"] == str(stale_files[2])
    assert all(not file_path.exists() for file_path in stale_files)


def test_drifting_gratings_phase_is_keyword_compatible_and_cast(monkeypatch):
    signature = inspect.signature(lgn_generator.create_drifting_gratings_generator)
    parameters = list(signature.parameters)
    assert parameters.index("phase") > parameters.index("seed")

    captured = {}

    def make_movie(**kwargs):
        captured["phase"] = kwargs["phase"]
        return tf.zeros(
            (kwargs["image_duration"], kwargs["row_size"], kwargs["col_size"]),
            dtype=kwargs["dtype"],
        )

    class DummyLGNNetwork:
        n_nodes = 2

        def spatial_response(self, videos, bmtk_compat):
            return (videos,)

        def firing_rates_from_spatial(self, videos):
            return tf.ones((5, self.n_nodes), dtype=tf.float16)

    monkeypatch.setattr(lgn_generator, "make_drifting_grating_stimulus", make_movie)
    monkeypatch.setattr(
        lgn_generator,
        "movies_concat",
        lambda movie, pre_delay, post_delay, dtype: movie,
    )

    dataset = lgn_generator.create_drifting_gratings_generator(
        DummyLGNNetwork(),
        5,
        [0],
        2,
        0.04,
        0.8,
        3,
        4,
        current_input=True,
        dtype=tf.float16,
        phase=90,
    )
    next(iter(dataset))

    assert captured["phase"].dtype == tf.float16
