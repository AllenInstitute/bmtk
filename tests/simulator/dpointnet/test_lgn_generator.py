from pathlib import Path
import inspect
from types import SimpleNamespace

import tensorflow as tf

import bmtk.simulator.dpointnet.input_modules.lgn_generator as lgn_generator


class _DummyLGN:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs


def _build_generator(tmp_path, monkeypatch, **kwargs):
    monkeypatch.setattr(lgn_generator, 'LGN', _DummyLGN)
    input_network = SimpleNamespace(
        name='lgn',
        n_nodes=696,
        _cache_file=str(tmp_path / 'network_cache' / 'lgn_network.pkl'),
    )
    return lgn_generator.LGNGenerator(
        rnn=SimpleNamespace(),
        name='lgn_inputs',
        input_network=input_network,
        stimulus_type='gray_screen',
        stimulus_options={'row_size': 80, 'col_size': 120, 'contrast': 0.0},
        **kwargs,
    )


def test_lgn_generator_uses_network_cache_dir_by_default(tmp_path, monkeypatch):
    _build_generator(tmp_path, monkeypatch)

    kwargs = _DummyLGN.last_kwargs
    cache_dir = tmp_path / 'network_cache' / 'lgn_cache'

    assert Path(kwargs['spon_frs_path']).parent == cache_dir
    assert Path(kwargs['temp_krns_path']).parent == cache_dir
    assert Path(kwargs['spatial_krns_path']).parent == cache_dir
    assert Path(kwargs['spon_frs_path']).name == 'lgn_696_80x120.spontaneous.pkl'
    assert Path(kwargs['temp_krns_path']).name == 'lgn_696_80x120.temporal.pkl'
    assert Path(kwargs['spatial_krns_path']).name == 'lgn_696_80x120.spatial.pkl'


def test_lgn_generator_honors_custom_cache_file_and_overwrite(tmp_path, monkeypatch):
    cache_file = tmp_path / 'custom_cache' / 'lgn_inputs.pkl'
    prefix = cache_file.with_suffix('')
    stale_files = [
        prefix.parent / f'{prefix.name}.spontaneous.pkl',
        prefix.parent / f'{prefix.name}.temporal.pkl',
        prefix.parent / f'{prefix.name}.spatial.pkl',
    ]
    for file_path in stale_files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text('stale')

    _build_generator(
        tmp_path,
        monkeypatch,
        cache_file=str(cache_file),
        cache_overwrite=True,
    )

    kwargs = _DummyLGN.last_kwargs
    assert kwargs['spon_frs_path'] == str(stale_files[0])
    assert kwargs['temp_krns_path'] == str(stale_files[1])
    assert kwargs['spatial_krns_path'] == str(stale_files[2])
    assert all(not file_path.exists() for file_path in stale_files)


def test_drifting_gratings_phase_is_keyword_compatible_and_cast(monkeypatch):
    signature = inspect.signature(lgn_generator.create_drifting_gratings_generator)
    parameters = list(signature.parameters)
    assert parameters.index('phase') > parameters.index('seed')

    captured = {}

    def make_movie(**kwargs):
        captured['phase'] = kwargs['phase']
        return tf.zeros((kwargs['image_duration'], kwargs['row_size'], kwargs['col_size']), dtype=kwargs['dtype'])

    class DummyLGNNetwork:
        n_nodes = 2

        def spatial_response(self, videos, bmtk_compat):
            return (videos,)

        def firing_rates_from_spatial(self, videos):
            return tf.ones((5, self.n_nodes), dtype=tf.float16)

    monkeypatch.setattr(lgn_generator, 'make_drifting_grating_stimulus', make_movie)
    monkeypatch.setattr(
        lgn_generator,
        'movies_concat',
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

    assert captured['phase'].dtype == tf.float16
