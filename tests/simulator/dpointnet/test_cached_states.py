import pickle

import numpy as np

from bmtk.simulator.dpointnet.state_modules.cached_states import CachedInitState


def test_load_pkl_reads_state_without_modifying_cache(tmp_path):
    cache_path = tmp_path / "initial_state.pkl"
    expected = (np.array([[1.0, 2.0]], dtype=np.float32),)
    cache_path.write_bytes(pickle.dumps(expected))
    original_bytes = cache_path.read_bytes()

    actual = CachedInitState._load_pkl(cache_path, rnn=None)

    np.testing.assert_array_equal(actual[0], expected[0])
    assert cache_path.read_bytes() == original_bytes
