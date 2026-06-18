from types import SimpleNamespace

import numpy as np
import pytest

from bmtk.simulator.dpointnet.loss_functions import loss_utils
from bmtk.simulator.dpointnet.loss_functions.synchronization_loss import SynchronizationLoss


def _build_dummy_rnn():
    return SimpleNamespace(
        recurrent_network={
            'n_nodes': 4,
            'node_params': {
                'pop_name': ['e0', 'i0', 'e1', 'i1'],
            },
        }
    )


def test_synchronization_loss_uses_millisecond_window(monkeypatch):
    loaded_paths = []

    monkeypatch.setattr(loss_utils, 'get_pop_names', lambda network: np.array(['e0', 'i0', 'e1', 'i1']))
    monkeypatch.setattr('os.path.exists', lambda path: True)

    def fake_load(path, allow_pickle=True):
        loaded_paths.append(path)
        return np.ones((3, 20), dtype=np.float32)

    monkeypatch.setattr(np, 'load', fake_load)

    loss = SynchronizationLoss(
        _build_dummy_rnn(),
        t_start=200,
        t_end=500,
        neuropixels_data_dir='Synchronization_data',
    )

    assert loss._t_start_idx == 200
    assert loss._t_end_idx == 500
    assert loaded_paths == ['Synchronization_data/Fano_factor_v1/v1_fano_running_300ms_evoked.npy']


def test_synchronization_loss_rejects_second_style_window(monkeypatch):
    monkeypatch.setattr(loss_utils, 'get_pop_names', lambda network: np.array(['e0', 'i0', 'e1', 'i1']))
    monkeypatch.setattr('os.path.exists', lambda path: True)
    monkeypatch.setattr(np, 'load', lambda path, allow_pickle=True: np.ones((3, 20), dtype=np.float32))

    with pytest.raises(ValueError, match='multiply by 1000.*t_start=200, t_end=500'):
        SynchronizationLoss(_build_dummy_rnn(), t_start=0.2, t_end=0.5)
