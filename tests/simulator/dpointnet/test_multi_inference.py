from types import SimpleNamespace

import pytest

from bmtk.simulator.dpointnet.rnn_model import Inference, RNN
from bmtk.simulator.dpointnet.state_modules.cached_states import CachedInitState


class DummyResults:
    def __init__(self, name):
        self.name = name
        self.saved_outputs = []

    def save_results(self, **kwargs):
        self.saved_outputs.append(kwargs)


class DummyRNN(RNN):
    def __init__(self):
        self._model_built = True
        self._inferences = []
        self.runs = []

    def build(self):
        self._model_built = True

    @property
    def training_engine(self):
        return None

    def run_inference(self, spikes=None, initial_state=None, inference=None, **kwargs):
        inference_obj = self._select_inference(inference)
        self.runs.append(inference_obj.name)
        return DummyResults(inference_obj.name)


class DummyStateModule:
    def __init__(self):
        self.batch_sizes = []

    def get_state(self, batch_size=None):
        self.batch_sizes.append(batch_size)
        return f'state-batch-{batch_size}'


def test_normalized_inference_configs_keeps_legacy_schema():
    inference = {
        'inputs': ['lgn_dg0', 'bkg'],
        'initial_state': {'module': 'zero_state'},
        'output': {'output_dir': 'out', 'spikes_file': 'spikes.h5'},
    }

    assert RNN._normalized_inference_configs(inference) == [inference]


def test_normalized_inference_configs_expands_parameter_schema():
    inference = {
        'initial_state': {'module': 'zero_state'},
        'output': {'log_level': 'INFO'},
        'parameters': [
            {
                'name': 'dg0',
                'inputs': ['lgn_dg0', 'bkg'],
                'output': {'output_dir': 'out_dg0', 'spikes_file': 'spikes_dg0.h5'},
            },
            {
                'name': 'dg180',
                'inputs': ['lgn_dg180', 'bkg'],
                'output': {'output_dir': 'out_dg180', 'spikes_file': 'spikes_dg180.h5'},
            },
        ],
    }

    configs = RNN._normalized_inference_configs(inference)

    assert [config['name'] for config in configs] == ['dg0', 'dg180']
    assert configs[0]['initial_state'] == {'module': 'zero_state'}
    assert configs[0]['output'] == {
        'log_level': 'INFO',
        'output_dir': 'out_dg0',
        'spikes_file': 'spikes_dg0.h5',
    }
    assert configs[1]['output'] == {
        'log_level': 'INFO',
        'output_dir': 'out_dg180',
        'spikes_file': 'spikes_dg180.h5',
    }


def test_run_executes_all_inferences_and_returns_results_by_name():
    network = DummyRNN()
    dg0 = Inference(network, name='dg0')
    dg0.output_params = {'output_dir': 'out_dg0', 'spikes_file': 'spikes_dg0.h5'}
    dg180 = Inference(network, name='dg180')
    dg180.output_params = {'output_dir': 'out_dg180', 'spikes_file': 'spikes_dg180.h5'}
    network.add_inference(dg0)
    network.add_inference(dg180)

    results = network.run()

    assert network.runs == ['dg0', 'dg180']
    assert list(results) == ['dg0', 'dg180']
    assert results['dg0'].saved_outputs == [{'output_dir': 'out_dg0', 'spikes_file': 'spikes_dg0.h5'}]
    assert results['dg180'].saved_outputs == [{'output_dir': 'out_dg180', 'spikes_file': 'spikes_dg180.h5'}]


def test_add_inferences_from_config_builds_named_conditions(monkeypatch):
    network = DummyRNN()
    parsed_inputs = []

    def parse_input_mods_from_config(input_names):
        parsed_inputs.append(input_names)
        return [(input_name, SimpleNamespace(name=input_name)) for input_name in input_names]

    network.parse_input_mods_from_config = parse_input_mods_from_config
    config = SimpleNamespace(output=None)
    inference = {
        'initial_state': {'module': 'zero_state'},
        'batch_size': 1,
        'seq_len': 500,
        'parameters': [
            {
                'name': 'dg0',
                'inputs': ['lgn_dg0', 'bkg'],
                'output': {'output_dir': 'out_dg0'},
            },
            {
                'name': 'dg180',
                'inputs': ['lgn_dg180', 'bkg'],
                'output': {'output_dir': 'out_dg180'},
            },
        ],
    }

    class ZeroState:
        def __init__(self, rnn, **kwargs):
            self.rnn = rnn
            self.kwargs = kwargs

    monkeypatch.setattr(
        'bmtk.simulator.dpointnet.rnn_model.StateModules.get_init_state_module',
        lambda self, module: ZeroState,
    )

    RNN._add_inferences_from_config(network, config, inference)

    assert [inference.name for inference in network._inferences] == ['dg0', 'dg180']
    assert [inference.batch_size for inference in network._inferences] == [1, 1]
    assert [inference.seq_len for inference in network._inferences] == [500, 500]
    assert parsed_inputs == [['lgn_dg0', 'bkg'], ['lgn_dg180', 'bkg']]
    assert [inference.output_params for inference in network._inferences] == [
        {'output_dir': 'out_dg0'},
        {'output_dir': 'out_dg180'},
    ]
    assert all(inference.init_mod.kwargs == {'module': 'zero_state'} for inference in network._inferences)


def test_run_inference_requires_name_when_multiple_inferences_are_configured():
    network = DummyRNN()
    network.add_inference(Inference(network, name='dg0'))
    network.add_inference(Inference(network, name='dg180'))

    with pytest.raises(ValueError, match='Multiple inference conditions'):
        network._select_inference()

    assert network._select_inference('dg180').name == 'dg180'


def test_inference_initial_state_uses_inference_batch_size():
    network = SimpleNamespace(adjusted_batch_size=2, adjusted_seq_len=500)
    inference = Inference(network, name='dg0', batch_size=1, seq_len=500)
    inference.init_mod = DummyStateModule()

    assert inference.get_initial_state() == 'state-batch-1'
    assert inference.init_mod.batch_sizes == [1]


def test_cached_init_state_requires_rnn(tmp_path):
    cache_file = tmp_path / 'state.npz'
    cache_file.write_bytes(b'not-used')
    init_state = CachedInitState(str(cache_file), file_type='npz')

    with pytest.raises(ValueError, match='requires an RNN instance'):
        init_state.get_state()
