import pytest
from bmtk.utils.sonata.config import SonataConfig


def test_valid_config():
    _ = pytest.importorskip('jsonschema')
    cfg = SonataConfig.from_dict({
        "manifest": {
            "$BASE": "${configdir}"
        },
        "target_simulator": "NEURON",
        "target_simulator_version": ">=7.4",
        'run': {
            'tstop': 3000.0,
            'dt': 0.001
        },
        "networks": {
            "nodes": [
                {
                    "nodes_file": "nodes.h5",
                    "node_types_file": "node_types.csv"
                },
                {
                    "nodes_file": "nodes2.h5",
                    "node_types_file": "node_types2.csv"
                }
            ]
        },
        "output": {
            'output_dir': 'output',
            'spikes_file': "null"
        },
        "inputs": {
            "input1": {
                'input_type': 'spikes',
                'input_file': 'myspikes.csv'
            },
            "input2": {
                'input_type': 'voltage_clamp'
            }
        }
    })

    assert(cfg.validate())


def test_negative_tstop():
    _ = pytest.importorskip('jsonschema')
    cfg = SonataConfig.from_dict({
        'run': {
            'tstop': -1.0
        }
    })

    with pytest.raises(Exception):
        cfg.validate()


def test_missing_nodes_file():
    _ = pytest.importorskip('jsonschema')
    cfg = SonataConfig.from_dict({
        "networks": {
            "nodes": [
                {
                    "node_types_file": "node_types.csv"
                },
            ]
        }
    })

    with pytest.raises(Exception):
        cfg.validate()


def test_inputs():
    _ = pytest.importorskip('jsonschema')
    # valid inputs section
    cfg = SonataConfig.from_dict({
        "inputs": {
            "input1": {
                'input_type': 'str1',
                'input_file': 'str2',
                'trial': 'str2',
                'module': 'str',
                'electrode_file': 'str',
                'node_set': 'str',
                'random_seed': 100
            }
        }
    })
    assert(cfg.validate())

    # Base inputs
    cfg = SonataConfig.from_dict({
        "inputs": [{
            'input_type': 'spikes',
            'input_file': 'myspikes.csv'
        }]
    })
    with pytest.raises(Exception):
        cfg.validate()

    # missing input_type
    cfg = SonataConfig.from_dict({
        "inputs": {
            "input1": {
                'input_file': 'str2',
                'trial': 'str2',
                'module': 'myspikes.csv',
                'electrode_file': 'myspikes.csv',
                'node_set': 'myspikes.csv',
                'random_seed': 100
            }
        }
    })
    with pytest.raises(Exception):
        cfg.validate()


if __name__ == '__main__':
    test_valid_config()
    test_negative_tstop()
    test_missing_nodes_file()
    test_inputs()
