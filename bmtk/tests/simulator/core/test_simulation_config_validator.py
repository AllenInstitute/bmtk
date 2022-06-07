import pytest
import tempfile

from bmtk.simulator.core.simulation_config import SimulationConfig


def test_validate_network():
    # Check that validate() passes when all the files exists
    tmp_h5_file = tempfile.NamedTemporaryFile(suffix='.h5')
    tmp_csv_types_file = tempfile.NamedTemporaryFile(suffix='.csv')
    cfg = SimulationConfig.from_dict({
        "networks": {
            "nodes": [{
                "nodes_file": tmp_h5_file.name,
                "node_types_file": tmp_csv_types_file.name
            }],
            "edges": [{
                "edges_file": tmp_h5_file.name,
                "edge_types_file": tmp_csv_types_file.name
            }]
        }
    })
    assert(cfg.validate())

    cfg = SimulationConfig.from_dict({
        "networks": {
            "nodes": [{
                "nodes_file": tmp_h5_file.name,
                "node_types_file": 'nameoffilethatdoesntexists.csv'
            }]
        }
    })
    with pytest.raises(Exception):
        cfg.validate()

    # edges_file does not exist should raise error
    cfg = SimulationConfig.from_dict({
        "networks": {
            "edges": [{
                "edges_file": 'blah',
                "edge_types_file": tmp_csv_types_file.name
            }]
        }
    })
    with pytest.raises(Exception):
        cfg.validate()


def test_validate_components():
    cfg = SimulationConfig.from_dict({
        "components": {
            "synaptic_models_dir": tempfile.mkdtemp(),
            "morphologies_dir": tempfile.mkdtemp(),
            "biophysical_neuron_models_dir": tempfile.mkdtemp()
        }
    })
    assert(cfg.validate())

    cfg = SimulationConfig.from_dict({
        "components": {
            "point_neuron_models_dir": tempfile.mkdtemp(),
            "templates_dir": 'invaliddir',
        }
    })

    with pytest.raises(Exception):
        assert(cfg.validate())


def test_validate_inputs():
    spikes_file = tempfile.NamedTemporaryFile(suffix='.h5')
    cfg = SimulationConfig.from_dict({
        "inputs": {
            "valid_input": {
                "input_type": "spikes",
                "input_file": spikes_file.name
            }
        }
    })
    assert(cfg.validate())

    cfg = SimulationConfig.from_dict({
        "inputs": {
            "valid_input": {
                "input_type": "spikes",
                "input_file": 'notmyspikesfile.h5'
            }
        }
    })

    with pytest.raises(Exception):
        assert(cfg.validate())


if __name__ == '__main__':
    test_validate_network()
    # test_validate_components()
    # test_validate_inputs()
