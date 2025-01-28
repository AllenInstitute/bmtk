import pytest

from bmtk.simulator.utils.simulation_inputs import SimInput, from_config


def test_sim_inputs():
    sim_input = SimInput.build(
        'spikes_inputs',
        params={
            "input_type": "spikes",
            "module": "sonata",
            "input_file": "spikes.h5",
            "node_set": "lgn"
        }
    )

    assert(sim_input.name == 'spikes_inputs')
    assert(sim_input.input_type == 'spikes')
    assert(sim_input.module == 'sonata')
    assert(sim_input.params == {'input_file': 'spikes.h5', 'node_set': 'lgn'})
    assert(sim_input.node_set == 'lgn')
    assert(sim_input.enabled)


def test_sim_inputs_disable():
    sim_input = SimInput.build(
        'spikes_inputs',
        params={
            "input_type": "spikes",
            "module": "sonata",
            "input_file": "spikes.h5",
            "node_set": "lgn",
            "enabled": False
        }
    )

    assert(sim_input.name == 'spikes_inputs')
    assert(sim_input.input_type == 'spikes')
    assert(sim_input.module == 'sonata')
    assert(sim_input.params == {'input_file': 'spikes.h5', 'node_set': 'lgn'})
    assert(sim_input.node_set == 'lgn')
    assert(not sim_input.enabled)


def test_no_module():
    with pytest.raises(Exception):
        SimInput.build(
            'spikes_inputs',
            params={
                "input_type": "spikes",
                "input_file": "spikes.h5",
                "node_set": "lgn"
            }
        )


def test_no_input_type():
    with pytest.raises(Exception):
        SimInput.build(
            'spikes_inputs',
            params={
                "module": "sonata",
                "input_file": "spikes.h5",
                "node_set": "lgn"
            }
        )


def test_registry():
    class NWbInput(SimInput):
        def __init__(self, *params):
            super(NWbInput, self).__init__(*params)
            self.name = 'MY_NWB_MOD'

        @staticmethod
        def avail_modules():
            return ['nwb', 'nwb_mod']

    SimInput.register_module(NWbInput)

    input_mod = SimInput.build(
        'my_nwb_input',
        params={
            "module": "nwb_mod",
            "input_type": "spikes",
            "input_file": "spikes.h5",
            "node_set": "lgn"
        }
    )
    assert(isinstance(input_mod, NWbInput))
    assert(input_mod.name == 'MY_NWB_MOD')

    input_mod = SimInput.build(
        'my_nwb_input',
        params={
            "module": "nwb",
            "input_type": "spikes",
            "input_file": "spikes.h5",
            "node_set": "lgn"
        }
    )
    assert(isinstance(input_mod, NWbInput))
    assert(input_mod.name == 'MY_NWB_MOD')


if __name__ == '__main__':
    # test_sim_inputs()
    # test_sim_inputs_disable()
    # test_no_module()
    # test_no_input_type()
    test_registry()
