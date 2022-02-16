import pytest
import os
import json
import numpy as np

nest = pytest.importorskip('nest')
from bmtk.simulator.pointnet.glif_utils import convert_aibs2nest
from bmtk.simulator.pointnet.nest_utils import NEST_SPIKE_DETECTOR

try:
    nest.Install('glifmodule')
except Exception as e:
    pass


@pytest.mark.parametrize('model_name,dynamics_params,expected', [
    ('nest:glif_lif_psc', 'glif_models/637930677_lif.json', [14.77547828]),
    ('nest:glif_lif_r_psc', 'glif_models/637930677_lif_r.json', [14.95315601, 26.00083926]),
    ('nest:glif_lif_asc_psc', 'glif_models/637930677_lif_asc.json', [14.46548367]),
    ('nest:glif_lif_r_asc_psc', 'glif_models/637930677_lif_r_asc.json', [14.52419554]),
    ('nest:glif_lif_r_asc_a_psc', 'glif_models/637930677_lif_r_asc_a.json', [13.70442843])
])
def test_converter_psc(model_name, dynamics_params, expected):
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.001})

    dyn_params_path = os.path.join(os.path.dirname(__file__), dynamics_params)
    aibs_params = json.load(open(dyn_params_path, 'r'))
    model_name, model_params = convert_aibs2nest(model_name, aibs_params)
    model_name = model_name.split(':')[-1]
    nrn = nest.Create(model_name, 1, model_params)

    sr = nest.Create(NEST_SPIKE_DETECTOR)
    espikes = nest.Create(
        'spike_generator',
        params={'spike_times': [5.0, 7.0, 8.0, 9.0, 10.0, 12.0], 'spike_weights': [100.0]*6}
    )
    nest.Connect(espikes, nrn, syn_spec={"receptor_type": 1})
    nest.Connect(nrn, sr)

    nest.Simulate(50.0)
    spikes = nest.GetStatus(sr, 'events')[0]['times']
    assert(len(spikes) == len(expected))
    assert(np.allclose(spikes, expected, atol=1.0, equal_nan=True))


if __name__ == '__main__':
    test_converter_psc(model_name='nest:glif_lif_psc', dynamics_params='glif_models/637930677_lif.json', expected=[14.77547828])
    test_converter_psc(model_name='nest:glif_lif_r_psc', dynamics_params='glif_models/637930677_lif_r.json', expected=[14.95315601, 26.00083926])
    test_converter_psc(model_name='nest:glif_lif_asc_psc', dynamics_params='glif_models/637930677_lif_asc.json', expected=[14.46548367])
    test_converter_psc(model_name='nest:glif_lif_r_asc_psc', dynamics_params='glif_models/637930677_lif_r_asc.json', expected=[14.52419554])
    test_converter_psc(model_name='nest:glif_lif_r_asc_a_psc', dynamics_params='glif_models/637930677_lif_r_asc_a.json', expected=[13.70442843])
    print('passed all')
