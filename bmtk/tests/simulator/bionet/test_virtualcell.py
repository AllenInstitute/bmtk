import pytest
import numpy as np
from .conftest import *
from neuron import h


# try:
#     h.load_file('stdrun.hoc')
# except Exception as e:
#     pass

class NRNPythonObj(object):
    def post_fadvance(self):
        pass

try:
    load_neuron_modules(mechanisms_dir='components/mechanisms', templates_dir='.')
    h.pysim = NRNPythonObj()
    has_mechanism = True
except AttributeError as ae:
    has_mechanism = False


class MockNode(object):
    node_id = 0


class MockSpikes(object):
    def __init__(self, spike_times):
        self.spikes = spike_times

    def get_times(self, node_id):
        return self.spikes


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
@pytest.mark.parametrize('spike_times', [
    [],
    np.array([]),
    None,
    [0.0, 0.2, 0.4, 0.6, 0.8],
    np.array([0.0, 0.2, 0.4, 0.6, 0.8]),
    [0.8, 0.2, 0.4, 0.6, 0.0],
    [0.5]
])
def test_spiketrain(spike_times):
    from bmtk.simulator.bionet.virtualcell import VirtualCell

    h.load_file('stdrun.hoc')
    vc = VirtualCell(node=MockNode(), population='test_pop', spike_train_dataset=MockSpikes(spike_times))

    # Create simple cell (soma) attach synapse using spike times from virtual cell and run, making sure
    soma = h.Section(name='soma')
    syn = h.Exp2Syn(0.5, sec=soma)
    nc = h.NetCon(vc.hobj, syn)
    nc.weight[0] = 2.0
    v = h.Vector().record(soma(0.5)._ref_v)
    h.run(1.0)

    assert(list(v))
    assert(len(v) > 0)


@pytest.mark.skipif(not has_mechanism, reason='Mechanisms has not been compiled, run nrnivmodl mechanisms.')
@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
@pytest.mark.parametrize('spike_times', [
    [-1.0, 0.0, 1.0],
    np.array([-1.0, 0.0, 1.0])
])
def test_spiketrains_negative(spike_times):
    from bmtk.simulator.bionet.virtualcell import VirtualCell

    # BMTK should raise it's own exception if spike-times contains a negative value
    with pytest.raises(Exception):
        vc = VirtualCell(node=MockNode(), population='test_pop', spike_train_dataset=MockSpikes(spike_times))


if __name__ == '__main__':
    test_spiketrain([0.001, 1.0, 5.0, 10.0])
    # test_spiketrains_negative()
    # test_spiketrain()
