import pytest
from .conftest import *


@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_weight():
    def wmax(v1, v2):
        return max(v1, v2)

    def wmin(v1, v2):
        return min(v1, v2)

    add_weight_function(wmax)
    add_weight_function(wmin, 'minimum')

    assert('wmax' in py_modules.synaptic_weights)
    assert('minimum' in py_modules.synaptic_weights)
    assert('wmin' not in py_modules.synaptic_weights)
    wmax_fnc = py_modules.synaptic_weight('wmax')
    assert(wmax_fnc(1, 2) == 2)

    wmin_fnc = py_modules.synaptic_weight('minimum')
    assert(wmin_fnc(1, 2) == 1)
    py_modules.clear()


@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_weight_decorator():
    @synaptic_weight
    def wmax(v1, v2):
        return max(v1, v2)

    @synaptic_weight(name='minimum')
    def wmin(v1, v2):
        return min(v1, v2)

    assert('wmax' in py_modules.synaptic_weights)
    assert('minimum' in py_modules.synaptic_weights)
    assert('wmin' not in py_modules.synaptic_weights)
    wmax_fnc = py_modules.synaptic_weight('wmax')
    assert(wmax_fnc(1, 2) == 2)

    wmin_fnc = py_modules.synaptic_weight('minimum')
    assert(wmin_fnc(1, 2) == 1)
    py_modules.clear()


@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_synapse_model():
    def syn1():
        return 'Syn1'

    def syn2(p1, p2):
        return p1, p2

    add_synapse_model(syn1)
    add_synapse_model(syn2, 'synapse_2')

    assert('syn1' in py_modules.synapse_models)
    assert('synapse_2' in py_modules.synapse_models)
    assert('syn2' not in py_modules.synapse_models)

    syn_fnc = py_modules.synapse_model('syn1')
    assert(syn_fnc() == 'Syn1')

    syn_fnc = py_modules.synapse_model('synapse_2')
    assert(syn_fnc(1, 2) == (1, 2))
    py_modules.clear()


@pytest.mark.skipif(not nrn_installed, reason='NEURON is not installed')
def test_synapse_model_decorator():
    @synapse_model
    def syn1():
        return 'Syn1'

    @synapse_model(name='synapse_2')
    def syn2(p1, p2):
        return p1, p2

    assert('syn1' in py_modules.synapse_models)
    assert('synapse_2' in py_modules.synapse_models)
    assert('syn2' not in py_modules.synapse_models)

    syn_fnc = py_modules.synapse_model('syn1')
    assert(syn_fnc() == 'Syn1')

    syn_fnc = py_modules.synapse_model('synapse_2')
    assert(syn_fnc(1, 2) == (1, 2))
    py_modules.clear()


@pytest.mark.skip()
def test_cell_model():
    def hoc1():
        return "hoc"

    def hoc2(p1):
        return p1

    add_cell_model(hoc1)
    add_cell_model(hoc2, name='hoc_function')

    assert('hoc1' in py_modules.cell_models)
    assert('hoc_function' in py_modules.cell_models)
    assert('hoc2' not in py_modules.cell_models)

    hoc_fnc = py_modules.cell_model('hoc1')
    assert(hoc_fnc() == 'hoc')

    hoc_fnc = py_modules.cell_model('hoc_function')
    assert(hoc_fnc(1.0) == 1.0)


@pytest.mark.skip()
def test_cell_model_decorator():
    @cell_model
    def hoc1():
        return "hoc"

    @cell_model(name='hoc_function')
    def hoc2(p1):
        return p1

    assert('hoc1' in py_modules.cell_models)
    assert('hoc_function' in py_modules.cell_models)
    assert('hoc2' not in py_modules.cell_models)

    hoc_fnc = py_modules.cell_model('hoc1')
    assert(hoc_fnc() == 'hoc')

    hoc_fnc = py_modules.cell_model('hoc_function')
    assert(hoc_fnc(1.0) == 1.0)


@pytest.mark.skip()
def test_load_py_modules():
    import set_weights
    import set_syn_params
    import set_cell_params

    load_py_modules(cell_models=set_cell_params, syn_models=set_syn_params, syn_weights=set_weights)
    assert(all(n in py_modules.cell_models for n in ['Biophys1', 'IntFire1']))
    assert(isinstance(py_modules.cell_model('Biophys1'), types.FunctionType))
    assert (isinstance(py_modules.cell_model('IntFire1'), types.FunctionType))

    assert (all(n in py_modules.synapse_models for n in ['exp2syn']))
    assert (isinstance(py_modules.synapse_model('exp2syn'), types.FunctionType))

    assert (all(n in py_modules.synaptic_weights for n in ['wmax', 'gaussianLL']))
    assert (isinstance(py_modules.synaptic_weight('wmax'), types.FunctionType))
    assert (isinstance(py_modules.synaptic_weight('gaussianLL'), types.FunctionType))
