from bmtk.simulator.core.pyfunction_cache import *


def test_add_cell_model():
    assert(not py_modules.has_cell_model(directive='test1', model_type='biophysical'))

    add_cell_model(func=lambda *_: 100, directive='test1', model_type='biophysical')
    assert(py_modules.has_cell_model(directive='test1', model_type='biophysical'))
    cell_func = py_modules.cell_model('test1', 'biophysical')
    assert(cell_func() == 100)

    add_cell_model(func=lambda *_: 200, directive='test1', model_type='biophysical', overwrite=False)
    cell_func = py_modules.cell_model('test1', 'biophysical')
    assert(cell_func() == 100)

    add_cell_model(func=lambda *_: 300, directive='test1', model_type='biophysical', overwrite=True)
    cell_func = py_modules.cell_model('test1', 'biophysical')
    assert(cell_func() == 300)


def test_add_cell_model_decorator():
    # Decorator with no arguments should use function name as directive
    @cell_model
    def cell_model_dec1():
        return 100
    assert(py_modules.has_cell_model(directive='cell_model_dec1', model_type='*'))
    cell_func = py_modules.cell_model('cell_model_dec1', '*')
    assert(cell_func() == 100)

    # Decorator with "directive" and "model_type" argument specified
    @cell_model('test2', model_type='biophysical')
    def cell_model_dec_2():
        return 200
    assert(py_modules.has_cell_model(directive='test2', model_type='biophysical'))
    cell_func = py_modules.cell_model('test2', 'biophysical')
    assert(cell_func() == 200)

    # Test "overwrite" option
    @cell_model('test2', model_type='biophysical', overwrite=False)
    def cell_model_dec_3():
        return 300
    assert(py_modules.has_cell_model(directive='test2', model_type='biophysical'))
    cell_func = py_modules.cell_model('test2', 'biophysical')
    assert(cell_func() == 200)

    @cell_model('test2', model_type='biophysical', overwrite=True)
    def cell_model_dec_3():
        return 300
    assert(py_modules.has_cell_model(directive='test2', model_type='biophysical'))
    cell_func = py_modules.cell_model('test2', 'biophysical')
    assert(cell_func() == 300)


def test_add_model_nomodeltype():
    # add function for directive="test3" and model_type="mt1", no wildcard/fallthrough option
    add_cell_model(func=lambda *_: 100, directive='test3', model_type='mt1')
    assert(py_modules.has_cell_model(directive='test3', model_type='mt1'))
    assert(not py_modules.has_cell_model(directive='test3', model_type='mt2'))
    assert(not py_modules.has_cell_model(directive='test3', model_type='*'))
    assert(not py_modules.has_cell_model(directive='test3'))

    # Add function for any directive="test3" with any model_type
    add_cell_model(func=lambda *_: 200, directive='test3', model_type='*')
    assert(py_modules.has_cell_model(directive='test3', model_type='mt1'))
    assert(not py_modules.has_cell_model(directive='test3', model_type='mt2'))
    assert(py_modules.has_cell_model(directive='test3', model_type='*'))
    assert(py_modules.has_cell_model(directive='test3', model_type=''))
    assert(py_modules.has_cell_model(directive='test3', model_type=None))
    assert(py_modules.has_cell_model(directive='test3'))

    # Make sure it py_modules finds valid cell_model() function with "model_type=mt1"
    cell_func = py_modules.cell_model('test3', 'mt1')
    assert(cell_func() == 100)

    # No cell_model() function with "model_type=mt2", should instead return the one with "model_type=*"
    cell_func = py_modules.cell_model('test3', 'mt2')
    assert (cell_func() == 200)

    cell_func = py_modules.cell_model('test3')
    assert (cell_func() == 200)


def test_synpatic_weight():
    assert(not py_modules.has_synaptic_weight('syn_weight_fnc1'))
    add_weight_function(func=lambda *_: 0.01, name='syn_weight_fnc1')
    assert('syn_weight_fnc1' in py_modules.synaptic_weights)
    assert(py_modules.has_synaptic_weight(name='syn_weight_fnc1'))
    weight_func = py_modules.synaptic_weight(name='syn_weight_fnc1')
    assert(weight_func() == 0.01)

    assert(not py_modules.has_synaptic_weight(name='syn_weight2'))
    @synaptic_weight(name='syn_weight2')
    def syn_weight():
        return 0.02
    assert(py_modules.has_synaptic_weight(name='syn_weight2'))
    weight_func = py_modules.synaptic_weight(name='syn_weight2')
    assert(weight_func() == 0.02)

    assert(not py_modules.has_synaptic_weight(name='syn_weight3'))
    @synaptic_weight
    def syn_weight3():
        return 0.03
    assert(py_modules.has_synaptic_weight(name='syn_weight3'))
    weight_func = py_modules.synaptic_weight(name='syn_weight3')
    assert(weight_func() == 0.03)


def test_synpase_model():
    assert(not py_modules.has_synapse_model('syn_model_fnc1'))
    add_synapse_model(func=lambda *_: 'A', name='syn_model_fnc1')
    assert('syn_model_fnc1' in py_modules.synapse_models)
    assert(py_modules.has_synapse_model(name='syn_model_fnc1'))
    syn_func = py_modules.synapse_model(name='syn_model_fnc1')
    assert(syn_func() == 'A')

    assert(not py_modules.has_synapse_model(name='syn_model2'))
    @synapse_model(name='syn_model2')
    def syn_model():
        return 'B'
    assert(py_modules.has_synapse_model(name='syn_model2'))
    syn_func = py_modules.synapse_model(name='syn_model2')
    assert(syn_func() == 'B')

    assert(not py_modules.has_synapse_model(name='syn_model3'))
    @synapse_model
    def syn_model3():
        return 'C'
    assert(py_modules.has_synapse_model(name='syn_model3'))
    syn_func = py_modules.synapse_model(name='syn_model3')
    assert(syn_func() == 'C')


def test_cell_processor():
    assert (not py_modules.has_synapse_model('cell_proc1'))
    add_cell_processor(func=lambda *_: 'A', name='cell_proc1')
    assert('cell_proc1' in py_modules.cell_processors)
    assert(py_modules.has_cell_processor(name='cell_proc1'))
    cell_proc_func = py_modules.cell_processor(name='cell_proc1')
    assert(cell_proc_func() == 'A')

    assert(not py_modules.has_synapse_model(name='cell_proc2'))
    @synapse_model(name='cell_proc2')
    def cell_proc():
        return 'B'
    assert(py_modules.has_synapse_model(name='cell_proc2'))
    cell_proc_func = py_modules.synapse_model(name='cell_proc2')
    assert(cell_proc_func() == 'B')

    assert(not py_modules.has_synapse_model(name='cell_proc3'))
    @synapse_model
    def cell_proc3():
        return 'C'
    assert(py_modules.has_synapse_model(name='cell_proc3'))
    cell_proc_func = py_modules.synapse_model(name='cell_proc3')
    assert(cell_proc_func() == 'C')


# @cell_model(name='boo')
# def cell_model_dec():
#     pass

if __name__ == '__main__':
    # test_add_cell_model()
    # test_add_cell_model_decorator()
    # test_add_model_nomodeltype()
    # test_synpatic_weight()
    # test_synpase_model()
    test_cell_processor()
