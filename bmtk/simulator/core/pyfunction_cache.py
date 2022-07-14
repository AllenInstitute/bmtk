# Copyright 2020. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import types
import six
import warnings
from functools import wraps


class _PyFunctions(object):
    """Structure that allows users to add functions that will overwrite certain parts of bmtk, in particular; how
    cell-objects are created, how cells are processed, how synapatic objects are created, and how synaptic weights are
    calculated. To use the user must write a custom function in their own scripts and register it using the add()
    method or decorator. ex run_bionet.py::
        from bmtk.simulator.bionet import synaptic_weight

        @synaptic_weight
        def calc_custom_weight(edge_props, src_cell, trg_cell):
            syn_weight = ... # calulate updated weight using edge and cells props
            return syn_weight

    This will register the user function "calc_custom_weight(...)" in the py_modules object. Any edge object that has
    attribute weight_function=="calc_custom_weight" will use this new user-defined function to readjust weights. ie
    in the bmtk code where syn_weight value is calculate for each edge::
        from bmtk.simulator.bionet.pyfunction_cache import py_modules

        if 'weight_function' in edge_props:
            fnc_name = edge_props['weight_function']
            assert(py_modules.has_synaptic_weight(name=fnc_name)
            weight_fnc = py_modules.synaptic_weight(name=fnc_name)
            syn_weight = weight_fnc(edge_props, src_cell, trg_cell)
        else:
            syn_weight = ...  # default way to get weight


    The following functionality can be overwritten by the user.
        cell_model: function to create a NEURON/NEST/DiPDE cell object(s). Uses SONATA properties "model_template" and
            "model_type".

        cell_processor: Used to do some post-processing of cell object after being created, ex fixing part of morphology
            or updating channel variable parameters. Uses "model_processing" property.

        synapse_model: Used to create a synapse between two (sets of) cells. Uses SONATA edges "model_template"
            property.

        synaptic_weight: Used to update/recalculate synaptic weight of an edge (synapse or junction). Uses
            "weight_function" (not a default SONATA variable).
    """
    def __init__(self):
        self.__syn_weights = {}
        self.__cell_models = {}
        self.__synapse_models = {}
        self.__cell_processors = {}

    def clear(self):
        self.__syn_weights.clear()
        self.__cell_models.clear()
        self.__synapse_models.clear()
        self.__cell_processors.clear()

    def add_synaptic_weight(self, name, func, overwrite=True):
        """stores synaptic function for given name"""
        if overwrite or name not in self.__syn_weights:
            self.__syn_weights[name] = func

    @property
    def synaptic_weights(self):
        """return list of the names of all available synaptic weight functions"""
        return self.__syn_weights.keys()

    def synaptic_weight(self, name):
        """return the synaptic weight function"""
        return self.__syn_weights[name]

    def has_synaptic_weight(self, name):
        return name in self.__syn_weights

    def __cell_model_key(self, directive, model_type):
        return (directive, model_type)

    def add_cell_model(self, directive, model_type, func, overwrite=True):
        if not model_type:
            # use * to indicate "model_type" is not specified and therefore a wildcard option.
            model_type = '*'

        key = self.__cell_model_key(directive, model_type)
        if overwrite or key not in self.__cell_models:
            self.__cell_models[key] = func

    @property
    def cell_models(self):
        return self.__cell_models.keys()

    def cell_model(self, directive, model_type='*'):
        # Check to see if function exists with corresponding "directive" and "model_type". If not see if there is a
        # function with "directive" but "model_type=*" to act as fall-through option
        if self.has_cell_model(directive=directive, model_type=model_type):
            return self.__cell_models[self.__cell_model_key(directive, model_type)]

        elif model_type != '*' and self.has_cell_model(directive=directive, model_type='*'):
            return self.__cell_models[self.__cell_model_key(directive, '*')]

        else:
            raise ValueError('Could not find cell_model() function with directive="{}" and model_type="{}"'.format(
                directive, model_type
            ))

    def has_cell_model(self, directive, model_type='*'):
        if not model_type:
            model_type = '*'
        return self.__cell_model_key(directive, model_type) in self.__cell_models

    def add_synapse_model(self, name, func, overwrite=True):
        if overwrite or name not in self.__synapse_models:
            self.__synapse_models[name] = func

    def has_synapse_model(self, name):
        return name in self.__synapse_models

    @property
    def synapse_models(self):
        return self.__synapse_models.keys()

    def synapse_model(self, name):
        return self.__synapse_models[name]

    def has_cell_processor(self, name):
        return name in self.__cell_processors

    @property
    def cell_processors(self):
        return self.__cell_processors.keys()

    def cell_processor(self, name):
        return self.__cell_processors[name]

    def add_cell_processor(self, name, func, overwrite=True):
        if overwrite or name not in self.__syn_weights:
            self.__cell_processors[name] = func

    def __repr__(self):
        rstr = '{}: {}\n'.format('cell_models', self.cell_models)
        rstr += '{}: {}\n'.format('synapse_models', self.synapse_models)
        rstr += '{}: {}\n'.format('synaptic_weights', self.synaptic_weights)
        rstr += '{}: {}'.format('cell_processors', self.cell_processors)
        return rstr


py_modules = _PyFunctions()


def synaptic_weight(*wargs, **wkwargs):
    """A decorator for registering a function as a synaptic weight function.
    To use either::

        @synaptic_weight
        def weight_function():
            ...

    or::

        @synaptic_weight(name='name_in_edge_types')
        def weight_function():
            ...

    Once the decorator has been attached and imported the functions will automatically be added to py_modules and BMTK
    will when assigning synaptic/gap junction weights for edges with matching "weight_function" attribute
    """
    if len(wargs) == 1 and callable(wargs[0]):
        # for the case without decorator arguments, grab the function object in wargs and create a decorator
        func = wargs[0]
        py_modules.add_synaptic_weight(func.__name__, func)  # add function assigned to its original name

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # for the case with decorator arguments
        assert(all(k in ['name'] for k in wkwargs.keys()))
        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_synaptic_weight(wkwargs['name'], func)

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return func_wrapper
        return decorator


def cell_model(*wargs, **wkwargs):
    """A decorator for registering NEURON cell loader functions."""
    if len(wargs) == 1 and callable(wargs[0]):
        # for the case without decorator arguments, grab the function object in wargs and create a decorator
        func = wargs[0]
        py_modules.add_cell_model(
            directive=func.__name__,  # add function assigned to its original name
            model_type='*',
            func=func
        )

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # look for directive/name argument inside decorator arguments, otherwise default
        if 'directive' in wkwargs:
            directive = wkwargs['directive']
        elif 'name' in wkwargs:
            directive = wkwargs['name']
        elif len(wargs) >= 1 and isinstance(wargs[0], six.string_types):
            directive = wargs[0]
        elif len(wargs) >= 1 and callable(wargs[0]):
            directive = wargs[0].__name__
        else:
            raise ValueError('Please specify "directive" name in call_model() arguments')

        model_type = wkwargs.get('model_type', '*')
        overwrite = wkwargs.get('overwrite', True)

        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_cell_model(directive=directive, model_type=model_type, func=func, overwrite=overwrite)

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return func_wrapper
        return decorator


def synapse_model(*wargs, **wkwargs):
    """A decorator for registering NEURON synapse loader functions."""
    if len(wargs) == 1 and callable(wargs[0]):
        # for the case without decorator arguments, grab the function object in wargs and create a decorator
        func = wargs[0]
        py_modules.add_synapse_model(func.__name__, func)  # add function assigned to its original name

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # for the case with decorator arguments
        assert(all(k in ['name'] for k in wkwargs.keys()))

        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_synapse_model(wkwargs['name'], func)

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return func_wrapper
        return decorator


def add_weight_function(func, name=None, overwrite=True):
    assert(callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_synaptic_weight(func_name, func, overwrite)


def add_cell_model(func, directive, model_type='*', overwrite=True):
    assert(callable(func))
    py_modules.add_cell_model(directive=directive, model_type=model_type, func=func, overwrite=overwrite)


def add_cell_processor(func, name=None, overwrite=True):
    assert(callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_cell_processor(func_name, func, overwrite)


def add_synapse_model(func, name=None, overwrite=True):
    assert (callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_synapse_model(func_name, func, overwrite)


def load_py_modules(cell_models=None, syn_models=None, syn_weights=None, cell_processors=None):
    # py_modules.clear()
    warnings.warn('Do not call this method directly', DeprecationWarning)
    if cell_models is not None:
        assert(isinstance(cell_models, types.ModuleType))
        for f in [cell_models.__dict__.get(f) for f in dir(cell_models)]:
            if isinstance(f, types.FunctionType):
                py_modules.add_cell_model(f.__name__, f)

    if syn_models is not None:
        assert(isinstance(syn_models, types.ModuleType))
        for f in [syn_models.__dict__.get(f) for f in dir(syn_models)]:
            if isinstance(f, types.FunctionType):
                py_modules.add_synapse_model(f.__name__, f)

    if syn_weights is not None:
        assert(isinstance(syn_weights, types.ModuleType))
        for f in [syn_weights.__dict__.get(f) for f in dir(syn_weights)]:
            if isinstance(f, types.FunctionType):
                py_modules.add_synaptic_weight(f.__name__, f)

    if cell_processors is not None:
        assert(isinstance(cell_processors, types.ModuleType))
        for f in [cell_processors.__dict__.get(f) for f in dir(cell_processors)]:
            if isinstance(f, types.FunctionType):
                py_modules.add_cell_processor(f.__name__, f)
