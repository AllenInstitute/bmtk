# Copyright 2017. Allen Institute. All rights reserved
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
from functools import wraps


class _PyFunctions(object):
    """Structure for holding custom user-defined python functions.

    Will store a set of functions created by the user. Should not access this directly but rather user the
    decorators or setter functions, and use the py_modules class variable to access individual functions. Is divided
    up into
    synaptic_weight: functions for calcuating synaptic weight.
    cell_model: should return NEURON cell hobj.
    synapse model: should return a NEURON synapse object.
    """
    def __init__(self):
        self.__cell_processors = {}

    def clear(self):
        self.__cell_processors.clear()

    @property
    def cell_processors(self):
        return self.__cell_processors.keys()

    def cell_processor(self, name):
        return self.__cell_processors[name]

    def add_cell_processor(self, name, func, overwrite=True):
        if overwrite or name not in self.__cell_processors:
            self.__cell_processors[name] = func

    def __repr__(self):
        return self.__cell_processors


py_modules = _PyFunctions()


def cell_processor(*wargs, **wkwargs):
    """A decorator for registering NEURON cell loader functions."""
    if len(wargs) == 1 and callable(wargs[0]):
        # for the case without decorator arguments, grab the function object in wargs and create a decorator
        func = wargs[0]
        py_modules.add_cell_processor(func.__name__, func)  # add function assigned to its original name

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # for the case with decorator arguments
        assert(all(k in ['name'] for k in wkwargs.keys()))

        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_cell_processor(wkwargs['name'], func)

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return func_wrapper
        return decorator


def add_cell_processor(func, name=None, overwrite=True):
    assert(callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_cell_processor(func_name, func, overwrite)


def load_py_modules(cell_processors):
    # py_modules.clear()
    assert (isinstance(cell_processors, types.ModuleType))
    for f in [cell_processors.__dict__.get(f) for f in dir(cell_processors)]:
        if isinstance(f, types.FunctionType):
            py_modules.add_cell_processor(f.__name__, f)
