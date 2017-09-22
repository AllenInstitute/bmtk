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
        self.__syn_weights = {}
        self.__cell_models = {}
        self.__synapse_models = {}

    def clear(self):
        self.__syn_weights.clear()
        self.__cell_models.clear()
        self.__synapse_models.clear()

    def add_synaptic_weight(self, name, func, overwrite=True):
        """stores synpatic fuction for given name"""
        if overwrite or name not in self.__syn_weights:
            self.__syn_weights[name] = func

    @property
    def synaptic_weights(self):
        """return list of the names of all available synaptic weight functions"""
        return self.__syn_weights.keys()

    def synaptic_weight(self, name):
        """return the synpatic weight function"""
        return self.__syn_weights[name]

    def add_cell_model(self, name, func, overwrite=True):
        if overwrite or name not in self.__cell_models:
            self.__cell_models[name] = func

    @property
    def cell_models(self):
        return self.__cell_models.keys()

    def cell_model(self, name):
        return self.__cell_models[name]

    def add_synapse_model(self, name, func, overwrite=True):
        if overwrite or name not in self.__synapse_models:
            self.__synapse_models[name] = func

    @property
    def synapse_models(self):
        return self.__synapse_models.keys()

    def synapse_model(self, name):
        return self.__synapse_models[name]

    def __repr__(self):
        rstr = '{}: {}\n'.format('cell_models', self.cell_models)
        rstr += '{}: {}\n'.format('synapse_models', self.synapse_models)
        rstr += '{}: {}'.format('synaptic_weights', self.synaptic_weights)
        return rstr

py_modules = _PyFunctions()


def synaptic_weight(*wargs, **wkwargs):
    """A decorator for registering a function as a synaptic weight function.
    To use either
      @synaptic_weight
      def weight_function(): ...

    or
      @synaptic_weight(name='name_in_edge_types')
      def weight_function(): ...

    Once the decorator has been attached and imported the functions will automatically be added to py_modules.
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
        py_modules.add_cell_model(func.__name__, func)  # add function assigned to its original name

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # for the case with decorator arguments
        assert(all(k in ['name'] for k in wkwargs.keys()))

        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_cell_model(wkwargs['name'], func)

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


def add_cell_model(func, name=None, overwrite=True):
    assert(callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_cell_model(func_name, func, overwrite)


def add_synapse_model(func, name=None, overwrite=True):
    assert (callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_synapse_model(func_name, func, overwrite)


def load_py_modules(cell_models=None, syn_models=None, syn_weights=None):
    py_modules.clear()

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
