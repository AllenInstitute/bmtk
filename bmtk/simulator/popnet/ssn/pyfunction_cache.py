from functools import wraps


class _PyFunctions(object):
    def __init__(self):
        self.__user_functions = {}
        self.__activation_funcs = {}

    @property
    def activation_functions(self):
        return self.__activation_funcs.keys()
    
    def activation_function(self, name):
        return self.__activation_funcs[name]
    
    def add_activation_function(self, name, func, overwrite=True):
        if overwrite or name not in self.__activation_funcs:
            self.__activation_funcs[name] = func

    @property
    def user_functions(self):
        return self.__user_functions.keys()
    
    def user_function(self, name):
        return self.__user_functions[name]
    
    def add_user_functions(self, name, func, overwrite=True):
        if overwrite or name not in self.__user_functions:
            self.__user_functions[name] = func


py_modules = _PyFunctions()


def inputs_generator(*wargs, **wkwargs):
    if len(wargs) == 1 and callable(wargs[0]):
        # for the case without decorator arguments, grab the function object in wargs and create a decorator
        func = wargs[0]
        py_modules.add_user_functions(func.__name__, func)  # add function assigned to its original name

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # for the case with decorator arguments
        assert(all(k in ['name'] for k in wkwargs.keys()))

        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_user_functions(wkwargs['name'], func)

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return func_wrapper
        return decorator

init_function = inputs_generator


def activation_function(*wargs, **wkwargs):
    if len(wargs) == 1 and callable(wargs[0]):
        # for the case without decorator arguments, grab the function object in wargs and create a decorator
        func = wargs[0]
        py_modules.add_activation_function(func.__name__, func)  # add function assigned to its original name

        @wraps(func)
        def func_wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return func_wrapper
    else:
        # for the case with decorator arguments
        assert(all(k in ['name'] for k in wkwargs.keys()))

        def decorator(func):
            # store the function in py_modules but under the name given in the decorator arguments
            py_modules.add_activation_function(wkwargs['name'], func)

            @wraps(func)
            def func_wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return func_wrapper
        return decorator


def add_function(func, name=None, overwrite=True):
    assert(callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_spikes_generator(func_name, func, overwrite)


def add_activation_function(func, name=None, overwrite=True):
    assert(callable(func))
    func_name = name if name is not None else func.__name__
    py_modules.add_activation_function(func_name, func, overwrite)
