import tensorflow as tf
import numpy as np
from functools import partial

from bmtk.simulator.dpointnet.io_tools import io


def binary(shape, dtype, firing_rate, dt, batch_size):
    _shape = (batch_size, shape)
    ms_fr = (dt/1000.0)*firing_rate
    return tf.cast(tf.random.uniform(shape=_shape, minval=0.0, maxval=1.0, dtype=dtype) < ms_fr, dtype)


def uniform(shape, dtype, low, high, batch_size):
    _shape = (batch_size, shape, )
    return tf.random.uniform(shape=_shape, minval=low, maxval=high, dtype=dtype)


def gaussian(shape, dtype, mean, stddev, batch_size):
    _shape = (batch_size, shape)
    return tf.random.normal(shape=_shape, mean=mean, stddev=stddev, dtype=dtype)


def constant(shape, dtype, value, batch_size):
    _shape = (batch_size, )
    return tf.constant(shape=_shape, value=value)


class RandomizedStateModule:
    def __init__(self, rnn, params, **kwargs):
        self._rnn = rnn
        self.params = params
        # self._rnn = kwargs.get('rnn', None)
        self._function_ptrs = None
        
    def _build_funcs(self, func_opts):
        fnc_name = func_opts['random_func']
        dtype = func_opts.get('dtype', self._rnn.dtype)
        if isinstance(dtype, str):
            dtype = tf.dtypes.as_dtype(dtype)
        shape = func_opts.get('shape', ())
       
        if fnc_name == 'binary':
            return partial(binary, shape=shape, dtype=dtype, firing_rate=func_opts['firing_rate'], dt=self._rnn.dt)

        elif fnc_name == 'gaussian':
            return partial(gaussian, shape=shape, mean=func_opts['mean'], stddev=func_opts['std'], dtype=dtype)

        elif fnc_name == 'uniform':
            return partial(uniform, shape=shape, low=func_opts['low'], high=func_opts['high'], dtype=dtype)

        elif fnc_name == 'const':
            return partial(constant, shape=shape, dtype=dtype, value=func_opts['value'])

        else:
            raise ValueError(f'Unknown random_func "{fnc_name}"')

    @property
    def function_pointers(self):
        if self._function_ptrs is None:
            if self._rnn is None:
                raise ValueError(f'{self.__class__}: Please set rnn before attempting to get params')
            
            zs_params, state_names = self._rnn.cell.zero_state(
                batch_size=self._rnn.batch_size, 
                dtype=self._rnn.dtype, 
                with_names=True
            )
            
            rand_funcs = []
            if isinstance(self.params, dict):
                if set(state_names) > set(self.params.keys()):
                    missing = set(state_names) - set(self.params.keys())
                    raise ValueError(f'{self.__class__}: The following state parameters are missing from RNN Column class {self._rnn.cell.__class__}; {missing}.')
                elif set(state_names) < set(self.params.keys()):
                    extra = set(self.params.keys()) - set(state_names)
                    io.log_warning(f'{self.__class__.__name__}: Contains extra parameters that are not used by {self._rnn.cell.__class__.__name__}: {list(extra)}')

                for name in state_names:
                    rand_funcs.append(self._build_funcs(self.params[name]))

            else:
                for func_params in self.params:
                    rand_funcs.append(self._build_funcs(func_params))

            self._function_ptrs = rand_funcs

        return self._function_ptrs

    def get_state(self, rnn=None, batch_size=None, **kwargs):
        if rnn is not None:
            self._rnn = rnn
        batch_size = batch_size or self._rnn.batch_size
        state_vals = [f(batch_size=batch_size) for f in self.function_pointers]
        return state_vals
