from functools import wraps
import tensorflow as tf

from .inputs_base import InputsGeneratorMod


_spike_functions = {}


def add_spikes_function(func, name=None, overwrite=True):
    if not callable(func):
        raise TypeError(f'add_spikes_function - {func} is not callable.')

    name = name or func.__name__
    if name in _spike_functions.keys() and not overwrite:
        raise RuntimeError(f'spikes generator function with name {name} already exists.'
                           ' Use "name" option to provide an alias or set overwrite=True')

    _spike_functions[name] = func


def spikes_function(_func=None, *_, **wkwargs):
    def decorator(func):
        fname = wkwargs.get('name', None)
        overwrite = wkwargs.get('overwrite', True)
        add_spikes_function(func=func, name=fname, overwrite=overwrite)

        @wraps(func)
        def add_function(*args, **kwargs):
            return func(*args, **kwargs)
        return add_function

    return decorator if _func is None else decorator(_func)


class SpikesFunctionGenerator(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, function_name, function_type='function', **kwargs):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        # self.name = name
        # self.network = input_network
        self.fnc_name = function_name
        self.fnc_ptr = _spike_functions[function_name]
        self.n_nodes = self.network.n_nodes
        self.input_network.options['input_type'] = 'spikes'
        
        _target_dtype = kwargs.get('target_dtype', tf.string)
        self.target_dtype = tf.dtypes.as_dtype(_target_dtype).name

    @staticmethod
    def module():
        return 'custom_spikes_functions'
       
    @staticmethod
    def input_type():
        return 'spikes'

    def create_generator(self, seq_len, dt=1.0, dtype=tf.float32, **kwargs):
        def _generator():
            while True:
                results = self.fnc_ptr(
                    seq_len=seq_len,
                    n_nodes=self.n_nodes,
                    network=self.network,
                    dtype=dtype,
                    dt=dt
                )
                if isinstance(results, (list, tuple)):
                    spikes = results[0]
                    target = results[1:]
                else:
                    spikes = results
                    target = self.fnc_name

                yield spikes, target


        data_set = tf.data.Dataset.from_generator(
            _generator, 
            output_signature=(
                tf.TensorSpec(shape=(seq_len, self.n_nodes), dtype=tf.float32),
                tf.TensorSpec(shape=(1,), dtype=self.target_dtype)
            )
        )

        return data_set
