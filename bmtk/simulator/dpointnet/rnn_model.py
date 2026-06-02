import tensorflow as tf
import numpy as np
import pandas as pd
import pickle as pkl
from numbers import Number
from pathlib import Path

from bmtk.simulator.core.simulation_config import SimulationConfig
from bmtk.simulator.core.io_tools import io
from .network_adaptor import NetworkAdaptor
from .cell_models import cell_models, GLIF3Cell
from . import tf_utils
from .io_tools import io
from .input_modules import InputModules
from .state_modules import StateModules
from .results import RNNExtractorResults
from .loss_functions import LossModules
from . import optimizers
from .training import TrainingEngine
from .callbacks import callback_classes
from .id_maps import TFIDMap
from .weights import ModelWeights
from .data_iterator import DataIterator


class Inference:
    def __init__(self, rnn):
        self.rnn = rnn
        self.init_mod = None
        self.input_mods = []
        self._data_itr = None

        self.output_params = None

    @property
    def data_itr(self):
        if self._data_itr is None:
            self._data_itr = DataIterator(
                input_mods=self.input_mods,
                batch_size=self.rnn.adjusted_batch_size,
                seq_len=self.rnn.adjusted_seq_len,
                ordered_populations=self.rnn.ordered_inputs_populations
            )
            self._data_itr.build()
        
        return self._data_itr

    def get_initial_state(self):
        if self.init_mod is None:
            return None
        else:
            return self.init_mod.get_state()

    def close(self):
        if self._data_itr is not None:
            self._data_itr.close()
            self._data_itr = None

class RNN:
    """The primary model and simulation class for dpointnet.

    This class is used to store and coordinate all the neccessary information to build, train, and run predctions
    on the GLIF Recurrent Neural Network model. Includes storing information about the original (sonata) network, 
    the RNN model, and any spiking and state inputs used during training and prediction. It can also be used to 
    extract weights and results from any previous training/predictions.


    
    """

    def __init__(self, seq_len=None, batch_size=None, dtype='float32', dt=1.0, default_seed=None, cell_cls=None, cell_params=None, **kwargs):
        if cell_cls is None:
            self.cell_cls = cell_models['default']
        elif isinstance(cell_cls, str):
            if cell_cls not in cell_cls.keys():
                raise ValueError(f'Unrecognized RNN Cell class {cell_cls}')
            self.cell_cls = cell_models[cell_cls]
        else:
            self.cell_cls = cell_cls
        
        self.cell_params = cell_params
        self._components = {}
        self._node_populations = {}
        self.io = io
        self.model = None
        self._state_only_model = None
        self._rsnn_layer = None
        self.extractor_model = None
        self.zero_state = None
        self._cell = None
        self._training_engine = None

        self._recurrent_networks = {}
        self._built_recurrent_net = None

        self._model_built = False
        
        
        self._input_networks = {}  # 
        self._inputs_order = []
        self._inputs_dict = None

        self._input_generators_mods = {}
        # self._inputs_dict = None
        self._init_state = None

        self._inferences = []
        # self._inference_inputs = {}

        self.precision_module, self.dtype = tf_utils.get_precision_policy_and_dtype(dtype)
        self.seq_len = seq_len
        self.dt = dt
        self.default_seed = default_seed

        self.batch_size = batch_size
        self._adjusted_batch_size = None

        tf_utils.enable_gpu_memory_growth()
        tf_utils.enable_tensorflow_optimizations()
        # MirroredStrategy (matches the reference V1_GLIF_model). With variables created in
        # strategy.scope() (see build()), the connectivity-variable reads are hoisted out of
        # the RNN while_loop as loop-invariant captures instead of being stacked per timestep;
        # OneDeviceStrategy does NOT do this and OOMs on the full network. Single visible GPU
        # => effectively one device.
        self.strategy = tf.distribute.MirroredStrategy()

    @property
    def input_modules(self):
        return self._input_generators_mods

    def get_inputs_mods(self, mod_names=None):
        if mod_names is None:
            return list(self._input_generators_mods.values())
        elif isinstance(mod_names, str):
            return self._input_generators_mods[mod_names]
        else:
            mods = []
            for mname in mod_names:
                mods.append(self._input_generators_mods[mname])

    def get_init_state(self, state_names=None):
        return self._init_state

    @property
    def inference(self):
        return self._inferences[0]

    @property
    def node_populations(self):
        return self._node_populations.values()

    @property
    def training_engine(self):
        return self._training_engine

    @property
    def adjusted_batch_size(self):
        if self.training_engine is not None:
            return self.training_engine.adjusted_batch_size
        else:
            return self.batch_size

    @property
    def adjusted_seq_len(self):
        if self.training_engine is not None:
            return self.training_engine.adjusted_seq_len
        else:
            return self.seq_len

    @property
    def inputs_populations(self):
        return self._inputs_order

    @property
    def ordered_inputs_populations(self):
        return self._inputs_order

    def get_input_network(self, population_name):
        return self._input_networks[population_name]
    
    # @property
    # def inputs(self):
    #     if self._inputs_dict is None:
    #         self._inputs_dict = {i.name: i.to_dict() for i in self._input_networks.values()}

    #     return self._inputs_dict

    def add_inference(self, inference):
        self._inferences.append(inference)

    def add_component(self, name, path):
        self._components[name] = path

    def has_component(self, name):
        return name in self._components

    def get_component(self, name):
        if name not in self._components:
            self.io.log_exception(f'No network component set with name "{name}"')
        else:
            return self._components[name]

    def add_nodes(self, node_population):
        pop_name = node_population.name
        if pop_name in self._node_populations:
            # Make sure their aren't any collisions
            self.io.log_exception('There are multiple node populations with name {}.'.format(pop_name))

        node_population.initialize(self)
        self._node_populations[pop_name] = node_population
        # if node_population.mixed_nodes:
        #     # We'll allow a population to have virtual and non-virtual nodes but it is not ideal
        #     self.io.log_warning(('Node population {} contains both virtual and non-virtual nodes which can cause ' +
        #                          'memory and build-time inefficency. Consider separating virtual nodes into their ' +
        #                          'own population').format(pop_name))

        # Used in inputs/reports when needed to get all gids belonging to a node population
        # self._node_sets[pop_name] = NodeSet({'population': pop_name}, self)

    def add_network(self, network, **args):
        if isinstance(network, NetworkAdaptor):
            rec_nets, input_nets = [], []
            if network.network_type == 'recurrent':
                if network.name in self._recurrent_networks:
                    raise ValueError(f'Multiple recurrent networks with name {network.name}.')
                rec_nets.append(network)
                self._recurrent_networks[network.name] = network

            elif network.network_type == 'input':
                if network.name in self._input_networks:
                    raise ValueError(f'Multiple input networks with name {network.name}.')
                input_nets.append(input_nets)
                self._inputs_order.append(network.name)
                self._input_networks[network.name] = network

        elif isinstance(network, dict):
            rec_nets, input_nets = NetworkAdaptor.from_dict(network)

            for rec_net in rec_nets:
                if rec_net.name in self._recurrent_networks:
                    raise ValueError(f'Multiple recurrent networks with name {rec_net.name}; {network._recurrent_networks[rec_net.name].file_path}, {rec_net.file_path}')
                self._recurrent_networks[rec_net.name] = rec_net

            for in_net in input_nets:
                if in_net.name in self._input_networks:
                    raise ValueError(f'Multiple input networks with name {in_net.name}; {network._input_networks[in_net.name].file_path}, {in_net.file_path}')
                self._input_networks[in_net.name] = in_net

        else:
            raise NotImplementedError

        return rec_nets, input_nets

    '''
    def add_network_orig(
                self, cache_file=None, network_type=None, name=None, 
                sonata_nodes=None, sonata_node_types=None, 
                sonata_edges=None, sonata_edge_types=None, 
                **opt_args
            ):
        
        rec_nets, input_nets = NetworkAdaptor.from_params(
            cache_file=cache_file, network_type=network_type, network_name=name
        )

        for rec_net in rec_nets:
            if rec_net.name in self._recurrent_networks:
                raise ValueError(f'Multiple recurrent networks with name {rec_net.name}; {self._recurrent_networks[rec_net.name].file_path}, {rec_net.file_path}')
            self._recurrent_networks[rec_net.name] = rec_net

        for in_net in input_nets:
            if in_net.name in self._input_networks:
                raise ValueError(f'Multiple input networks with name {in_net.name}; {self._input_networks[in_net.name].file_path}, {in_net.file_path}')
            self._input_networks[in_net.name] = in_net
    '''
    
    @property
    def cell(self):
        if self._cell is None:
            self.build()

        return self._cell

    def get_model_weights(self, trainable_only=False, deep_copy=True):
        # TODO: Check that model is built
        return ModelWeights(
            rnn=self, 
            cell=self.cell,
            trainable_only=trainable_only,
            deep_copy=deep_copy
        )

    def get_network(self, network_name):
        if network_name in self._recurrent_networks:
            return self._recurrent_networks[network_name]
        elif network_name in self._input_networks:
            return self._input_networks[network_name]
        elif network_name == '<recurrent>':
            return self.get_recurrent_network()
        else:
            raise ValueError(f'No network named {network_name}') 

    @property
    def recurrent_network(self):
        if self._built_recurrent_net:
            return self._built_recurrent_net
        
        for network in self._recurrent_networks.values():
            self._built_recurrent_net = network.to_dict()

        return self._built_recurrent_net

    def add_input(self, name, mod):
        self._input_generators_mods[name] = mod

    def build(self, rebuild=False, seq_len=None, dtype=tf.float32, use_dummy_state_input=False, use_state_input=True, n_output=2, return_state=True, batch_size=None, **kwargs):
        if self._model_built and not rebuild:
            io.log_debug('Model already built. Skipping.')
            return

        _batch_size = batch_size or self.adjusted_batch_size
        _seq_len = seq_len or self.adjusted_seq_len

        # Pre-populate the global syn_id mapping by scanning all networks (recurrent + inputs)
        # BEFORE building any dict. This ensures the recurrent network's basis_weights table
        # includes entries for input network dynamics_params as well.
        from bmtk.simulator.dpointnet.network_adaptor import SONATANetwork
        SONATANetwork.reset_global_syn_id_mapping()
        all_networks = list(self._recurrent_networks.values()) + list(self._input_networks.values())
        for net in all_networks:
            if hasattr(net, '_synaptic_dyn_params'):
                net._synaptic_dyn_params()

        # Force rebuild of recurrent network dict (don't use cached version)
        self._built_recurrent_net = None
        network = self.recurrent_network
        n_spiking_inputs = sum(i.n_spiking_nodes for i in self._input_networks.values())
        n_neurons = network['n_nodes']
        inputs_dicts = {i.name: i.to_dict() for i in self._input_networks.values()}

        extrn_inputs = tf.keras.layers.Input(
            shape=(None, n_spiking_inputs,),
            dtype=self.dtype,
            name='external inputs'
        )

        if use_dummy_state_input:
            internal_state_inputs = tf.keras.layers.Input(
                shape=(None, n_neurons),
                name='internal state inputs'
            )
            full_inputs = tf.concat((extrn_inputs, internal_state_inputs), -1)
        else:
            state_input_holder = None
            full_inputs = extrn_inputs

        # Build the model (cell + RNN + variables) INSIDE the distribution strategy scope.
        # Required for MirroredStrategy (variables must be created in scope), and it is also
        # what makes the loop-invariant connectivity-variable reads be hoisted out of the RNN
        # while_loop instead of stacked per timestep (the cause of full-network OOM). Mirrors
        # the reference V1_GLIF_model, which builds create_model() within strategy.scope().
        cell_params = {} if self.cell_params is None else self.cell_params
        with self.strategy.scope():
            self._cell = self.cell_cls(network, inputs=inputs_dicts, train_recurrent_per_type=False, **cell_params)
            self.zero_state, state_names = self._cell.zero_state(self.batch_size, self.dtype, with_names=True)

            if use_state_input:
                initial_state_holder = tuple(
                    tf.keras.layers.Input(shape=s.shape[1:], dtype=s.dtype, name=n)
                    for s, n in zip(self.zero_state, state_names)
                )
                rnn_initial_state = tf.nest.map_structure(tf.identity, initial_state_holder)
            else:
                initial_state_holder = None
                rnn_initial_state = self.zero_state

            rnn = tf.keras.layers.RNN(self._cell, return_sequences=True, return_state=return_state, name='rsnn')
            # Keep the cell's provided state dtypes instead of letting Keras autocast them to the
            # compute dtype. Matches the reference (V1_GLIF_model create_model).
            rnn._autocast = False
            rnn_layer = rnn(full_inputs, initial_state=rnn_initial_state)

            rnn_out = rnn_layer[0] if return_state else rnn_layer
            spikes_output = rnn_out[0]

            output = tf.keras.layers.Dense(n_output, name='projection', trainable=False)(spikes_output)

            if use_state_input:
                if use_dummy_state_input:
                    inputs = [extrn_inputs, state_input_holder, initial_state_holder]
                else:
                    inputs = [extrn_inputs, initial_state_holder]
            else:
                if use_dummy_state_input:
                    inputs = [extrn_inputs, state_input_holder]
                else:
                    inputs = [extrn_inputs]

            self.model = tf.keras.Model(inputs=inputs, outputs=[output])
            self.model.build((_batch_size, _seq_len, n_spiking_inputs))
        self._model_built = True

    @property
    def rsnn_layer(self):
        if self._rsnn_layer is None:
            self._rsnn_layer = self.model.get_layer('rsnn')

        return self._rsnn_layer

    @property
    def state_only_model(self):
        if self._state_only_model is None:
            rnn_inputs = self.rsnn_layer.input
            if isinstance(rnn_inputs, (list, tuple)):
                full_inputs = rnn_inputs[0]
                state_inputs = list(rnn_inputs[1:])
            else:
                full_inputs = rnn_inputs
                state_inputs = []
            state_rnn = tf.keras.layers.RNN(
                self.rsnn_layer.cell, 
                return_sequences=False, 
                return_state=True, 
                name='rsnn_state'
            )
            # Preserve heterogeneous state dtypes for the state rollout path as well.
            state_rnn._autocast = False
            if state_inputs:
                state_out = state_rnn(full_inputs, initial_state=state_inputs)
            else:
                state_out = state_rnn(full_inputs)
            self._state_only_model = tf.keras.Model(
                inputs=self.model.inputs, 
                outputs=state_out[1:]
            )

        return self._state_only_model

    def run_inference(self, spikes=None, initial_state=None, **kwargs):
        if not self._model_built:
            self.build()
        
        # Fetch the spikes
        if spikes is None:
            if len(self._inferences) > 1:
                raise NotImplementedError()
            elif len(self._inferences) == 1:
                spikes, y = self._inferences[0].data_itr.next_spikes()
        
        elif isinstance(spikes, DataIterator):
            spikes, y = spikes.next_spikes()

        # elif isinstance(spikes, (list, tuple)):
        #     spikes = tf.concat(spikes, axis=2)

        # Fetch model init state
        if initial_state is None:
            if len(self._inferences) > 1:
                raise NotImplementedError()
            elif len(self._inferences) == 1:
                initial_state = self._inferences[0].get_initial_state()
       
        # Get version of model for pass-through only and reutrns spikes, voltages, and model states
        if self.extractor_model is None:
            self.extractor_model = tf.keras.Model(
                inputs=self.model.inputs, 
                outputs=self.model.get_layer('rsnn').output
            )

        # Run inputs through the model; fetch, package and return results
        out = self.extractor_model((spikes, initial_state))
        extractor_results = RNNExtractorResults(
            seq_len=self.seq_len,
            dt=self.dt,
            batch_size=self.batch_size,
            extractor_results=out
        )
        return extractor_results

    def cleanup(self):
        for inference in self._inferences:
            inference.close()

    def train(self, training_engine=None):
        with self.strategy.scope():
            if self.extractor_model is None:
                self.extractor_model = tf.keras.Model(
                    inputs=self.model.inputs,
                    outputs=self.model.get_layer('rsnn').output
                )

        training_engine = training_engine or self.training_engine
        if training_engine is None:
            io.log_debug('No training condition has been set, skipping training.')

        ## Build the optimizer (in strategy scope so its slot variables are created correctly)
        with self.strategy.scope():
            optimizer = training_engine.optimizer
            optimizer.build(self.model.trainable_variables)

        if self.dtype == 'float16':
            # Prevent gradient underflow in mixed-float16 training. The wrapped optimizer
            # must be set back on the engine so the train step scales the loss / unscales
            # the gradients (scale_loss_for_optimizer only acts on a LossScaleOptimizer).
            from tensorflow.keras import mixed_precision as mixed_precision_module
            optimizer = mixed_precision_module.LossScaleOptimizer(optimizer)
            training_engine.set_optimizer(optimizer)

        training_engine.train()

    def run(self):
        """A generic function to run a full execution of the RNN including building, training, running inference, and 
        saving results (only training/inference/output parameters have been added to the RNN in the config or API). 

        This is primarily designed for use with the SONATA config, so that user may run::

            $ python run_dpointnet.py config.json

        and not have to change any of the python code - only the json file. For more grainular control you can call
        the `build(...)`, `train(...)`, and `run_inference(...)` function directly through the Python API.
        """
        io.log_info('RNN.run() starting...')
        if not self._model_built:
            io.log_info('Building Model.')
            self.build()
        
        if self.training_engine:
            io.log_info('Training Model.')
            self.train()

        results = None
        if len(self._inferences) > 1:
            raise NotImplementedError()
        elif len(self._inferences) == 1:
            io.log_info('Runing Inference on Model.')
            results = self.run_inference()
            inference = self._inferences[0]
            if inference.output_params:
                io.log_info('Saving Results to file.')
                results.save_results(**inference.output_params)

        io.log_info('RNN.run() completed.')
        return results


    def add_init_state(self, mod):
        self._init_state = mod

    '''
    def predict(self, spike_inputs, initial_state=None, **kwargs):
        if self.extractor_model is None:
            self.extractor_model = tf.keras.Model(
                inputs=self.model.inputs, 
                outputs=self.model.get_layer('rsnn').output
            )
        
        if initial_state is None:
            initial_state = self.zero_state

        out = self.extractor_model((spike_inputs, initial_state))
        return out[0]
    '''

    @staticmethod
    def parse_initial_states_from_config(network, json_config):
        if isinstance(json_config, list):
            init_state_list = [(None, is_params) for is_params in json_config]
        elif isinstance(json_config, dict):
            init_state_list = list(json_config.items())
        else:
            raise RuntimeError()

        for name, params in init_state_list:
            if isinstance(params, str):
                mod = network.parent.get_init_state(name=name)
                network.add_initial_state(mod)
            
            else:
                enabled = params.get('enabled', True)
                if not enabled:
                    continue

                if name is None:
                    if 'name' not in params:
                        raise ValueError(f'Missing "name" for ')
                    name = params['name']

                    mod_cls = StateModules().get_init_state_module(params['module'])
                    network.add_init_state(mod_cls(name=name, rnn_net=network, **params))

    def parse_input_mods_from_config(self, json_config):
        inputs_modules_lu = InputModules()
        
        mod_params_list = []
        # mod_names, mod_params = [], []
        
        mod_instances = []
        if json_config is None or len(json_config) == 0:
            return []
        elif isinstance(json_config, list):
            mod_params_list = [(None, params) for params in json_config]
        else:
            mod_params_list = json_config.items()
        for mod_name, mod_params in mod_params_list:
            if isinstance(mod_params, str):
                if mod_params not in self.input_modules:
                    raise ValueError(f'Could not find pre-instanced input module {mod_params}. Available options: {", ".join(self.input_modules.keys())}')
                mod = self.input_modules[mod_params]
                if mod_name is None:
                    mod_name = mod.name
                mod_instances.append((mod_name, mod))
            else:
                enabled = mod_params.get('enabled', True)
                if not enabled:
                    continue

                if mod_name is None:
                    if 'name' not in mod_params:
                        raise ValueError(f'Missing "name" for ')
                    mod_name = mod_params['name']

                input_population = mod_params['node_set']
                input_network = self.get_input_network(input_population)

                # Honor config-level input weight options (e.g. trainable, weight_scale) so the
                # cell's input weights follow the config. These end up in the network's options
                # dict, which is what to_dict() exposes and the GLIF cell reads when creating the
                # input weight Variable. Without this the config's per-input "trainable" was a
                # no-op and inputs (incl. the background) could never be trained (cf. the
                # reference V1_GLIF_model which trains the bkg "rest_of_brain" weights).
                for _opt in ('trainable', 'weight_scale'):
                    if _opt in mod_params:
                        input_network.options[_opt] = mod_params[_opt]

                io.log_info(f'Building "{mod_name}" inputs for {input_population}')
                module_cls = inputs_modules_lu.get_module(
                    input_type=mod_params['input'],
                    module_name=mod_params['module']
                )
                mod = module_cls(
                    rnn=self,
                    name=mod_name,
                    input_network=input_network,
                    **mod_params
                )
                mod_instances.append((mod_name, mod))

        return mod_instances

    def set_training(self, training_engine=None, **training_params):
        if training_engine is not None:
            self._training_engine = training_engine
        else:
            self._training_engine = TrainingEngine(**training_params)

        return self._training_engine

    # def split_train_step(self, x, y, init_state):
    #     self.distributed_train_step(x, y, init_state)

    def get_recurrent_network(self):
        if len(self._recurrent_networks) > 1:
            raise NotImplementedError()
        else:
            return list(self._recurrent_networks.values())[0]



    @property
    def input_populations(self):
        pass


    @classmethod
    def from_config(cls, config, **kwargs):        
        # load components
        if isinstance(config, SimulationConfig):
            config = config
        else:
            try:
                config = SimulationConfig.load(config)
            except Exception as e:
                io.log_exception(f'Could not convert {config} (type "{type(config)}") to json.')

        # for name, value in config.components.items():
        #     network.add_component(name, value)

        # cell_type = config.get('RNN_cell', GLIF3Cell)
        
        cell_model_params = config['rnn_cell_params']
        cell_model_str = cell_model_params.pop('cell_model')
        cell_model_cls = cell_models.get(cell_model_str, GLIF3Cell)

        network = cls(
            cell_cls=cell_model_cls,
            cell_params=cell_model_params, 
            **config.run
        )

        rec_nets, input_nets = NetworkAdaptor.from_dict(config.networks)
        for net in rec_nets + input_nets:
            network.add_network(net)

        """
        for rec_net in rec_nets:
            if rec_net.name in network._recurrent_networks:
                raise ValueError(f'Multiple recurrent networks with name {rec_net.name}; {network._recurrent_networks[rec_net.name].file_path}, {rec_net.file_path}')
            network._recurrent_networks[rec_net.name] = rec_net

        for in_net in input_nets:
            if in_net.name in network._input_networks:
                raise ValueError(f'Multiple input networks with name {in_net.name}; {network._input_networks[in_net.name].file_path}, {in_net.file_path}')
            network._input_networks[in_net.name] = in_net
        """
            
        if config.components:
            for net in network._recurrent_networks.values():
                net.add_components_dirs(config.components)
            for net in network._input_networks.values():
                net.add_components_dirs(config.components)

        for input_name, input_mod in network.parse_input_mods_from_config(config.inputs):
            network.add_input(name=input_name, mod=input_mod)
            # network._inference_inputs[input_name] = input_mod

        # inputs_modules_lu = InputModules()
        # for input_name, input_props in config.inputs.items():
        #     enabled = input_props.get('enabled', True)
        #     if not enabled:
        #         continue

        #     input_population = input_props['node_set']
        #     input_network = network.get_input_network(input_population)

        #     io.log_info(f'Building "{input_name}" inputs for {input_population}')
        #     module_cls = inputs_modules_lu.get_module(
        #         input_type=input_props['input'],
        #         module_name=input_props['module']
        #     )
        #     input_mod = module_cls(
        #         rnn=network,
        #         name=input_name,
        #         input_network=input_network,
        #         **input_props
        #     )
        #     network.add_input(name=input_name, mod=input_mod)

        #     network._inference_inputs[input_name] = input_mod

        init_state_params = config.get('initial_state', {})
        if init_state_params:
            mod_cls = StateModules().get_init_state_module(init_state_params['module'])
            network.add_init_state(mod_cls(rnn=network, **init_state_params))


        # init_states = config.get('initial_states', None)
        # if init_states:
        #     network.parse_initial_states_from_config(init_states)


        train_dict = config.get('training', None)
        if train_dict:                        
            n_epochs = train_dict['n_epochs']
            steps_per_epoch = train_dict['steps_per_epoch']
            training_approach = train_dict.get('training_approach', None)
            gradient_checkpointing = train_dict.get('gradient_checkpointing', False)
            training_engine = network.set_training(
                rnn=network,
                n_epochs=n_epochs,
                steps_per_epoch=steps_per_epoch,
                training_approach=training_approach,
                gradient_checkpointing=gradient_checkpointing
            )

            learning_rate = train_dict['learning_rate']
            if isinstance(learning_rate, dict):
                learning_sched = learning_rate['schedule']
                learning_params = learning_rate
                learning_rate = optimizers.build_learning_rate(learning_sched, **learning_params)
            elif isinstance(learning_rate, (Number, np.number)):
                pass
            else:
                raise ValueError('Learning rate must by a dictionary or number.')

            training_engine.set_learning_rate(learning_rate)
            
            # Note: This won't immedietly call optimizer.build() - that must be done only at the
            # begging of training.
            optimizer_params = train_dict['optimizer']
            optimizer_name = optimizer_params['name']
            optimizer = optimizers.create_optimizer(
                optimizer=optimizer_name, 
                learning_rate=learning_rate,
                optimizer_params=optimizer_params, 
            )
            training_engine.set_optimizer(optimizer)

            ## Process "init_states"
            training_init_state_params = train_dict.get('initial_state', {})
            if training_init_state_params:
                mod_cls = StateModules().get_init_state_module(training_init_state_params['module'])
                training_engine.set_init_state(mod_cls(rnn=network, **training_init_state_params))

            ## process "callbacks" class
            callback_params = train_dict.get('callbacks', False)
            if callback_params:
                cb_name = callback_params.pop('class')
                cb_class = callback_classes[cb_name]
                callbacks = cb_class(rnn=network, **callback_params)
                training_engine.set_callbacks(callbacks=callbacks)

            for train_params_dict in train_dict['parameters']:
                pname = train_params_dict['name']
                pbatch_size = train_dict.get('batch_size', network.batch_size)
                pseq_len = train_dict.get('seq_len', network.seq_len)
                training_params = training_engine.add_parameters(pname, batch_size=pbatch_size, seq_len=pseq_len)
                
                ## For each "parameter" process the spike input modules
                for input_name, input_mod in network.parse_input_mods_from_config(train_params_dict['inputs']):
                    training_params.add_inputs_generator(name=input_name, input_mod=input_mod)

                ## For each "parameter" process the loss functions
                for loss_fnc_name, loss_fnc_params in train_params_dict['loss_functions'].items():
                    if not loss_fnc_params.get('enabled', True):
                        continue
                    loss_mod = LossModules().get_module(loss_fnc_params['module'])
                    training_params.add_loss_function(
                        name=loss_fnc_name,
                        loss_mod=loss_mod(rnn=network, **loss_fnc_params)
                    )

        inference_dict = config.get('inference', None)
        if inference_dict:
            inference = Inference(rnn=network)
            for input_name, input_mod in network.parse_input_mods_from_config(inference_dict['inputs']):
                inference.input_mods.append(input_mod)

            init_state_dict = inference_dict.get('initial_state', {})
            if init_state_dict:
                module = init_state_dict['module']
                mod_cls = StateModules().get_init_state_module(module)
                inference.init_mod = mod_cls(rnn=network, **init_state_dict)

            output = inference_dict.get('output', config.output)
            if output is not None:
                inference.output_params = output

            network.add_inference(inference)


        return network
