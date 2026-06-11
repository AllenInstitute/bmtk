import tensorflow as tf
import numpy as np
import pandas as pd
import copy

from . import optimizers
from .callbacks import Callbacks
from .callbacks import callback_classes
from .io_tools import io
from .data_iterator import DataIterator


class TrainingParameters:
    def __init__(self, name, batch_size=None, seq_len=None):
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self._loss_functions = {}

        self._inputs_mods_byname = {}
        self._inputs_mods_bypop = {}

    def add_inputs_generator(self, name, input_mod):
        # self._inputs.append(input_mod)
        if name in self._inputs_mods_byname:
            raise ValueError(f'Training parameters "{self.name}" already contains inputs-generator named "{name}".')
        self._inputs_mods_byname[name] = input_mod
        self._inputs_mods_bypop[input_mod.population_name] = input_mod

    def get_inputs_generator(self, mod_name=None, population_name=None):
        if mod_name is None and population_name is None:
            raise ValueError(f'{self.__class__.name}.get_inputs_generator(): Please specify either a mod_name or generator')
        
        mod = None
        if mod_name is not None:
            mod = self._inputs_mods_byname[mod_name]

        if population_name is not None:
            pmod = self._inputs_mods_bypop[population_name]
            if mod is not None and pmod != mod:
                raise ValueError(f'{self.__class__.name}.get_inputs_generator(): different input mods with name="{mod_name}" and population="{population_name}"')
            else:
                mod = pmod

        return mod            

    @property
    def input_generators(self):
        return list(self._inputs_mods_byname.values())


    # @property
    # def input_mods(self):
    #     return self._inputs

    def add_loss_function(self, name, loss_mod):
        self._loss_functions[name] = loss_mod

    @property
    def loss_functions(self):
        return self._loss_functions


class InputsSignatureFactory:
    class InputsSignature:
        def __init__(self, lutables, ysigs):
            self.ysigs = ysigs
            self.lu_tables = lutables

        def get_value(self, population, name):
            return self.ysigs[self.lu_tables['input_pops_order_lu'][population]][name]

        def to_dataframe(self):
            dfs = []
            for y_dict, ipop, imod in zip(self.ysigs, self.lu_tables['input_pops'], self.lu_tables['input_mods']):
                in_df = pd.DataFrame({name: value.numpy().flatten() for name, value in y_dict.items()})
                in_df['population'] = ipop
                in_df['input_mod'] = imod
                dfs.append(in_df)

            if len(dfs) == 1:
                return dfs[0]
            else:
                return pd.concat(dfs, axis=0, ignore_index=True)


    def __init__(self, pnames, ordered_input_pops, ordered_input_mods):
        assert(len(pnames) == len(ordered_input_pops) == len(ordered_input_mods))
        self.lu_tables = []
        for pname, ipops, imods in zip(pnames, ordered_input_pops, ordered_input_mods):
            self.lu_tables.append({
                'pname': pname,
                'input_pops': ipops,
                'input_pops_order_lu': {name: order for order, name in enumerate(ipops)},
                'input_mods': imods,
                'input_mods_order_lu': {name: order for order, name in enumerate(imods)},
            })

    def build(self, input_sigs):
        return [InputsSignatureFactory.InputsSignature(copy.deepcopy(lutable), y) for (lutable,y) in zip(self.lu_tables, input_sigs)]


class TrainingEngine:
    def __init__(self, rnn, n_epochs, steps_per_epoch, training_approach=None, **kwargs):
        self.rnn = rnn
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch

        self._learning_rate = None
        self._optimizer = None
        self._parameters = []
        self._init_state_mod = None
        self._training_approach = training_approach
        self._training_fnc = None

        # Gradient checkpointing (a.k.a. recompute_grad): recompute the per-timestep
        # RNN activations during the backward pass instead of storing all of them across
        # the unrolled sequence. This is required to fit full-network BPTT (seq_len ~500)
        # on a single GPU; mirrors the reference V1_GLIF_model implementation.
        # NOTE: default off for now. The recompute_grad wrapping does not yet reach the
        # reference's memory profile (the per-timestep constant retention persists and, in
        # at least one case, checkpointing increased peak memory). Re-enable by default once
        # checkpointing is fixed to match V1_GLIF_model.
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        self._extractor_forward = None

        self._batch_indices = None

        self._inputs_sig_factory = None

        self._callbacks = None
        self._normalizers = None

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def adjusted_batch_size(self):
        if self.training_approach == 'single':
            return self._parameters[0].batch_size
        
        elif self.training_approach in ['series', 'series_accumulate']:
            p0 = self._parameters[0]
            batch_size = p0.batch_size
            for pi in self._parameters[1:]:
                if pi.batch_size != batch_size:
                    raise ValueError(f'Training Parameters for {p0.name} and {pi.name} have different batch_size ({p0.batch_size} != {pi.batch_size}).'
                                     f' For "{self.training_approach}" training approach they must be the same')
            return batch_size
        
        elif self.training_approach == 'parallel':
            batch_size = 0
            for p in self._parameters:
                batch_size += p.batch_size
            return batch_size
        
        else:
            raise ValueError(f'Unknown "training_approach" {self.training_approach}')

    @property
    def adjusted_seq_len(self):
        if self._training_approach == 'single':
            return self._parameters[0].seq_len
        
        elif self._training_approach in ['series', 'series_accumulate', 'parallel']:
            p0 = self._parameters[0]
            seq_len = p0.seq_len
            for pi in self._parameters[1:]:
                if pi.seq_len != seq_len:
                    raise ValueError(f'Training Parameters for {p0.name} and {pi.name} have different seq_len ({p0.seq_len} != {pi.seq_len}).'
                                     f' For "{self._training_approach}" training approach they must be the same')
            return seq_len
        
        else:
            raise ValueError(f'Unknown "training_approach" {self._training_approach}')

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate
        
    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def add_parameters(self, name, batch_size=None, seq_len=None):
        training_params = TrainingParameters(name, batch_size=batch_size, seq_len=seq_len)
        self._parameters.append(training_params)
        return training_params
    
    def set_init_state(self, init_state_mod):
        self._init_state_mod = init_state_mod

    @property
    def parameters(self):
        return self._parameters

    @property
    def n_parameters(self):
        return len(self._parameters)
    
    @property
    def batch_indices(self):
        if self._batch_indices is None:
            self._batch_indices = np.cumsum([p.batch_size for p in self.parameters], dtype=int)

        return self._batch_indices

    @property
    def init_state(self):
        return self._init_state_mod

    # def get_generator(self):
    #     return self._parameters[0].inputs[0]

    @property
    def callbacks(self):
        if self._callbacks is None:
            self._callbacks = Callbacks(rnn=self.rnn)
        
        return self._callbacks

    def set_callbacks(self, callbacks):
        self._callbacks = callbacks

    @property
    def training_approach(self):
        if self._training_approach is None:
            if self.n_parameters == 1:
                self._training_approach = 'single'
            
            else:
                io.log_debug(f'{self.__class__.__name__}: "training_approach" not set for multi-parameter training, defaulting to "parallel"')
                self._training_approach = 'parallel'

        if self._training_approach == 'single' and self.n_parameters > 1:
            raise ValueError(f'{self.__class__.__name__}: Attempting to use "single" training approach when more than one parameter. Please use options: parallel, series, series_accumulate.')

        return self._training_approach

    @property
    def step_train_function(self):
        if self._training_fnc is None:
            if self.n_parameters == 1:
                self._training_fnc = self._train_step_single
            
            else:
                if self._training_approach is None or self._training_approach == '':
                    raise ValueError('Training Error: When using more than one training parameter please specify "training_approach", Options: parallel, series, series_accumulate')
                elif self._training_approach == 'parallel':
                    self._training_fnc = self._train_step_parallel
                elif self._training_approach in ['series', 'series_accumulate']:
                    self._training_fnc = self._train_step_series
                else:
                    raise ValueError(f'Training Error: Invalid training approach "{self._training_approach}"')

        return self._training_fnc

    def _uses_ema_normalizer(self):
        for parameter in self.parameters:
            for loss_fnc in parameter.loss_functions.values():
                if getattr(loss_fnc, 'uses_ema_normalizer', False):
                    return True
        return False

    def _prepare_normalizers(self):
        if self._normalizers is not None or not self._uses_ema_normalizer():
            return

        with self.rnn.strategy.scope():
            self._normalizers = {
                'v1_ema': tf.Variable(
                    tf.fill(
                        [self.rnn.recurrent_network['n_nodes']],
                        tf.constant(0.003, dtype=tf.float32),
                    ),
                    trainable=False,
                    name='V1_EMA',
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                )
            }

    @staticmethod
    def _has_orientation(y):
        if y is None:
            return False
        if hasattr(y, 'ysigs'):
            candidates = y.ysigs
        elif isinstance(y, dict):
            candidates = [y]
        elif isinstance(y, (list, tuple)):
            candidates = y
        else:
            return False

        for sig in candidates:
            if isinstance(sig, dict) and 'orientation' in sig:
                return True
        return False

    def _prepare_loss_kwargs(self, parameter, spikes, y):
        if self._normalizers is None:
            return {}

        if self._has_orientation(y):
            self._update_normalizers(parameter, spikes, y)
            return {
                'normalizer': tf.stop_gradient(tf.identity(self._normalizers['v1_ema'])),
                'batch_size_hint': parameter.batch_size,
            }

        return {}

    def _update_normalizers(self, parameter, spikes, y):
        if self._normalizers is None or not self._has_orientation(y):
            return

        for loss_fnc in parameter.loss_functions.values():
            update_normalizers = getattr(loss_fnc, 'update_normalizers', None)
            if update_normalizers is not None:
                update_normalizers(tf.stop_gradient(spikes), self._normalizers)
                break
        
    @property
    def inputs_signature_factory(self):
        if self._inputs_sig_factory is None:           
            pnames = []
            ordered_input_pops = []
            ordered_input_mods = []
            for p in self.parameters:
                pnames.append(p.name)
                _pops = []
                _mods = []
                for inpop in self.rnn.ordered_inputs_populations:
                    input_mod = p.get_inputs_generator(population_name=inpop)
                    _pops.append(inpop)
                    _mods.append(input_mod.name)

                ordered_input_pops.append(_pops)
                ordered_input_mods.append(_mods)

            self._inputs_sig_factory = InputsSignatureFactory(pnames, ordered_input_pops, ordered_input_mods)

        return self._inputs_sig_factory

    def _run_extractor(self, x, init_state):
        """Run the extractor (RNN) forward pass, optionally with gradient checkpointing.

        When ``gradient_checkpointing`` is enabled, the forward is wrapped in
        ``tf.recompute_grad`` so the per-timestep activations are recomputed during the
        backward pass rather than retained for the whole sequence. The forward path is
        deterministic (background spikes are supplied as explicit inputs), so the recompute
        reproduces the original forward exactly.
        """
        if self.gradient_checkpointing and self._extractor_forward is not None:
            return self._extractor_forward(x, init_state)
        return self.rnn.extractor_model((x, init_state))

    @staticmethod
    def _add_gradients(accumulated, current):
        if accumulated is None:
            return current
        if current is None:
            return accumulated
        if isinstance(accumulated, tf.IndexedSlices) or isinstance(current, tf.IndexedSlices):
            accumulated = tf.convert_to_tensor(accumulated)
            current = tf.convert_to_tensor(current)
        return accumulated + current

    def _train_step_single(self, x, y, init_state):
        """Main training step when there is only one parameter (eg. one set of inputs/loss functions). 
            1. Feedforward input is applied to spike inputs "x" with model state "init_state"
            2. Results are applied to each loss function
            3. losses are summed together to calculate gradients.
            4. Gradients are applied to optimizers to update weights.
        
        TODO: This function should be able to be merged with _train_step_series.
        """
        pname = self.parameters[0].name
        loss_vals = {pname: {}}
        input_spikes = x[0]
        with tf.GradientTape() as tape:
            _out = self._run_extractor(input_spikes, init_state)
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]
            loss_vals[pname]['__mean_rate'] = tf.cast(tf.reduce_mean(_spikes_out), tf.float32)

            _total_loss = 0.0
            loss_kwargs = self._prepare_loss_kwargs(self.parameters[0], _spikes_out, y)
            for loss_name, loss_fnc in self.parameters[0].loss_functions.items():
                _loss = loss_fnc(
                    spikes=_spikes_out, 
                    voltages=_v_out, 
                    model_state=_model_state,
                    y=y,
                    **loss_kwargs,
                )
                _total_loss += tf.cast(_loss, tf.float32)
                loss_vals[pname][loss_name] = _loss

            _total_loss = tf.cast(_total_loss, tf.float32)
            total_loss = tf.nn.scale_regularization_loss(_total_loss)
            loss_for_grad = optimizers.scale_loss_for_optimizer(self.optimizer, total_loss)
            loss_vals['__total_loss'] = tf.reduce_mean(total_loss)

        # Backpropagation of the model (gradients computation and application)
        grad = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
        grad = optimizers.unscale_gradients_for_optimizer(self.optimizer, grad)
        self.optimizer.apply_gradients(zip(grad, self.rnn.model.trainable_variables))

        return loss_vals

    def _train_step_parameter_gradients(self, x, y, init_state, parameter_index):
        """Compute one parameter set's gradients without applying them."""
        p = self.parameters[parameter_index]
        ysig = InputsSignatureFactory.InputsSignature(
            copy.deepcopy(self.inputs_signature_factory.lu_tables[parameter_index]),
            y
        )
        loss_vals = {p.name: {}}
        with tf.GradientTape() as tape:
            _out = self._run_extractor(x, init_state)
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]
            loss_vals[p.name]['__mean_rate'] = tf.cast(tf.reduce_mean(_spikes_out), tf.float32)

            _total_loss = 0.0
            loss_kwargs = self._prepare_loss_kwargs(p, _spikes_out, ysig)
            for loss_name, loss_fnc in p.loss_functions.items():
                _loss = loss_fnc(
                    spikes=_spikes_out,
                    voltages=_v_out,
                    model_state=_model_state,
                    y=ysig,
                    **loss_kwargs,
                )
                _total_loss += tf.cast(_loss, tf.float32)
                loss_vals[p.name][loss_name] = _loss

            _total_loss = tf.cast(_total_loss, tf.float32)
            total_loss = tf.nn.scale_regularization_loss(_total_loss)
            loss_for_grad = optimizers.scale_loss_for_optimizer(self.optimizer, total_loss)
            loss_vals['__total_loss'] = total_loss

        grads = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
        grads = optimizers.unscale_gradients_for_optimizer(self.optimizer, grads)
        return loss_vals, grads

    @tf.function
    def _distributed_train_step_parameter_gradients(self, x, y, init_state, parameter_index):
        return self.rnn.strategy.run(
            self._train_step_parameter_gradients,
            args=(x, y, init_state, parameter_index)
        )

    def _apply_gradients(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.rnn.model.trainable_variables))

    @tf.function
    def _distributed_apply_gradients(self, grads):
        return self.rnn.strategy.run(self._apply_gradients, args=(grads,))

    @staticmethod
    def _record_signature_metrics(loss_vals, pname, y):
        candidates = y.ysigs if hasattr(y, 'ysigs') else y
        if isinstance(candidates, dict):
            candidates = [candidates]
        elif not isinstance(candidates, (list, tuple)):
            return

        for sig in candidates:
            if isinstance(sig, dict) and 'orientation' in sig:
                orientation = tf.reshape(tf.cast(sig['orientation'], tf.float32), [-1])
                loss_vals[pname]['__orientation_mean'] = tf.reduce_mean(orientation)
                loss_vals[pname]['__orientation_first'] = orientation[0]
                return

    def _train_step_parallel(self, xs, ys, init_state):
        """Main training step for 'parallel' approach to training multiple parameters. All the inputs from
        each parameter is concated together so that if there are N parameters with inputs each "batch_size", 
        a single feedforward step of "N*batch_size" is executed instead. The results are then separated and
        loss functions applied to each parameter, summed together, then returned.


        """
        loss_vals = {}
        all_losses = []
        x_concat = tf.concat(xs, axis=0)
        sigs = self.inputs_signature_factory.build(ys)
        with tf.GradientTape() as tape:
            _out = self._run_extractor(x_concat, init_state)
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]

            _total_loss = 0.0
            pidx_beg = 0
            for p, pidx_end, ysig in zip(self.parameters, self.batch_indices, sigs):
                loss_vals[p.name] = {}
                _pspikes = _spikes_out[pidx_beg:pidx_end]
                _pvolts = _v_out[pidx_beg:pidx_end]
                _pstate = _model_state[pidx_beg:pidx_end]
                loss_vals[p.name]['__mean_rate'] = tf.cast(tf.reduce_mean(_pspikes), tf.float32)
                self._record_signature_metrics(loss_vals, p.name, ysig)
                loss_kwargs = self._prepare_loss_kwargs(p, _pspikes, ysig)
                for loss_name, loss_fnc in p.loss_functions.items():
                    _loss = loss_fnc(
                        spikes=_pspikes,
                        voltages=_pvolts,
                        model_state=_pstate,
                        y=ysig,
                        **loss_kwargs,
                    )
                    _total_loss += tf.cast(_loss, tf.float32)
                    loss_vals[p.name][loss_name] = _loss
                # Advance to this parameter's slice of the concatenated batch so the next
                # parameter's losses are computed on its own spikes/voltages (not [0:end]).
                pidx_beg = pidx_end

            _total_loss = tf.cast(_total_loss, tf.float32)
            total_loss = tf.nn.scale_regularization_loss(_total_loss)
            all_losses.append(total_loss)
            loss_for_grad = optimizers.scale_loss_for_optimizer(self.optimizer, total_loss)

        # Backpropagation of the model (gradients computation and application)
        grad = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
        grad = optimizers.unscale_gradients_for_optimizer(self.optimizer, grad)
        self.optimizer.apply_gradients(zip(grad, self.rnn.model.trainable_variables))

        loss_vals['__total_loss'] = tf.reduce_mean(all_losses)
        return loss_vals

    def _train_step_series(self, xs, ys, init_state):
        loss_vals = {}
        all_losses = []
        for p, x, y in zip(self.parameters, xs, ys):
            loss_vals[p.name] = {}
            with tf.GradientTape() as tape:
                _out = self._run_extractor(x, init_state)
                _spikes_out, _v_out = _out[0]
                _model_state = _out[1:]
                loss_vals[p.name]['__mean_rate'] = tf.cast(tf.reduce_mean(_spikes_out), tf.float32)
                self._record_signature_metrics(loss_vals, p.name, y)

                _total_loss = 0.0
                loss_kwargs = self._prepare_loss_kwargs(p, _spikes_out, y)
                for loss_name, loss_fnc in p.loss_functions.items():
                    _loss = loss_fnc(
                        spikes=_spikes_out, 
                        voltages=_v_out, 
                        model_state=_model_state,
                        y=y,
                        **loss_kwargs,
                    )
                    loss_vals[p.name][loss_name] = _loss
                    _total_loss += tf.cast(_loss, tf.float32)

                _total_loss = tf.cast(_total_loss, tf.float32)
                total_loss = tf.nn.scale_regularization_loss(_total_loss)
                loss_for_grad = optimizers.scale_loss_for_optimizer(self.optimizer, total_loss)
                all_losses.append(total_loss)

            grad = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
            grad = optimizers.unscale_gradients_for_optimizer(self.optimizer, grad)
            self.optimizer.apply_gradients(zip(grad, self.rnn.model.trainable_variables))

        loss_vals['__total_loss'] = tf.reduce_mean(all_losses)
        return loss_vals

    def _distributed_train_step_series_accumulate(self, xs, ys, init_state):
        """Series execution with one accumulated optimizer update.

        Each parameter set is traced/executed separately so TensorFlow can release the
        sequence activations between parameter sets. Gradients are accumulated and applied
        once, instead of applying one optimizer update per parameter as normal series does.
        """
        loss_vals = {}
        all_losses = []
        accum_grads = None

        for parameter_index, (p, x, y) in enumerate(zip(self.parameters, xs, ys)):
            param_loss_vals, grads = self._distributed_train_step_parameter_gradients(
                x, y, init_state, parameter_index
            )
            loss_vals[p.name] = param_loss_vals[p.name]
            all_losses.append(param_loss_vals['__total_loss'])
            if accum_grads is None:
                accum_grads = list(grads)
            else:
                accum_grads = [
                    self._add_gradients(accum_grad, grad)
                    for accum_grad, grad in zip(accum_grads, grads)
                ]

        self._distributed_apply_gradients(accum_grads)

        loss_vals['__total_loss'] = tf.reduce_mean(all_losses)
        return loss_vals

    @tf.function
    def _distributed_train_step(self, x, y, init_state):
        return self.rnn.strategy.run(
            self.step_train_function,
            args=(x, y, init_state)
        )

    def _validation_step(self, xs, ys, init_state, training_approach):
        if training_approach in ['series', 'series_accumulate', 'single']:
            loss_vals = {}
            all_losses = []
            for p, x, y in zip(self.parameters, xs, ys):
                loss_vals[p.name] = {}
                _out = self.rnn.extractor_model((x, init_state))
                _spikes_out, _v_out = _out[0]
                _model_state = _out[1:]
                loss_vals[p.name]['__mean_rate'] = tf.cast(tf.reduce_mean(_spikes_out), tf.float32)
                self._record_signature_metrics(loss_vals, p.name, y)

                _total_loss = 0.0
                loss_kwargs = self._prepare_loss_kwargs(p, _spikes_out, y)
                for loss_name, loss_fnc in p.loss_functions.items():
                    _loss = loss_fnc(
                        spikes=_spikes_out, 
                        voltages=_v_out, 
                        model_state=_model_state,
                        y=y,
                        **loss_kwargs,
                    )
                    loss_vals[p.name][loss_name] = _loss
                    _total_loss += tf.cast(_loss, tf.float32)

                _total_loss = tf.cast(_total_loss, tf.float32)
                total_loss = tf.nn.scale_regularization_loss(_total_loss)
                all_losses.append(total_loss)

            loss_vals['__total_loss'] = tf.reduce_mean(all_losses)
        
        elif training_approach == 'parallel':
            x_concat = tf.concat(xs, axis=0)
            sigs = self.inputs_signature_factory.build(ys)
            _out = self.rnn.extractor_model((x_concat, init_state))
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]

            loss_vals = {}
            _total_loss = 0.0
            pidx_beg = 0
            for p, pidx_end, ysig in zip(self.parameters, self.batch_indices, sigs):
                loss_vals[p.name] = {}
                _pspikes = _spikes_out[pidx_beg:pidx_end]
                _pvolts = _v_out[pidx_beg:pidx_end]
                _pstate = _model_state[pidx_beg:pidx_end]
                loss_vals[p.name]['__mean_rate'] = tf.cast(tf.reduce_mean(_pspikes), tf.float32)
                self._record_signature_metrics(loss_vals, p.name, ysig)
                loss_kwargs = self._prepare_loss_kwargs(p, _pspikes, ysig)
                for loss_name, loss_fnc in p.loss_functions.items():
                    _loss = loss_fnc(
                        spikes=_pspikes,
                        voltages=_pvolts,
                        model_state=_pstate,
                        y=ysig,
                        **loss_kwargs,
                    )
                    loss_vals[p.name][loss_name] = _loss
                    _total_loss += tf.cast(_loss, tf.float32)
                # Advance to this parameter's slice so the next parameter's validation losses
                # are computed on its own batch segment (mirrors the fix in _train_step_parallel).
                pidx_beg = pidx_end

            _total_loss = tf.cast(_total_loss, tf.float32)
            total_loss = tf.nn.scale_regularization_loss(_total_loss)
            loss_vals['__total_loss'] = total_loss

        else:
            raise ValueError(f'Unknown training approach method "{training_approach}"')

        return loss_vals


    @tf.function
    def _distributed_validation_step(self, xs, ys, init_state, training_approach):
        return self.rnn.strategy.run(
            self._validation_step,
            args=(xs, ys, init_state, training_approach)
        )

    @staticmethod
    def merge_generators(spikes_generators):
        spikes = []
        ys = []
        for gen in spikes_generators:
            if gen is None:
                continue
            s, y = next(gen)
            spikes.append(s)
            ys.append(y)

        concat_spikes = tf.concat(spikes, axis=2)
        return concat_spikes, ys

    def train(self):
        input_generators = []
        input_batch_sizes = []
        input_seq_lens = []
        for p in self.parameters:
            input_generators.append(p.input_generators)
            input_batch_sizes.append(p.batch_size)
            input_seq_lens.append(p.seq_len)
        
        input_itr = DataIterator(input_generators, input_batch_sizes, input_seq_lens, self.rnn.ordered_inputs_populations)

        # input_itrs = [DataIterator(p.input_generators, p.batch_size, p.seq_len, self.rnn.ordered_inputs_populations) for p in self.parameters]
        init_state = self.init_state.get_state()
        self._prepare_normalizers()

        # Build the gradient-checkpointed forward ONCE, eagerly, before the @tf.function
        # train step is traced. tf.recompute_grad must wrap the function in eager context;
        # applying it lazily during graph tracing does not establish the checkpoint
        # boundary and the per-timestep activations are retained anyway.
        if self.gradient_checkpointing and self._extractor_forward is None:
            @tf.recompute_grad
            def extractor_forward(x, fwd_init_state):
                return self.rnn.extractor_model((x, fwd_init_state))
            self._extractor_forward = extractor_forward

        try:
            self.callbacks.on_train_begin()
            for epoch in range(self.n_epochs):
                self.callbacks.on_epoch_start()
                
                for step in range(self.steps_per_epoch):
                    self.callbacks.on_step_start()
                    spikes, ys = input_itr.next_spikes()
                    if self.training_approach == 'series_accumulate':
                        step_loss_vals = self._distributed_train_step_series_accumulate(spikes, ys, init_state=init_state)
                    else:
                        step_loss_vals = self._distributed_train_step(spikes, ys, init_state=init_state)
                    # Propagate the just-updated master recurrent weights into the compute-dtype
                    # shadow used by the forward pass (no-op when not using a shadow).
                    cell = getattr(self.rnn, '_cell', None)
                    if cell is not None and hasattr(cell, 'refresh_recurrent_weight_shadow'):
                        cell.refresh_recurrent_weight_shadow()
                    self.callbacks.on_step_end(step_loss_vals)

                validation_loss = self._distributed_validation_step(spikes, ys, init_state=init_state, training_approach=self.training_approach)
                stop = self.callbacks.on_epoch_end(validation_loss)
                if stop:
                    break

            normalizers = None
            if self._normalizers is not None:
                normalizers = {name: value.numpy() for name, value in self._normalizers.items()}
            self.callbacks.on_train_end(normalizers=normalizers)
        finally:
            input_itr.close()
    

    '''
    def train_old(self):
        # For each parameter, creates spikes generators for each input-network. Should be a list of lists of size 
        # n_parameters x n_input_networks.
        data_itrs = []
        for p in self.parameters:
            _itrs = []
            for inpop in self.rnn.ordered_inputs_populations:
                input_mod = p.get_inputs_generator(population_name=inpop)
                generator = input_mod.create_generator(seq_len=self.rnn.seq_len)
                if generator is None:
                    _itrs.append(None)
                else:                    
                    dataset = generator.batch(self.rnn.batch_size).prefetch(tf.data.AUTOTUNE)
                    itr = iter(dataset)
                    _itrs.append(itr)
            
            data_itrs.append(_itrs)

        # InputsSigFac = self.inputs_signature_factory
        init_state = self.init_state.get_state()
        
        self.callbacks.on_train_begin()
        for epoch in range(self.n_epochs):
            self.callbacks.on_epoch_start()
            
            for step in range(self.steps_per_epoch):
                self.callbacks.on_step_start()

                spikes = []
                ys = []
                for p in data_itrs:
                    _spikes, y = TrainingEngine.merge_generators(p)
                    spikes.append(_spikes)
                    ys.append(y)
                           
                step_loss_vals = self._distributed_train_step(spikes, ys, init_state=init_state)
                self.callbacks.on_step_end(step_loss_vals)

            validation_loss = self._distributed_validation_step(spikes, ys, init_state=init_state, training_approach=self.training_approach)
            
            
            stop = self.callbacks.on_epoch_end(validation_loss)
            if stop:
                break

        self.callbacks.on_train_end()
    '''
