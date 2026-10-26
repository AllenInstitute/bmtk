import contextlib

import tensorflow as tf
import numpy as np
import pandas as pd
import copy

from . import optimizers
from ._options import UNSET, validate_bool_option
from .callbacks import Callbacks
from .callbacks import callback_classes
from .io_tools import io
from .data_iterator import DataIterator
from .learning_rules import BPTTLearningRule, LearningRule, LearningRules
from .learning_rules import LearningRuleObservations
from .segmented_recompute import FullBPTTGradientRunner, SegmentedRecomputeRunner
from .spike_statistics import SpikeStatisticsRunner
from .spike_summary import SpikeSummary
from .loss_functions import loss_utils
from .loss_functions.weight_regularization import weight_regularization_scope


def _live_master_gradients(variables, gradients):
    return gradients


def _binary_spike_checkpoint_state(rnn):
    """Whether the RNN cell's first state is the binary GLIF3 spike buffer."""
    from .cell_models.glif3_cell import GLIF3Cell

    cell = getattr(rnn, "_cell", None)
    if cell is None:
        cell_cls = getattr(rnn, "cell_cls", None)
        return isinstance(cell_cls, type) and issubclass(cell_cls, GLIF3Cell)
    if not isinstance(cell, GLIF3Cell):
        return False
    try:
        _, names = cell.zero_state(1, tf.float32, with_names=True)
    except TypeError:
        # A wrapped zero_state without the names keyword: the base cell's
        # first state is the binary spike buffer by construction.
        return type(cell) is GLIF3Cell
    except ValueError:
        return False
    return bool(names) and names[0] == "z0_buf"


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
            raise ValueError(
                f'Training parameters "{self.name}" already contains inputs-generator named "{name}".'
            )
        self._inputs_mods_byname[name] = input_mod
        self._inputs_mods_bypop[input_mod.population_name] = input_mod

    def get_inputs_generator(self, mod_name=None, population_name=None):
        if mod_name is None and population_name is None:
            raise ValueError(
                f"{self.__class__.name}.get_inputs_generator(): Please specify either a mod_name or generator"
            )

        mod = None
        if mod_name is not None:
            mod = self._inputs_mods_byname[mod_name]

        if population_name is not None:
            pmod = self._inputs_mods_bypop[population_name]
            if mod is not None and pmod != mod:
                raise ValueError(
                    f'{self.__class__.name}.get_inputs_generator(): different input mods with name="{mod_name}" and population="{population_name}"'
                )
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
            return self.ysigs[self.lu_tables["input_pops_order_lu"][population]][name]

        def to_dataframe(self):
            dfs = []
            for y_dict, ipop, imod in zip(
                self.ysigs, self.lu_tables["input_pops"], self.lu_tables["input_mods"]
            ):
                in_df = pd.DataFrame(
                    {name: value.numpy().flatten() for name, value in y_dict.items()}
                )
                in_df["population"] = ipop
                in_df["input_mod"] = imod
                dfs.append(in_df)

            if len(dfs) == 1:
                return dfs[0]
            else:
                return pd.concat(dfs, axis=0, ignore_index=True)

    def __init__(self, pnames, ordered_input_pops, ordered_input_mods):
        assert len(pnames) == len(ordered_input_pops) == len(ordered_input_mods)
        self.lu_tables = []
        for pname, ipops, imods in zip(pnames, ordered_input_pops, ordered_input_mods):
            self.lu_tables.append(
                {
                    "pname": pname,
                    "input_pops": ipops,
                    "input_pops_order_lu": {
                        name: order for order, name in enumerate(ipops)
                    },
                    "input_mods": imods,
                    "input_mods_order_lu": {
                        name: order for order, name in enumerate(imods)
                    },
                }
            )

    def build(self, input_sigs):
        return [
            InputsSignatureFactory.InputsSignature(copy.deepcopy(lutable), y)
            for (lutable, y) in zip(self.lu_tables, input_sigs)
        ]


class TrainingEngine:
    _record_signature_metrics_enabled = True
    _record_mean_rate_metrics_enabled = True
    stream_spike_statistics = False
    recorded_adjoint = False
    # Class-level fallbacks for engines built without __init__ (test doubles):
    # explicit dense requests, so preparation never resolves a route for them.
    _stream_spike_statistics_request = False
    _recorded_adjoint_request = False
    training_route_report = None

    def __init__(
        self, rnn, n_epochs, steps_per_epoch, training_approach=None, **kwargs
    ):
        self.rnn = rnn
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch

        self._learning_rate = None
        self._optimizer = None
        self._parameters = []
        self._init_state_mod = None
        self._training_approach = training_approach
        self._training_fnc = None
        self._learning_rule = BPTTLearningRule()

        self.gradient_checkpointing = kwargs.get("gradient_checkpointing", True)
        # Requests are kept apart from the resolved booleans: "auto" (the default)
        # resolves in prepare_gradient_checkpointing, once the cell, losses and
        # strategy are known. Until then both resolved flags read False.
        self._stream_spike_statistics_request = kwargs.get("stream_spike_statistics", "auto")
        if self._stream_spike_statistics_request != "auto":
            validate_bool_option(self._stream_spike_statistics_request, "stream_spike_statistics")
        self._recorded_adjoint_request = kwargs.get("recorded_adjoint", "auto")
        if self._recorded_adjoint_request not in ("auto", "direct"):
            validate_bool_option(self._recorded_adjoint_request, "recorded_adjoint")
        if self._recorded_adjoint_request not in ("auto", False) and self._stream_spike_statistics_request is False:
            raise ValueError("recorded_adjoint requires stream_spike_statistics=True.")
        self.stream_spike_statistics = self._stream_spike_statistics_request is True
        self.recorded_adjoint = (
            self._recorded_adjoint_request if self._recorded_adjoint_request in (True, "direct") else False
        )
        self.training_route_report = None
        chunk_size = kwargs.get("gradient_checkpoint_chunk_size", UNSET)
        self._gradient_checkpoint_chunk_size = (
            UNSET if chunk_size is UNSET else int(chunk_size)
        )
        self.pack_spike_checkpoints = kwargs.get("pack_spike_checkpoints", "auto")
        if chunk_size is not UNSET and self._gradient_checkpoint_chunk_size < 1:
            raise ValueError("gradient_checkpoint_chunk_size must be positive.")
        self.regenerate_initial_state_each_epoch = kwargs.get(
            "regenerate_initial_state_each_epoch", True
        )
        self._extractor_forward = None

        self._batch_indices = None

        self._inputs_sig_factory = None

        self._callbacks = None
        self._normalizers = None
        self._record_signature_metrics_enabled = bool(
            kwargs.get("record_signature_metrics", True)
        )
        self._record_mean_rate_metrics_enabled = bool(
            kwargs.get("record_mean_rate_metrics", True)
        )

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def pack_spike_checkpoints(self):
        """Whether segmented recomputation stores bit-packed spike checkpoints.

        Default ``"auto"`` packs (63 spikes per on-device int64 word) when the cell's
        first recurrent state is the binary spike buffer of the GLIF3 family; other
        cells keep unpacked checkpoints. Explicit booleans are used as given.
        """
        if self._pack_spike_checkpoints != "auto":
            return self._pack_spike_checkpoints
        return _binary_spike_checkpoint_state(self.rnn)

    @pack_spike_checkpoints.setter
    def pack_spike_checkpoints(self, value):
        if value is None or value is UNSET:
            value = "auto"
        if value != "auto":
            validate_bool_option(value, "pack_spike_checkpoints")
        self._pack_spike_checkpoints = value

    @property
    def pack_spike_checkpoints_explicit(self):
        """The explicit boolean request, or None when automatic."""
        return None if self._pack_spike_checkpoints == "auto" else self._pack_spike_checkpoints

    @property
    def adjusted_batch_size(self):
        if self.training_approach == "single":
            return self._parameters[0].batch_size

        elif self.training_approach in ["series", "series_accumulate"]:
            p0 = self._parameters[0]
            batch_size = p0.batch_size
            for pi in self._parameters[1:]:
                if pi.batch_size != batch_size:
                    raise ValueError(
                        f"Training Parameters for {p0.name} and {pi.name} have different batch_size ({p0.batch_size} != {pi.batch_size})."
                        f' For "{self.training_approach}" training approach they must be the same'
                    )
            return batch_size

        elif self.training_approach == "parallel":
            batch_size = 0
            for p in self._parameters:
                batch_size += p.batch_size
            return batch_size

        else:
            raise ValueError(f'Unknown "training_approach" {self.training_approach}')

    @property
    def adjusted_seq_len(self):
        if self._training_approach == "single":
            return self._parameters[0].seq_len

        elif self._training_approach in ["series", "series_accumulate", "parallel"]:
            p0 = self._parameters[0]
            seq_len = p0.seq_len
            for pi in self._parameters[1:]:
                if pi.seq_len != seq_len:
                    raise ValueError(
                        f"Training Parameters for {p0.name} and {pi.name} have different seq_len ({p0.seq_len} != {pi.seq_len})."
                        f' For "{self._training_approach}" training approach they must be the same'
                    )
            return seq_len

        else:
            raise ValueError(f'Unknown "training_approach" {self._training_approach}')

    @property
    def gradient_checkpoint_chunk_size(self):
        if self._gradient_checkpoint_chunk_size is not UNSET:
            return self._gradient_checkpoint_chunk_size
        sequence_length = (
            self.adjusted_seq_len if self._parameters
            else getattr(self.rnn, "seq_len", None)
        )
        return min(25, sequence_length) if sequence_length is not None else 25

    @gradient_checkpoint_chunk_size.setter
    def gradient_checkpoint_chunk_size(self, value):
        value = int(value)
        if value < 1:
            raise ValueError("gradient_checkpoint_chunk_size must be positive.")
        self._gradient_checkpoint_chunk_size = value

    def set_learning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        self._checkpoint_slots_optimizer = None

    def _optimizer_update_variables(self):
        if self.learning_rule.uses_bptt:
            model = getattr(self.rnn, "model", None)
            return None if model is None else list(model.trainable_variables)
        surfaces = getattr(self.learning_rule, "weight_surfaces", None)
        return [surface.variable for surface in surfaces] if surfaces else None

    def _build_optimizer_state(self):
        if optimizers.optimizer_is_built(self.optimizer):
            return True
        variables = self._optimizer_update_variables()
        if not variables:
            return False
        strategy = getattr(self.rnn, "strategy", None)
        with tf.init_scope(), (strategy.scope() if strategy is not None else contextlib.nullcontext()):
            optimizers.build_optimizer(self.optimizer, variables)
        return True

    def prepare_optimizer(self):
        """Return the optimizer that training applies, wrapping it once for float16.

        Mixed-float16 models (``rnn.dtype`` float16) train through a
        ``LossScaleOptimizer``; the wrapper replaces ``self.optimizer`` here, and
        ``RNN.train`` reuses it instead of creating another one.
        """
        optimizer = self.optimizer
        strategy = getattr(self.rnn, "strategy", None)
        with strategy.scope() if strategy is not None else contextlib.nullcontext():
            wrapped = optimizers.loss_scale_optimizer_for(
                optimizer, getattr(self.rnn, "dtype", None)
            )
        if wrapped is not optimizer:
            self.set_optimizer(wrapped)
            io.log_info(
                'Mixed-precision optimizer loss scaling enabled with LossScaleOptimizer.'
            )
        return wrapped

    def build_optimizer(self):
        """Build the optimizer that ``train`` will apply and bind CSR-master slots.

        Resume flow: build the RNN, call this, restore ``tf.train.Checkpoint(model=...,
        optimizer=engine.optimizer)``, then train. For a float16 model the optimizer is
        first wrapped in the ``LossScaleOptimizer`` that ``RNN.train`` uses, so read
        ``engine.optimizer`` after this call; restoring into that exact object restores
        the loss scale and the slots, and training continues with it. Restoring after
        the build also restores weights that Keras 3 checkpoints only through the
        optimizer's variable list (trainable named-input weights), which
        ``optimizer.build`` replaces. The update variables are the model's trainable
        variables for BPTT, else the built learning rule's weight surfaces. Returns
        False when they are not known yet.
        """
        if self.optimizer is None:
            raise ValueError("Set an optimizer before building it.")
        self.prepare_optimizer()
        if not self._build_optimizer_state():
            return False
        return self.bind_optimizer_checkpoint_slots()

    def bind_optimizer_checkpoint_slots(self, *, after_update=False):
        """Bind CSR-master optimizer slots to the canonical checkpoint format.

        ``train`` and ``train_step`` call this before the first update, building an
        unbuilt optimizer first, so CSR-ordered slots restored by a
        ``tf.train.Checkpoint`` before or after the build are converted correctly.
        Returns False when the optimizer could not be built yet. No-op without
        CSR-ordered masters.
        """
        cell = getattr(self.rnn, "_cell", None)
        if not getattr(cell, "_csr_master_weights", False):
            return True
        optimizer = self.optimizer
        if optimizer is None:
            return True
        if getattr(self, "_checkpoint_slots_optimizer", None) is optimizer:
            return True
        if not after_update and not self._build_optimizer_state():
            return False
        with tf.init_scope():
            cell.bind_optimizer_checkpoint_slots(optimizer, after_update=after_update)
        self._checkpoint_slots_optimizer = optimizer
        return True

    @property
    def learning_rule(self):
        if not hasattr(self, "_learning_rule"):
            self._learning_rule = BPTTLearningRule()
        return self._learning_rule

    def set_learning_rule(self, learning_rule="bptt", **params):
        if isinstance(learning_rule, str):
            learning_rule = LearningRules().get_rule(learning_rule)(**params)
        elif params:
            raise ValueError(
                "Learning-rule parameters cannot accompany an instantiated rule."
            )
        if not isinstance(learning_rule, LearningRule):
            raise TypeError(
                "learning_rule must be a registered name or LearningRule instance."
            )
        if (
            self.n_parameters
            and self.training_approach
            not in learning_rule.supported_training_approaches
        ):
            raise ValueError(
                f'Learning rule "{learning_rule.module()}" does not support '
                f'training_approach="{self.training_approach}".'
            )
        self._learning_rule = learning_rule
        self._training_fnc = None

    def add_parameters(self, name, batch_size=None, seq_len=None):
        training_params = TrainingParameters(
            name, batch_size=batch_size, seq_len=seq_len
        )
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
            self._batch_indices = np.cumsum(
                [p.batch_size for p in self.parameters], dtype=int
            )

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
                self._training_approach = "single"

            else:
                io.log_debug(
                    f'{self.__class__.__name__}: "training_approach" not set for multi-parameter training, defaulting to "parallel"'
                )
                self._training_approach = "parallel"

        if self._training_approach == "single" and self.n_parameters > 1:
            raise ValueError(
                f'{self.__class__.__name__}: Attempting to use "single" training approach when more than one parameter. Please use options: parallel, series, series_accumulate.'
            )

        return self._training_approach

    @property
    def step_train_function(self):
        prepare_training_model = getattr(self.rnn, "_prepare_training_model", None)
        if prepare_training_model is not None:
            prepare_training_model()
        if getattr(self.rnn, "_online_voltage_losses", ()) and not self.learning_rule.uses_bptt:
            raise ValueError("VoltageRateFloor requires the BPTT learning rule.")
        if self._training_fnc is None:
            if not self.learning_rule.uses_bptt:
                if self.n_parameters != 1:
                    raise ValueError(
                        f'Learning rule "{self.learning_rule.module()}" currently requires '
                        "exactly one training parameter."
                    )
                if (
                    self.training_approach
                    not in self.learning_rule.supported_training_approaches
                ):
                    raise ValueError(
                        f'Learning rule "{self.learning_rule.module()}" does not support '
                        f'training_approach="{self.training_approach}".'
                    )
                self._training_fnc = self._train_step_local_single
                return self._training_fnc
            if self.n_parameters == 1:
                self._training_fnc = self._train_step_single

            else:
                if self._training_approach is None or self._training_approach == "":
                    raise ValueError(
                        'Training Error: When using more than one training parameter please specify "training_approach", Options: parallel, series, series_accumulate'
                    )
                elif self._training_approach == "parallel":
                    self._training_fnc = self._train_step_parallel
                elif self._training_approach in ["series", "series_accumulate"]:
                    self._training_fnc = self._train_step_series
                else:
                    raise ValueError(
                        f'Training Error: Invalid training approach "{self._training_approach}"'
                    )

        return self._training_fnc

    def _train_step_local_single(self, x, y, init_state):
        parameter = self.parameters[0]
        pname = parameter.name
        loss_vals = {pname: {}}
        input_spikes = x[0]
        out = self._run_extractor(input_spikes, init_state)
        spikes_out, voltages_out = (tf.stop_gradient(value) for value in out[0])
        model_state = tf.nest.map_structure(tf.stop_gradient, out[1:])
        if self._record_mean_rate_metrics_enabled:
            loss_vals[pname]["__mean_rate"] = tf.cast(
                tf.reduce_mean(spikes_out), tf.float32
            )

        selected_variables = [
            surface.variable for surface in self.learning_rule.weight_surfaces
        ]
        selected_variable_ids = {id(variable) for variable in selected_variables}
        all_weight_variables = list(self.rnn.model.trainable_variables)

        with tf.GradientTape() as signal_tape, weight_regularization_scope():
            signal_tape.watch(spikes_out)
            signal_tape.watch(voltages_out)
            total_loss = tf.constant(0.0, dtype=tf.float32)
            loss_kwargs = self._prepare_loss_kwargs(parameter, spikes_out, y)
            for loss_name, loss_fnc in parameter.loss_functions.items():
                loss = loss_fnc(
                    spikes=spikes_out,
                    voltages=voltages_out,
                    model_state=model_state,
                    y=y,
                    **loss_kwargs,
                )
                total_loss += tf.cast(loss, tf.float32)
                loss_vals[pname][loss_name] = loss
            total_loss = tf.nn.scale_regularization_loss(total_loss)
            loss_for_signal = optimizers.scale_loss_for_optimizer(
                self.optimizer, total_loss
            )

        local_gradients = signal_tape.gradient(
            loss_for_signal,
            [spikes_out, voltages_out] + all_weight_variables,
        )
        local_gradients = optimizers.unscale_gradients_for_local_rule(
            self.optimizer, local_gradients
        )
        spike_learning_signal = local_gradients[0]
        if spike_learning_signal is None:
            spike_learning_signal = tf.zeros_like(spikes_out)
        voltage_learning_signal = local_gradients[1]
        if voltage_learning_signal is None:
            voltage_learning_signal = tf.zeros_like(voltages_out)

        weight_gradients_by_id = {}
        for variable, gradient in zip(all_weight_variables, local_gradients[2:]):
            if gradient is not None and id(variable) not in selected_variable_ids:
                raise ValueError(
                    f'Loss depends directly on excluded weight surface "{variable.name}". '
                    "Include that surface in training.learning_rule.surfaces."
                )
            weight_gradients_by_id[id(variable)] = gradient
        direct_weight_gradients = tuple(
            (
                weight_gradients_by_id.get(id(variable))
                if weight_gradients_by_id.get(id(variable)) is not None
                else tf.zeros_like(variable)
            )
            for variable in selected_variables
        )
        updates = self.learning_rule.compute_updates(
            LearningRuleObservations(
                input_spikes=input_spikes,
                spikes=spikes_out,
                voltages=voltages_out,
                initial_state=init_state,
                spike_learning_signal=spike_learning_signal,
                voltage_learning_signal=voltage_learning_signal,
                direct_weight_gradients=direct_weight_gradients,
                targets=y,
            )
        )
        self.learning_rule.apply_updates(self.optimizer, updates)
        loss_vals["__total_loss"] = tf.reduce_mean(total_loss)
        return loss_vals

    def _uses_ema_normalizer(self):
        for parameter in self.parameters:
            for loss_fnc in parameter.loss_functions.values():
                if getattr(loss_fnc, "uses_ema_normalizer", False):
                    return True
        return False

    def _prepare_normalizers(self):
        if self._normalizers is not None or not self._uses_ema_normalizer():
            return

        with self.rnn.strategy.scope():
            self._normalizers = {
                "v1_ema": tf.Variable(
                    tf.fill(
                        [self.rnn.recurrent_network["n_nodes"]],
                        tf.constant(0.003, dtype=tf.float32),
                    ),
                    trainable=False,
                    name="V1_EMA",
                    aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                )
            }

    @staticmethod
    def _has_orientation(y):
        if y is None:
            return False
        if hasattr(y, "ysigs"):
            candidates = y.ysigs
        elif isinstance(y, dict):
            candidates = [y]
        elif isinstance(y, (list, tuple)):
            candidates = y
        else:
            return False

        for sig in candidates:
            if isinstance(sig, dict) and "orientation" in sig:
                return True
        return False

    def _prepare_loss_kwargs(self, parameter, spikes, y, **cached):
        if self._normalizers is None:
            return {}

        if self._has_orientation(y):
            self._update_normalizers(parameter, spikes, y, **cached)
            return {
                "normalizer": tf.stop_gradient(
                    tf.identity(self._normalizers["v1_ema"])
                ),
                "batch_size_hint": parameter.batch_size,
            }

        return {}

    def _update_normalizers(self, parameter, spikes, y, **cached):
        if self._normalizers is None or not self._has_orientation(y):
            return

        for loss_fnc in parameter.loss_functions.values():
            update_normalizers = getattr(loss_fnc, "update_normalizers", None)
            if update_normalizers is not None:
                detached = spikes.stop_gradient() if isinstance(spikes, SpikeSummary) else tf.stop_gradient(spikes)
                update_normalizers(detached, self._normalizers, **cached)
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

            self._inputs_sig_factory = InputsSignatureFactory(
                pnames, ordered_input_pops, ordered_input_mods
            )

        return self._inputs_sig_factory

    def _run_extractor(self, x, init_state, parameters=None):
        """Run the extractor (RNN) forward pass, optionally with gradient checkpointing.

        When checkpointing is enabled, only chunk-boundary recurrent states are
        retained. Internal Poisson progression is part of that state, so each
        chunk is replayed exactly during the backward pass.
        """
        if self._extractor_forward is not None:
            if x.dtype == tf.bool:
                x = tf.cast(x, self.rnn.dtype)
            self.rnn.cell.advance_noise_seed()
            init_state = self.rnn.cell.reset_voltage_penalty_state(init_state)
            if self.stream_spike_statistics:
                from .loss_functions import SynchronizationLoss

                groups = self.parameters if parameters is None else parameters
                traces, keys, start = [], [], 0
                for parameter in groups:
                    for loss in parameter.loss_functions.values():
                        if type(loss) is SynchronizationLoss and loss._sync_cost > 0:
                            plan = loss.prepare_sampled_statistics(
                                parameter.batch_size, x.dtype, x.shape[1], start
                            )
                            if plan is not None:
                                traces.append(plan)
                                keys.append((loss.name, *plan.batch_range))
                    start += parameter.batch_size
                if start != x.shape[0]:
                    raise ValueError("Spike statistics batch groups do not match extractor inputs.")
                result = self._extractor_forward(
                    x, init_state, sampled_traces=tuple(traces)
                )
                spikes = SpikeSummary(
                    result.counts, self._extractor_forward.count_windows,
                    x.shape[1], result.traces, tuple(keys),
                )
                return ((spikes, *result.sequences), *result.final_state)
            flat_output = self._extractor_forward(x, init_state)
            return self.rnn._unpack_extractor_output(flat_output)
        return self.rnn.run_extractor(x, init_state)

    def _run_parameter_extractor(self, x, init_state, parameter):
        if self.stream_spike_statistics:
            return self._run_extractor(x, init_state, (parameter,))
        return self._run_extractor(x, init_state)

    @property
    def stream_spike_statistics_request(self):
        """The constructor request: ``"auto"`` (default) or an explicit boolean."""
        return self._stream_spike_statistics_request

    @property
    def recorded_adjoint_request(self):
        """The constructor request: ``"auto"`` (default), ``False``, ``True`` or ``"direct"``."""
        return self._recorded_adjoint_request

    def _resolve_training_route(self, cell):
        """Resolve ``"auto"`` streaming/adjoint requests into the engine's booleans.

        Automatic policy: stream spike statistics when every streaming
        prerequisite holds, and use the direct recorded adjoint when the cell
        and trainable surfaces satisfy its eligibility and the per-step record
        payload fits the device. Explicit requests are used as given and keep
        their validation errors. The decision and its reasons are recorded in
        ``training_route_report`` and the cell's acceleration report.
        """
        from .direct_recorded_adjoint import direct_records_supported, direct_record_bytes
        from .cell_models.glif3_cell import GLIF3Cell

        report = {
            "requested": {
                "stream_spike_statistics": self._stream_spike_statistics_request,
                "recorded_adjoint": self._recorded_adjoint_request,
            },
            "reasons": {},
        }
        if not self.gradient_checkpointing or getattr(cell, "temporal_gradient_precision", "compute") == "float32":
            auto_stream, stream_reason = False, "dense route kept: requires checkpointed compute-temporal gradients"
        else:
            unmet = self._spike_statistics_unmet(automatic=True)
            auto_stream = not unmet
            stream_reason = (
                "streamed FP32 counts and sampled traces: every streaming prerequisite holds"
                if auto_stream else "dense route kept: " + "; ".join(unmet)
            )
        stream = self._stream_spike_statistics_request
        if stream == "auto":
            stream = auto_stream
            report["reasons"]["stream_spike_statistics"] = stream_reason
        else:
            report["reasons"]["stream_spike_statistics"] = "explicit request"
        adjoint = self._recorded_adjoint_request
        auto_adjoint = None
        if adjoint == "auto" or (adjoint is not True and adjoint != "direct"):
            if not stream:
                auto_adjoint, reason = False, "recorded routes require streamed statistics"
                if adjoint == "auto":
                    adjoint = False
                    report["reasons"]["recorded_adjoint"] = reason
            else:
                unmet = list(direct_records_supported(cell))
                variables = getattr(getattr(self.rnn, "model", None), "trainable_variables", None)
                if variables is not None and type(cell) is GLIF3Cell:
                    masters = {id(cell.recurrent_weight_values)} | {
                        id(net["input_weight_values"]) for net in cell.inputs.values()
                    }
                    extra = [
                        getattr(v, "name", str(v)) for v in variables
                        if id(v) not in masters and id(getattr(v, "value", None)) not in masters
                    ]
                    if extra:
                        unmet.append("trainable surfaces limited to recurrent/named-input CSR masters "
                                     f"(found {', '.join(extra[:3])})")
                try:
                    batch, length = int(self.adjusted_batch_size), int(self.adjusted_seq_len)
                except (IndexError, TypeError, ValueError):
                    batch = length = None
                fits, memory_reason = direct_record_bytes(
                    cell, batch, length, self.gradient_checkpoint_chunk_size,
                )
                auto_adjoint = "direct" if not unmet else False
                reason = (
                    "direct recorded adjoint: eligible cell and CSR-master surfaces; " + memory_reason
                    if auto_adjoint else
                    "checkpointed streaming kept: requires " + "; ".join(unmet)
                )
                report["record_payload"] = {"fits": fits, "estimate": memory_reason}
                # The payload estimate is advisory: the fastest qualified route is
                # still selected, and the user is told when it may not fit.
                if auto_adjoint and adjoint == "auto" and fits is not True:
                    io.log_warning(
                        "Direct recorded adjoint selected automatically, but its record payload may "
                        "not fit the device (" + memory_reason + "). The rollout can fail with an "
                        "out-of-memory error; set recorded_adjoint=False to keep checkpointed "
                        "streaming, or reduce the batch, sequence length or network size."
                    )
            if adjoint == "auto":
                adjoint = auto_adjoint
                report["reasons"]["recorded_adjoint"] = reason
        if self._recorded_adjoint_request != "auto":
            report["reasons"]["recorded_adjoint"] = "explicit request"
        report["selected"] = {"stream_spike_statistics": bool(stream), "recorded_adjoint": adjoint}
        report["automatic"] = {"stream_spike_statistics": bool(auto_stream), "recorded_adjoint": auto_adjoint}
        # Explicit engine requests are the supported opt-out; name the ones that
        # leave the automatic (fastest qualified) route so stale flags are visible.
        report["explicit_deviations"] = {
            name: {"explicit": report["selected"][name], "automatic": report["automatic"][name]}
            for name in ("stream_spike_statistics", "recorded_adjoint")
            if getattr(self, f"_{name}_request") != "auto"
            and report["automatic"][name] is not None
            and report["selected"][name] != report["automatic"][name]
        }
        self.stream_spike_statistics = bool(stream)
        self.recorded_adjoint = adjoint
        self.training_route_report = report
        acceleration = getattr(cell, "acceleration_report", None)
        if isinstance(acceleration, dict):
            acceleration["training_route"] = report
        io.log_info(
            "Training route: stream_spike_statistics=%s (%s); recorded_adjoint=%s (%s)."
            % (report["selected"]["stream_spike_statistics"], report["reasons"]["stream_spike_statistics"],
               report["selected"]["recorded_adjoint"], report["reasons"]["recorded_adjoint"])
        )
        if report["explicit_deviations"]:
            io.log_warning(
                "Explicit training-route requests differ from the automatic (fastest qualified) "
                "choice: " + ", ".join(
                    f"{name}={value['explicit']!r} (auto: {value['automatic']!r})"
                    for name, value in report["explicit_deviations"].items()
                ) + ". Remove inherited stream_spike_statistics/recorded_adjoint flags unless the "
                "override is intended."
            )

    def prepare_gradient_checkpointing(self):
        prepare_training_model = getattr(self.rnn, "_prepare_training_model", None)
        if prepare_training_model is not None:
            prepare_training_model()
        cell = getattr(self.rnn, "cell", None)
        if self._extractor_forward is None:
            self._resolve_training_route(cell)
        if self.stream_spike_statistics:
            self._validate_spike_statistics()
        if getattr(cell, "temporal_gradient_precision", "compute") == "float32":
            if self.gradient_checkpointing and (
                cell.temporal_checkpoint_chunk_size
                != self.gradient_checkpoint_chunk_size
            ):
                raise ValueError(
                    "Set temporal_checkpoint_chunk_size before building the FP32-adjoint RNN; "
                    "an outer checkpoint wrapper would narrow temporal cotangents."
                )
            io.log_info(
                "FP32 temporal reverse/replay owns checkpointing and canonical gradient restore."
            )
            return
        use_direct_csr_gradient = getattr(
            cell, "_use_direct_csr_recurrent_gradient", False
        )
        csr_live_masters = getattr(cell, "_csr_master_weights", False)
        if self._extractor_forward is not None:
            return
        if not self.gradient_checkpointing and not use_direct_csr_gradient:
            return
        if self.rnn.extractor_model is None:
            raise ValueError(
                "Build rnn.extractor_model before preparing gradient checkpointing."
            )
        if not self.gradient_checkpointing:
            self._extractor_forward = FullBPTTGradientRunner(
                self.rnn.extractor_model,
                _live_master_gradients if csr_live_masters
                else cell.restore_segmented_variable_gradients,
            )
            io.log_info(
                "Full BPTT with live CSR-master gradients enabled."
                if csr_live_masters else
                "Full BPTT with one canonical recurrent-gradient restore enabled."
            )
            return
        runner_type = SpikeStatisticsRunner if self.stream_spike_statistics else SegmentedRecomputeRunner
        statistics_options = {"count_windows": self._statistics_windows()} if self.stream_spike_statistics else {}
        output_options = {"n_sequence_outputs": 2}
        if self.recorded_adjoint == "direct":
            from .direct_recorded_adjoint import DirectRecordedAdjointRunner
            runner_type = DirectRecordedAdjointRunner
            output_options = {"cell": cell}
        elif self.recorded_adjoint:
            from .recorded_adjoint import RecordedAdjointRunner
            runner_type = RecordedAdjointRunner
            output_options = {"cell": cell}
        self._extractor_forward = runner_type(
            self.rnn.extractor_model,
            sequence_length=self.adjusted_seq_len,
            chunk_size=self.gradient_checkpoint_chunk_size,
            **output_options,
            differentiate_inputs=False,
            pack_spike_checkpoints=self.pack_spike_checkpoints,
            variable_gradient_transform=(
                cell.restore_segmented_variable_gradients
                if use_direct_csr_gradient and not csr_live_masters
                else None
            ),
            **statistics_options,
        )
        io.log_info(
            "Segmented exact BPTT enabled: "
            f"{self._extractor_forward.n_chunks} chunks, "
            f"chunk_size={self.gradient_checkpoint_chunk_size}, "
            f"pack_spike_checkpoints={self.pack_spike_checkpoints}."
        )

    def _statistics_windows(self):
        length = self.adjusted_seq_len
        windows = [(0, length)]
        for parameter in self.parameters:
            for loss in parameter.loss_functions.values():
                if hasattr(loss, "_pre_delay"):
                    window = (loss._pre_delay or 0, length - (loss._post_delay or 0))
                    if not 0 <= window[0] < window[1] <= length:
                        raise ValueError(f"Invalid spike-statistics loss window {window}.")
                    if window not in windows:
                        windows.append(window)
        return tuple(windows)

    def _spike_statistics_unmet(self, automatic=False):
        """Streaming prerequisites that do not hold, as user-facing sentences.

        ``automatic`` adds the policy-only requirements of the ``"auto"``
        request (a GLIF3 cell whose first state is the binary spike buffer);
        explicit ``True`` keeps the contract of the original validation.
        """
        from . import loss_functions as losses

        supported = (
            losses.TargetFiringRate, losses.SpikeRateDistributionTarget,
            losses.OrientationSelectivityLoss, losses.SynchronizationLoss,
            losses.VoltageRegularization, losses.EMDWeightRegularization,
            losses.LowRateFloor, losses.VoltageRateFloor,
        )
        unmet = []
        if not self.gradient_checkpointing:
            unmet.append("stream_spike_statistics requires gradient_checkpointing.")
        if not isinstance(self.learning_rule, BPTTLearningRule):
            unmet.append("stream_spike_statistics currently requires BPTT.")
        cell = getattr(self.rnn, "cell", None)
        if automatic and not _binary_spike_checkpoint_state(self.rnn):
            unmet.append("automatic streaming requires a GLIF3 cell with the binary spike state")
        if getattr(cell, "temporal_gradient_precision", "compute") != "compute":
            unmet.append("stream_spike_statistics requires compute temporal gradients.")
        strategy = getattr(self.rnn, "strategy", None)
        if strategy is None:
            if automatic:
                unmet.append("automatic streaming requires the RNN's distribution strategy")
        elif strategy.num_replicas_in_sync != 1:
            unmet.append("stream_spike_statistics currently requires one replica.")
        if automatic and not self.parameters:
            unmet.append("automatic streaming requires at least one parameter group with losses")
        for parameter in self.parameters:
            sampled = set()
            loss_functions = getattr(parameter, "loss_functions", None)
            if not loss_functions:
                if automatic:
                    unmet.append("automatic streaming requires every parameter group to define losses")
                continue
            for loss in loss_functions.values():
                if type(loss) not in supported:
                    unmet.append(f"stream_spike_statistics does not support {type(loss).__name__}.")
                    continue
                if type(loss) is losses.OrientationSelectivityLoss and loss._method not in ("crowd_osi", "crowd_spikes"):
                    unmet.append("Spike statistics require a rate-based orientation objective.")
                if type(loss) is losses.SynchronizationLoss:
                    if id(loss) in sampled:
                        unmet.append("A synchronization loss cannot be reused twice in one statistics batch group.")
                    sampled.add(id(loss))
        if not unmet:
            try:
                self._statistics_windows()
            except ValueError as error:
                unmet.append(str(error))
        return unmet

    def _validate_spike_statistics(self):
        unmet = self._spike_statistics_unmet()
        if unmet:
            raise ValueError(unmet[0])

    @staticmethod
    def _mean_spikes(spikes):
        if isinstance(spikes, SpikeSummary):
            return tf.reduce_mean(spikes.temporal_sum()) / tf.cast(spikes.shape[1], tf.float32)
        return tf.cast(tf.reduce_mean(spikes), tf.float32)

    @staticmethod
    def _spike_shape(spikes):
        if isinstance(spikes, SpikeSummary):
            return tf.stack((
                tf.shape(spikes.counts[0])[0], spikes.shape[1],
                tf.shape(spikes.counts[0])[-1],
            ))
        return tf.shape(spikes)

    @staticmethod
    def _add_gradients(accumulated, current):
        if accumulated is None:
            return current
        if current is None:
            return accumulated
        if isinstance(accumulated, tf.IndexedSlices) or isinstance(
            current, tf.IndexedSlices
        ):
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
        loss_vals, gradients, statistics = self._single_gradients(x, y, init_state)
        self._apply_gradients(gradients, statistics)
        return loss_vals

    def _single_gradients(self, x, y, init_state):
        pname = self.parameters[0].name
        loss_vals = {pname: {}}
        input_spikes = x[0]
        with tf.GradientTape() as tape, weight_regularization_scope():
            _out = self._run_parameter_extractor(input_spikes, init_state, self.parameters[0])
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]
            if self._record_mean_rate_metrics_enabled:
                loss_vals[pname]["__mean_rate"] = tf.cast(
                    self._mean_spikes(_spikes_out), tf.float32
                )

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
            loss_for_grad = optimizers.scale_loss_for_optimizer(
                self.optimizer, total_loss
            )
            loss_vals["__total_loss"] = tf.reduce_mean(
                _total_loss if getattr(self.rnn, "data_parallel", None) else total_loss
            )

        # Backpropagation of the model (gradients computation and application)
        grad = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
        grad = optimizers.unscale_gradients_for_optimizer(self.optimizer, grad)
        return loss_vals, grad, self._voltage_floor_rate_statistics(_spikes_out)

    @tf.function
    def _distributed_single_gradients(self, x, y, init_state):
        return self.rnn.strategy.run(
            self._single_gradients, args=(x, y, init_state)
        )

    def _train_step_parameter_gradients(self, x, y, init_state, parameter_index):
        """Compute one parameter set's gradients without applying them."""
        p = self.parameters[parameter_index]
        ysig = InputsSignatureFactory.InputsSignature(
            copy.deepcopy(self.inputs_signature_factory.lu_tables[parameter_index]), y
        )
        loss_vals = {p.name: {}}
        with tf.GradientTape() as tape, weight_regularization_scope():
            _out = self._run_parameter_extractor(x, init_state, p)
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]
            if self._record_mean_rate_metrics_enabled:
                loss_vals[p.name]["__mean_rate"] = tf.cast(
                    self._mean_spikes(_spikes_out), tf.float32
                )

            _total_loss = 0.0
            loss_kwargs = self._prepare_loss_kwargs(p, _spikes_out, ysig)
            if getattr(self.rnn, "_online_voltage_losses", ()):
                loss_kwargs["voltage_rate_floor_batch_weight"] = (
                    self.n_parameters * p.batch_size
                    / sum(parameter.batch_size for parameter in self.parameters)
                )
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
            loss_for_grad = optimizers.scale_loss_for_optimizer(
                self.optimizer, total_loss
            )
            loss_vals["__total_loss"] = total_loss

        grads = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
        grads = optimizers.unscale_gradients_for_optimizer(self.optimizer, grads)
        if getattr(self.rnn, "_online_voltage_losses", ()):
            loss_vals["__voltage_floor_statistics"] = self._voltage_floor_rate_statistics(_spikes_out)
        return loss_vals, grads

    @tf.function
    def _distributed_train_step_parameter_gradients(
        self, x, y, init_state, parameter_index
    ):
        return self.rnn.strategy.run(
            self._train_step_parameter_gradients,
            args=(x, y, init_state, parameter_index),
        )

    def _voltage_floor_rate_statistics(self, spikes):
        if not getattr(self.rnn, "_online_voltage_losses", ()):
            return None
        # Detached FP32 chunk reductions never expand the full spike sequence.
        if isinstance(spikes, SpikeSummary):
            counts = tf.reduce_sum(tf.stop_gradient(spikes.temporal_sum()), axis=0)
        else:
            spikes = tf.stop_gradient(spikes)
            counts = tf.add_n([
                tf.reduce_sum(tf.cast(spikes[:, start:start + 25], tf.float32), axis=(0, 1))
                for start in range(0, spikes.shape[1], 25)
            ])
        shape = self._spike_shape(spikes)
        samples = tf.cast(shape[0] * shape[1], tf.float32)
        context = tf.distribute.get_replica_context()
        if context is not None:
            counts = context.all_reduce(tf.distribute.ReduceOp.SUM, counts)
            samples = context.all_reduce(tf.distribute.ReduceOp.SUM, samples)
        return counts, samples

    def _apply_gradients(self, grads, rate_statistics=None):
        parallel = getattr(self.rnn, "data_parallel", None)
        if getattr(parallel, "process_based", False):
            # External graph arguments have no device for Keras' local reduction.
            with tf.device(parallel.strategy.extended.worker_devices[0]):
                grads = [None if value is None else tf.identity(value) for value in grads]
        losses = getattr(self.rnn, "_online_voltage_losses", ())
        if not losses:
            self.optimizer.apply_gradients(zip(grads, self.rnn.model.trainable_variables))
            return
        if rate_statistics is None:
            raise ValueError("VoltageRateFloor requires measured rates for every optimizer update.")
        # Keras 2's dynamic scaler can increment iterations even when it skips
        # an update. Its finite-gradient check must accompany the counter check.
        finite = tf.reduce_all(tf.stack([
            tf.reduce_all(tf.math.is_finite(g.values if isinstance(g, tf.IndexedSlices) else g))
            for g in grads if g is not None
        ]))
        context = tf.distribute.get_replica_context()
        if context is not None:
            finite = context.all_reduce(tf.distribute.ReduceOp.SUM, tf.cast(finite, tf.int32))
            finite = finite == self.rnn.strategy.num_replicas_in_sync
        before = tf.identity(self.optimizer.iterations)
        with tf.control_dependencies([before]):
            self.optimizer.apply_gradients(zip(grads, self.rnn.model.trainable_variables))
        accepted = self.optimizer.iterations > before
        if optimizers.optimizer_supports_loss_scaling(self.optimizer):
            accepted = tf.logical_and(accepted, finite)
        counts, samples = rate_statistics

        def commit():
            for loss in losses:
                loss.commit_rates(counts * (1000.0 / loss.dt) / samples)
            return tf.constant(0)

        return tf.cond(accepted, commit, lambda: tf.constant(0))

    @tf.function
    def _distributed_apply_gradients(self, grads, rate_statistics=None):
        return self.rnn.strategy.run(self._apply_gradients, args=(grads, rate_statistics))

    @staticmethod
    def _record_signature_metrics(loss_vals, pname, y):
        candidates = y.ysigs if hasattr(y, "ysigs") else y
        if isinstance(candidates, dict):
            candidates = [candidates]
        elif not isinstance(candidates, (list, tuple)):
            return

        for sig in candidates:
            if isinstance(sig, dict) and "orientation" in sig:
                orientation = tf.reshape(tf.cast(sig["orientation"], tf.float32), [-1])
                loss_vals[pname]["__orientation_mean"] = tf.reduce_mean(orientation)
                loss_vals[pname]["__orientation_first"] = orientation[0]
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
        with tf.GradientTape() as tape, weight_regularization_scope():
            _out = self._run_extractor(x_concat, init_state)
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]

            _total_loss = 0.0
            pidx_beg = 0
            for p, pidx_end, ysig in zip(self.parameters, self.batch_indices, sigs):
                loss_vals[p.name] = {}
                _pspikes = _spikes_out[pidx_beg:pidx_end]
                _pvolts = _v_out[pidx_beg:pidx_end]
                _pstate = tf.nest.map_structure(
                    lambda state: state[pidx_beg:pidx_end], _model_state
                )
                if self._record_mean_rate_metrics_enabled:
                    loss_vals[p.name]["__mean_rate"] = tf.cast(
                        self._mean_spikes(_pspikes), tf.float32
                    )
                if self._record_signature_metrics_enabled:
                    self._record_signature_metrics(loss_vals, p.name, ysig)
                cached_spike_counts = None
                cached_spike_duration = None
                if _pspikes.shape[1] is not None:
                    cached_spike_counts = loss_utils.temporal_sum(
                        _pspikes, dtype=tf.float32
                    )
                    cached_spike_duration = tf.cast(self._spike_shape(_pspikes)[1], tf.float32)
                loss_kwargs = self._prepare_loss_kwargs(
                    p,
                    _pspikes,
                    ysig,
                    spike_counts=(
                        None
                        if cached_spike_counts is None
                        else tf.stop_gradient(cached_spike_counts)
                    ),
                    spike_count_duration=cached_spike_duration,
                )
                if cached_spike_counts is not None:
                    loss_kwargs["spike_counts"] = cached_spike_counts
                    loss_kwargs["spike_count_duration"] = cached_spike_duration
                if getattr(self.rnn, "_online_voltage_losses", ()):
                    loss_kwargs["voltage_rate_floor_batch_weight"] = tf.cast(
                        self.n_parameters * self._spike_shape(_pspikes)[0], tf.float32
                    ) / tf.cast(self._spike_shape(_spikes_out)[0], tf.float32)
                for loss_name, loss_fnc in p.loss_functions.items():
                    _loss = loss_fnc(
                        spikes=_pspikes,
                        voltages=_pvolts,
                        model_state=_pstate,
                        y=ysig,
                        **loss_kwargs,
                    )
                    _loss32 = tf.cast(_loss, tf.float32)
                    _total_loss += _loss32
                    loss_vals[p.name][loss_name] = _loss
                # Advance to this parameter's slice of the concatenated batch so the next
                # parameter's losses are computed on its own spikes/voltages (not [0:end]).
                pidx_beg = pidx_end

            _total_loss = tf.cast(_total_loss, tf.float32)
            total_loss = tf.nn.scale_regularization_loss(_total_loss)
            all_losses.append(total_loss)
            loss_for_grad = optimizers.scale_loss_for_optimizer(
                self.optimizer, total_loss
            )

        # Backpropagation of the model (gradients computation and application)
        grad = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
        grad = optimizers.unscale_gradients_for_optimizer(self.optimizer, grad)
        self._apply_gradients(grad, self._voltage_floor_rate_statistics(_spikes_out))

        total_metric = tf.reduce_mean(all_losses) / tf.cast(
            self.n_parameters, total_loss.dtype
        )
        loss_vals["__total_loss"] = total_metric
        return loss_vals

    def _train_step_series(self, xs, ys, init_state):
        loss_vals = {}
        all_losses = []
        for parameter_index, (p, x, y) in enumerate(zip(self.parameters, xs, ys)):
            loss_vals[p.name] = {}
            with tf.GradientTape() as tape, weight_regularization_scope():
                _out = self._run_parameter_extractor(x, init_state, p)
                _spikes_out, _v_out = _out[0]
                _model_state = _out[1:]
                if self._record_mean_rate_metrics_enabled:
                    loss_vals[p.name]["__mean_rate"] = tf.cast(
                        self._mean_spikes(_spikes_out), tf.float32
                    )
                if self._record_signature_metrics_enabled:
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
                loss_for_grad = optimizers.scale_loss_for_optimizer(
                    self.optimizer, total_loss
                )
                all_losses.append(total_loss)

            grad = tape.gradient(loss_for_grad, self.rnn.model.trainable_variables)
            grad = optimizers.unscale_gradients_for_optimizer(self.optimizer, grad)
            self._apply_gradients(grad, self._voltage_floor_rate_statistics(_spikes_out))
            if parameter_index + 1 < self.n_parameters:
                self.rnn.cell.refresh_recurrent_weight_shadow()

        loss_vals["__total_loss"] = tf.reduce_mean(all_losses)
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
        rate_statistics = None

        for parameter_index, (p, x, y) in enumerate(zip(self.parameters, xs, ys)):
            param_loss_vals, grads = self._distributed_train_step_parameter_gradients(
                x, y, init_state, parameter_index
            )
            loss_vals[p.name] = param_loss_vals[p.name]
            all_losses.append(param_loss_vals["__total_loss"])
            statistics = param_loss_vals.get("__voltage_floor_statistics")
            if statistics is not None:
                rate_statistics = statistics if rate_statistics is None else tf.nest.map_structure(
                    tf.add, rate_statistics, statistics
                )
            if accum_grads is None:
                accum_grads = list(grads)
            else:
                accum_grads = [
                    self._add_gradients(accum_grad, grad)
                    for accum_grad, grad in zip(accum_grads, grads)
                ]

        self._distributed_apply_gradients(accum_grads, rate_statistics)

        loss_vals["__total_loss"] = tf.reduce_mean(all_losses)
        return loss_vals

    @tf.function
    def _distributed_train_step(self, x, y, init_state):
        if getattr(getattr(self.rnn, "data_parallel", None), "process_based", False):
            raise ValueError("MPI collectives require eager train_step(), not a traced step.")
        if getattr(self.rnn, "data_parallel", None) is not None:
            self.rnn.data_parallel.validate_training(self)
        return self.rnn.strategy.run(self.step_train_function, args=(x, y, init_state))

    def train_step(self, x, y, init_state):
        """Apply one update; process collectives run outside TensorFlow graphs."""
        ready = self.bind_optimizer_checkpoint_slots()
        metrics = self._train_step_unbound(x, y, init_state)
        if not ready:
            # The update built the optimizer; restored-but-unbound CSR slots now raise.
            self.bind_optimizer_checkpoint_slots(after_update=True)
        return metrics

    def _train_step_unbound(self, x, y, init_state):
        parallel = getattr(self.rnn, "data_parallel", None)
        if not getattr(parallel, "process_based", False):
            return self._distributed_train_step(x, y, init_state)
        parallel.prepare(self)
        metrics, gradients, statistics = self._distributed_single_gradients(x, y, init_state)
        if statistics is not None:
            raise ValueError("MPI online population-rate state is not yet qualified.")
        gradients = parallel.average_gradients(gradients)
        self._distributed_apply_gradients(gradients)
        return metrics

    def _validation_step(self, xs, ys, init_state, training_approach):
        if training_approach in ["series", "series_accumulate", "single"]:
            loss_vals = {}
            all_losses = []
            for p, x, y in zip(self.parameters, xs, ys):
                loss_vals[p.name] = {}
                _out = self.rnn.run_extractor(x, init_state)
                _spikes_out, _v_out = _out[0]
                _model_state = _out[1:]
                if self._record_mean_rate_metrics_enabled:
                    loss_vals[p.name]["__mean_rate"] = tf.cast(
                        self._mean_spikes(_spikes_out), tf.float32
                    )
                if self._record_signature_metrics_enabled:
                    self._record_signature_metrics(loss_vals, p.name, y)

                _total_loss = 0.0
                loss_kwargs = self._prepare_loss_kwargs(p, _spikes_out, y)
                if training_approach == "series_accumulate" and getattr(self.rnn, "_online_voltage_losses", ()):
                    loss_kwargs["voltage_rate_floor_batch_weight"] = (
                        self.n_parameters * p.batch_size
                        / sum(parameter.batch_size for parameter in self.parameters)
                    )
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
                all_losses.append(
                    _total_loss if getattr(self.rnn, "data_parallel", None) else total_loss
                )

            loss_vals["__total_loss"] = tf.reduce_mean(all_losses)

        elif training_approach == "parallel":
            x_concat = tf.concat(xs, axis=0)
            sigs = self.inputs_signature_factory.build(ys)
            _out = self.rnn.run_extractor(x_concat, init_state)
            _spikes_out, _v_out = _out[0]
            _model_state = _out[1:]

            loss_vals = {}
            _total_loss = 0.0
            pidx_beg = 0
            for p, pidx_end, ysig in zip(self.parameters, self.batch_indices, sigs):
                loss_vals[p.name] = {}
                _pspikes = _spikes_out[pidx_beg:pidx_end]
                _pvolts = _v_out[pidx_beg:pidx_end]
                _pstate = tf.nest.map_structure(
                    lambda state: state[pidx_beg:pidx_end], _model_state
                )
                if self._record_mean_rate_metrics_enabled:
                    loss_vals[p.name]["__mean_rate"] = tf.cast(
                        self._mean_spikes(_pspikes), tf.float32
                    )
                if self._record_signature_metrics_enabled:
                    self._record_signature_metrics(loss_vals, p.name, ysig)
                loss_kwargs = self._prepare_loss_kwargs(p, _pspikes, ysig)
                if getattr(self.rnn, "_online_voltage_losses", ()):
                    loss_kwargs["voltage_rate_floor_batch_weight"] = tf.cast(
                        self.n_parameters * self._spike_shape(_pspikes)[0], tf.float32
                    ) / tf.cast(self._spike_shape(_spikes_out)[0], tf.float32)
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
            loss_vals["__total_loss"] = total_loss / tf.cast(
                self.n_parameters, total_loss.dtype
            )

        else:
            raise ValueError(f'Unknown training approach method "{training_approach}"')

        return loss_vals

    @tf.function
    def _distributed_validation_step(self, xs, ys, init_state, training_approach):
        return self.rnn.strategy.run(
            self._validation_step, args=(xs, ys, init_state, training_approach)
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

    def _next_spikes_with_retry(self, input_itr, max_retries=8):
        """Fetch the next input batch, skipping rare malformed batches.

        The LGN tf.data generator pipeline intermittently raises
        InvalidArgumentError ("Shapes of all inputs must match ... Op:Pack")
        when assembling a batch (~1 per 16-43 epochs), which otherwise kills
        long runs. Re-fetching advances past the offending batch; a skipped
        batch is negligible over thousands of steps. This is a robustness guard
        around a pre-existing pipeline bug, not a correctness change.
        """
        last_err = None
        for attempt in range(max_retries):
            try:
                return input_itr.next_spikes()
            except (tf.errors.InvalidArgumentError, tf.errors.UnknownError) as e:
                last_err = e
                io.log_debug(
                    f"{self.__class__.__name__}: next_spikes() raised {type(e).__name__} "
                    f"(attempt {attempt + 1}/{max_retries}); rebuilding iterator and skipping batch."
                )
                # A tf.data iterator that raised is in an undefined state; rebuild it
                # (the LGN generators restart, which only reshuffles the stimulus stream).
                try:
                    input_itr.close()
                    input_itr.build()
                except Exception:
                    pass
        raise last_err

    def train(self):
        data_parallel = getattr(self.rnn, "data_parallel", None)
        process_parallel = getattr(data_parallel, "process_based", False)
        chief = not process_parallel or data_parallel.is_chief
        if data_parallel is not None:
            data_parallel.validate_training(self)
        prepare_training_model = getattr(self.rnn, "_prepare_training_model", None)
        if prepare_training_model is not None:
            prepare_training_model()
        input_generators = []
        input_batch_sizes = []
        input_seq_lens = []
        for p in self.parameters:
            input_generators.append(p.input_generators)
            input_batch_sizes.append(
                p.batch_size * data_parallel.replicas if data_parallel is not None else p.batch_size
            )
            input_seq_lens.append(p.seq_len)

        # fetch_in_graph=False: fetch the next batch eagerly. The in-graph path wraps the
        # Python-generator fetch in @tf.function(reduce_retracing=True), which intermittently
        # fails with an Op:Pack shape mismatch ([seq,n_input] vs []) on long runs (crashed
        # ~epoch 16-22). Eager fetch avoids the tf.function packing; tf.data prefetch still
        # overlaps the LGN generation with compute, and the model forward dominates step time.
        input_itr = DataIterator(
            input_generators,
            input_batch_sizes,
            input_seq_lens,
            self.rnn.ordered_inputs_populations,
            fetch_in_graph=False,
            use_device_generation=data_parallel is None,
            output_dtype=(
                self.rnn.model.inputs[0].dtype
                if getattr(getattr(self.rnn, "cell", None), "_use_combined_current_sweep", False)
                else None
            ),
        ) if chief else None

        # input_itrs = [DataIterator(p.input_generators, p.batch_size, p.seq_len, self.rnn.ordered_inputs_populations) for p in self.parameters]
        init_state = None
        if not self.regenerate_initial_state_each_epoch:
            init_state = (
                self.rnn.strategy.run(self.init_state.get_state)
                if data_parallel is not None else self.init_state.get_state()
            )
        self._prepare_normalizers()

        self.prepare_gradient_checkpointing()

        try:
            if chief:
                self.callbacks.on_train_begin()
            for epoch in range(self.n_epochs):
                if chief:
                    self.callbacks.on_epoch_start()
                if self.regenerate_initial_state_each_epoch:
                    init_state = (
                        self.rnn.strategy.run(self.init_state.get_state)
                        if data_parallel is not None else self.init_state.get_state()
                    )

                for step in range(self.steps_per_epoch):
                    if chief:
                        self.callbacks.on_step_start()
                    values = self._next_spikes_with_retry(input_itr) if chief else None
                    if data_parallel is not None:
                        spikes, ys = data_parallel.distribute_batch(values)
                    else:
                        spikes, ys = values
                    if self.training_approach == "series_accumulate":
                        ready = self.bind_optimizer_checkpoint_slots()
                        step_loss_vals = self._distributed_train_step_series_accumulate(
                            spikes, ys, init_state=init_state
                        )
                        if not ready:
                            self.bind_optimizer_checkpoint_slots(after_update=True)
                    else:
                        step_loss_vals = self.train_step(
                            spikes, ys, init_state=init_state
                        )
                    self.learning_rule.on_global_step_end()
                    # Propagate the just-updated master recurrent weights into the compute-dtype
                    # shadow used by the forward pass (no-op when not using a shadow).
                    cell = getattr(self.rnn, "_cell", None)
                    if cell is not None and hasattr(
                        cell, "refresh_recurrent_weight_shadow"
                    ):
                        if data_parallel is not None:
                            self.rnn.strategy.run(cell.refresh_recurrent_weight_shadow)
                        else:
                            cell.refresh_recurrent_weight_shadow()
                    if data_parallel is not None:
                        step_loss_vals = data_parallel.reduce_metrics(step_loss_vals)
                    if chief:
                        self.callbacks.on_step_end(step_loss_vals)

                validation_loss = self._distributed_validation_step(
                    spikes,
                    ys,
                    init_state=init_state,
                    training_approach=self.training_approach,
                )
                if data_parallel is not None:
                    validation_loss = data_parallel.reduce_metrics(validation_loss)
                stop = self.callbacks.on_epoch_end(validation_loss) if chief else None
                if process_parallel:
                    stop = data_parallel.broadcast_stop(stop)
                if stop:
                    break

            normalizers = None
            if self._normalizers is not None:
                normalizers = {
                    name: value.numpy() for name, value in self._normalizers.items()
                }
            if chief:
                self.callbacks.on_train_end(normalizers=normalizers)
        finally:
            if input_itr is not None:
                input_itr.close()

    """
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
    """
