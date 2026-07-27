import tensorflow as tf

from .base import LearningRule, WeightSurface


class EPropLearningRule(LearningRule):
    """Sparse three-factor eligibility-propagation updates.

    Synaptic-rise, postsynaptic-current, and voltage eligibility states follow
    the same basis-weighted GLIF3 dynamics as the forward cell. The voltage
    eligibility is combined with the local factor
    ``d loss/dz_j[t] * psi_j[t] + d loss/dv_j[t]``. Edges are processed in
    bounded chunks, avoiding sequence-by-edge tensors.
    """

    def __init__(
            self,
            surrogate_dampening=None,
            edge_chunk_size=65536,
            surfaces=None,
            learning_signal_clip=None,
            gradient_clip_norm=None,
            min_weight=None,
            max_weight=None,
            name='eprop'):
        super().__init__(name=name)
        if edge_chunk_size <= 0:
            raise ValueError('edge_chunk_size must be greater than zero.')
        if min_weight is not None and max_weight is not None and min_weight > max_weight:
            raise ValueError('min_weight must not exceed max_weight.')

        self.surrogate_dampening = (
            None if surrogate_dampening is None else float(surrogate_dampening)
        )
        self.edge_chunk_size = int(edge_chunk_size)
        if isinstance(surfaces, str):
            raise TypeError(
                'surfaces must be a sequence of surface names, not a single string.'
            )
        self.surface_names = None if surfaces is None else tuple(surfaces)
        if self.surface_names is not None:
            if not self.surface_names:
                raise ValueError('surfaces must contain at least one surface name.')
            if any(not isinstance(surface, str) for surface in self.surface_names):
                raise TypeError('Every surfaces entry must be a string.')
            if len(set(self.surface_names)) != len(self.surface_names):
                raise ValueError('surfaces contains duplicate surface names.')
        self.learning_signal_clip = learning_signal_clip
        self.gradient_clip_norm = gradient_clip_norm
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_step = tf.Variable(
            0, dtype=tf.int64, trainable=False, name='update_step'
        )

    @classmethod
    def module(cls):
        return 'eprop'

    def _surface_enabled(self, name):
        return self.surface_names is None or name in self.surface_names

    def _validate_requested_surfaces(self, cell):
        if self.surface_names is None:
            return

        available = {'<recurrent>': cell.recurrent_weight_values}
        available.update(
            (name, props['input_weight_values'])
            for name, props in cell.inputs.items()
        )
        unknown = sorted(set(self.surface_names) - set(available))
        if unknown:
            choices = ', '.join(sorted(available))
            raise ValueError(
                f'Unknown learning-rule surface(s): {", ".join(unknown)}. '
                f'Available surfaces: {choices}.'
            )

        non_trainable = [
            name for name in self.surface_names if not available[name].trainable
        ]
        if non_trainable:
            raise ValueError(
                f'Requested learning-rule surface(s) are not trainable: '
                f'{", ".join(non_trainable)}. Enable training for each surface '
                'or remove it from training.learning_rule.surfaces.'
            )

    def _build_weight_surfaces(self, rnn):
        cell = rnn._cell
        if cell._hard_reset:
            raise ValueError('eprop does not currently support hard_reset=True.')
        self._validate_requested_surfaces(cell)
        surfaces = []
        if self._surface_enabled('<recurrent>') and cell.recurrent_weight_values.trainable:
            surfaces.append(WeightSurface(
                name='<recurrent>',
                variable=cell.recurrent_weight_values,
                indices=cell.recurrent_indices,
                synapse_types=cell.syn_ids,
                source='recurrent',
            ))

        for input_name, input_props in cell.inputs.items():
            variable = input_props['input_weight_values']
            if not self._surface_enabled(input_name) or not variable.trainable:
                continue
            if input_props['input_type'] not in ('spikes', 'current'):
                raise ValueError(
                    f'eprop cannot observe internally generated input surface "{input_name}" '
                    f'with input_type="{input_props["input_type"]}".'
                )
            surfaces.append(WeightSurface(
                name=input_name,
                variable=variable,
                indices=input_props['input_indices'],
                synapse_types=input_props['input_syn_ids'],
                source=input_name,
            ))

        if not surfaces:
            requested = 'all trainable surfaces' if self.surface_names is None else self.surface_names
            raise ValueError(f'eprop did not find any requested trainable weight surfaces: {requested}.')
        return surfaces

    def _local_factor(self, observations, pseudo_derivative):
        return (
            tf.cast(observations.spike_learning_signal, tf.float32)
            * pseudo_derivative
            + tf.cast(observations.voltage_learning_signal, tf.float32)
        )

    def _prepare_rule_state(self, observations, pseudo_derivative):
        del observations, pseudo_derivative
        return ()

    def _initial_chunk_state(self, batch_size, edge_count):
        del batch_size, edge_count
        return ()

    def _accumulate_gradient(
            self,
            time,
            post_ids,
            post_local_factor,
            membrane_eligibility,
            pseudo_derivative,
            observations,
            rule_state,
            chunk_state):
        del time, post_ids, pseudo_derivative, observations, rule_state
        gradient = tf.reduce_mean(
            post_local_factor * membrane_eligibility, axis=0
        )
        return gradient, chunk_state

    def _surrogate_derivative(self, observations):
        cell = self.rnn._cell
        voltages = observations.voltages
        dampening = self.surrogate_dampening
        if dampening is None:
            dampening = cell._dampening_factor
        dtype = voltages.dtype
        derivative = (
            tf.cast(dampening, dtype)
            * tf.maximum(
                tf.cast(1.0, dtype) - tf.abs(voltages - tf.cast(cell.v_th, dtype)),
                tf.cast(0.0, dtype),
            )
        )
        if len(observations.initial_state) <= 2:
            return derivative

        batch_size = tf.shape(observations.spikes, out_type=tf.int32)[0]
        seq_len = tf.shape(observations.spikes, out_type=tf.int32)[1]
        n_neurons = cell._n_neurons
        refractory_state = tf.cast(observations.initial_state[2], dtype)
        previous_spikes = tf.cast(
            observations.initial_state[0][:, :n_neurons], dtype
        )
        refractory_masks = tf.TensorArray(dtype=tf.bool, size=seq_len)

        def condition(time, refractory, previous, masks):
            del refractory, previous, masks
            return time < seq_len

        def body(time, refractory, previous, masks):
            refractory = tf.maximum(
                refractory
                + previous * tf.cast(cell.t_ref_steps, dtype)
                - tf.cast(1.0, dtype),
                tf.cast(0.0, dtype),
            )
            masks = masks.write(time, refractory > tf.cast(0.0, dtype))
            previous = tf.cast(observations.spikes[:, time, :], dtype)
            return time + 1, refractory, previous, masks

        _, _, _, refractory_masks = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0, tf.int32),
                refractory_state,
                previous_spikes,
                refractory_masks,
            ),
            parallel_iterations=1,
        )
        refractory_masks = tf.transpose(refractory_masks.stack(), [1, 0, 2])
        return tf.where(refractory_masks, tf.zeros_like(derivative), derivative)

    def _edge_gradients(self, surface, observations, pseudo_derivative):
        indices = tf.cast(surface.indices, tf.int32)
        n_edges = tf.shape(indices, out_type=tf.int32)[0]
        chunk_size = tf.constant(self.edge_chunk_size, dtype=tf.int32)
        batch_size = tf.shape(observations.spikes, out_type=tf.int32)[0]
        seq_len = tf.shape(observations.spikes, out_type=tf.int32)[1]
        pseudo_derivative = tf.cast(pseudo_derivative, tf.float32)
        local_factor = self._local_factor(observations, pseudo_derivative)
        if self.learning_signal_clip is not None:
            clip = tf.cast(self.learning_signal_clip, local_factor.dtype)
            local_factor = tf.clip_by_value(local_factor, -clip, clip)
        local_factor_by_time = tf.transpose(local_factor, [1, 2, 0])
        rule_state = self._prepare_rule_state(observations, pseudo_derivative)
        cell = self.rnn._cell
        n_basis = cell._n_syn_basis
        synaptic_basis_weights = tf.cast(cell.synaptic_basis_weights, tf.float32)
        synaptic_decay = tf.reshape(
            tf.cast(cell.syn_decay, tf.float32),
            [cell._n_neurons, n_basis],
        )[0]
        psc_initial = tf.reshape(
            tf.cast(cell.psc_initial, tf.float32),
            [cell._n_neurons, n_basis],
        )[0]
        membrane_decay = tf.reshape(tf.cast(cell.decay, tf.float32), [-1])
        current_factor = tf.reshape(tf.cast(cell.current_factor, tf.float32), [-1])
        voltage_gradient_factor = tf.cast(
            1.0 - cell._voltage_gradient_dampening, tf.float32
        )
        dt = tf.cast(cell._dt, tf.float32)
        lr_scale = tf.cast(cell._lr_scale, tf.float32)

        if surface.source == 'recurrent':
            n_neurons = self.rnn._cell._n_neurons
            max_delay = self.rnn._cell.max_delay
            initial_z_buffer = tf.reshape(
                observations.initial_state[0],
                [batch_size, max_delay, n_neurons],
            )
            initial_history = tf.reverse(initial_z_buffer, axis=[1])
            pre_history = tf.concat(
                [tf.cast(initial_history, tf.float32),
                 tf.cast(observations.spikes, tf.float32)],
                axis=1,
            )
            pre_by_time = tf.transpose(pre_history, [1, 2, 0])
        else:
            input_props = self.rnn._cell.inputs[surface.source]
            input_index = list(self.rnn._cell.inputs).index(surface.source)
            begin = self.rnn._cell.inputs_idx[input_index]
            end = begin + input_props['input_dim']
            pre_by_time = tf.transpose(
                tf.cast(observations.input_spikes[:, :, begin:end], tf.float32),
                [1, 2, 0],
            )
            max_delay = 0

        gradients = tf.TensorArray(
            tf.float32, size=0, dynamic_size=True, infer_shape=False,
            element_shape=tf.TensorShape([None]),
        )

        def chunk_condition(start, chunk_index, result):
            del chunk_index, result
            return start < n_edges

        def chunk_body(start, chunk_index, result):
            end = tf.minimum(start + chunk_size, n_edges)
            chunk_indices = indices[start:end]
            chunk_synapse_types = tf.cast(
                surface.synapse_types[start:end], tf.int32
            )
            post_ids = chunk_indices[:, 0]
            raw_pre_ids = chunk_indices[:, 1]
            if surface.source == 'recurrent':
                delays = raw_pre_ids // self.rnn._cell._n_neurons
                pre_ids = raw_pre_ids % self.rnn._cell._n_neurons
            else:
                delays = tf.zeros_like(raw_pre_ids)
                pre_ids = raw_pre_ids

            edge_count = end - start
            synaptic_rise_eligibility = tf.zeros(
                [batch_size, edge_count, n_basis], dtype=tf.float32
            )
            psc_eligibility = tf.zeros(
                [batch_size, edge_count, n_basis], dtype=tf.float32
            )
            voltage_eligibility = tf.zeros(
                [batch_size, edge_count], dtype=tf.float32
            )
            gradient = tf.zeros([edge_count], dtype=tf.float32)
            chunk_state = self._initial_chunk_state(batch_size, edge_count)
            edge_basis = tf.gather(synaptic_basis_weights, chunk_synapse_types)
            post_membrane_decay = tf.gather(membrane_decay, post_ids)[tf.newaxis, :]
            post_current_factor = tf.gather(current_factor, post_ids)[tf.newaxis, :]

            def time_condition(time, rise_eligibility, current_eligibility,
                               membrane_eligibility, edge_gradient, local_chunk_state):
                del rise_eligibility, current_eligibility
                del membrane_eligibility, edge_gradient, local_chunk_state
                return time < seq_len

            def time_body(time, rise_eligibility, current_eligibility,
                          membrane_eligibility, edge_gradient, local_chunk_state):
                post_local_factor = tf.transpose(
                    tf.gather(local_factor_by_time[time], post_ids)
                )
                if surface.source == 'recurrent':
                    source_times = time + max_delay - 1 - delays
                    pre_spikes = tf.transpose(tf.gather_nd(
                        pre_by_time, tf.stack([source_times, pre_ids], axis=1)
                    ))
                else:
                    pre_spikes = tf.transpose(tf.gather(pre_by_time[time], pre_ids))

                new_membrane_eligibility = (
                    post_membrane_decay * voltage_gradient_factor * membrane_eligibility
                    + post_current_factor * tf.reduce_sum(current_eligibility, axis=-1)
                )
                # GLIF3 stops gradients through spikes entering ASC, so ASC has no
                # weight-eligibility contribution in the cell's differentiation contract.
                gradient_increment, local_chunk_state = self._accumulate_gradient(
                    time=time,
                    post_ids=post_ids,
                    post_local_factor=post_local_factor,
                    membrane_eligibility=new_membrane_eligibility,
                    pseudo_derivative=pseudo_derivative,
                    observations=observations,
                    rule_state=rule_state,
                    chunk_state=local_chunk_state,
                )
                edge_gradient += gradient_increment
                new_current_eligibility = (
                    current_eligibility * synaptic_decay
                    + dt * synaptic_decay * rise_eligibility
                )
                new_rise_eligibility = (
                    rise_eligibility * synaptic_decay
                    + pre_spikes[:, :, tf.newaxis]
                    * edge_basis[tf.newaxis, :, :]
                    * psc_initial
                    * lr_scale
                )
                return (
                    time + 1,
                    new_rise_eligibility,
                    new_current_eligibility,
                    new_membrane_eligibility,
                    edge_gradient,
                    local_chunk_state,
                )

            _, _, _, _, gradient, _ = tf.while_loop(
                time_condition,
                time_body,
                (
                    tf.constant(0, tf.int32),
                    synaptic_rise_eligibility,
                    psc_eligibility,
                    voltage_eligibility,
                    gradient,
                    chunk_state,
                ),
                parallel_iterations=1,
            )
            return end, chunk_index + 1, result.write(chunk_index, gradient)

        _, _, gradients = tf.while_loop(
            chunk_condition,
            chunk_body,
            (tf.constant(0, tf.int32), tf.constant(0, tf.int32), gradients),
            parallel_iterations=1,
        )
        gradient = gradients.concat()
        if self.gradient_clip_norm is not None:
            gradient = tf.clip_by_norm(gradient, self.gradient_clip_norm)
        return tf.cast(gradient, surface.variable.dtype)

    def compute_updates(self, observations):
        pseudo_derivative = self._surrogate_derivative(observations)
        return tuple(
            (
                surface,
                self._edge_gradients(surface, observations, pseudo_derivative)
                + tf.cast(direct_gradient, surface.variable.dtype),
            )
            for surface, direct_gradient in zip(
                self.weight_surfaces, observations.direct_weight_gradients
            )
        )

    def apply_updates(self, optimizer, updates):
        updates = tuple(updates)
        super().apply_updates(optimizer, updates)
        for surface, _ in updates:
            variable = surface.variable
            value = variable
            if self.min_weight is not None or self.max_weight is not None:
                lower = self.min_weight if self.min_weight is not None else -float('inf')
                upper = self.max_weight if self.max_weight is not None else float('inf')
                value = tf.clip_by_value(value, lower, upper)
            constraint = getattr(variable, 'constraint', None)
            if constraint is not None:
                value = constraint(value)
            variable.assign(value)
    def on_global_step_end(self):
        self.update_step.assign_add(1)

    def get_config(self):
        return {
            'name': self.module(),
            'surrogate_dampening': self.surrogate_dampening,
            'edge_chunk_size': self.edge_chunk_size,
            'surfaces': self.surface_names,
            'learning_signal_clip': self.learning_signal_clip,
            'gradient_clip_norm': self.gradient_clip_norm,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
        }
