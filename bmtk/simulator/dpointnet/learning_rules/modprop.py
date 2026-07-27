import numpy as np
import tensorflow as tf

from .eprop import EPropLearningRule


class ModPropLearningRule(EPropLearningRule):
    """GLIF3-compatible hybrid of e-prop and published ModProp.

    The additional term follows Liu et al. (2022), using fixed filters
    ``F[alpha, beta, s] = mu**s * A**(s + 1)[alpha, beta]`` and past
    eligibility traces. ``A`` is the initial effective recurrent matrix
    averaged over every possible pair of neurons in each node-type pair.
    Axis ``alpha`` is the downstream/modulating type; ``beta`` is the
    postsynaptic type of the synapse being trained.
    The base local factor retains DPointNet's voltage-loss contribution.
    """

    def __init__(
            self,
            filter_taps=None,
            mean_activity=None,
            modulatory_filters=None,
            **kwargs):
        configured_filters = (
            None if modulatory_filters is None
            else np.asarray(modulatory_filters, dtype=np.float32)
        )
        if configured_filters is not None and configured_filters.ndim != 3:
            raise ValueError('modulatory_filters must be a rank-3 array.')
        if configured_filters is not None and mean_activity is not None:
            raise ValueError(
                'mean_activity is irrelevant when modulatory_filters is supplied; '
                'remove one of these options.'
            )
        if filter_taps is None:
            filter_taps = (
                configured_filters.shape[2]
                if configured_filters is not None else 3
            )
        if filter_taps <= 0:
            raise ValueError('filter_taps must be greater than zero.')
        self.filter_taps = int(filter_taps)
        self.mean_activity = (
            None if configured_filters is not None
            else float(1.0 if mean_activity is None else mean_activity)
        )
        self._configured_modulatory_filters = configured_filters
        self.modulatory_filters = None
        self.node_type_values = None
        self._dense_node_type_ids = None
        super().__init__(name='modprop', **kwargs)

    @classmethod
    def module(cls):
        return 'modprop'

    def build(self, rnn):
        super().build(rnn)
        self._build_modulatory_filters()

    def _build_modulatory_filters(self):
        cell = self.rnn._cell
        type_values, dense_type_ids = np.unique(
            np.asarray(cell._node_type_ids), return_inverse=True
        )
        n_types = len(type_values)
        self.node_type_values = tf.Variable(
            type_values.astype(np.int64),
            trainable=False,
            dtype=tf.int64,
            name='modprop_node_type_values',
        )
        self._dense_node_type_ids = tf.constant(dense_type_ids, dtype=tf.int32)

        if self._configured_modulatory_filters is not None:
            filters = self._configured_modulatory_filters
            expected_prefix = (n_types, n_types)
            if filters.ndim != 3 or filters.shape[:2] != expected_prefix:
                raise ValueError(
                    'modulatory_filters must have shape '
                    f'[{n_types}, {n_types}, filter_taps] in sorted node-type order '
                    f'{type_values.tolist()}, got {filters.shape}.'
                )
            if filters.shape[2] != self.filter_taps:
                raise ValueError(
                    f'modulatory_filters provides {filters.shape[2]} taps, but '
                    f'filter_taps={self.filter_taps}.'
                )
        else:
            dense_type_ids_tf = tf.constant(dense_type_ids, dtype=tf.int32)
            type_pair_sums = tf.zeros([n_types * n_types], dtype=tf.float32)
            n_edges = int(cell.recurrent_indices.shape[0])
            for start in range(0, n_edges, self.edge_chunk_size):
                end = min(start + self.edge_chunk_size, n_edges)
                indices = tf.cast(cell.recurrent_indices[start:end], tf.int32)
                post_types = tf.gather(dense_type_ids_tf, indices[:, 0])
                pre_ids = indices[:, 1] % cell._n_neurons
                pre_types = tf.gather(dense_type_ids_tf, pre_ids)
                pair_ids = post_types * n_types + pre_types
                type_pair_sums += tf.math.unsorted_segment_sum(
                    tf.cast(cell.recurrent_weight_values[start:end], tf.float32),
                    pair_ids,
                    n_types * n_types,
                )

            effective_matrix = tf.reshape(
                type_pair_sums, [n_types, n_types]
            ).numpy()
            type_counts = np.bincount(dense_type_ids, minlength=n_types)
            pair_counts = type_counts[:, None] * type_counts[None, :]
            effective_matrix = np.divide(
                effective_matrix,
                pair_counts,
                out=np.zeros_like(effective_matrix),
                where=pair_counts > 0,
            )

            filters = np.empty(
                (n_types, n_types, self.filter_taps), dtype=np.float32
            )
            matrix_power = effective_matrix.copy()
            for tap in range(self.filter_taps):
                filters[:, :, tap] = (
                    self.mean_activity ** tap * matrix_power
                ).astype(np.float32)
                matrix_power = matrix_power @ effective_matrix

        self.modulatory_filters = tf.Variable(
            filters,
            trainable=False,
            dtype=tf.float32,
            name='modprop_filters',
        )

    def _prepare_rule_state(self, observations, pseudo_derivative):
        spike_modulation = (
            tf.cast(observations.spike_learning_signal, tf.float32)
            * pseudo_derivative
        )
        type_membership = tf.one_hot(
            self._dense_node_type_ids,
            depth=tf.shape(self.node_type_values)[0],
            dtype=tf.float32,
        )
        population_modulation = tf.einsum(
            'btn,nc->btc', spike_modulation, type_membership
        )
        return (tf.transpose(population_modulation, [1, 0, 2]),)

    def _initial_chunk_state(self, batch_size, edge_count):
        return (
            tf.zeros(
                [self.filter_taps, batch_size, edge_count], dtype=tf.float32
            ),
        )

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
        del observations
        base_gradient = tf.reduce_mean(
            post_local_factor * membrane_eligibility, axis=0
        )
        population_modulation_by_time, = rule_state
        eligibility_history, = chunk_state

        post_pseudo = tf.gather(
            pseudo_derivative[:, time, :], post_ids, axis=1
        )
        current_eligibility = post_pseudo * membrane_eligibility
        post_types = tf.gather(self._dense_node_type_ids, post_ids)
        edge_filters = tf.gather(
            self.modulatory_filters, post_types, axis=1
        )
        modulation_coefficients = tf.einsum(
            'bc,ces->bes',
            population_modulation_by_time[time],
            edge_filters,
        )
        modprop_gradient = tf.reduce_mean(
            tf.reduce_sum(
                tf.transpose(modulation_coefficients, [2, 0, 1])
                * eligibility_history,
                axis=0,
            ),
            axis=0,
        )
        new_history = tf.concat(
            [current_eligibility[tf.newaxis, ...], eligibility_history[:-1]],
            axis=0,
        )
        return base_gradient + modprop_gradient, (new_history,)

    def get_config(self):
        config = super().get_config()
        if self.modulatory_filters is not None:
            serialized_filters = self.modulatory_filters.numpy().tolist()
            serialized_mean_activity = None
        else:
            serialized_filters = (
                None if self._configured_modulatory_filters is None
                else self._configured_modulatory_filters.tolist()
            )
            serialized_mean_activity = self.mean_activity
        config.update({
            'filter_taps': self.filter_taps,
            'mean_activity': serialized_mean_activity,
            'modulatory_filters': serialized_filters,
        })
        return config
