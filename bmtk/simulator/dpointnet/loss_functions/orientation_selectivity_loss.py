import tensorflow as tf
import os
import numpy as np
import pandas as pd

from . import loss_utils


class OrientationSelectivityLoss:
    def __init__(self, rnn, network=None, osi_cost=1e-5, pre_delay=None, post_delay=None, dtype=tf.float32, 
                 core_mask=None, method="crowd_osi", subtraction_ratio=1.0, layer_info=None,
                 neuropixels_df="Neuropixels_data/v1_OSI_DSI_DF.csv"):
        self._rnn = rnn
        self._network = network
        self._osi_cost = osi_cost
        self._pre_delay = pre_delay
        self._post_delay = post_delay
        self._dtype = dtype
        self._core_mask = core_mask
        self._method = method
        self._subtraction_ratio = subtraction_ratio  # only for crowd_spikes method
        self._tf_pi = tf.constant(np.pi, dtype=dtype)
        self._neuropixels_df = neuropixels_df
        if (self._core_mask is not None) and (self._method == "crowd_spikes" or self._method == "crowd_osi"):
            self.np_core_mask = self._core_mask.numpy()
            core_tuning_angles = network['tuning_angle'][self.np_core_mask]
            self._tuning_angles = tf.constant(core_tuning_angles, dtype=dtype)
        else:
            self._tuning_angles = tf.constant(network['tuning_angle'], dtype=dtype)
        
        if self._method == "neuropixels_fr":
            self._layer_info = layer_info  # needed for neuropixels_fr method
            # the layer_info should be a dictionary that contains
            # the cell id of the corresponding layer.
            # the keys should be something like "EXC_L23" or "PV_L5"   

        elif self._method == "crowd_osi":
            # Get the target OSI
            self._target_osi_dsi = self.get_neuropixels_osi_dsi()
            self._min_rates_threshold = tf.constant(0.0005, dtype=self._dtype)
            # sum the core_mask
            n_nodes = len(self._tuning_angles)
            # self.node_type_ids = tf.zeros(n_nodes, dtype=tf.int32)
            node_type_ids = np.zeros(n_nodes, dtype=np.int32)
            osi_target_values = []
            dsi_target_values = []
            cell_type_count = []
            for node_type_id, (key, value) in enumerate(self._target_osi_dsi.items()):
                node_ids = value['ids']
                osi_target_values.append(value['OSI'])
                dsi_target_values.append(value['DSI'])
                cell_type_count.append(len(node_ids))
                # update the ndoe_type_ids tensor in positions node_ids with the node_type_id
                # self.node_type_ids = tf.tensor_scatter_nd_update(self.node_type_ids, indices=tf.expand_dims(node_ids, axis=1), updates=tf.fill(tf.shape(node_ids), node_type_id))
                node_type_ids[node_ids] = node_type_id

            self.osi_target_values = tf.constant(osi_target_values, dtype=self._dtype)
            self.dsi_target_values = tf.constant(dsi_target_values, dtype=self._dtype)
            self.cell_type_count = tf.constant(cell_type_count, dtype=self._dtype)
            self.node_type_ids = tf.constant(node_type_ids, dtype=tf.int32)
            self._n_node_types = len(self._target_osi_dsi)

    @staticmethod
    def module():
        return 'OrientationSelectivityLoss'
    

    def calculate_delta_angle(self, stim_angle, tuning_angle):
        # angle unit is degrees.
        # this function calculates the difference between stim_angle and tuning_angle,
        # but it is fine to have the opposite direction.
        # so, delta angle is always between -90 and 90.
        # they are both vector, so dimension matche is needed.
        # stim_angle is a length of batch size
        # tuning_angle is a length of n_neurons

        # delta_angle = stim_angle - tuning_angle
        delta_angle = tf.expand_dims(stim_angle, axis=1) - tuning_angle
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)
        # # do it twice to make sure everything is between -90 and 90.
        delta_angle = tf.where(delta_angle > 90, delta_angle - 180, delta_angle)
        delta_angle = tf.where(delta_angle < -90, delta_angle + 180, delta_angle)

        return delta_angle

    def get_neuropixels_osi_dsi(self):
        """
        Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        # neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'
        neuropixels_data_path = self._neuropixels_df
        # if the default one is specified and the file doesn't exist, process the data
        if neuropixels_data_path == "Neuropixels_data/v1_OSI_DSI_DF.csv":
            if not os.path.exists(neuropixels_data_path):
                loss_utils.process_neuropixels_data(path=neuropixels_data_path)
        else:
            print(f"> Using custom neuropixels data file for OSI/DSI loss: {neuropixels_data_path}")
        features_to_load = ['ecephys_unit_id', 'cell_type', 'OSI', 'DSI', "Ave_Rate(Hz)", "max_mean_rate(Hz)"]
        osi_dsi_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
        
        nonresponding = osi_dsi_df["max_mean_rate(Hz)"] < 0.5
        osi_dsi_df.loc[nonresponding, "OSI"] = np.nan
        osi_dsi_df.loc[nonresponding, "DSI"] = np.nan
        osi_dsi_df = osi_dsi_df[osi_dsi_df["Ave_Rate(Hz)"] != 0]
        osi_dsi_df.dropna(inplace=True)
        osi_dsi_df["cell_type"] = osi_dsi_df["cell_type"].apply(loss_utils.neuropixels_cell_type_to_cell_type)
        osi_target = osi_dsi_df.groupby("cell_type")['OSI'].mean()
        dsi_target = osi_dsi_df.groupby("cell_type")['DSI'].mean()

        original_pop_names = loss_utils.get_pop_names(self._network)
        if self._core_mask is not None:
            original_pop_names = original_pop_names[self.np_core_mask] 

        cell_types = np.array([loss_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True) for pop_name in original_pop_names])
        node_ids = np.arange(len(cell_types))
        cell_ids = {key: node_ids[cell_types == key] for key in set(osi_dsi_df['cell_type'])}

        # osi_target = osi_df.groupby("cell_type")['OSI'].mean()
        # osi_target = osi_df.groupby("cell_type")['OSI'].median()
        # osi_df.groupby("cell_type")['OSI'].median()
        # convert to dict
        osi_dsi_exp_dict = {key: {'OSI': val, 'DSI': dsi_target[key], 'ids': cell_ids[key]} for key, val in osi_target.to_dict().items()}

        return osi_dsi_exp_dict

    def crowd_spikes_loss(self, spikes, angle):
        # I need to access the tuning angle. of all the neurons.
        angle = tf.cast(angle, self._dtype)

        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
            
        delta_angle = self.calculate_delta_angle(angle, self._tuning_angles)
        # sum spikes in _z, and multiply with delta_angle.
        mean_spikes = tf.reduce_mean(spikes, axis=[1]) 
        mean_angle = mean_spikes * delta_angle
        # Here, the expected value with random firing to subtract
        # (this prevents the osi loss to drive the firing rates to go to zero.)
        expected_sum_angle = tf.reduce_mean(mean_spikes) * 45
        
        angle_loss = tf.reduce_mean(tf.abs(mean_angle)) - expected_sum_angle * self._subtraction_ratio
        
        return angle_loss * self._osi_cost
    
    def crowd_osi_loss(self, spikes, angle, normalizer=None):  
        # Ensure angle is [batch_size] and cast to correct dtype
        angle = tf.cast(tf.reshape(angle, [-1]), self._dtype)  # [batch_size]
        # Compute delta_angle with broadcasting
        delta_angle = angle[:, tf.newaxis] - self._tuning_angles[tf.newaxis, :]  # [batch_size, n_neurons_core]
        radians_delta_angle = delta_angle * (self._tf_pi / 180)
            
        # Compute rates over time dimension
        rates = tf.reduce_mean(spikes, axis=1)  # [batch_size, n_neurons]
        if self._core_mask is not None:
            rates = tf.boolean_mask(rates, self._core_mask, axis=1)

        if normalizer is not None:
            if self._core_mask is not None:
                normalizer = tf.boolean_mask(normalizer, self._core_mask, axis=0)
            # Use tf.maximum to ensure each element of normalizer does not fall below min_normalizer_value
            normalizer = tf.maximum(normalizer, self._min_rates_threshold)
            rates = rates / normalizer

        # Instead of complex numbers, use cosine and sine separately
        weighted_osi_cos_responses = rates * tf.math.cos(2.0 * radians_delta_angle)
        weighted_dsi_cos_responses = rates * tf.math.cos(radians_delta_angle)

        batch_size = tf.shape(rates)[0]
        # Adjust segment_ids for batch dimension
        batch_offsets = tf.range(batch_size, dtype=self.node_type_ids.dtype) * self._n_node_types  # [batch_size]
        batch_offsets_expanded = batch_offsets[:, tf.newaxis]  # [batch_size, 1]

        segment_ids = self.node_type_ids[tf.newaxis, :]  # [1, n_neurons_core]
        segment_ids = tf.tile(segment_ids, [batch_size, 1])  # [batch_size, n_neurons_core]
        segment_ids = segment_ids + batch_offsets_expanded  # [batch_size, n_neurons_core]

        # Flatten data and segment_ids
        data_flat_rates = tf.reshape(rates, [-1])  # [batch_size * n_neurons_core]
        data_flat_weighted_osi = tf.reshape(weighted_osi_cos_responses, [-1])
        data_flat_weighted_dsi = tf.reshape(weighted_dsi_cos_responses, [-1])
        segment_ids_flat = tf.reshape(segment_ids, [-1])

        num_segments = batch_size * self._n_node_types

        # Compute denominators and numerators
        approximated_denominator = tf.math.unsorted_segment_mean(data_flat_rates, segment_ids_flat, num_segments=num_segments)
        approximated_denominator = tf.reshape(approximated_denominator, [batch_size, self._n_node_types])
        approximated_denominator = tf.maximum(approximated_denominator, self._min_rates_threshold)

        osi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_osi, segment_ids_flat, num_segments=num_segments)
        osi_numerator = tf.reshape(osi_numerator, [batch_size, self._n_node_types])

        dsi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_dsi, segment_ids_flat, num_segments=num_segments)
        dsi_numerator = tf.reshape(dsi_numerator, [batch_size, self._n_node_types])

        # Compute approximations
        osi_approx_type = osi_numerator / approximated_denominator  # [batch_size, n_node_types]
        dsi_approx_type = dsi_numerator / approximated_denominator

        # Average over batch size
        osi_approx_type = tf.reduce_mean(osi_approx_type, axis=0)
        dsi_approx_type = tf.reduce_mean(dsi_approx_type, axis=0)

        # Compute losses
        # osi_target_values = self.osi_target_values[tf.newaxis, :]  # [1, n_node_types]
        # dsi_target_values = self.dsi_target_values[tf.newaxis, :]  # [1, n_node_types]
        osi_loss_type = tf.math.square(osi_approx_type - self.osi_target_values)  # [n_node_types]
        dsi_loss_type = tf.math.square(dsi_approx_type - self.dsi_target_values)
    
        # cell_type_count = self.cell_type_count[tf.newaxis, :]  # [1, n_node_types]
        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * self.cell_type_count)  # [1]
        denominator = tf.reduce_sum(self.cell_type_count)  # Scalar

        # total_loss_per_batch = numerator / denominator  # [batch_size]
        # total_loss = tf.reduce_mean(total_loss_per_batch) * self._osi_cost

        total_loss = (numerator / denominator) * self._osi_cost

        return total_loss

    def __call__(self, spikes, angle, trim, normalizer=None):

        spikes = loss_utils.spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        if self._method == "crowd_osi":
            return self.crowd_osi_loss(spikes, angle, normalizer=normalizer)
        elif self._method == "crowd_spikes":
            return self.crowd_spikes_loss(spikes, angle)
        elif self._method == "neuropixels_fr":
            return self.neuropixels_fr_loss(spikes, angle)
