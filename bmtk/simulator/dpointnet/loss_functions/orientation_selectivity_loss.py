import tensorflow as tf
import os
import numpy as np
import pandas as pd

from . import loss_utils


class OrientationSelectivityLoss:
    def __init__(self, rnn, osi_cost=20.0, pre_delay=None, post_delay=None, dtype=tf.float32,
                 core_mask=None, method="crowd_osi", subtraction_ratio=1.0, layer_info=None,
                 data_dir='GLIF_network', neuropixels_df="Neuropixels_data/v1_OSI_DSI_DF.csv",
                 ema_decay=0.95, use_ema_normalizer=True, annulus_loss_weight=0.0, **kwargs):
        self._rnn = rnn
        self._network = rnn.recurrent_network
        self._osi_cost = osi_cost
        self._annulus_loss_weight = float(annulus_loss_weight or 0.0)
        self._pre_delay = None if pre_delay is None else int(pre_delay)
        self._post_delay = None if post_delay is None else int(post_delay)
        self._dtype = dtype
        self._method = method
        self._subtraction_ratio = subtraction_ratio  # only for crowd_spikes method
        self._tf_pi = tf.constant(np.pi, dtype=dtype)
        self._neuropixels_df = neuropixels_df
        self._data_dir = data_dir
        self._use_ema_normalizer = use_ema_normalizer
        self._ema_decay = tf.constant(ema_decay, dtype=tf.float32)

        # Resolve core mask from an explicit mask or a core_radius (matches reference loss_core_radius).
        self._core_mask = loss_utils.resolve_core_mask(
            self._network, core_mask, kwargs.get('core_radius'), data_dir
        )

        tuning_angles = loss_utils.get_tuning_angles(self._network, data_dir)
        if self._core_mask is not None:
            self.np_core_mask = np.asarray(self._core_mask, dtype=bool)
            self._core_mask = tf.constant(self.np_core_mask, dtype=tf.bool)
            self._tuning_angles = tf.constant(tuning_angles[self.np_core_mask], dtype=dtype)
        else:
            self.np_core_mask = None
            self._tuning_angles = tf.constant(tuning_angles, dtype=dtype)

        if self._method == "neuropixels_fr":
            self._layer_info = layer_info  # needed for neuropixels_fr method
            # the layer_info should be a dictionary that contains
            # the cell id of the corresponding layer.
            # the keys should be something like "EXC_L23" or "PV_L5"   

        elif self._method == "crowd_osi":
            self._min_rates_threshold = tf.constant(0.0005, dtype=self._dtype)
            self._target_osi_dsi = self.get_neuropixels_osi_dsi(self.np_core_mask)
            self._assign_crowd_osi_attributes(self._target_osi_dsi, attr_prefix='')
            self._annulus_crowd_osi = None
            if self._annulus_loss_weight > 0.0 and self.np_core_mask is not None:
                np_annulus_mask = ~self.np_core_mask
                annulus_target_osi_dsi = self.get_neuropixels_osi_dsi(np_annulus_mask)
                self._annulus_crowd_osi = self._make_crowd_osi_selection(
                    mask=tf.constant(np_annulus_mask, dtype=tf.bool),
                    tuning_angles=tf.constant(tuning_angles[np_annulus_mask], dtype=dtype),
                    target_osi_dsi=annulus_target_osi_dsi,
                    osi_cost=self._annulus_loss_weight * self._osi_cost,
                )

    @staticmethod
    def module():
        return 'OrientationSelectivityLoss'

    @property
    def uses_ema_normalizer(self):
        return self._use_ema_normalizer and self._method == "crowd_osi"

    def update_normalizers(self, spikes, normalizers, trim=True):
        if not self.uses_ema_normalizer:
            return

        spikes = loss_utils.spike_trimming(
            spikes,
            pre_delay=self._pre_delay,
            post_delay=self._post_delay,
            trim=trim,
        )
        evoked_rates = tf.cast(tf.reduce_mean(spikes, axis=[0, 1]), tf.float32)
        v1_ema = normalizers['v1_ema']
        v1_ema.assign(self._ema_decay * v1_ema + (1.0 - self._ema_decay) * evoked_rates)
    

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

    def get_neuropixels_osi_dsi(self, selection_mask=None):
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

        original_pop_names = loss_utils.get_pop_names(self._network, data_dir=self._data_dir)
        if selection_mask is not None:
            original_pop_names = original_pop_names[np.asarray(selection_mask, dtype=bool)] 

        cell_types = np.array([loss_utils.pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True) for pop_name in original_pop_names])
        node_ids = np.arange(len(cell_types))
        cell_ids = {key: node_ids[cell_types == key] for key in set(osi_dsi_df['cell_type'])}

        # osi_target = osi_df.groupby("cell_type")['OSI'].mean()
        # osi_target = osi_df.groupby("cell_type")['OSI'].median()
        # osi_df.groupby("cell_type")['OSI'].median()
        # convert to dict
        osi_dsi_exp_dict = {key: {'OSI': val, 'DSI': dsi_target[key], 'ids': cell_ids[key]} for key, val in osi_target.to_dict().items()}

        return osi_dsi_exp_dict

    def _make_crowd_osi_selection(self, mask, tuning_angles, target_osi_dsi, osi_cost):
        n_nodes = int(tuning_angles.shape[0])
        node_type_ids = np.zeros(n_nodes, dtype=np.int32)
        osi_target_values = []
        dsi_target_values = []
        cell_type_count = []
        for node_type_id, (_key, value) in enumerate(target_osi_dsi.items()):
            node_ids = value['ids']
            osi_target_values.append(value['OSI'])
            dsi_target_values.append(value['DSI'])
            cell_type_count.append(len(node_ids))
            node_type_ids[node_ids] = node_type_id

        return {
            'mask': mask,
            'tuning_angles': tuning_angles,
            'node_type_ids': tf.constant(node_type_ids, dtype=tf.int32),
            'n_node_types': len(target_osi_dsi),
            'osi_target_values': tf.constant(osi_target_values, dtype=self._dtype),
            'dsi_target_values': tf.constant(dsi_target_values, dtype=self._dtype),
            'cell_type_count': tf.constant(cell_type_count, dtype=self._dtype),
            'osi_cost': tf.constant(osi_cost, dtype=self._dtype),
        }

    def _assign_crowd_osi_attributes(self, target_osi_dsi, attr_prefix):
        selection = self._make_crowd_osi_selection(
            mask=self._core_mask,
            tuning_angles=self._tuning_angles,
            target_osi_dsi=target_osi_dsi,
            osi_cost=self._osi_cost,
        )
        setattr(self, f'{attr_prefix}node_type_ids', selection['node_type_ids'])
        setattr(self, f'{attr_prefix}_n_node_types', selection['n_node_types'])
        setattr(self, f'{attr_prefix}osi_target_values', selection['osi_target_values'])
        setattr(self, f'{attr_prefix}dsi_target_values', selection['dsi_target_values'])
        setattr(self, f'{attr_prefix}cell_type_count', selection['cell_type_count'])

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
    
    @tf.function(jit_compile=False)
    def _compute_osi_dsi_core(self, rates, radians_delta_angle, batch_size, node_type_ids, n_node_types):
        """Core of crowd_osi: cos weighting + per-(batch,type) segment means.

        """
        weighted_osi_cos_responses = rates * tf.math.cos(2.0 * radians_delta_angle)
        weighted_dsi_cos_responses = rates * tf.math.cos(radians_delta_angle)

        batch_offsets = tf.range(batch_size, dtype=node_type_ids.dtype) * n_node_types
        segment_ids = node_type_ids[tf.newaxis, :] + batch_offsets[:, tf.newaxis]

        data_flat_rates = tf.reshape(rates, [-1])
        data_flat_weighted_osi = tf.reshape(weighted_osi_cos_responses, [-1])
        data_flat_weighted_dsi = tf.reshape(weighted_dsi_cos_responses, [-1])
        segment_ids_flat = tf.reshape(segment_ids, [-1])

        num_segments = batch_size * n_node_types

        approximated_denominator = tf.math.unsorted_segment_mean(data_flat_rates, segment_ids_flat, num_segments=num_segments)
        approximated_denominator = tf.reshape(approximated_denominator, [batch_size, n_node_types])
        approximated_denominator = tf.maximum(approximated_denominator, self._min_rates_threshold)

        osi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_osi, segment_ids_flat, num_segments=num_segments)
        osi_numerator = tf.reshape(osi_numerator, [batch_size, n_node_types])

        dsi_numerator = tf.math.unsorted_segment_mean(data_flat_weighted_dsi, segment_ids_flat, num_segments=num_segments)
        dsi_numerator = tf.reshape(dsi_numerator, [batch_size, n_node_types])

        osi_approx_type = tf.reduce_mean(osi_numerator / approximated_denominator, axis=0)
        dsi_approx_type = tf.reduce_mean(dsi_numerator / approximated_denominator, axis=0)
        return osi_approx_type, dsi_approx_type

    def _crowd_osi_loss_for_selection(self, spikes, angle, normalizer, batch_size_hint, selection):
        delta_angle = angle[:, tf.newaxis] - selection['tuning_angles'][tf.newaxis, :]
        radians_delta_angle = delta_angle * (self._tf_pi / 180)

        rates = tf.reduce_mean(spikes, axis=1)
        rates = tf.cast(rates, self._dtype)
        if selection['mask'] is not None:
            rates = tf.boolean_mask(rates, selection['mask'], axis=1)
            rates.set_shape([None, selection['node_type_ids'].shape[0]])

        if normalizer is not None:
            if selection['mask'] is not None:
                normalizer = tf.boolean_mask(normalizer, selection['mask'], axis=0)
            normalizer = tf.maximum(normalizer, self._min_rates_threshold)
            rates = rates / normalizer

        batch_size = batch_size_hint if batch_size_hint is not None else tf.shape(rates)[0]
        osi_approx_type, dsi_approx_type = self._compute_osi_dsi_core(
            rates,
            radians_delta_angle,
            batch_size,
            selection['node_type_ids'],
            selection['n_node_types'],
        )

        osi_loss_type = tf.math.square(osi_approx_type - selection['osi_target_values'])
        dsi_loss_type = tf.math.square(dsi_approx_type - selection['dsi_target_values'])
        numerator = tf.reduce_sum((osi_loss_type + dsi_loss_type) * selection['cell_type_count'])
        denominator = tf.reduce_sum(selection['cell_type_count'])
        return (numerator / denominator) * selection['osi_cost']

    def crowd_osi_loss(self, spikes, angle, normalizer=None, batch_size_hint=None):
        # Ensure angle is [batch_size] and cast to correct dtype
        angle = tf.cast(tf.reshape(angle, [-1]), self._dtype)  # [batch_size]
        loss = self._crowd_osi_loss_for_selection(
            spikes,
            angle,
            normalizer,
            batch_size_hint,
            {
                'mask': self._core_mask,
                'tuning_angles': self._tuning_angles,
                'node_type_ids': self.node_type_ids,
                'n_node_types': self._n_node_types,
                'osi_target_values': self.osi_target_values,
                'dsi_target_values': self.dsi_target_values,
                'cell_type_count': self.cell_type_count,
                'osi_cost': tf.constant(self._osi_cost, dtype=self._dtype),
            },
        )
        if self._annulus_crowd_osi is not None:
            loss += self._crowd_osi_loss_for_selection(
                spikes,
                angle,
                normalizer,
                batch_size_hint,
                self._annulus_crowd_osi,
            )
        return loss

    @staticmethod
    def _extract_orientation(y):
        """Pull the per-sample stimulus orientation tensor out of the dpointnet `y` signature.

        `y` is the per-parameter input signature: a list of per-input-population dicts (series),
        a single dict, or an InputsSignature wrapping `ysigs`. The drifting-gratings LGN
        generator emits an 'orientation' entry; spontaneous inputs do not.
        """
        if y is None:
            return None
        if hasattr(y, 'ysigs'):
            candidates = list(y.ysigs)
        elif isinstance(y, dict):
            candidates = [y]
        elif isinstance(y, (list, tuple)):
            candidates = list(y)
        else:
            return None
        for sig in candidates:
            if isinstance(sig, dict) and 'orientation' in sig:
                return sig['orientation']
        return None

    def __call__(self, spikes, voltages=None, model_state=None, y=None, trim=True, **kwargs):
        # OSI/DSI is an evoked-only loss; it needs the drifting-gratings orientation from `y`.
        # If absent (e.g. a spontaneous parameter set), contribute nothing.
        angle = self._extract_orientation(y)
        if angle is None:
            return tf.constant(0.0, dtype=tf.float32)

        # Keep spikes in the compute dtype (fp16 under mixed precision); only the small reduced
        # rate tensors are upcast to fp32 below. Casting the full [batch, seq, n_neurons] spikes
        # tensor here previously allocated ~1 GiB of fp32 transients per call.
        spikes = loss_utils.spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        normalizer = kwargs.get('normalizer')
        if normalizer is None and self._use_ema_normalizer:
            normalizers = kwargs.get('normalizers')
            if normalizers is not None:
                normalizer = normalizers.get('v1_ema')
        if normalizer is not None:
            normalizer = tf.cast(normalizer, self._dtype)

        if self._method == "crowd_osi":
            loss = self.crowd_osi_loss(
                spikes,
                angle,
                normalizer=normalizer,
                batch_size_hint=kwargs.get('batch_size_hint'),
            )
        elif self._method == "crowd_spikes":
            loss = self.crowd_spikes_loss(spikes, angle)
        elif self._method == "neuropixels_fr":
            loss = self.neuropixels_fr_loss(spikes, angle)
        else:
            raise ValueError(f"Unknown OSI method '{self._method}'.")

        return tf.cast(loss, tf.float32)
