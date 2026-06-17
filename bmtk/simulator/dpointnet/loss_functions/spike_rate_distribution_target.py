import os
import tensorflow as tf
import numpy as np
import pandas as pd

from . import loss_utils


class SpikeRateDistributionTarget:
    def __init__(
            self, 
            rnn, 
            stimulus_type='drifting_gratings', 
            rate_cost=.5, 
            pre_delay=None, 
            post_delay=None,
            data_dir='GLIF_network', 
            core_mask=None, 
            rates_dampening=1.0, 
            seed=42, 
            dtype=tf.float32,
            neuropixels_df=None, # 'Neuropixels_data/v1_OSI_DSI_DF.csv', 
            annulus_loss_weight=0.0,
            **kwargs
        ):
        self._rnn = rnn
        self._network = rnn.recurrent_network
        self._rate_cost = rate_cost
        self._annulus_loss_weight = float(annulus_loss_weight or 0.0)
        self._pre_delay = int(pre_delay)
        self._post_delay = int(post_delay)
        self._rates_dampening = rates_dampening
        self._data_dir = data_dir
        self._dtype = dtype
        self._seed = seed
        self._neuropixels_df = neuropixels_df

        # Restrict the rate-matching loss to a central core (matches the reference
        # V1_GLIF_model's loss_core_radius). With no core, the loss averages over all
        # ~66k neurons, which dilutes the per-neuron (and per-weight) gradient ~4x.
        self._core_mask = loss_utils.resolve_core_mask(
            self._network, core_mask, kwargs.get('core_radius'), self._data_dir
        )

        # Mapping of stimulus type to neuropixels feature
        # If deprecated arguments are used, map them to stimulus_type
        if kwargs.get('spontaneous_fr'):
            stimulus_type = 'spontaneous'
        elif kwargs.get('natural_images'):
            stimulus_type = 'natural_stimuli'

        if stimulus_type in ['spontaneous', 'gray']:
            self.neuropixels_feature = 'firing_rate_sp'
        elif stimulus_type in ['natural_stimuli', 'natural_images']:
            self.neuropixels_feature = 'firing_rate_ns'
        elif stimulus_type == 'drifting_gratings':
            self.neuropixels_feature = 'Ave_Rate(Hz)'
        else:
            raise ValueError(f"Unknown stimulus_type: {stimulus_type}. Choose among 'spontaneous/gray', 'drifting_gratings', or 'natural_stimuli'.")

        self._target_rates = self.get_neuropixels_firing_rates(self._core_mask)
        self._annulus_target_rates = None
        if self._annulus_loss_weight > 0.0 and self._core_mask is not None:
            annulus_mask = ~np.asarray(self._core_mask, dtype=bool)
            self._annulus_target_rates = self.get_neuropixels_firing_rates(annulus_mask)

    @staticmethod
    def module():
        return 'SpikeRateDistributionTarget'

    def get_neuropixels_firing_rates(self, core_mask=None):
        """Processes neuropixels data to obtain neurons average firing rates.

        Returns:
            dict: Dictionary containing rates and node_type_ids for each population query.
        """
        # Load data
        # neuropixels_data_path = f'Neuropixels_data/v1_OSI_DSI_DF.csv'

        neuropixels_data_path = self._neuropixels_df
        if neuropixels_data_path == 'Neuropixels_data/v1_OSI_DSI_DF.csv':
            if not os.path.exists(neuropixels_data_path):
                loss_utils.process_neuropixels_data(path=neuropixels_data_path)
        else: # just inform the user that the custom file is loading.
            print(f"Using custom neuropixels data file for FR loss: {neuropixels_data_path}")

        # New dataset has Spont_Rate(Hz) instead of firing_rate_sp.
        # if reading firing_rate_sp fails, replace it with Spont_Rate(Hz) and try again.
        features_to_load = ['ecephys_unit_id', 'cell_type', 'firing_rate_sp', 'Ave_Rate(Hz)']
        try:
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
        except ValueError:
            print(f"Neuropixels data file {neuropixels_data_path} does not contain firing_rate_sp. Using Spont_Rate(Hz) instead.")
            features_to_load = ['ecephys_unit_id', 'cell_type', 'Spont_Rate(Hz)', 'Ave_Rate(Hz)']
            np_df = pd.read_csv(neuropixels_data_path, index_col=0, sep=" ", usecols=features_to_load).dropna(how='all')
            # Rename the column to match the original
            np_df.rename(columns={'Spont_Rate(Hz)': 'firing_rate_sp'}, inplace=True)
        # Ensure they use the new names
        np_df["cell_type"] = np_df["cell_type"].apply(loss_utils.neuropixels_cell_type_to_cell_type)
        type_rates_dict = {
            cell_type: np.append(subdf[self.neuropixels_feature].dropna().values / 1000, 0)
            for cell_type, subdf in np_df.groupby("cell_type")
        }
        population_ids = loss_utils.get_population_neuron_ids(
            self._network, data_dir=self._data_dir, core_mask=core_mask
        )

        target_firing_rates = {}
        for cell_type in loss_utils.CELL_TYPE_ORDER:
            rates = type_rates_dict.get(cell_type, np.array([0.0], dtype=np.float32))
            neuron_ids = population_ids[cell_type]
            type_n_neurons = len(neuron_ids)
            target_firing_rates[cell_type] = {
                "rates": rates,
                "neuron_ids": tf.convert_to_tensor(neuron_ids, dtype=tf.int32),
                "sorted_target_rates": tf.convert_to_tensor(
                    self._rates_dampening
                    * loss_utils.sample_firing_rates(rates, type_n_neurons, self._seed),
                    dtype=self._dtype,
                ),
            }

        return target_firing_rates

    def __call__(self, spikes, trim=True, **kwargs):
        # if trim:
        #     if self._pre_delay is not None:
        #         spikes = spikes[:, self._pre_delay:, :]
        #     if self._post_delay is not None and self._post_delay != 0:
        #         spikes = spikes[:, :-self._post_delay, :]

        spikes = loss_utils.spike_trimming(spikes, pre_delay=self._pre_delay, post_delay=self._post_delay, trim=trim)

        if spikes.dtype != self._dtype:
            spikes = tf.cast(spikes, self._dtype)

        rates = tf.reduce_mean(spikes, (0, 1)) # calculate the mean firing rate over time and batch

        reg_loss = loss_utils.compute_spike_rate_target_loss(rates, self._target_rates, dtype=self._dtype)
        if self._annulus_target_rates is not None:
            annulus_loss = loss_utils.compute_spike_rate_target_loss(
                rates,
                self._annulus_target_rates,
                dtype=self._dtype,
            )
            reg_loss += self._annulus_loss_weight * annulus_loss

        return reg_loss * self._rate_cost
    