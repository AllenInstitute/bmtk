import tensorflow as tf
import os
import numpy as np

from . import loss_utils


class SynchronizationLoss(tf.keras.layers.Layer):
    def __init__(self, rnn, sync_cost=10, t_start=0.0, t_end=0.5, n_samples=50,
                 neuropixels_data_dir='Synchronization_data',
                 data_dir='GLIF_network',
                 session=None, dtype=tf.float32, core_mask=None, seed=42,
                 stimulus_type='drifting_gratings', **kwargs):
        super(SynchronizationLoss, self).__init__(dtype=dtype)
        self._rnn = rnn
        self._network = rnn.recurrent_network
        self._sync_cost = sync_cost
        self._t_start = t_start
        self._t_end = t_end
        self._t_start_seconds = int(t_start * 1000)
        self._t_end_seconds = int(t_end * 1000)
        self._data_dir = data_dir
        # Resolve core mask from an explicit mask or a core_radius (matches reference loss_core_radius).
        self._core_mask = loss_utils.resolve_core_mask(
            self._network, core_mask, kwargs.get('core_radius'), data_dir
        )
        self._neuropixels_data_dir = neuropixels_data_dir
        self._dtype = dtype
        self._n_samples = n_samples
        self._base_seed = seed
        if session is None:
            if stimulus_type in ['spontaneous', 'gray']:
                session = 'spont'
            elif stimulus_type == 'drifting_gratings':
                session = 'evoked'
            else:
                raise ValueError(
                    f"Unknown stimulus_type: {stimulus_type}. Choose among "
                    "'spontaneous', 'gray', or 'drifting_gratings'."
                )

        pop_names = loss_utils.get_pop_names(self._network)
        if self._core_mask is not None:
            pop_names = pop_names[self._core_mask]
        node_ei = np.array([pop_name[0] for pop_name in pop_names])
        node_id = np.arange(len(node_ei))
        
        # Get the IDs for excitatory neurons
        node_id_e = node_id[node_ei == 'e']
        self.node_id_e = tf.constant(node_id_e, dtype=tf.int32) # 14423
        
        # Pre-define bin sizes (same as experimental data)
        bin_sizes = np.logspace(-3, 0, 20)
        
        # using the simulation length, limit bin_sizes to define at least 2 bins
        bin_sizes_mask = bin_sizes < (self._t_end - self._t_start)/2
        bin_sizes = bin_sizes[bin_sizes_mask]
        self._bin_sizes_ms = tuple(max(1, int(round(v * 1000.0))) for v in bin_sizes)
        self._bin_sizes_ms_tf = tf.constant(self._bin_sizes_ms, dtype=tf.int32)
        self._epsilon_tf = tf.constant(1e-7, dtype=self._dtype)

        # Load the experimental data
        duration = str(int((t_end - t_start) * 1000))
        experimental_data_path = os.path.join(self._neuropixels_data_dir, f'Fano_factor_v1', f'v1_fano_running_{duration}ms_{session}.npy')
        
        # experimental_data_path = os.path.join(data_dir, f'all_fano_300ms_{session}.npy')
        assert os.path.exists(experimental_data_path), f'File not found: {experimental_data_path}'
        experimental_fanos = np.load(experimental_data_path, allow_pickle=True)
        experimental_fanos_mean = np.nanmean(experimental_fanos[:, bin_sizes_mask], axis=0)
        self.experimental_fanos_mean = tf.constant(experimental_fanos_mean, dtype=self._dtype)
    
    @staticmethod
    def module():
        return 'SynchronizationLoss'
    
    def pop_fano_tf(self, spikes):
        spikes = tf.expand_dims(spikes, axis=-1)
        fanos = tf.TensorArray(dtype=self._dtype, size=len(self._bin_sizes_ms))
        for i, bin_size in enumerate(self._bin_sizes_ms):
            # Use convolution for efficient binning
            kernel = tf.ones((bin_size, 1, 1), dtype=self._dtype)
            convolved = tf.nn.conv1d(spikes, kernel, stride=bin_size, padding='VALID')
            sp_counts = tf.squeeze(convolved, axis=-1)

            # Compute mean and variance of spike counts
            mean_count = tf.reduce_mean(sp_counts, axis=1)
            var_count = tf.math.reduce_variance(sp_counts, axis=1)
            mean_count = tf.maximum(mean_count, self._epsilon_tf)

            fano_per_sample = var_count / mean_count
            fano = tf.reduce_mean(fano_per_sample)
            fanos = fanos.write(i, fano)

        return fanos.stack()
    
    def __call__(self, spikes, trim=True, **kwargs):
        if self._core_mask is not None:
            spikes = tf.boolean_mask(spikes, self._core_mask, axis=2)
        
        if trim:
            spikes = spikes[:, self._t_start_seconds:self._t_end_seconds, :]
        duration_ms = tf.cast(tf.shape(spikes)[1], tf.int32)
        bin_limit_ms = duration_ms // 2
        bin_sizes_mask = self._bin_sizes_ms_tf < bin_limit_ms
        experimental_fanos_mean = tf.boolean_mask(self.experimental_fanos_mean, bin_sizes_mask)
        
        spikes = tf.cast(spikes, self._dtype)

        # Choose random trials to sample from (usually only have 1 trial to sample from)
        n_trials = tf.shape(spikes)[0]

        # increase the base seed to avoid the same random neurons to be selected in every instantation of the class
        self._base_seed += 1
        sample_trials = tf.random.uniform([self._n_samples], minval=0, maxval=n_trials, dtype=tf.int32, seed=self._base_seed)

        # Gernate sample counts with a normal distribution
        sample_size = 70
        sample_std = 30
        sample_counts = tf.cast(tf.random.normal([self._n_samples], mean=sample_size, stddev=sample_std, seed=self._base_seed), tf.int32)
        sample_counts = tf.clip_by_value(sample_counts, clip_value_min=15, clip_value_max=tf.shape(self.node_id_e)[0])

        # Randomize the neuron ids
        shuffled_e_ids = tf.random.shuffle(self.node_id_e, seed=self._base_seed)
        selected_spikes_sample = tf.TensorArray(self._dtype, size=self._n_samples)
        previous_id = tf.constant(0, dtype=tf.int32)
        for i in tf.range(self._n_samples):
            sample_num = sample_counts[i]
            sample_trial = sample_trials[i]

            # Randomly choose sample_num ids from shuffled_ids without replacement
            if previous_id + sample_num > tf.size(shuffled_e_ids):
                shuffled_e_ids = tf.random.shuffle(shuffled_e_ids, seed=self._base_seed)
                previous_id = tf.constant(0, dtype=tf.int32)

            sample_ids = shuffled_e_ids[previous_id:previous_id+sample_num]
            previous_id += sample_num

            selected_spikes = tf.reduce_sum(tf.gather(spikes[sample_trial], sample_ids, axis=1), axis=-1)
            selected_spikes_sample = selected_spikes_sample.write(i, selected_spikes)

        selected_spikes_sample = selected_spikes_sample.stack()
        if selected_spikes_sample.dtype != self._dtype:
            selected_spikes_sample = tf.cast(selected_spikes_sample, self._dtype)

        fanos_mean = self.pop_fano_tf(selected_spikes_sample)
        fanos_mean = tf.boolean_mask(fanos_mean, bin_sizes_mask)

        # Calculate MSE between experimental and calculated Fano Factors
        mse_loss = tf.cond(
            tf.size(experimental_fanos_mean) > 0,
            lambda: tf.reduce_mean(tf.square(experimental_fanos_mean - fanos_mean)),
            lambda: tf.constant(0.0, dtype=self._dtype),
        )

        # Calculate the synchronization loss
        sync_loss = self._sync_cost * mse_loss

        return sync_loss
