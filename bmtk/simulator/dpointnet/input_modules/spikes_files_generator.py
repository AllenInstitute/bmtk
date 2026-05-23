import numpy as np
from glob import glob
from scipy import sparse
import tensorflow as tf

from bmtk.utils.reports.spike_trains import SpikeTrains
from bmtk.simulator.dpointnet.io_tools import io
from .inputs_base import InputsGeneratorMod


def add_background_noise(spikes_table, noise_rate, dt):
    if noise_rate == 0.0:
        return spikes_table
    
    step_prob = noise_rate*dt  # equals the prob. of noise occuring at each step
    noise = np.random.rand(*spikes_table) < step_prob
    return spikes_table | noise


def add_jitter(spikes_table, jitter_var, dt):
    jitter_var_step = jitter_var / dt # if jitter_var is is ms, then this calculates the vars in terms of steps

    # Each row represents where spikes occur for each neuron. To apply jitter get each rows where spike occurs, 
    # apply noise to each noise which will either move the row location ahead, behind, or stay the same. Make sure
    # that updated rows don't go over the time window
    rows, cols = np.where(spikes_table)
    row_shift = np.round(
        np.random.normal(0.0, jitter_var_step, size=len(rows))
    ).astype(int)
    updated_rows = rows + row_shift
    valid_indices = (updated_rows >= 0) & (updated_rows < spikes_table.shape[0])

    updated_spikes_table = np.zeros_like(spikes_table)
    updated_spikes_table[updated_rows[valid_indices], cols[valid_indices]] = True
    return updated_spikes_table


def add_dropout(spikes_table, dropout_prob):
    keep = np.random.rand(*spikes_table) >= dropout_prob
    return spikes_table & keep


class SpikesFilesGenerator(InputsGeneratorMod):
    def __init__(self, rnn, name, input_network, spikes_paths, cache_spikes=False, 
                 background_noise_rate=0.0, dropout_prob=0.0, jitter_var=0.0, 
                 **kwargs
        ):
        super().__init__(rnn=rnn, name=name, input_network=input_network, **kwargs)
        self.nodes_pop = self.input_network.population_name
        self.n_nodes = self.input_network.n_nodes

        self.background_noise_rate = background_noise_rate
        self.dropout_prob = dropout_prob
        self.jitter_var = jitter_var

        # Create a list of all spike files, potentially combining multiple directories and files
        spikes_paths = [spikes_paths] if isinstance(spikes_paths, str) else spikes_paths
        self.combined_spike_files = []
        for path_exp in spikes_paths:
            spike_files_list = list(glob(path_exp))
            if len(spike_files_list) == 0:
                raise ValueError(f'Could not find spike files {path_exp}')
            self.combined_spike_files += spike_files_list

        self.cache_spikes = cache_spikes
        self.spikes_table_cache = {} 

    @staticmethod
    def module():
        return 'spikes_files'
    
    @staticmethod
    def input_type():
        return 'spikes'
    
    def create_generator(self, seq_len, dtype=tf.float32, dt=1.0, **kwargs):
        def _generator():
            while True:
                selected_spike_train = np.random.choice(self.combined_spike_files)
                if self.cache_spikes and selected_spike_train in self.spikes_table_cache:
                    spikes_table = self.spikes_table_cache[selected_spike_train]
                
                else:
                    spikes = SpikeTrains.load(
                        selected_spike_train,
                    )
                    if self.nodes_pop in spikes.populations:
                        spikes_file_pop = self.nodes_pop
                    elif len(spikes.populations) == 1:
                        spikes_file_pop = spikes.populations[0]
                        io.log_debug(f'Unable to find population {self.nodes_pop} in {selected_spike_train}. Defaulting to using {spikes_file_pop} population.')
                    else:
                        raise ValueError(f'Unable to find appropiate spike-population in {selected_spike_train}.')
                        # logger.error(f'Unable to find appropiate spike-population in {selected_spike_train}.')
                    
                    spikes_df = spikes.to_dataframe(populations=spikes_file_pop)
                    node_ids = spikes_df['node_ids'].values
                    timestamps = spikes_df['timestamps'].values
                    
                    ids_sorted = np.all(np.diff(node_ids) >= 0)
                    if not ids_sorted:
                        sort_idx = np.argsort(node_ids)
                        node_ids = node_ids[sort_idx]
                        timestamps = timestamps[sort_idx]
                            
                    step_ids = np.floor(timestamps / dt).astype(int)
                    data = np.ones(len(step_ids), dtype=bool)

                    spikes_table = sparse.coo_matrix(
                        (data, (step_ids, node_ids)),
                        shape=(seq_len, self.n_nodes),
                    ).tocsr()

                    spikes_table = spikes_table.toarray()
                    if self.cache_spikes:
                        self.spikes_table_cache[selected_spike_train] = spikes_table

                if self.background_noise_rate > 0.0:
                    spikes_table = add_background_noise(spikes_table, self.background_noise_rate, dt)

                if self.jitter_var > 0.0:
                    spikes_table = add_jitter(spikes_table, self.jitter_var, dt)

                if self.dropout_prob > 0.0:
                    spikes_table = add_dropout(spikes_table, self.dropout_prob)

                yield spikes_table, {'file_path': selected_spike_train}

        data_set = tf.data.Dataset.from_generator(
            _generator, 
            output_signature=(
                tf.TensorSpec(shape=(seq_len, self.n_nodes), dtype=dtype),
                {
                    'file_path': tf.TensorSpec(shape=(), dtype=tf.string),
                }
            )
        )

        return data_set
