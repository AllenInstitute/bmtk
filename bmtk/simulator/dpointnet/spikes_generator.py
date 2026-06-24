import tensorflow as tf
from glob import glob
from scipy import sparse
import numpy as np
import logging


from bmtk.simulator.dpointnet.network_adaptor import NetworkAdaptor
from bmtk.utils.reports.spike_trains import SpikeTrains


logger = logging.getLogger(__name__)


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


def sonata_spike_files_generator(
        network, 
        spikes_paths,
        seq_len,
        dt=1.0,
        dtype=tf.float32,
        cache_spikes_table=False,
        background_noise_rate=0.0,
        dropout_prob=0.0,
        jitter_var=0.0,

    ):
    # dt = 1.0
    # seq_len = 500
    # dtype = tf.float32

    # _, inputs = NetworkAdaptor.from_dict({
    #     "networks": {
    #         "nodes": [
    #         {
    #             "nodes_file": "GLIF_network/network/lgn_nodes.h5",
    #             "node_types_file": "GLIF_network/network/lgn_node_types.csv"
    #         }
    #         ]
    #     }
    # })
    # lng_net = inputs[0]
    assert(isinstance(network, NetworkAdaptor))

    nodes_pop = network.population_name
    n_nodes = network.n_nodes

    # spike_train_paths = 'pregen_spikes/lgn_spikes*.h5'
    spikes_paths = [spikes_paths] if isinstance(spikes_paths, str) else spikes_paths

    combined_spike_files = []
    for path_exp in spikes_paths:
        spike_files_list = list(glob(path_exp))
        if len(spike_files_list) == 0:
            # logger.error(f'Could not find spike files {spike_paths}')
            raise ValueError(f'Could not find spike files {path_exp}')
        combined_spike_files += spike_files_list

    spikes_table_cache = {}
    def spike_trains_generator():
        while True:
            selected_spike_train = np.random.choice(combined_spike_files)
            if cache_spikes_table and selected_spike_train in spikes_table_cache:
                spikes_table = spikes_table_cache[selected_spike_train]
            
            else:
                spikes = SpikeTrains.load(
                    selected_spike_train,
                )
                # node_ids = spikes.node_ids()
                if nodes_pop in spikes.populations:
                    spikes_file_pop = nodes_pop
                elif len(spikes.populations) == 1:
                    spikes_file_pop = spikes.populations[0]
                    logger.debug(f'Unable to find population {nodes_pop} in {selected_spike_train}. Defaulting to using {spikes_file_pop} population.')
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
                    shape=(seq_len, n_nodes),
                ).tocsr()

                spikes_table = spikes_table.toarray()
                if cache_spikes_table:
                    spikes_table_cache[selected_spike_train] = spikes_table

            if background_noise_rate > 0.0:
                spikes_table = add_background_noise(spikes_table, background_noise_rate, dt)

            if jitter_var > 0.0:
                spikes_table = add_jitter(spikes_table, jitter_var, dt)

            if dropout_prob > 0.0:
                spikes_table = add_dropout(spikes_table, dropout_prob)

            yield spikes_table, {'file_path': selected_spike_train}

    data_set = tf.data.Dataset.from_generator(
        spike_trains_generator, 
        output_signature=(
            tf.TensorSpec(shape=(seq_len, n_nodes), dtype=dtype),
            {
                'file_path': tf.TensorSpec(shape=(), dtype=tf.string),
            }
        )
    )

    return data_set


def create_random_spikes_generator(
        network,
        firing_rate,
        seq_len,
        dt=1.0,
        dtype=tf.float32,
    ):
    n_nodes = network.n_nodes
    lam = firing_rate*dt/1000.0
    
    def _g():
        while True:
            spikes = np.random.rand(seq_len, n_nodes) < lam # .astype(np.bool)
            yield spikes, {'firing_rate': firing_rate}

    data_set = tf.data.Dataset.from_generator(
        _g, 
        output_signature=(
            tf.TensorSpec(shape=(seq_len, n_nodes), dtype=dtype),
            {
                'firing_rate': tf.TensorSpec(shape=(), dtype=tf.float32),
            }
        )
    )

    return data_set


