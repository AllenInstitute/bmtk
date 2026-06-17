import tensorflow as tf
import os
import pandas as pd
import numpy as np
import h5py



CELL_TYPE_QUERY_MAPPING = {
    "i1H": "L1 Htr3a",
    "e23": "L2/3 Exc",
    "i23P": "L2/3 PV",
    "i23S": "L2/3 SST",
    "i23V": "L2/3 VIP",
    "e4": "L4 Exc",
    "i4P": "L4 PV",
    "i4S": "L4 SST",
    "i4V": "L4 VIP",
    "e5": "L5 Exc",
    "i5P": "L5 PV",
    "i5S": "L5 SST",
    "i5V": "L5 VIP",
    "e6": "L6 Exc",
    "i6P": "L6 PV",
    "i6S": "L6 SST",
    "i6V": "L6 VIP",
}

CELL_TYPE_ORDER = tuple(CELL_TYPE_QUERY_MAPPING.values())


def spike_trimming(spikes, pre_delay=50, post_delay=50, trim=True):
    pre = pre_delay or 0
    if trim:
        post = -post_delay if post_delay else None
        spikes = spikes[:, pre:post, :]
    else:
        spikes = spikes[:, pre:, :]
    return spikes



def compute_spike_rate_target_loss(rates, target_rates, dtype=tf.float32):
    # TODO: define this function
    # target_rates is a dictionary that contains all the cell types.
    # I should iterate on them, and add the cost for each one at the end.
    # spikes will have a shape of (batch_size, n_steps, n_neurons)
    # rates = tf.reduce_mean(_spikes, (0, 1))
    total_loss = tf.constant(0.0, dtype=dtype)
    num_neurons = tf.constant(0, dtype=tf.int32)
    # if core_mask is not None:
    #     core_neurons_ids = np.where(core_mask)[0]

    for key, value in target_rates.items():
        neuron_ids = value["neuron_ids"]
        if len(neuron_ids) != 0:
            _rate_type = tf.gather(rates, neuron_ids)
            target_rate = value["sorted_target_rates"]
            # if core_mask is not None:
            #     key_core_mask = np.isin(value["neuron_ids"], core_neurons_ids)
            #     neuron_ids =  np.where(key_core_mask)[0]
            #     _rate_type = tf.gather(rates, neuron_ids)
            #     target_rate = value["sorted_target_rates"][key_core_mask]
            # else:
            #     _rate_type = tf.gather(rates, value["neuron_ids"])
            #     target_rate = value["sorted_target_rates"]

            loss_type = compute_spike_rate_distribution_loss(_rate_type, target_rate, dtype=dtype)
            total_loss += tf.reduce_sum(loss_type)
            num_neurons += tf.size(neuron_ids)

    total_loss /= tf.cast(num_neurons, dtype=dtype)

    return total_loss


def compute_spike_rate_distribution_loss(_rates, target_rate, dtype=tf.float32):
    # Firstly we shuffle the current model rates to avoid bias towards a particular tuning angles (inherited from neurons ordering in the network)
    ind = tf.range(target_rate.shape[0])
    rand_ind = tf.random.shuffle(ind)
    _rates = tf.gather(_rates, rand_ind)
    sorted_rate = tf.sort(_rates)
    # u = target_rate - sorted_rate
    u = sorted_rate - target_rate
    n = tf.shape(target_rate)[0]
    tau = (tf.cast(tf.range(n), dtype) + 1) / tf.cast(n, dtype)
    loss = huber_quantile_loss(u, tau, 0.002, dtype=dtype)
    # loss = huber_quantile_loss(u, tau, 0.1, dtype=dtype)

    return loss


def process_neuropixels_data(path=''):
    # Load data
    neuropixels_data_path = f'Neuropixels_data/cortical_metrics_1.4.csv'
    df_all = pd.read_csv(neuropixels_data_path, sep=",")
    # Exc and PV have sufficient number of cells, so we'll filter out non-V1 Exc and PV.
    # SST and VIP are small populations, so let's keep also non-V1 neurons
    exclude = (df_all["cell_type"].isnull() | df_all["cell_type"].str.contains("EXC") | df_all["cell_type"].str.contains("PV")) \
            & (df_all["ecephys_structure_acronym"] != 'VISp')
    df = df_all[~exclude]
    print(f"Original: {df_all.shape[0]} cells,   filtered: {df.shape[0]} cells")

    # Some cells have very large values of RF. They are likely not-good fits, so ignore.
    df.loc[(df["width_rf"] > 100), "width_rf"] = np.nan
    df.loc[(df["height_rf"] > 100), "height_rf"] = np.nan

    # Save the processed table
    df.to_csv(f'Neuropixels_data/v1_OSI_DSI_DF.csv', sep=" ", index=False)
    # return df


def sample_firing_rates(firing_rates, n_neurons, rnd_seed):
    # Sort the original firing rates
    sorted_firing_rates = np.sort(firing_rates)
    # Calculate the empirical cumulative distribution function (CDF)
    percentiles = np.linspace(0, 1, sorted_firing_rates.size)
    # Generate random uniform values from 0 to 1
    rate_rd = np.random.RandomState(seed=rnd_seed)
    x_rand = rate_rd.uniform(low=0, high=1, size=n_neurons)
    # Use inverse transform sampling: interpolate the uniform values to find the firing rates
    target_firing_rates = np.sort(np.interp(x_rand, percentiles, sorted_firing_rates))
    # target_firing_rates = np.interp(x_rand, percentiles, sorted_firing_rates)
    return target_firing_rates


def huber_quantile_loss(u, tau, kappa, dtype=tf.float32):
    tau = tf.cast(tau, dtype)
    abs_u = tf.abs(u)
    num = tf.abs(tau - tf.cast(u <= 0, dtype))
    branch_1 = num / (2 * kappa) * tf.square(u)
    branch_2 = num * (abs_u - 0.5 * kappa)
    return tf.where(abs_u <= kappa, branch_1, branch_2)



def get_pop_names(network, core_radius = None, n_selected_neurons=None, data_dir='', return_node_type_ids=False):
    if data_dir != '':  # if changed from default, use as is.
        pass
    elif "data_dir" in network:  # if defined in the network, use it.
        data_dir = network["data_dir"]
    else:
        print("No data_dir defined in the network. Using the default one.")
        data_dir = 'GLIF_network'  # if none is the cae, use the default one
    path_to_csv = os.path.join(data_dir, 'network/v1_node_types.csv')
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')

    # Read data
    node_types = pd.read_csv(path_to_csv, sep=' ')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        # Create mapping from node_type_id to pop_name
        node_types.set_index('node_type_id', inplace=True)
        node_type_id_to_pop_name = node_types['pop_name'].to_dict()

        # Map node_type_id to pop_name for all neurons and select population names of neurons in the present network 
        ### node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'][()])[network['tf_id_to_bmtk_id']]
        node_type_ids = np.array(node_h5['nodes']['v1']['node_type_id'][()])
        true_pop_names = np.array([node_type_id_to_pop_name[nid] for nid in node_type_ids])

    if core_radius is not None:
        selected_mask = isolate_core_neurons(network, radius=core_radius, data_dir=data_dir)
    elif n_selected_neurons is not None:
        selected_mask = isolate_core_neurons(network, n_selected_neurons=n_selected_neurons, data_dir=data_dir)
    else:
        selected_mask = np.full(len(true_pop_names), True)
        
    true_pop_names = true_pop_names[selected_mask]
    node_type_ids = node_type_ids[selected_mask]

    if return_node_type_ids:
        return true_pop_names, node_type_ids
    else:
        return true_pop_names


def isolate_core_neurons(network, radius=None, n_selected_neurons=None, data_dir='GLIF_network'):
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        x = np.array(node_h5['nodes']['v1']['0']['x'][()])
        z = np.array(node_h5['nodes']['v1']['0']['z'][()])
    # The reference V1_GLIF_model reorders neurons (tf vs bmtk id); dpointnet keeps the
    # native SONATA order (no 'tf_id_to_bmtk_id'), in which case the file order is the
    # model order and no reindexing is needed.
    if isinstance(network, dict) and ('tf_id_to_bmtk_id' in network):
        x = x[network['tf_id_to_bmtk_id']]
        z = z[network['tf_id_to_bmtk_id']]

    r = np.sqrt(x ** 2 + z ** 2)
    if radius is not None:
        selected_mask = r < radius
    elif n_selected_neurons is not None:
        selected_mask = np.argsort(r)[:n_selected_neurons]
        selected_mask = np.isin(np.arange(len(r)), selected_mask)
    
    return selected_mask


def get_tuning_angles(network, data_dir=''):
    """Per-neuron preferred orientation (degrees) in the model's node order.

    Mirrors get_pop_names' data_dir resolution and isolate_core_neurons' native-order
    handling: dpointnet keeps the SONATA file order (no 'tf_id_to_bmtk_id' reindexing),
    so the file order is the model order unless the reference reordering map is present.
    """
    if data_dir != '':  # if changed from default, use as is.
        pass
    elif isinstance(network, dict) and ("data_dir" in network):
        data_dir = network["data_dir"]
    else:
        data_dir = 'GLIF_network'
    path_to_h5 = os.path.join(data_dir, 'network/v1_nodes.h5')
    with h5py.File(path_to_h5, mode='r') as node_h5:
        tuning_angle = np.array(node_h5['nodes']['v1']['0']['tuning_angle'][()], dtype=np.float32)
    if isinstance(network, dict) and ('tf_id_to_bmtk_id' in network):
        tuning_angle = tuning_angle[network['tf_id_to_bmtk_id']]
    return tuning_angle


def resolve_core_mask(network, core_mask=None, core_radius=None, data_dir='GLIF_network'):
    """Resolve a boolean core mask for a loss function.

    If an explicit ``core_mask`` is given it is used as-is. Otherwise, if a ``core_radius``
    is given, the central-core mask is computed from neuron positions (sqrt(x^2+z^2)<radius),
    matching the reference V1_GLIF_model's ``loss_core_radius``. Returns ``None`` if neither
    is provided (loss applies to all neurons).
    """
    if core_mask is not None:
        return core_mask
    if core_radius is not None:
        return isolate_core_neurons(network, radius=core_radius, data_dir=data_dir)
    return None


def pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=False):
    """convert pop_name in the old format to cell types.
    for example,
    'e4Rorb' -> 'L4 Exc'
    'i4Pvalb' -> 'L4 PV'
    'i23Sst' -> 'L2/3 SST'
    'e5ET' -> 'L5 ET'
    """
    shift = 0  # letter shift for L23
    layer = pop_name[1]
    if layer == "2":
        layer = "2/3"
        shift = 1
    elif layer == "1":
        return "L1 Htr3a"  # special case

    class_name = pop_name[2 + shift :]
    if class_name == "Pvalb":
        subclass = "PV"
    elif class_name == "Sst":
        subclass = "SST"
    elif (class_name == "Vip") or (class_name == "Htr3a"):
        subclass = "VIP"
    else:  # excitatory
        if layer == "5" and not ignore_l5e_subtypes:
            subclass = class_name
        else:
            subclass = "Exc"

    return f"L{layer} {subclass}" 



def _core_mask_to_numpy(core_mask, n_nodes):
    if core_mask is None:
        return np.ones(n_nodes, dtype=bool)
    if hasattr(core_mask, "numpy"):
        core_mask = core_mask.numpy()
    core_mask = np.asarray(core_mask, dtype=bool)
    if core_mask.shape[0] != n_nodes:
        raise ValueError(
            f"core_mask has length {core_mask.shape[0]}, expected {n_nodes}."
        )
    return core_mask




def get_population_neuron_ids(network, data_dir="GLIF_network", core_mask=None, reindex_selected=False):
    pop_names = get_pop_names(network, data_dir=data_dir)
    np_core_mask = _core_mask_to_numpy(core_mask, len(pop_names))
    if reindex_selected:
        selected_ids = np.arange(np.count_nonzero(np_core_mask), dtype=np.int32)
    else:
        selected_ids = np.flatnonzero(np_core_mask)
    selected_pop_names = pop_names[np_core_mask]
    selected_cell_types = np.array(
        [
            pop_name_to_cell_type(pop_name, ignore_l5e_subtypes=True)
            for pop_name in selected_pop_names
        ]
    )

    grouped_ids = {}
    for cell_type in CELL_TYPE_ORDER:
        grouped_ids[cell_type] = selected_ids[selected_cell_types == cell_type]
    return grouped_ids


def neuropixels_cell_type_to_cell_type(pop_name):
    if not isinstance(pop_name, str):
        return pop_name
    if ' ' in pop_name:  # This is already new. No need to update.
        return pop_name

    # Convert pop_name in the neuropixels cell type to cell types. E.g, 'EXC_L23' -> 'L2/3 Exc', 'PV_L5' -> 'L5 PV'
    layer = pop_name.split('_')[1]
    class_name = pop_name.split('_')[0]
    if "2" in layer:
        layer = "L2/3"
    elif layer == "L1":
        return "L1 Htr3a"  # special case
    if class_name == "EXC":
        class_name = "Exc"
    if class_name == 'Htr3a':
        class_name = 'VIP'

    return f"{layer} {class_name}"