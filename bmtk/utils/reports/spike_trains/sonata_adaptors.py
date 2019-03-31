import os
import h5py
import numpy as np

from .core import SortOrder
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


def write_sonata(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, **kwargs):
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with h5py.File(path, mode=mode) as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        for pop_name in spiketrain_reader.populations:
            spikes_grp = h5.create_group('/spikes/{}/'.format(pop_name))
            if sort_order != SortOrder.unknown:
                spikes_grp.attrs['sorting'] = sort_order.value

            n_spikes = spiketrain_reader.n_spikes(pop_name)
            timestamps_ds = spikes_grp.create_dataset('timestamps', shape=(n_spikes,), dtype=np.float64)
            node_ids_ds = spikes_grp.create_dataset('node_ids', shape=(n_spikes,), dtype=np.uint64)
            for i, spk in enumerate(spiketrain_reader.spikes(populations=pop_name, sort_order=sort_order)):
                timestamps_ds[i] = spk[0]
                node_ids_ds[i] = spk[2]
