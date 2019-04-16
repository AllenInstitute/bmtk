import os
import six
import h5py
import pandas as pd
import numpy as np

from .core import SortOrder, STReader
from .core import col_node_ids, col_timestamps, col_population, pop_na, find_conversion
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    barrier = comm.Barrier
except:
    MPI_rank = 0
    MPI_size = 1
    barrier = lambda: None


GRP_spikes_root = 'spikes'
DATASET_timestamps = 'timestamps'
DATASET_node_ids = 'node_ids'


sorting_attrs = {
    'time': SortOrder.by_time,
    'by_time': SortOrder.by_time,
    'id': SortOrder.by_id,
    'by_id': SortOrder.by_id,
    'none': SortOrder.none,
    'unknown': SortOrder.unknown
}


def write_sonata(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, units='ms', **kwargs):
    path_dir = os.path.dirname(path)
    if path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir)

    conv_factor = find_conversion(spiketrain_reader.units, units)
    with h5py.File(path, mode=mode) as h5:
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        for pop_name in spiketrain_reader.populations:
            n_spikes = spiketrain_reader.n_spikes(pop_name)
            if n_spikes <= 0:
                continue

            spikes_grp = h5.create_group('/spikes/{}/'.format(pop_name))
            if sort_order != SortOrder.unknown:
                spikes_grp.attrs['sorting'] = sort_order.value

            timestamps_ds = spikes_grp.create_dataset('timestamps', shape=(n_spikes,), dtype=np.float64)
            timestamps_ds.attrs['units'] = units
            node_ids_ds = spikes_grp.create_dataset('node_ids', shape=(n_spikes,), dtype=np.uint64)
            for i, spk in enumerate(spiketrain_reader.spikes(populations=pop_name, sort_order=sort_order)):
                timestamps_ds[i] = spk[0]*conv_factor
                node_ids_ds[i] = spk[2]


class SonataSTReader(STReader):
    # TODO: Split into multi-children so we can handle reading different version

    def __init__(self, path, **kwargs):
        self._path = path
        self._h5_handle = h5py.File(self._path, 'r')
        self._DATASET_node_ids = 'node_ids'
        self._n_spikes = None
        # TODO: Create a function for looking up population and can return errors if more than one
        self._default_pop = None
        # self._node_list = None

        self._indexed = False
        self._index_nids = {}

        if GRP_spikes_root not in self._h5_handle:
            raise Exception('Could not find /{} root'.format(GRP_spikes_root))
        else:
            self._spikes_root = self._h5_handle[GRP_spikes_root]

        # get a map of 'pop_name' -> pop_group
        self._population_map = {}
        for name, h5_obj in self._h5_handle[GRP_spikes_root].items():
            if isinstance(h5_obj, h5py.Group):
                self._population_map[name] = h5_obj

        if not self._population_map:
            self._population_map[pop_na] = self._h5_handle[GRP_spikes_root]
            self._DATASET_node_ids = 'gids'

        self._default_pop = list(self._population_map.keys())  # [0]

        self._population_sorting_map = {}
        for pop_name, pop_grp in self._population_map.items():
            if 'sorting' in pop_grp[self._DATASET_node_ids].attrs.keys():
                attr_str = pop_grp[self._DATASET_node_ids].attrs['sorting']
                sort_order = sorting_attrs.get(attr_str, SortOrder.unknown)

            elif 'sorting' in pop_grp.attrs.keys():
                attr_str = pop_grp.attrs['sorting']
                sort_order = sorting_attrs.get(attr_str, SortOrder.unknown)

            else:
                sort_order = SortOrder.unknown

            self._population_sorting_map[pop_name] = sort_order

        # TODO: Add option to skip building indices
        self._build_node_index()

        # units are not instrinsic to a csv file, but allow users to pass it in if they know
        # TODO: Should check the populations for the units
        self._units = kwargs.get('units', 'unknown')


    def _build_node_index(self):
        self._indexed = False
        for pop_name, pop_grp in self._population_map.items():
            sort_order = self._population_sorting_map[pop_name]
            nodes_indices = {}
            node_ids_ds = pop_grp[self._DATASET_node_ids]
            if sort_order == SortOrder.by_id:
                indx_beg = 0
                last_id = node_ids_ds[0]
                for indx, cur_id in enumerate(node_ids_ds):
                    # print cur_id
                    if cur_id != last_id:
                        # nodes_indices[last_id] = np.arange(indx_beg, indx)
                        nodes_indices[last_id] = slice(indx_beg, indx)
                        last_id = cur_id
                        indx_beg = indx
                # nodes_indices[last_id] = np.arange(indx_beg, indx + 1)
                nodes_indices[last_id] = slice(indx_beg, indx + 1)  # capture the last node_id
            else:
                nodes_indices = {int(node_id): [] for node_id in np.unique(node_ids_ds)}
                for indx, node_id in enumerate(node_ids_ds):
                    nodes_indices[node_id].append(indx)

            self._index_nids[pop_name] = nodes_indices

        self._indexed = True

    @property
    def populations(self):
        return list(self._population_map.keys())

    @property
    def units(self):
        return self.units

    @units.setter
    def units(self, v):
        self._units = v

    def sort_order(self, population):
        return self._population_sorting_map[population]

    def nodes(self, populations=None):
        if populations is None:
            populations = self._default_pop  # [self._default_pop]

        if isinstance(populations, six.string_types) or np.isscalar(populations):
            populations = [populations]

        node_list = []
        for pop_name, pop_grp in self._population_map.items():
            # print(pop_name)
            # print populations
            if pop_name in populations:
                # TODO: Check memory profile, may be better to iterate node_ids than convert entire set
                # print 'BLAH'
                node_list.extend((pop_name, node_id) for node_id in np.unique(pop_grp[self._DATASET_node_ids][()]))

        return node_list

    def n_spikes(self, population=None):
        if population not in self._population_map:
            return 0

        return len(self._population_map[population][DATASET_timestamps])

    def time_range(self, populations=None):
        if populations is None:
            populations = self._default_pop  # [self._default_pop]

        if isinstance(populations, six.string_types) or np.isscalar(populations):
            populations = [populations]

        min_time = np.inf
        max_time = -np.inf
        for pop_name, pop_grp in self._population_map.items():
            if pop_name in populations:
                # TODO: Check if sorted by time
                for ts in pop_grp[DATASET_timestamps]:
                    if ts < min_time:
                        min_time = ts

                    if ts > max_time:
                        max_time = ts

        return min_time, max_time

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if populations is None:
            populations = self._default_pop  # [self._default_pop]

        if isinstance(populations, six.string_types) or np.isscalar(populations):
            populations = [populations]

        def build_mask(selected_df):
            mask = True
            if node_ids is not None:
                if np.isscalar(node_ids):
                    mask &= (selected_df[col_node_ids] == node_ids)
                else:
                    mask &= (selected_df[col_node_ids].isin(node_ids))

            if time_window is not None:
                min_time, max_time = time_window
                mask &= (min_time <= selected_df[col_timestamps]) & (selected_df[col_timestamps] <= max_time)

            return mask

        ret_df = pd.DataFrame({
            col_timestamps: pd.Series(dtype=np.float),
            col_population: pd.Series(dtype=np.string_),
            col_node_ids: pd.Series(dtype=np.uint64)
        })
        for pop_name, pop_grp in self._population_map.items():
            if pop_name in populations:
                pop_df = pd.DataFrame({
                    col_timestamps: pop_grp[DATASET_timestamps],
                    col_population: pop_name,
                    col_node_ids: pop_grp[self._DATASET_node_ids]
                })

                mask = build_mask(pop_df)
                if isinstance(mask, pd.Series):
                    pop_df = pop_df[mask]

                ret_df = ret_df.append(pop_df)

        if sort_order == SortOrder.by_time:
            ret_df.sort_values(by=col_timestamps, inplace=True)
        elif sort_order == SortOrder.by_id:
            ret_df.sort_values(by=col_node_ids, inplace=True)

        return ret_df

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        if population is None:
            if len(self._default_pop) > 1:
                raise Exception('Error: Multiple populations, must select one.')

            population = self._default_pop[0]

        elif population not in self.populations:
            return []

        # print population, self._index_nids[population].keys()
        spikes_index = self._index_nids[population][node_id]
        # print(spikes_index)
        spike_times = self._population_map[population][DATASET_timestamps][spikes_index]

        if time_window is not None:
            spike_times = spike_times[(time_window[0] <= spike_times) & (spike_times <= time_window[1])]

        return spike_times

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if populations is None:
            populations = self._default_pop  # [self._default_pop]

        if sort_order == SortOrder.by_id:
            for pop_name in populations:
                if pop_name not in self.populations:
                    continue

                timestamps_ds = self._population_map[pop_name][DATASET_timestamps]
                index_map = self._index_nids[pop_name]
                node_ids = list(index_map.keys())
                node_ids.sort()
                for node_id in node_ids:
                    st_indices = index_map[node_id]
                    for st in timestamps_ds[st_indices]:  # st_indices:
                        yield st, pop_name, node_id

        elif sort_order == SortOrder.by_time:
            index_ranges = []
            for pop_name in populations:
                if pop_name not in self.populations:
                    continue

                pop_grp = self._population_map[pop_name]
                if self._population_sorting_map[pop_name] == SortOrder.by_time:
                    ts_ds = pop_grp[DATASET_timestamps]
                    index_ranges.append([pop_name, 0, len(ts_ds), np.arange(len(ts_ds)), pop_grp, ts_ds[0]])
                else:
                    ts_ds = pop_grp[DATASET_timestamps]
                    ts_indices = np.argsort(ts_ds[()])
                    index_ranges.append([pop_name, 0, len(ts_ds), ts_indices, pop_grp, ts_ds[ts_indices[0]]])

            while index_ranges:
                selected_r = index_ranges[0]
                for i, r in enumerate(index_ranges[1:]):
                    if r[5] < selected_r[5]:
                        selected_r = r

                ds_index = selected_r[1]
                timestamp = selected_r[5]  # pop_grp[DATASET_timestamps][ds_index]
                node_id = selected_r[4][self._DATASET_node_ids][ds_index]
                pop_name = selected_r[0]
                ds_index += 1
                if ds_index >= selected_r[2]:
                    index_ranges.remove(selected_r)
                else:
                    selected_r[1] = ds_index
                    ts_index = selected_r[3][ds_index]
                    next_ts = self._population_map[pop_name][DATASET_timestamps][selected_r[3][ds_index]]
                    selected_r[5] = next_ts  # pop_grp[DATASET_timestamps][selected_r[3][ds_index]]

                yield timestamp, pop_name, node_id

        else:
            for pop_name in populations:
                if pop_name not in self.populations:
                    continue

                pop_grp = self._population_map[pop_name]
                for i in range(len(pop_grp[DATASET_timestamps])):
                    yield pop_grp[DATASET_timestamps][i], pop_name, pop_grp[self._DATASET_node_ids][i]

    def __len__(self):
        if self._n_spikes is None:
            self._n_spikes = 0
            for _, pop_grp in self._population_map.items():
                self._n_spikes += len(pop_grp[self._DATASET_node_ids])

        return self._n_spikes
