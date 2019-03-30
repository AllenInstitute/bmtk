import csv
import six
from enum import Enum

import pandas as pd
import numpy as np
import h5py


class SortOrder(Enum):
    '''
    none = 0
    by_id = 1
    by_time = 2
    unknown = 3
    '''
    none = 'none'
    by_id = 'by_id'
    by_time = 'by_time'
    unknown = 'unknown'



class SpikeTrains(object):
    @classmethod
    def from_csv(cls, path, **kwargs):
        return CSVSTReader(path, **kwargs)

    @classmethod
    def from_sonata(cls, path, **kwargs):
        return SONATASTReader(path, **kwargs)

    @classmethod
    def from_nwb(cls, path, **kwargs):
        return NWBSTReader(path, **kwargs)


class STReader(object):
    @property
    def populations(self):
        raise NotImplementedError()

    def nodes(self, populations=None):
        raise NotImplementedError()

    def time_range(self, populations=None):
        raise NotImplementedError()

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        raise NotImplementedError()

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.to_dataframe())


GRP_spikes_root = 'spikes'
DATASET_timestamps = 'timestamps'
DATASET_node_ids = 'node_ids'


class NWBSTReader(STReader):
    def __init__(self, path, **kwargs):
        self._path = path
        self._h5_file = h5py.File(self._path, 'r')
        self._n_nodes = None
        self._spikes_df = None

        # TODO: Check for other versions
        self._population = kwargs.get('population', pop_na)
        if 'trial' in kwargs.keys():
            self._trial = kwargs['trial']
        elif len(self._h5_file['/processing']) == 1:
            self._trial = self._h5_file['/processing'].keys()[0]
        else:
            raise Exception('Please specify a trial')

        self._trial_grp = self._h5_file['processing'][self._trial]['spike_train']

    @property
    def populations(self):
        return [self._population]

    def nodes(self, populations=None):
        if populations is None:
            populations = [self._population]
        elif isinstance(populations, six.string_types):
            populations = [populations]

        if self._population not in populations:
            return []

        return [(self._population, np.uint64(node_id)) for node_id in self._trial_grp.keys()]

    def time_range(self, populations=None):
        data_df = self.to_dataframe()
        return data_df[col_timestamps].agg([np.min, np.max]).values

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        if population != self._population:
            return []

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if self._spikes_df is None:
            self._spikes_df[0]


    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        if self._n_nodes is None:
            self._n_nodes = 0
            for node_id in self._trial_grp.keys():
                self._n_nodes += len(self._trial_grp[node_id]['data'])

        return self._n_nodes

sorting_attrs = {
    'time': SortOrder.by_time,
    'by_time': SortOrder.by_time,
    'id': SortOrder.by_id,
    'by_id': SortOrder.by_id,
    'none': SortOrder.none,
    'unknown': SortOrder.unknown
}


class SONATASTReader(STReader):
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

        if len(self._population_map) == 1:
            self._default_pop = list(self._population_map.keys())[0]

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
                    if cur_id != last_id:
                        cur_id[cur_id] = slice(indx_beg, indx)
                        last_id = cur_id
                        indx_beg = indx
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

    def nodes(self, populations=None):
        if populations is None:
            populations = [self._default_pop]

        if isinstance(populations, six.string_types) or np.isscalar(populations):
            populations = [populations]

        node_list = []
        for pop_name, pop_grp in self._population_map.items():
            if pop_name in populations:
                # TODO: Check memory profile, may be better to iterate node_ids than convert entire set
                node_list.extend((pop_name, node_id) for node_id in np.unique(pop_grp[self._DATASET_node_ids][()]))

        return node_list

    def time_range(self, populations=None):
        if populations is None:
            populations = [self._default_pop]

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
            populations = [self._default_pop]

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
            population = self._default_pop

        elif population not in self.populations:
            return []

        spikes_index = self._index_nids[population][node_id]
        spike_times = self._population_map[population][DATASET_timestamps][spikes_index]

        if time_window is not None:
            spike_times = spike_times[(time_window[0] <= spike_times) & (spike_times <= time_window[1])]

        return spike_times

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if populations is None:
            populations = [self._default_pop]

        if sort_order == SortOrder.by_id:
            for pop_name in populations:
                if pop_name not in self.populations:
                    continue

                timestamps_ds = self._population_map[pop_name][DATASET_timestamps]
                index_map = self._index_nids[pop_name]
                node_ids = index_map.keys()
                node_ids.sort()
                for node_id in node_ids:
                    st_indices = index_map[node_id]
                    for st_indx in st_indices:
                        yield timestamps_ds[st_indx], pop_name, node_id

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
                    if selected_r[4] < r[4]:
                        selected_r = r

                ds_index = selected_r[1]
                timestamp = pop_grp[DATASET_timestamps][ds_index]
                node_id = pop_grp[self._DATASET_node_ids][ds_index]
                pop_name = selected_r[0]
                ds_index += 1
                if ds_index >= selected_r[2]:
                    index_ranges.remove(selected_r)
                else:
                    selected_r[1] = ds_index
                    selected_r[5] = pop_grp[DATASET_timestamps][ds_index]

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


col_timestamps = 'timestamps'
col_node_ids = 'node_ids'
col_population = 'population'
csv_headers = [col_timestamps, col_population, col_node_ids]
pop_na = '<sonata:none>'


class CSVSTReader(STReader):
    def __init__(self, path, sep=' ', **kwargs):
        self._n_spikes = None
        self._populations = None

        try:
            # check to see if file contains headers
            with open(path, 'r') as csvfile:
                sniffer = csv.Sniffer()
                has_headers = sniffer.has_header(csvfile.read(1024))
        except Exception:
            has_headers = True

        self._spikes_df = pd.read_csv(path, sep=sep, header=0 if has_headers else None)

        if not has_headers:
            self._spikes_df.columns = csv_headers[0::2]

        if col_population not in self._spikes_df.columns:
            pop_name = kwargs.get(col_population, pop_na)
            self._spikes_df[col_population] = pop_name

        # TODO: Check all the necessary columns exits
        self._spikes_df = self._spikes_df[csv_headers]

    @property
    def populations(self):
        if self._populations is None:
            self._populations = self._spikes_df['population'].unique()

        return self._populations

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_by=SortOrder.none, **kwargs):
        selected = self._spikes_df.copy()

        mask = True
        if populations is not None:
            if isinstance(populations, six.string_types) or np.isscalar(populations):
                mask &= selected[col_population] == populations
            else:
                mask &= selected[col_population].isin(populations)

        if node_ids is not None:
            node_ids = [node_ids] if np.isscalar(node_ids) else node_ids
            mask &= selected[col_node_ids].isin(node_ids)

        if time_window is not None:
            mask &= (selected[col_timestamps] >= time_window[0]) & (selected[col_timestamps] <= time_window[1])

        if isinstance(mask, pd.Series):
            selected = selected[mask]

        if sort_by == SortOrder.by_time:
            selected.sort_values(by=col_timestamps, inplace=True)
        elif sort_by == SortOrder.by_id:
            selected.sort_values(by=col_node_ids, inplace=True)

        selected.index = pd.RangeIndex(len(selected.index))
        return selected

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        selected = self._spikes_df.copy()
        mask = (selected[col_node_ids] == node_id)

        if population is not None:
            mask &= (selected[col_population] == population)

        if time_window is not None:
            mask &= (selected[col_timestamps] >= time_window[0]) & (selected[col_timestamps] <= time_window[1])

        return np.array(self._spikes_df[mask][col_timestamps])

    def nodes(self, populations=None):
        selected = self._spikes_df.copy()
        mask = True
        if populations is not None:
            if isinstance(populations, six.string_types) or np.isscalar(populations):
                mask = selected[col_population] == populations
            else:
                mask = selected[col_population].isin(populations)

        if isinstance(mask, pd.Series):
            selected = selected[mask]
        return selected.groupby(by=[col_population, col_node_ids]).indices.keys()

    def time_range(self, populations=None):
        selected = self._spikes_df.copy()
        if populations is not None:
            if isinstance(populations, six.string_types) or np.isscalar(populations):
                mask = selected[col_population] == populations
            else:
                mask = selected[col_population].isin(populations)

            selected = selected[mask]

        return selected[col_timestamps].agg([np.min, np.max]).values

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        selected = self._spikes_df.copy()

        mask = True
        if populations is not None:
            if isinstance(populations, six.string_types) or np.isscalar(populations):
                mask &= selected[col_population] == populations
            else:
                mask &= selected[col_population].isin(populations)

        if node_ids is not None:
            node_ids = [node_ids] if np.isscalar(node_ids) else node_ids
            mask &= selected[col_node_ids].isin(node_ids)

        if time_window is not None:
            mask &= (selected[col_timestamps] >= time_window[0]) & (selected[col_timestamps] <= time_window[1])

        if isinstance(mask, pd.Series):
            selected = selected[mask]

        if sort_order == SortOrder.by_time:
            selected.sort_values(by=col_timestamps, inplace=True)
        elif sort_order == SortOrder.by_id:
            # print('BLAH')
            selected.sort_values(by=col_node_ids, inplace=True)

        indicies = selected.index.values
        for indx in indicies:
            yield tuple(self._spikes_df.iloc[indx])
            #exit()

            #yield self._spikes_df.iloc[indx]

        #print(selected.head()
        #print(selected.index.values)
        #exit()

    def __len__(self):
        if self._n_spikes is None:
            self._n_spikes = len(self._spikes_df)

        return self._n_spikes

