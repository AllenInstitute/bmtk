# Copyright 2020. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import csv
import pandas as pd
import numpy as np
import h5py
import six
from collections import defaultdict
import warnings

from .spike_trains_api import SpikeTrainsReadOnlyAPI
from .core import SortOrder, csv_headers, col_population, col_timestamps, col_node_ids, pop_na


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


def load_sonata_file(path, version=None, **kwargs):
    """Loads a Sonata file reader, making sure it matches the correct version.

    :param path:
    :param version:
    :param kwargs:
    :return:
    """
    try:
        with h5py.File(path, 'r') as h5:
            spikes_root = h5[GRP_spikes_root]
            for name, h5_obj in spikes_root.items():
                if isinstance(h5_obj, h5py.Group):
                    # In case there exists a population subgroup
                    return SonataSTReader(path, **kwargs)
    except Exception:
        pass

    try:
        with h5py.File(path, 'r') as h5:
            spikes_root = h5[GRP_spikes_root]
            if 'gids' in spikes_root and 'timestamps' in spikes_root:
                return SonataOldReader(path, **kwargs)
    except Exception:
        pass

    try:
        with h5py.File(path, 'r') as h5:
            if '/spikes' in h5:
                return EmptySonataReader(path, **kwargs)
    except Exception:
        pass

    raise Exception('Could not open file {}, does not contain SONATA spike-trains'.format(path))


def to_list(v):
    if v is not None and np.isscalar(v):
        return [v]
    else:
        return v


class SonataSTReader(SpikeTrainsReadOnlyAPI):
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

        if 'population' in kwargs:
            pop_filter = to_list(kwargs['population'])
        elif 'populations' in kwargs:
            pop_filter = to_list(kwargs['populations'])
        else:
            pop_filter = None

        # get a map of 'pop_name' -> pop_group
        self._population_map = {}
        for name, h5_obj in self._h5_handle[GRP_spikes_root].items():
            if isinstance(h5_obj, h5py.Group):
                if pop_filter is not None and name not in pop_filter:
                    continue

                if 'node_ids' not in h5_obj or 'timestamps' not in h5_obj:
                    warnings.warn('population {} in {} is missing spikes, skipping.'.format(name, path))
                    continue

                self._population_map[name] = h5_obj

        if not self._population_map:
            # In old version of the sonata standard there was no 'population' subgroup. For backwards compatability
            # use a default dictionary
            # TODO: Remove so we only have to support latest version of SONATA
            self._population_map = defaultdict(lambda: self._h5_handle[GRP_spikes_root])
            self._population_map[pop_na] = self._h5_handle[GRP_spikes_root]
            self._DATASET_node_ids = 'gids'

        self._default_pop = kwargs.get('default_population', list(self._population_map.keys())[0])

        self._population_sorting_map = {}
        for pop_name, pop_grp in self._population_map.items():
            if 'sorting' in pop_grp[self._DATASET_node_ids].attrs.keys():
                # Found a few existing sonata files put the 'sorting' attribute in the node_ids dataset, remove later
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
        self._units_maps = {}
        for pop_name, pop_grp in self._population_map.items():
            if 'units' in pop_grp['timestamps'].attrs:
                pop_units = pop_grp['timestamps'].attrs['units']
            elif 'units' in kwargs:
                pop_units = kwargs['units']
            else:
                pop_units = 'ms'

            self._units_maps[pop_name] = pop_units

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

    def units(self, population=None):
        population = population if population is not None else self._default_pop
        return self._units_maps[population]

    def set_units(self, u, population=None):
        self._units_maps[population] = u

    def sort_order(self, population=None):
        return self._population_sorting_map[population]

    def node_ids(self, population=None):
        population = population if population is not None else self._default_pop
        pop_grp = self._population_map[population]
        return np.unique(pop_grp[self._DATASET_node_ids][()])

    def n_spikes(self, population=None):
        population = population if population is not None else self._default_pop
        return len(self._population_map[population][DATASET_timestamps])

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

    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
        populations = populations if populations is not None else self.populations
        if isinstance(populations, six.string_types) or np.isscalar(populations):
            populations = [populations]

        ret_df = None
        for pop_name, pop_grp in self._population_map.items():
            if pop_name in populations:
                pop_df = pd.DataFrame({
                    col_timestamps: pop_grp[DATASET_timestamps],
                    # col_population: pop_name,
                    col_node_ids: pop_grp[self._DATASET_node_ids]
                })

                if with_population_col:
                    pop_df['population'] = pop_name

                if sort_order == SortOrder.by_id:
                    pop_df = pop_df.sort_values('node_ids')
                elif sort_order == SortOrder.by_time:
                    pop_df = pop_df.sort_values('timestamps')

                ret_df = pop_df if ret_df is None else pd.concat((ret_df, pop_df))

        if sort_order == SortOrder.by_time:
            ret_df.sort_values(by=col_timestamps, inplace=True)
        elif sort_order == SortOrder.by_id:
            ret_df.sort_values(by=col_node_ids, inplace=True)

        return ret_df

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        if population is None:
            if not isinstance(self._default_pop, six.string_types) and len(self._default_pop) > 1:
                raise Exception('Error: Multiple populations, must select one.')

            population = self._default_pop

        elif population not in self._population_map:
            return []

        spikes_index = self._index_nids[population].get(node_id, None)
        if spikes_index is None:
            return []
        spike_times = self._population_map[population][DATASET_timestamps][spikes_index]

        if time_window is not None:
            spike_times = spike_times[(time_window[0] <= spike_times) & (spike_times <= time_window[1])]

        return spike_times

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        populations = populations or self.populations
        if np.isscalar(populations):
            populations = [populations]

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
            # TODO: Reimplement using a heap
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


class SonataOldReader(SonataSTReader):
    """Older version of SONATA
    """

    def node_ids(self, population=None):
        return super(SonataOldReader, self).node_ids(population=None)

    def n_spikes(self, population=None):
        return super(SonataOldReader, self).n_spikes(population=self._default_pop)

    def time_range(self, populations=None):
        return super(SonataOldReader, self).time_range(populations=None)

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        return super(SonataOldReader, self).to_dataframe(node_ids=node_ids, populations=None,
                                                         time_window=time_window, sort_order=sort_order, **kwargs)

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        return super(SonataOldReader, self).get_times(node_id=node_id, population=None,
                                                      time_window=time_window, **kwargs)

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        return super(SonataOldReader, self).spikes(node_ids=node_ids, populations=None,
                                                   time_window=time_window, sort_order=sort_order, **kwargs)


class EmptySonataReader(SpikeTrainsReadOnlyAPI):
    """A Hack that is needed for when a simulation produces a file with no spikes, since there won't/can't be
    <population_name> subgroup and/or gids/timestamps datasets.

    """
    def __init__(self, path, **kwargs):
        pass

    @property
    def populations(self):
        return []

    def node_ids(self, population=None):
        return []

    def n_spikes(self, population=None):
        return 0

    def time_range(self, populations=None):
        return None

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        return pd.DataFrame(columns=csv_headers)

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        return []

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        return []


class CSVSTReader(SpikeTrainsReadOnlyAPI):
    def __init__(self, path, sep=' ', default_population=None, **kwargs):
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

        self._defaul_population = default_population if default_population is not None \
            else self._spikes_df[col_population][0]

        if col_population not in self._spikes_df.columns:
            pop_name = kwargs.get(col_population, self._defaul_population)
            self._spikes_df[col_population] = pop_name

        # TODO: Check all the necessary columns exits
        self._spikes_df = self._spikes_df[csv_headers]

    @property
    def populations(self):
        if self._populations is None:
            self._populations = self._spikes_df['population'].unique()

        return self._populations

    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
        selected = self._spikes_df.copy()

        mask = True
        if populations is not None:
            if isinstance(populations, six.string_types) or np.isscalar(populations):
                mask &= selected[col_population] == populations
            else:
                mask &= selected[col_population].isin(populations)

        if isinstance(mask, pd.Series):
            selected = selected[mask]

        if sort_order == SortOrder.by_time:
            selected.sort_values(by=col_timestamps, inplace=True)
        elif sort_order == SortOrder.by_id:
            selected.sort_values(by=col_node_ids, inplace=True)

        if not with_population_col:
            selected = selected.drop(col_population, axis=1)

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

    def node_ids(self, population=None):
        population = population if population is not None else self._defaul_population
        return np.unique(self._spikes_df[self._spikes_df[col_population] == population][col_node_ids])

    def n_spikes(self, population=None):
        population = population if population is not None else self._defaul_population
        return len(self.to_dataframe(populations=population))

    # def time_range(self, populations=None):
    #     selected = self._spikes_df.copy()
    #     if populations is not None:
    #         if isinstance(populations, six.string_types) or np.isscalar(populations):
    #             mask = selected[col_population] == populations
    #         else:
    #             mask = selected[col_population].isin(populations)
    #
    #         selected = selected[mask]
    #
    #     return selected[col_timestamps].agg([np.min, np.max]).values

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
            selected.sort_values(by=col_node_ids, inplace=True)

        indicies = selected.index.values
        for indx in indicies:
            yield tuple(self._spikes_df.iloc[indx])

    def __len__(self):
        if self._n_spikes is None:
            self._n_spikes = len(self._spikes_df)

        return self._n_spikes


class NWBSTReader(SpikeTrainsReadOnlyAPI):
    def __init__(self, path, **kwargs):
        self._path = path
        self._h5_file = h5py.File(self._path, 'r')
        self._n_spikes = None
        self._spikes_df = None

        # TODO: Check for other versions
        self._population = kwargs.get('population', pop_na)
        if 'trial' in kwargs.keys():
            self._trial = kwargs['trial']
        elif len(self._h5_file['/processing']) == 1:
            self._trial = list(self._h5_file['/processing'].keys())[0]
        else:
            raise Exception('Please specify a trial')

        self._trial_grp = self._h5_file['processing'][self._trial]['spike_train']

    @property
    def populations(self):
        return [self._population]

    def node_ids(self, population=None):
        # if populations is None:
        #     populations = [self._population]
        # elif isinstance(populations, six.string_types):
        #     populations = [populations]

        if self._population != population:
            return []
        return [(self._population, np.uint64(node_id)) for node_id in self._trial_grp.keys()]

    def n_spikes(self, population=None):
        if population != self._population:
            return 0

        return self.__len__()

    # def time_range(self, populations=None):
    #     data_df = self.to_dataframe()
    #     return data_df[col_timestamps].agg([np.min, np.max]).values

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        try:
            spiketimes = self._trial_grp[str(node_id)]['data'][()]

            if time_window is not None:
                spiketimes = spiketimes[(time_window[0] <= spiketimes) & (spiketimes <= time_window[1])]

            return spiketimes
        except KeyError:
            return []

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if self._spikes_df is None:
            self._spikes_df = pd.DataFrame({
                col_timestamps: pd.Series(dtype=np.float),
                col_population: pd.Series(dtype=np.string_),
                col_node_ids: pd.Series(dtype=np.uint64)
            })
            for node_id, node_grp in self._trial_grp.items():
                timestamps = node_grp['data'][()]
                node_df = pd.DataFrame({
                    col_timestamps: timestamps,
                    col_population: self._population,
                    col_node_ids: np.uint64(node_id)
                })
                self._spikes_df = self._spikes_df.append(node_df, ignore_index=True)

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
            selected.sort_values(by=col_node_ids, inplace=True)

        return selected

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if populations is not None:
            if np.isscalar(populations) and populations != self._population:
                raise StopIteration
            elif self._population not in populations:
                raise StopIteration

        if sort_order == SortOrder.by_time:
            spikes_df = self.to_dataframe()
            spikes_df = spikes_df.sort_values(col_timestamps)
            for indx in spikes_df.index:
                r = spikes_df.loc[indx]
                yield (r[col_timestamps], r[col_population], r[col_node_ids])
        else:
            node_ids = np.array(list(self._trial_grp.keys()), dtype=np.uint64)
            if sort_order == SortOrder.by_id:
                node_ids.sort()

            for node_id in node_ids:
                timestamps = self._trial_grp[str(node_id)]['data']
                for ts in timestamps:
                    yield (ts, self._population, node_id)

    def __len__(self):
        if self._n_spikes is None:
            self._n_spikes = 0
            for node_id in self._trial_grp.keys():
                self._n_spikes += len(self._trial_grp[node_id]['data'])

        return self._n_spikes
