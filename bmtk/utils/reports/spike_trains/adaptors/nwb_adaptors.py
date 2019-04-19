import six
import h5py
import numpy as np
import pandas as pd

from ..core import STReader, SortOrder
from ..core import pop_na, col_timestamps, col_population, col_node_ids


# TODO: Get rid of population filters,

class NWBSTReader(STReader):
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

    def nodes(self, populations=None):
        if populations is None:
            populations = [self._population]
        elif isinstance(populations, six.string_types):
            populations = [populations]

        if self._population not in populations:
            return []

        return [(self._population, np.uint64(node_id)) for node_id in self._trial_grp.keys()]

    def n_spikes(self, population=None):
        if population != self._population:
            return 0

        return self.__len__()

    def time_range(self, populations=None):
        data_df = self.to_dataframe()
        return data_df[col_timestamps].agg([np.min, np.max]).values

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
