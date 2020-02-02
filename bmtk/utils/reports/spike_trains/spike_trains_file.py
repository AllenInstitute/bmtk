from enum import Enum
import numpy as np


MPI_Rank = 0
MPI_Size = 1


class SortOrder(Enum):
    none = 'none'
    by_id = 'by_id'
    by_time = 'by_time'
    unknown = 'unknown'


class SpikeBuffer(object):
    def __init__(self):
        pass

    def insert(self, times, node_id):
        pass

    def add(self, times, node_id):
        pass

    def cache(self):
        pass


class STReadAdapter(object):
    def to_dataframe(self):
        raise NotImplementedError()

    def get_spikes(self, node_ids=None, time_window=None):
        raise NotImplementedError()

    def time_min(self):
        raise NotImplementedError()

    def time_max(self):
        raise NotImplementedError()

    @property
    def sorted(self):
        raise NotImplementedError()

import pandas as pd
import csv

col_timestamps = 'timestamps'
col_node_ids = 'node_ids'
col_population = 'population'
csv_headers = [col_timestamps, col_node_ids, col_population]
pop_na = 'NONE'


class CSVReader(STReadAdapter):
    def __init__(self, path, sep=' ', **kwargs):
        self._node_ids = None
        self._min_time = None
        self._max_time = None
        # self._dt = None

        try:
            # check to see if file contains headers
            with open(path, 'r') as csvfile:
                sniffer = csv.Sniffer()
                has_headers = sniffer.has_header(csvfile.read(1024))
        except Exception:
            has_headers = True

        self._spikes_df = pd.read_csv(path, sep=sep, header=0 if has_headers else None)

        if not has_headers:
            self._spikes_df.columns = csv_headers[:self._spikes_df.shape[1]]

        if col_population not in self._spikes_df.columns:
            pop_name = kwargs.get(col_population, pop_na)
            self._spikes_df[col_population] = pop_name

    @property
    def node_ids(self):
        if self._node_ids is not None:
            return self._node_ids
        else:
            self._node_ids = np.unique(self._spikes_df[col_node_ids].values)
            return self._node_ids

    def get_spikes(self, node_ids=None, time_window=None):
        spike_times = np.array([])
        if node_ids is not None:
            pass

    def get_spike_times(self, node_id, time_window=None, population=None, sorted=False):
        mask = (self._spikes_df[col_node_ids] == node_id)
        mask &= (self._spikes_df[col_population] == population) if population else True

        # spike_times = self._spikes_df[self._spikes_df[col_node_ids] == node_id]# [col_timestamps].values
        # spike_times = self._spikes_df[self._spikes_df[col_node_ids] == node_id][col_timestamps].values
        spike_times = self._spikes_df[mask][col_timestamps].values

        if time_window is not None:
            spike_times = spike_times[(spike_times >= time_window[0]) & (spike_times <= time_window[1])]

        if sorted:
            spike_times.sort()

        return spike_times

    def get_spikes_df(self, node_ids=None, time_window=None, population=None, sort_by=SortOrder.none):
        selected = self._spikes_df.copy()

        #mask = (selected_spikes[col_node_ids].isin(node_ids)) if node_ids else True
        #mask &= (selected_spikes[col_population] == population) if population else True
        #mask &= ()
        mask = True

        if population is not None:
            mask &= selected[col_population] == population

        if node_ids is not None:
            mask &= selected[col_node_ids].isin(node_ids)
            # selected_spikes = selected_spikes[selected_spikes[col_node_ids].isin(node_ids)]

        if time_window is not None:
            mask &= (selected[col_timestamps] >= time_window[0]) & (selected[col_timestamps] <= time_window[1])
            #selected_spikes = selected_spikes[(selected_spikes[col_timestamps] >= time_window[0]) &
            #                                  (selected_spikes[col_timestamps] <= time_window[1])]

        if isinstance(mask, pd.Series):
            selected = selected[mask]
        # print(type(mask))
        # exit()

        if sort_by == SortOrder.by_time:
            selected.sort_values(by=col_timestamps, inplace=True)
        elif sort_by == SortOrder.by_id:
            selected.sort_values(by=col_node_ids, inplace=True)

        selected.index = pd.RangeIndex(len(selected.index))
        return selected
