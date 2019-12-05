import os
import csv
import six
import pandas as pd
import numpy as np

from ..core import STReader, SortOrder, find_conversion
from ..core import csv_headers, col_population, pop_na, col_timestamps, col_node_ids

# TODO: BMTK won't work with a non-population column csv file. Update so that if there is no populations then it will
#   do a lookup by node_id only.


def write_csv(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, include_header=True,
              include_population=True, units='ms', **kwargs):
    path_dir = os.path.dirname(path)
    if path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir)

    conv_factor = find_conversion(spiketrain_reader.units, units)
    with open(path, mode=mode) as f:
        if include_population:
            # Saves the Population column
            csv_writer = csv.writer(f, delimiter=' ')
            if include_header:
                csv_writer.writerow(csv_headers)
            for spk in spiketrain_reader.spikes(sort_order=sort_order):
                csv_writer.writerow([spk[0]*conv_factor, spk[1], spk[2]])

        else:
            # Don't write the Population column
            csv_writer = csv.writer(f, delimiter=' ')
            if include_header:
                csv_writer.writerow([c for c in csv_headers if c != col_population])
            for spk in spiketrain_reader.spikes(sort_order=sort_order):
                csv_writer.writerow([spk[0]*conv_factor, spk[2]])


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

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
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
        return list(selected.groupby(by=[col_population, col_node_ids]).indices.keys())

    def n_spikes(self, population=None):
        return len(self.to_dataframe(populations=population))

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
            selected.sort_values(by=col_node_ids, inplace=True)

        indicies = selected.index.values
        for indx in indicies:
            yield tuple(self._spikes_df.iloc[indx])

    def __len__(self):
        if self._n_spikes is None:
            self._n_spikes = len(self._spikes_df)

        return self._n_spikes