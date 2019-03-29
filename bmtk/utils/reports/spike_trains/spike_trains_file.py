import os
import csv
from enum import Enum

import numpy as np


MPI_Rank = 0
MPI_Size = 1


class SortOrder(Enum):
    none = 'none'
    by_id = 'by_id'
    by_time = 'by_time'
    unknown = 'unknown'


class SpikeTrains(object):
    # _read_adaptor = {}

    def __init__(self, population_name=None, use_caching=False, output_dir='.', **opt_args):
        # TODO: implement as lazy attribute
        self._read_adapter = None  # Used for reading spike trains from a file
        self._buffer = None  # Used for adding new spikes

        self._population_name = population_name

        '''
        if use_caching or MPI_Size > 0:
            self._spikes_cache = STDiskCache(output_dir=output_dir, **opt_args)
        else:
            self._spikes_cache = STMemoryCache(**opt_args)
        '''

        self._sort_by = SortOrder.unknown

        ## self._itr_order = SortOrder.none

    @property
    def sorted_by(self):
        return self._sort_by

    @property
    def population(self):
        return self._population_name

    @property
    def node_ids(self):
        return self._read_adapter.node_ids
        # return self._spikes_cache.node_ids()

    @property
    def gids(self):
        # TODO: Depreciate
        raise NotImplementedError()

    @property
    def n_ids(self):
        return len(self.node_ids)

    @property
    def n_spikes(self):
        return self._spikes_cache.count_spikes()

    @classmethod
    def from_csv(cls, path, **kwargs):
        obj = cls()
        obj._read_adapter = CSVReader(path, **kwargs)
        return obj

    #@staticmethod
    #def from_csv(path, sep=' '):
    #    raise NotImplementedError()

    @staticmethod
    def from_sonata():
        raise NotImplementedError()

    @staticmethod
    def from_nwb():
        raise NotImplementedError()

    @staticmethod
    def from_file():
        raise NotImplementedError()

    def to_csv(self, file_name, mode='w', sort_order=SortOrder.none, **opt_params):
        pass
        # csv_writer = CSVWriter(spike_trains=self, file_name=file_name, mode=mode, sort_order=sort_order, **opt_params)
        # csv_writer.save()

    def to_sonata(self):
        pass

    def to_nwb(self):
        raise NotImplementedError()

    def add_spike(self, spike_time, node_id):
        self._spikes_cache.add_spikes(times=[spike_time], node_id=node_id)

    def add_spikes(self, spike_times, node_id):
        self._spikes_cache.add_spikes(times=spike_times, node_id=node_id)

    def get_spikes_df(self, node_ids=None, time_window=None, population=None, sort_by=SortOrder.none):
        return self._read_adapter.get_spikes_df(node_ids=node_ids, time_window=time_window, population=population,
                                                sort_by=sort_by)
        # return self._spikes_cache.get_spikes(gid, time_window=None)
        # pass

    #def spikes(self, sort_order):
    #    return self._spikes_cache.spikes(sort_order)

    def get_spike_times(self, node_id, time_window=None, population=None, sorted=False):
        return self._read_adapter.get_spike_times(node_id=node_id, time_window=time_window, population=population,
                                                  sorted=sorted)

    def flush(self):
        raise NotImplementedError()


    def close(self):
        self._spikes_cache.close()


class SpikeBuffer(object):
    def __init__(self):
        pass

    def insert(self, times, node_id):
        pass

    def add(self, times, node_id):
        pass

    def cache(self):
        pass


class SpikeTrainRecorder(object):
    """
    class CachedFileMetadata(object):
        def __init__(self, file_name, sort_order=None):
            self.file_name = file_name
            self.sort_order = sort_order
    """

    def __init__(self):
        self._n_total_spikes = 0
        self._node_ids = []
        self._spikes_buffer = SpikeBuffer()

        self._disk_cache = None
        # self._use_cache = True

    def _get_tmp_filename(self, rank):
        return os.path.join(self._tmp_dir, '.bmtk_spikes_cache_{}.csv'.format(rank))

    def add_spikes(self, times, node_id):
        if node_id in self._spikes.keys():
            self._spikes_buffer.add(times, node_id)
        else:
            self._spikes_buffer.insert(times, node_id)
            self._node_ids.append(node_id)
            self._node_id_count += 1

        self._n_total_spikes += len(times)

    def flush(self):
        self._spikes_buffer.cache()


    def count_node_ids(self):
        return self._node_id_count

    def count_spikes(self):
        return self._spike_count

    def node_ids(self):
        return list(self._spikes.keys())


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


'''
class SonataSTFile(SpikeTrains):
    pass




class CSVWriter(object):
    def __init__(self, spike_trains, file_name, mode, sort_order, **opt_params):
        self._st_obj = spike_trains
        self._sort_order = sort_order

        self._population_column = opt_params.get('population_column', False)
        delimiter = opt_params.get('delimiter', ' ')
        self._csv_handle = open(file_name, mode)
        self._csv_writer = csv.writer(self._csv_handle, delimiter=delimiter)

    def save(self):
        if self._population_column:
            pop_name = self._st_obj.population_name
            self._csv_writer.write_row(['time', 'node_id', 'population'])
            for node_id, spike_time in self._st_obj.spikes(sorted_by=self._sort_order):
                self._csv_writer.writerow([spike_time, node_id, pop_name])

        else:
            self._csv_writer.writerow(['time', 'node_id'])
            for node_id, spike_time in self._st_obj.spikes(sort_order=self._sort_order):
                self._csv_writer.writerow([spike_time, node_id])




class ISTCache(object):
    def add_spikes(self, times, node_id):
        raise NotImplementedError()


class STMemoryCache(ISTCache):
    def __init__(self, **opt_args):
        self._spikes = {}
        self._node_id_count = 0
        self._spike_count = 0

        self._spike_itr_indx = {}

    def _next_spike(self, node_id):
        spike_train = self._spikes[node_id]
        c_spike_indx = self._spike_itr_indx[node_id]
        # print(self._spikes)
        # print(spike_train)

        if c_spike_indx >= len(spike_train):
            return None
        else:
            c_spike = spike_train[c_spike_indx]
            self._spike_itr_indx[node_id] += 1
            return c_spike

    def add_spikes(self, times, node_id):
        if node_id in self._spikes.keys():
            self._spikes[node_id].extend(times)
        else:
            self._spikes[node_id] = times
            self._node_id_count += 1

        self._spike_count += len(times)

    def count_node_ids(self):
        return self._node_id_count

    def count_spikes(self):
        return self._spike_count

    def node_ids(self):
        return list(self._spikes.keys())

    def spikes(self, sort_order):
        if sort_order == SortOrder.unknown or sort_order == SortOrder.none:
            for nid, spike_times in self._spikes.items():
                for st in spike_times:
                    yield nid, st

        elif sort_order == SortOrder.by_id:
            #print('HERE')
            node_ids_sorted = list(self._spikes.keys()).copy()
            node_ids_sorted.sort()
            #print(list(self._spikes.keys()).sort())
            #node_ids_sorted = list(self._spikes.keys()).sort()
            #print(node_ids_sorted)
            for nid in node_ids_sorted:
                spike_trains = self._spikes[nid]
                for st in spike_trains:
                    yield nid, st

        elif sort_order == SortOrder.by_time:
            # make sure spike times are ordered
            for nid, spike_times in self._spikes.items():
                self._spikes[nid].sort()

            self._spike_itr_indx = {nid: 0 for nid in self._spikes.keys()}

            spikes = []
            for nid in self._spikes.keys():  # range(self._mpi_size):
                spike = self._next_spike(nid)
                if spike is not None:
                    spikes.append((nid, spike))
            #print(spikes)
            #exit()

            # Iterate through all the ranks and find the first spike. Write that spike/gid to the output, then
            # replace that data point with the next spike on the selected rank
            indx = 0
            while spikes:
                # print('>>', spikes)
                # find which rank has the first spike
                selected_index = 0
                selected_val = spikes[0][1]
                for i, spike in enumerate(spikes[1:]):
                    # print(spike)
                    if spike[1] < selected_val:
                        selected_index = i + 1
                        selected_val = spike[1]

                # write the spike to the file
                row = spikes.pop(selected_index)
                #print(row)
                #exit()
                # file_write_fnc(float(row[self.time_col]), int(row[self.gid_col]), indx)
                indx += 1

                # get the next spike on that rank and replace in spikes table
                another_spike = self._next_spike(row[0])
                if another_spike is not None:
                    spikes.append((row[0], another_spike))

                yield row[0], selected_val


class STDiskCache(ISTCache):
    class TmpFileMetadata(object):
        def __init__(self, file_name, sort_order=None):
            self.file_name = file_name
            self.sort_order = sort_order

    def __init__(self, output_dir, **opt_args):
        # For NEST/NEURON based simulations it is prefereable not to use mpi4py, so let the parent simulator determine
        # MPI rank and size
        #self._mpi_rank = MPI_Rank
        #self._mpi_size = MPI_Size

        # used to temporary save spike files since for large simulations saving spikes into memory can crash the
        # system. Requires the user to create the directory
        self._tmp_dir = output_dir
        if self._tmp_dir is None or not os.path.exists(self._tmp_dir):
            raise Exception('Directory path {} does not exists'.format(self._tmp_dir))
        self._all_tmp_files = [self.TmpFileMetadata(self._get_tmp_filename(r)) for r in range(MPI_Size)]
        # TODO: Determine best buffer size.
        self._tmp_file_handle = open(self._all_tmp_files[MPI_Rank].file_name, 'w')

        self._tmp_spikes_handles = []  # used when sorting mulitple file
        # self._spike_count = -1

        # Nest gid files uses tab seperators and a different order for tmp spike files.
        self.delimiter = ' '  # delimiter for temporary file
        self.time_col = 0
        self.gid_col = 1

        # self._

    """
    def _next_spike(self, node_id):
        spike_train = self._spikes[node_id]
        c_spike_indx = self._spike_itr_indx[node_id]
        # print(self._spikes)
        # print(spike_train)

        if c_spike_indx >= len(spike_train):
            return None
        else:
            c_spike = spike_train[c_spike_indx]
            self._spike_itr_indx[node_id] += 1
            return c_spike
    """

    def _get_tmp_filename(self, rank):
        return os.path.join(self._tmp_dir, '.bmtk_spikes_cache_{}.csv'.format(rank))

    def add_spikes(self, times, node_id):
        for t in times:
            self.add_spike(t, node_id)

    def add_spike(self, time, gid):
        self._tmp_file_handle.write('{:.6f} {}\n'.format(time, gid))

    def count_node_ids(self):
        return len(self.node_ids())

    def count_spikes(self):
        spike_count = 0
        for tmp_file in self._all_tmp_files:
            with open(tmp_file.file_name, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=self.delimiter)
                spike_count += sum(1 for _ in csv_reader)

        return spike_count

    def node_ids(self):
        node_id_set = set()
        for tmp_file in self._all_tmp_files:
            with open(tmp_file.file_name, 'r') as csvfile:
                # print(csvfile)
                csv_reader = csv.reader(csvfile, delimiter=self.delimiter)
                # print(csv_reader)
                for r in csv_reader:
                    node_id_set.add(int(r[1]))

        return list(node_id_set)

    def spikes(self, sort_order):
        if sort_order == SortOrder.unknown or sort_order == SortOrder.none:
            for nid, spike_times in self._spikes.items():
                for st in spike_times:
                    yield nid, st

        elif sort_order == SortOrder.by_id:
            #print('HERE')
            node_ids_sorted = list(self._spikes.keys()).copy()
            node_ids_sorted.sort()
            #print(list(self._spikes.keys()).sort())
            #node_ids_sorted = list(self._spikes.keys()).sort()
            #print(node_ids_sorted)
            for nid in node_ids_sorted:
                spike_trains = self._spikes[nid]
                for st in spike_trains:
                    yield nid, st

        elif sort_order == SortOrder.by_time:
            # make sure spike times are ordered
            for nid, spike_times in self._spikes.items():
                self._spikes[nid].sort()

            self._spike_itr_indx = {nid: 0 for nid in self._spikes.keys()}

            spikes = []
            for nid in self._spikes.keys():  # range(self._mpi_size):
                spike = self._next_spike(nid)
                if spike is not None:
                    spikes.append((nid, spike))
            #print(spikes)
            #exit()

            # Iterate through all the ranks and find the first spike. Write that spike/gid to the output, then
            # replace that data point with the next spike on the selected rank
            indx = 0
            while spikes:
                # print('>>', spikes)
                # find which rank has the first spike
                selected_index = 0
                selected_val = spikes[0][1]
                for i, spike in enumerate(spikes[1:]):
                    # print(spike)
                    if spike[1] < selected_val:
                        selected_index = i + 1
                        selected_val = spike[1]

                # write the spike to the file
                row = spikes.pop(selected_index)
                #print(row)
                #exit()
                # file_write_fnc(float(row[self.time_col]), int(row[self.gid_col]), indx)
                indx += 1

                # get the next spike on that rank and replace in spikes table
                another_spike = self._next_spike(row[0])
                if another_spike is not None:
                    spikes.append((row[0], another_spike))

                yield row[0], selected_val

    def flush(self):
        self._tmp_file_handle.flush()

    def close(self):
        self._tmp_file_handle.close()
'''