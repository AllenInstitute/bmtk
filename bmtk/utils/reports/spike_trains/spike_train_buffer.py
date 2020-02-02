import os
import numpy as np
import pandas as pd
import csv
from mpi4py import MPI
from array import array
import fcntl

from .core import pop_na, STReader, STBuffer
from .core import SortOrder
from bmtk.utils.io import bmtk_world_comm


comm = MPI.COMM_WORLD


def _spikes_filter1(p, t, time_window, populations):
    return p in populations and time_window[0] <= t <= time_window[1]


def _spikes_filter2(p, t, populations):
    return p in populations


def _spikes_filter3(p, t, time_window):
    return time_window[0] <= t <= time_window[1]


def _create_filter(populations, time_window):
    from functools import partial

    if populations is None and time_window is None:
        return lambda p, t: True
    if populations is None:
        return partial(_spikes_filter3, time_window=time_window)

    populations = [populations] if np.isscalar(populations) else populations
    if time_window is None:
        return partial(_spikes_filter2, populations=populations)
    else:
        return partial(_spikes_filter1, populations=populations, time_window=time_window)


class STMemoryBuffer(STBuffer, STReader):
    """ A Class for creating, storing and reading multi-population spike-trains - especially for saving the spikes of a
    large scale network simulation. Keeps a running tally of the (timestamp, population-name, node_id) for each
    individual spike.

    The spikes are stored in memory and very large and/or epiletic simulations may run into memory issues. Not designed
    to work with parallel simulations.
    """
    def __init__(self, default_population=None, store_type='array', **kwargs):
        self._default_population = default_population or pop_na
        self._store_type = store_type
        # self._pop_counts = {self._default_population: 0}  # A count of spikes per population
        self._units = kwargs.get('units', 'ms')  # for backwards compatability default to milliseconds
        self._pops = {}

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        population = population or self._default_population

        if population not in self._pops:
            self._create_store(population)
        self._pops[population]['node_ids'].append(node_id)
        self._pops[population]['timestamps'].append(timestamp)

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        population = population or self._default_population
        if np.isscalar(node_ids):
            node_ids = [node_ids]*len(timestamps)

        if len(node_ids) != len(timestamps):
            raise ValueError('node_ids and timestamps must by of the same length')

        if population not in self._pops:
            self._create_store(population)
        pop_data = self._pops[population]
        pop_data['node_ids'].extend(node_ids)
        pop_data['timestamps'].extend(timestamps)

    def _create_store(self, population):
        """Helper for creating storage data struct of a population, so add_spike/add_spikes is consistent."""

        # Benchmark Notes:
        #   Tested with numpy, lists and arrays. np.concate/append is too slow to consider. regular list is ~2-3x
        #   faster than array, but require 2-4x the amount of memory. For larger and parallelized applications
        #   (> 100 million spikes) use array since the amount of memory can required can exceed amount available. But
        #   if memory is not an issue use list.
        if self._store_type == 'list':
            self._pops[population] = {'node_ids': [], 'timestamps': []}

        elif self._store_type == 'array':
            self._pops[population] = {'node_ids': array('I'), 'timestamps': array('d')}

        else:
            raise AttributeError('Uknown store type {} for SpikeTrains'.format(self._store_type))

    def import_spikes(self, obj, **kwargs):
        pass

    def flush(self):
        pass  # not necessary since everything is stored in memory

    def close(self):
        pass  # don't need to do anything

    @property
    def populations(self):
        return list(self._pops.keys())

    def node_ids(self, population=None):
        population = population if population is not None else self._default_population
        return np.unique(self._pops[population]['node_ids']).astype(np.uint)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, v):
        self._units = v

    def n_spikes(self, population=None):
        population = population if population is not None else self._default_population
        return len(self._pops[population]['timestamps'])

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        population = population if population is not None else self._default_population
        pop = self._pops[population]

        # filter by node_id and (if specified) by time.
        mask = np.array(pop['node_ids']) == node_id
        ts = np.array(pop['timestamps'])
        if time_window:
            mask &= (time_window[0] <= ts) & (ts <= time_window[1])
        return ts[mask]

    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
        if populations is None:
            selelectd_pops = list(self.populations)
        elif np.isscalar(populations):
            selelectd_pops = [populations]
        else:
            selelectd_pops = populations

        ret_df = None
        for pop_name in selelectd_pops:
            pop_data = self._pops[pop_name]
            pop_df = pd.DataFrame({
                'node_ids': pop_data['node_ids'],
                'timestamps': pop_data['timestamps']
            })
            if with_population_col:
                pop_df['population'] = pop_name

            if sort_order == SortOrder.by_id:
                pop_df = pop_df.sort_values('node_ids')
            elif sort_order == SortOrder.by_time:
                pop_df = pop_df.sort_values('timestamps')

            if ret_df is None:
                ret_df = pop_df
            else:
                ret_df = ret_df.append(pop_df)

        return ret_df

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if populations is None:
            populations = self.populations
        elif np.isscalar(populations):
            populations = [populations]

        for pop_name in populations:
            pop_data = self._pops[pop_name]
            timestamps = pop_data['timestamps']
            node_ids = pop_data['node_ids']

            if sort_order == SortOrder.by_id:
                sort_indx = np.argsort(node_ids)
            elif sort_order == SortOrder.by_time:
                sort_indx = np.argsort(timestamps)
            else:
                sort_indx = range(len(timestamps))

            filter = _create_filter(populations, time_window)
            for i in sort_indx:
                t = timestamps[i]
                p = pop_name
                if filter(p=p, t=t):
                    yield t, p, node_ids[i]

        return

    def metrics(self, **kwargs):
        metrics_dict = {}
        for p in self.populations.keys():
            metrics_dict[p] = self.n_spikes(p)

    def __len__(self):
        return len(self.to_dataframe())


class STMPIBuffer(STMemoryBuffer):
    def __init__(self, default_population=None, store_type='array', **kwargs):
        self.mpi_rank = kwargs.get('MPI_rank', bmtk_world_comm.MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', bmtk_world_comm.MPI_size)
        super(STMPIBuffer, self).__init__(default_population=default_population, store_type=store_type, **kwargs)

    def metrics(self, on_rank='all'):
        comm = bmtk_world_comm.comm

        def collect_metrics(metrics_data):
            all_metrics = metrics_data[0]
            for rank_dict in metrics_data[1:]:
                for pop, pop_count in rank_dict.items():
                    if pop in all_metrics:
                        all_metrics[pop] += pop_count
                    else:
                        all_metrics[pop] = pop_count
            return all_metrics

        local_metrics = {p: len(v['timestamps']) for p, v in self._pops.items()}

        if on_rank == 'local':
            return local_metrics

        elif on_rank == 'all':
            metrics_data = comm.allgather(local_metrics)
            return collect_metrics(metrics_data)

        elif on_rank == 'root':
            metrics_data = comm.gather(local_metrics, root=0)
            if bmtk_world_comm.MPI_rank == 0:
                return collect_metrics(metrics_data)
            else:
                return None

        # print(local_metrics)
        # exit()
        #
        #
        # metrics_data = comm.allgather(self._pop_counts)
        # all_metrics = metrics_data[0]
        # for rank_dict in metrics_data[1:]:
        #     for pop, pop_count in rank_dict.items():
        #         if pop in all_metrics:
        #             all_metrics[pop] += pop_count
        #         else:
        #             all_metrics[pop] = pop_count
        #
        # return all_metrics

    def _dataframe_Allgatherv(self, population):
        comm = bmtk_world_comm.comm
        size = bmtk_world_comm.MPI_size
        rank = bmtk_world_comm.MPI_rank

        local_n_spikes = super(STMPIBuffer, self).n_spikes('v1')
        sizes = comm.allgather(local_n_spikes)
        offsets = np.zeros(size, dtype=np.int)
        offsets[1:] = np.cumsum(sizes)[:-1]
        all_n_spikes = np.sum(sizes)

        # local_node_ids = np.array(self._node_ids, dtype=np.uint64)
        local_node_ids = np.array(self._pops[population]['node_ids'], dtype=np.uint64)
        all_node_ids = np.zeros(all_n_spikes, dtype=np.uint64)
        comm.Gatherv(local_node_ids, [all_node_ids, sizes, offsets, MPI.UINT64_T])

        # local_timestamps = np.array(self._timestamps, dtype=np.double)
        local_timestamps = np.array(self._pops[population]['timestamps'], dtype=np.double)
        all_timestamps = np.zeros(all_n_spikes, dtype=np.double)
        comm.Gatherv(local_timestamps, [all_timestamps, sizes, offsets, MPI.DOUBLE])

        if rank == 0:
            return pd.DataFrame({
                #'population': 'v1',
                'node_ids': all_node_ids,
                'timestamps': all_timestamps
            })
        else:
            return None

    def _dataframe_gather(self):
        comm = bmtk_world_comm.comm
        rank = bmtk_world_comm.MPI_rank

        spikes_df = pd.DataFrame({
            'node_ids': self._node_ids,
            'timestamps': self._timestamps
        })
        data = comm.gather(spikes_df, root=0)

        if bmtk_world_comm.MPI_rank == 0:
            #print('gather_')
            all_spikes_df = data[0]
            for df in data[1:]:
                all_spikes_df = all_spikes_df.append(df)

            all_spikes_df['population'] = 'v1'
        else:
            all_spikes_df = None

        return all_spikes_df


    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
        # collected_df = self._dataframe_gather()
        collected_df = self._dataframe_Allgatherv(populations)

        # if sort_order == SortOrder.by_id:
        #     collected_df = collected_df.sort_values('node_ids')
        # elif sort_order == SortOrder.by_time:
        #     collected_df = collected_df.sort_values('timestamps')

        return collected_df



    def gather_spikes(self):
        comm = bmtk_world_comm.comm
        data = comm.allgather(self.to_dataframe())
        #metrics_data = comm.allgather(self._pop_counts)
        #print(metrics_data)
        all_spikes_df = data[0]
        for df in data[1:]:
            all_spikes_df = all_spikes_df.append(df)

        print(len(all_spikes_df))
        # print(data[0])
        exit()

    @property
    def populations(self):
        print(self._pop_counts)
        exit()

        self._gather()
        return list(self._pop_counts.keys())

    def n_spikes(self, population=None):
        comm = bmtk_world_comm.comm
        n = super(STMPIBuffer, self).n_spikes(population)
        return comm.allreduce(n, MPI.SUM)

    def _gather(self):
        self._pop_counts = {}
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue
            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    pop = row[1]
                    self._pop_counts[pop] = self._pop_counts.get(pop, 0) + 1

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        self.flush()

        filter = _create_filter(populations, time_window)
        if sort_order == SortOrder.by_time or sort_order == SortOrder.by_id:
            for file_name in self._all_cached_files():
                if not os.path.exists(file_name):
                    continue

                self._sort_buffer_file(file_name, sort_order)

            return self._sorted_itr(filter, 0 if sort_order == SortOrder.by_time else 2)
        else:
            return self._unsorted_itr(filter)

    def _unsorted_itr(self, filter):
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue

            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    t = float(row[0])
                    p = row[1]
                    if filter(p=p, t=t):
                        yield t, p, int(row[2])

        return

    def _sorted_itr(self, filter, sort_col):
        """Iterates through all the spikes on each rank, returning them in the specified order"""
        import heapq

        def next_row(csv_reader):
            try:
                rn = next(csv_reader)
                row = [float(rn[0]), rn[1], int(rn[2])]
                return row[sort_col], row, csv_reader
            except StopIteration:
                return None

        # Assumes all the ranked cached files have already been sorted. Pop the top row off of each rank onto the
        # heap, pull next spike off the heap and replace. Repeat until all spikes on all ranks have been poped.
        h = []
        readers = [next_row(csv.reader(open(fn, 'r'), delimiter=' ')) for fn in self._all_cached_files()]
        for r in readers:
            if r is not None:  # Some ranks may not have produced any spikes
                heapq.heappush(h, r)

        while h:
            v, row, csv_reader = heapq.heappop(h)
            n = next_row(csv_reader)
            if n:
                heapq.heappush(h, n)

            if filter(row[1], row[2]):
                yield row


class STCSVBuffer(STBuffer, STReader):
    """ A Class for creating, storing and reading multi-population spike-trains - especially for saving the spikes of a
    large scale network simulation. Keeps a running tally of the (timestamp, population-name, node_id) for each
    individual spike.

    Uses a caching mechanism to periodically save spikes to the disk. Will encure a runtime performance penality but
    will always have an upper bound on the maximum memory used.

    If running parallel simulations should use the STMPIBuffer adaptor instead.
    """

    def __init__(self, cache_dir=None, default_population=None, cache_name='spikes', **kwargs):
        self._default_population = default_population or pop_na

        # Keep a file handle open for writing spike information
        self._cache_dir = cache_dir or '.'
        self._cache_name = cache_name
        self._buffer_filename = self._cache_fname(self._cache_dir)
        self._buffer_handle = open(self._buffer_filename, 'w')
        # fcntl.lockf(self._buffer_handle, fcntl.LOCK_EX)

        # print(self._default_population)
        #self._pop_counts = {} #{self._default_population: 0}
        #self._pop_node_ids = {}
        #self._nspikes = 0
        self._units = kwargs.get('units', 'ms')

        self._pop_metadata = {}

    def _cache_fname(self, cache_dir):
        # TODO: Potential problem if multiple SpikeTrains are opened at the same time, add salt to prevent collisions
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return os.path.join(cache_dir, '.bmtk.{}.cache.csv'.format(self._cache_name))

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        population = population or self._default_population
        # print(population)

        # NOTE: I looked into using a in-memory buffer to save data and caching only when they reached a threshold,
        # however on my computer it was actually slower than just calling file.write() each time. Likely the python
        # file writer is more efficent than what I could write. However still would like to benchmark on a NSF.
        self._buffer_handle.write('{} {} {}\n'.format(timestamp, population, node_id))

        if population not in self._pop_metadata:
            self._pop_metadata[population] = {'node_ids': set(), 'n_spikes': 0}
        self._pop_metadata[population]['node_ids'].add(node_id)
        self._pop_metadata[population]['n_spikes'] += 1

        #self._nspikes += 1
        #self._pop_counts[population] = self._pop_counts.get(population, 0) + 1
        #self._pop_node_ids[population] = self._pop_node_ids.get(population, [])


    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        if np.isscalar(node_ids):
            for ts in timestamps:
                self.add_spike(node_ids, ts, population)
        else:
            for node_id, ts in zip(node_ids, timestamps):
                self.add_spike(node_id, ts, population)

    @property
    def populations(self):
        return list(self._pop_metadata.keys())

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, v):
        self._units = v

    def node_ids(self, population=None):
        population = population if population is not None else self._default_population
        return list(self._pop_metadata[population]['node_ids'])

    def n_spikes(self, population=None):
        population = population if population is not None else self._default_population
        return self._pop_metadata[population]['n_spikes']

    def time_range(self, populations=None):
        return None  # TODO: keep track of largest and smallest values
        # return np.min(self._timestamps), np.max(self._timestamps)

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        self.flush()
        population = population if population is not None else self._default_population
        return np.array([t[0] for t in self.spikes(populations=population, time_window=time_window) if t[2] == node_id])

    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
        self.flush()

        sorting_cols = ['population']
        if sort_order == SortOrder.by_time:
            sorting_cols = ['population', 'timestamps']
        elif sort_order == SortOrder.by_id:
            sorting_cols = ['population', 'node_ids']

        ret_df = pd.read_csv(
            self._buffer_filename, sep=' ', names=['timestamps', 'population', 'node_ids']
        ).sort_values(sorting_cols)

        # filter by population
        if np.isscalar(populations):
            ret_df = ret_df[ret_df['population'] == populations]
        elif populations is not None:
            ret_df = ret_df[ret_df['population'].isin(populations)]

        if not with_population_col:
            ret_df = ret_df.drop('population', axis=1)

        return ret_df

    def flush(self):
        self._buffer_handle.flush()

    def close(self):
        self._buffer_handle.close()
        if os.path.exists(self._buffer_filename):
            os.remove(self._buffer_filename)

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        self.flush()

        self._sort_buffer_file(self._buffer_filename, sort_order)
        filter = _create_filter(populations, time_window)
        with open(self._buffer_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                t = float(row[0])
                p = row[1]
                if filter(p=p, t=t):
                    yield t, p, int(row[2])

        return

    def _sort_buffer_file(self, file_name, sort_order):
        if sort_order == SortOrder.by_time:
            sort_col = 'time'
        elif sort_order == SortOrder.by_id:
            sort_col = 'node'
        else:
            return

        tmp_spikes_ds = pd.read_csv(file_name, sep=' ', names=['time', 'population', 'node'])
        tmp_spikes_ds = tmp_spikes_ds.sort_values(by=sort_col)
        tmp_spikes_ds.to_csv(file_name, sep=' ', index=False, header=False)



class STCSVMPIBuffer(STCSVBuffer):
    def __init__(self, cache_dir=None, default_population=None, cache_name='spikes', **kwargs):
        self.mpi_rank = kwargs.get('MPI_rank', bmtk_world_comm.MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', bmtk_world_comm.MPI_size)
        self._cache_name = cache_name
        super(STCSVMPIBuffer, self).__init__(cache_dir, default_population=default_population, **kwargs)

    def _cache_fname(self, cache_dir):
        if self.mpi_rank == 0:
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
        bmtk_world_comm.barrier()

        return os.path.join(self._cache_dir, '.bmtk.{}.cache.node{}.csv'.format(self._cache_name, self.mpi_rank))

    def _all_cached_files(self):
        return [os.path.join(self._cache_dir, '.bmtk.{}.cache.node{}.csv'.format(self._cache_name, r)) for r in range(bmtk_world_comm.MPI_size)]

    @property
    def populations(self):
        self._gather()
        return list(self._pop_counts.keys())

    def n_spikes(self, population=None):
        self._gather()
        return self._pop_counts.get(population, 0)

    def _gather(self):
        self._pop_counts = {}
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue
            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    pop = row[1]
                    self._pop_counts[pop] = self._pop_counts.get(pop, 0) + 1

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        self.flush()

        filter = _create_filter(populations, time_window)
        if sort_order == SortOrder.by_time or sort_order == SortOrder.by_id:
            for file_name in self._all_cached_files():
                if not os.path.exists(file_name):
                    continue

                self._sort_buffer_file(file_name, sort_order)

            return self._sorted_itr(filter, 0 if sort_order == SortOrder.by_time else 2)
        else:
            return self._unsorted_itr(filter)

    def _unsorted_itr(self, filter):
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue

            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    t = float(row[0])
                    p = row[1]
                    if filter(p=p, t=t):
                        yield t, p, int(row[2])

        return

    def _sorted_itr(self, filter, sort_col):
        """Iterates through all the spikes on each rank, returning them in the specified order"""
        import heapq

        def next_row(csv_reader):

            try:
                rn = next(csv_reader)
                # print(rn)
                row = [float(rn[0]), rn[1], int(rn[2])]
                return row[sort_col], row, csv_reader
            except StopIteration:
                return None
            except ValueError as ie:
                print(ie)
                exit()

        # Assumes all the ranked cached files have already been sorted. Pop the top row off of each rank onto the
        # heap, pull next spike off the heap and replace. Repeat until all spikes on all ranks have been poped.
        h = []
        readers = [next_row(csv.reader(open(fn, 'r'), delimiter=' ')) for fn in self._all_cached_files()]
        for r in readers:
            if r is not None:  # Some ranks may not have produced any spikes
                heapq.heappush(h, r)

        while h:
            v, row, csv_reader = heapq.heappop(h)
            n = next_row(csv_reader)
            if n:
                heapq.heappush(h, n)

            if filter(row[1], row[2]):
                yield row

class STCSVMPIBufferV2(STCSVMPIBuffer):
    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        final_df = None

        if bmtk_world_comm.MPI_rank == 0:
            for file_name in self._all_cached_files():
                if not os.path.exists(file_name):
                    continue

                df = pd.read_csv(file_name, sep=' ', names=['timestamps', 'population', 'node_ids'])
                if final_df is None:
                    final_df = df
                else:
                    final_df = final_df.append(df)

        return final_df
