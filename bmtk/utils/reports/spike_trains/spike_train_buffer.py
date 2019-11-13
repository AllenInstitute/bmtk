import os
import numpy as np
import pandas as pd
import six
import csv

from .core import pop_na, STReader, STBuffer
from .core import SortOrder
from .core import csv_headers
from bmtk.utils.io import bmtk_world_comm


"""
class DiskBuffer(object):
    def __init__(self, cache_dir, **kwargs):
        self.cache_dir = cache_dir
        self.mpi_rank = kwargs.get('MPI_rank', MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', MPI_size)
        self.cached_fname = os.path.join(self.cache_dir, '.spikes.cache.node{}.csv'.format(self.mpi_rank))

        if self.mpi_rank == 0:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
        barrier()

        self.cached_fhandle = open(self.cached_fname, 'w')
        self._csv_reader = None

    def append(self, timestamp, population, node_id):
        self.cached_fhandle.write('{} {} {}\n'.format(timestamp, population, node_id))

    def extend(self, timestamps, population, node_id):
        pass

    def __iter__(self):
        self.cached_fhandle.flush()
        barrier()
        self._csv_reader = csv.reader(open(self.cached_fname, 'r'), delimiter=' ')
        return self

    def __next__(self):
        #try:
        r = next(self._csv_reader)
        return [np.float64(r[0]), r[1], np.uint64(r[2])]
        #except StopIteration:
        #    return StopIteration
        #exit()
        #return next(self._csv_reader)

    next = __next__


class MemoryBuffer(object):
    pass


class STBufferedWriter(STBuffer, STReader):
    def __init__(self, buffer_dir=None, default_pop=None, **kwargs):
        if default_pop is None or isinstance(default_pop, six.string_types) or np.isscalar(default_pop):
            self.default_pop = pop_na
        elif len(default_pop) == 1:
            self.default_pop = default_pop[0]
        else:
            self.default_pop = Exception

        self._buffer = DiskBuffer('cache')

        # self._populations = set([default_pop])
        self._populations_counts = {self.default_pop: 0}
        self._units = kwargs.get('units', 'ms')

    def add_spike(self, node_id, timestamp, population=None):
        if population is None:
            population = self.default_pop

        if population not in self._populations_counts:
            self._populations_counts[population] = 0
        self._populations_counts[population] += 1
        # self._populations.add(population)
        self._buffer.write(timestamp=timestamp, population=population, node_id=node_id)

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        if population is None:
            population = self.default_pop

        if np.isscalar(node_ids):
            for ts in timestamps:
                self.add_spike(node_ids, ts, population)
        else:
            if len(node_ids) != len(timestamps):
                raise Exception('timestamps and node_ids must be of the same length.')

            for node_id, ts in zip(node_ids, timestamps):
                self.add_spike(node_id, ts, population)

    def import_spikes(self, obj):
        pass

    def flush(self):
        pass

    @property
    def populations(self):
        return list(self._populations_counts.keys())

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, v):
        self._units = v

    def nodes(self, populations=None):
        raise NotImplementedError()

    def n_spikes(self, population=None):
        return self._populations_counts[population]

    def time_range(self, populations=None):
        raise NotImplementedError()

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        raise NotImplementedError()

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if sort_order == SortOrder.by_time or sort_order == SortOrder.by_time:
            raise Exception("Can't sort by time or node-id")

        for n in self._buffer:
            if n[1] == populations:
                yield n

        raise StopIteration

    def __len__(self):
        return len(self.to_dataframe())
"""

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

    def __init__(self, default_population=None, **kwargs):
        self._default_population = default_population or pop_na

        # look into storing data using numpy arrays or pandas series.
        self._node_ids = []
        self._timestamps = []
        self._populations = []
        self._pop_counts = {self._default_population: 0}  # A count of spikes per population
        self._units = kwargs.get('units', 'ms')  # for backwards compatability default to milliseconds

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        population = population or self._default_population
        self._node_ids.append(node_id)
        self._timestamps.append(timestamp)
        self._populations.append(population)

        self._pop_counts[population] = self._pop_counts.get(population, 0) + 1

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        if np.isscalar(node_ids):
            node_ids = [node_ids]*len(timestamps)

        for node_id, ts in zip(node_ids, timestamps):
            self.add_spike(node_id, ts, population)

    def import_spikes(self, obj, **kwargs):
        pass

    def flush(self):
        pass  # not necessary since everything is stored in memory

    def close(self):
        pass  # don't need to do anything

    @property
    def populations(self):
        return list(self._pop_counts.keys())

    def nodes(self, populations=None):
        return list(set(self._node_ids))

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, v):
        self._units = v

    def n_spikes(self, population=None):
        return self._pop_counts.get(population, 0)

    def time_range(self, populations=None):
        return np.min(self._timestamps), np.max(self._timestamps)

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        population = population or self._default_population
        mask = (np.array(self._node_ids) == node_id) & (np.array(self._populations) == population)
        ts = np.array(self._timestamps)
        if time_window:
            mask &= (time_window[0] <= ts) & (ts <= time_window[1])

        return ts[mask]

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        # raise NotImplementedError()
        populations = populations or self._default_population
        # TODO: Filter by population, node-id and time
        # TODO: Sort dataframe if needed
        return pd.DataFrame({'node_id': self._node_ids, 'population': self._populations, 'timestamps': self._timestamps})


    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if sort_order == SortOrder.by_time:
            sort_indx = np.argsort(self._timestamps)
        elif sort_order == SortOrder.by_time:
            sort_indx = np.argsort(self._node_ids)
        else:
            sort_indx = range(len(self._timestamps))

        filter = _create_filter(populations, time_window)
        for i in sort_indx:
            t = self._timestamps[i]
            p = self._populations[i]
            if filter(p=p, t=t):
                yield t, p, self._node_ids[i]

        return
        #raise StopIteration

    def __len__(self):
        return len(self.to_dataframe())


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

        self._pop_counts = {self._default_population: 0}
        self._nspikes = 0
        self._units = kwargs.get('units', 'ms')

    def _cache_fname(self, cache_dir):
        # TODO: Potential problem if multiple SpikeTrains are opened at the same time, add salt to prevent collisions
        if not os.path.exists(self._cache_dir):
            os.mkdirs(self._cache_dir)
        return os.path.join(cache_dir, '.bmtk.{}.cache.csv'.format(self._cache_name))

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        population = population or pop_na

        # NOTE: I looked into using a in-memory buffer to save data and caching only when they reached a threshold,
        # however on my computer it was actually slower than just calling file.write() each time. Likely the python
        # file writer is more efficent than what I could write. However still would like to benchmark on a NSF.
        self._buffer_handle.write('{} {} {}\n'.format(timestamp, population, node_id))
        self._nspikes += 1
        self._pop_counts[population] = self._pop_counts.get(population, 0) + 1

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        if np.isscalar(node_ids):
            for ts in timestamps:
                self.add_spike(node_ids, ts, population)
        else:
            for node_id, ts in zip(node_ids, timestamps):
                self.add_spike(node_id, ts, population)

    @property
    def populations(self):
        return list(self._pop_counts.keys())

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, v):
        self._units = v

    def nodes(self, populations=None):
        return list(set(self._node_ids))

    def n_spikes(self, population=None):
        return self._pop_counts.get(population, 0)

    def time_range(self, populations=None):
        return np.min(self._timestamps), np.max(self._timestamps)

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        return np.array([t[0] for t in self.spikes(population=population, time_window=time_window) if t[1] == node_id])

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def flush(self):
        self._buffer_handle.flush()

    def close(self):
        self._buffer_handle.close()
        if os.path.exists(self._buffer_filename):
            os.remove(self._buffer_filename)

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
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
        #raise StopIteration

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


class STMPIBuffer(STCSVBuffer):
    def __init__(self, cache_dir=None, default_population=None, cache_name='spikes', **kwargs):
        self.mpi_rank = kwargs.get('MPI_rank', bmtk_world_comm.MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', bmtk_world_comm.MPI_size)
        self._cache_name = cache_name
        super(STMPIBuffer, self).__init__(cache_dir, default_population=default_population, **kwargs)


    def _cache_fname(self, cache_dir):
        if self.mpi_rank == 0:
            if not os.path.exists(self._cache_dir):
                os.mkdirs(self._cache_dir)
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
        #raise StopIteration

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
