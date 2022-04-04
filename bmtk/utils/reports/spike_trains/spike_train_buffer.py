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
import os
import numpy as np
import pandas as pd
import csv
import time
from array import array

from .core import SortOrder, pop_na, comm, MPI_size, MPI_rank, comm_barrier
from .core import col_node_ids, col_population, col_timestamps
from .spike_trains_api import SpikeTrainsAPI


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


def _create_empty_df(with_population_col=True):
    columns = [col_timestamps, col_population, col_node_ids] if with_population_col else [col_timestamps, col_node_ids]
    return pd.DataFrame(columns=columns)


class STMemoryBuffer(SpikeTrainsAPI):
    """ A Class for creating, storing and reading multi-population spike-trains - especially for saving the spikes of a
    large scale network simulation. Keeps a running tally of the (timestamp, population-name, node_id) for each
    individual spike.

    The spikes are stored in memory and very large and/or epiletic simulations may run into memory issues. Not designed
    to work with parallel simulations.
    """
    def __init__(self, default_population=None, store_type='array', **kwargs):
        self._default_population = default_population or kwargs.get('population', None) or pop_na
        self._store_type = store_type
        # self._pop_counts = {self._default_population: 0}  # A count of spikes per population
        self._units = kwargs.get('units', 'ms')  # for backwards compatability default to milliseconds
        self._pops = {}

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        population = population or self._default_population

        if population not in self._pops:
            self._create_store(population)
        self._pops[population][col_node_ids].append(node_id)
        self._pops[population][col_timestamps].append(timestamp)

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        population = population or self._default_population
        if np.isscalar(node_ids):
            node_ids = [node_ids]*len(timestamps)

        if len(node_ids) != len(timestamps):
            raise ValueError('node_ids and timestamps must by of the same length')

        if population not in self._pops:
            self._create_store(population)
        pop_data = self._pops[population]
        pop_data[col_node_ids].extend(node_ids)
        pop_data[col_timestamps].extend(timestamps)

    def _create_store(self, population):
        """Helper for creating storage data struct of a population, so add_spike/add_spikes is consistent."""

        # Benchmark Notes:
        #   Tested with numpy, lists and arrays. np.concate/append is too slow to consider. regular list is ~2-3x
        #   faster than array, but require 2-4x the amount of memory. For larger and parallelized applications
        #   (> 100 million spikes) use array since the amount of memory can required can exceed amount available. But
        #   if memory is not an issue use list.
        if self._store_type == 'list':
            self._pops[population] = {col_node_ids: [], col_timestamps: []}

        elif self._store_type == 'array':
            self._pops[population] = {col_node_ids: array('I'), col_timestamps: array('d')}

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
        if population not in self._pops:
            return []
        return np.unique(self._pops[population][col_node_ids]).astype(np.uint)

    def units(self, population=None):
        return self._units

    def set_units(self, v, population=None):
        self._units = v

    def n_spikes(self, population=None):
        population = population if population is not None else self._default_population
        if population not in self._pops:
            return 0
        return len(self._pops[population][col_timestamps])

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        population = population if population is not None else self._default_population
        pop = self._pops[population]

        # filter by node_id and (if specified) by time.
        mask = np.array(pop[col_node_ids]) == node_id
        ts = np.array(pop[col_timestamps])
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
            pop_data = self._pops.get(pop_name, {col_node_ids: [], col_timestamps: []})
            pop_df = pd.DataFrame({
                col_node_ids: pop_data[col_node_ids],
                col_timestamps: pop_data[col_timestamps]
            })
            if with_population_col:
                pop_df[col_population] = pop_name

            if sort_order == SortOrder.by_id:
                pop_df = pop_df.sort_values(col_node_ids)
            elif sort_order == SortOrder.by_time:
                pop_df = pop_df.sort_values(col_timestamps)

            if ret_df is None:
                ret_df = pop_df
            else:
                ret_df = pd.concat((ret_df, pop_df))

        # Make sure ret_df is not None
        ret_df = _create_empty_df(with_population_col) if ret_df is None else ret_df

        return ret_df

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        if populations is None:
            populations = self.populations
        elif np.isscalar(populations):
            populations = [populations]

        for pop_name in populations:
            pop_data = self._pops.get(pop_name, {col_node_ids: [], col_timestamps: []})
            timestamps = pop_data[col_timestamps]
            node_ids = pop_data[col_node_ids]

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

    def __len__(self):
        return len(self.to_dataframe())


class STMPIBuffer(STMemoryBuffer):
    def __init__(self, default_population=None, store_type='array', **kwargs):
        self.mpi_rank = kwargs.get('MPI_rank', MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', MPI_size)
        super(STMPIBuffer, self).__init__(default_population=default_population, store_type=store_type, **kwargs)

    def _gatherv(self, population, on_all_ranks=True):
        from mpi4py import MPI

        local_n_spikes = super(STMPIBuffer, self).n_spikes(population)
        sizes = comm.allgather(local_n_spikes)
        offsets = np.zeros(MPI_size, dtype=np.int64)
        offsets[1:] = np.cumsum(sizes)[:-1]
        all_n_spikes = np.sum(sizes)

        local_population = self._pops.get(population, {col_node_ids: [], col_timestamps: []})  # if pop not on rank
        local_node_ids = np.array(local_population[col_node_ids], dtype=np.uint64)
        all_node_ids = np.zeros(all_n_spikes, dtype=np.uint64)
        if on_all_ranks:
            comm.Allgatherv(local_node_ids, [all_node_ids, sizes, offsets, MPI.UINT64_T])
        else:
            comm.Gatherv(local_node_ids, [all_node_ids, sizes, offsets, MPI.UINT64_T], root=0)
            if MPI_rank != 0:
                all_node_ids = None

        local_timestamps = np.array(local_population[col_timestamps], dtype=np.double)
        all_timestamps = np.zeros(all_n_spikes, dtype=np.double)
        if on_all_ranks:
            comm.Allgatherv(local_timestamps, [all_timestamps, sizes, offsets, MPI.DOUBLE])
        else:
            comm.Gatherv(local_timestamps, [all_timestamps, sizes, offsets, MPI.DOUBLE], root=0)
            if MPI_rank != 0:
                all_timestamps = None

        return all_node_ids, all_timestamps

    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, on_rank='all',
                     **kwargs):
        comm_barrier()
        if on_rank == 'local':
            return super(STMPIBuffer, self).to_dataframe(populations=populations, sort_order=sort_order,
                                                         with_population_col=with_population_col, **kwargs)

        if on_rank not in ['local', 'all', 'root']:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

        if populations is None:
            selelectd_pops = list(self.get_populations(on_rank='all'))
        elif np.isscalar(populations):
            selelectd_pops = [populations]
        else:
            selelectd_pops = populations

        # Make sure the list of populations is exactly the same (including order) across all ranks so that
        # _gatherv is called in the same sequence across all ranks.
        selelectd_pops.sort()
        ret_df = None
        for pop_name in selelectd_pops:
            if on_rank == 'all':
                node_ids, timestamps = self._gatherv(population=pop_name, on_all_ranks=True)
                pop_df = pd.DataFrame({
                    col_node_ids: node_ids,
                    col_timestamps: timestamps
                })

                if with_population_col:
                    pop_df[col_population] = pop_name

                if sort_order == SortOrder.by_id:
                    pop_df = pop_df.sort_values(col_node_ids)
                elif sort_order == SortOrder.by_time:
                    pop_df = pop_df.sort_values(col_timestamps)

                if ret_df is None:
                    ret_df = pop_df
                else:
                    ret_df = pd.concat((ret_df, pop_df))

            elif on_rank == 'root':
                node_ids, timestamps = self._gatherv(population=pop_name, on_all_ranks=False)
                if MPI_rank != 0:
                    continue

                pop_df = pd.DataFrame({
                    col_node_ids: node_ids,
                    col_timestamps: timestamps
                })

                if with_population_col:
                    pop_df[col_population] = pop_name

                if sort_order == SortOrder.by_id:
                    pop_df = pop_df.sort_values(col_node_ids)
                elif sort_order == SortOrder.by_time:
                    pop_df = pop_df.sort_values(col_timestamps)

                if ret_df is None:
                    ret_df = pop_df
                else:
                    ret_df = pd.concat((ret_df, pop_df))

        comm_barrier()
        if on_rank == 'all' or MPI_rank == 0:
            # If using 'all' or on rank 0 a dataframe is expected even if there are no spikes
            ret_df = _create_empty_df(with_population_col) if ret_df is None else ret_df

        return ret_df

    @property
    def populations(self):
        return self.get_populations(on_rank='all')

    def get_populations(self, on_rank='all'):
        local_pops = list(super(STMPIBuffer, self).populations)
        if on_rank == 'local':
            return local_pops

        if on_rank == 'all':
            gathered_pops = comm.allgather(local_pops)
        elif on_rank == 'root':
            gathered_pops = comm.gather(local_pops, 0)
        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

        if gathered_pops is None:
            return None
        else:
            all_populations = set()
            for pops in gathered_pops:
                all_populations |= set(pops)

            # WARNING: For a number of parallel applications it's important that the list of populations returned
            # is the same across all ranks (eg ranks don't iterate through each population in different sequences)
            all_populations = list(all_populations)
            all_populations.sort()

            return all_populations

    def node_ids(self, population=None, on_rank='all'):
        local_node_ids = super(STMPIBuffer, self).node_ids(population)
        if on_rank == 'local':
            return local_node_ids

        if on_rank == 'all':
            gathered_nodes = comm.allgather(local_node_ids)
        elif on_rank == 'root':
            gathered_nodes = comm.gather(local_node_ids, 0)
        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

        if gathered_nodes is None:
            return None
        else:
            return np.unique(np.concatenate(gathered_nodes)).astype(np.uint)

    def n_spikes(self, population=None, on_rank='all'):
        from mpi4py import MPI

        local_n = super(STMPIBuffer, self).n_spikes(population)
        if on_rank == 'local':
            return local_n
        elif on_rank == 'all':
            return comm.allreduce(local_n, MPI.SUM)
        elif on_rank == 'root':
            return comm.reduce(local_n, MPI.SUM)
        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

    def get_times(self, node_id, population=None, time_window=None, on_rank='all', **kwargs):
        local_times = super(STMPIBuffer, self).get_times(node_id=node_id, population=population,
                                                         time_window=time_window, **kwargs)

        if on_rank == 'local':
            return local_times
        elif on_rank == 'all':
            all_times = comm.allgather(local_times)
        elif on_rank == 'root':
            all_times = comm.gather(local_times, 0)
        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

        if all_times is not None:
            return np.sort(np.concatenate(all_times))
        else:
            return None

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, on_rank='all', **kwargs):
        if on_rank == 'local':
            for i in super(STMPIBuffer, self).spikes(populations=populations, time_window=time_window,
                                                     sort_order=sort_order, **kwargs):
                yield i
            return

        if populations is None:
            populations = self.populations
        elif np.isscalar(populations):
            populations = [populations]

        populations.sort()
        for pop_name in populations:
            node_ids, timestamps = self._gatherv(pop_name, on_all_ranks=(on_rank == 'all'))
            if node_ids is None:
                continue

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


class STCSVBuffer(SpikeTrainsAPI):
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
        self._units = kwargs.get('units', 'ms')
        self._pop_metadata = {}
        self._spike_counts = 0  # all spikes added on rank, for each individual pop spike count stored in _pop_metadata

    def _cache_fname(self, cache_dir):
        # TODO: Potential problem if multiple SpikeTrains are opened at the same time, add salt to prevent collisions
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return os.path.join(cache_dir, '.bmtk.{}.cache.csv'.format(self._cache_name))

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        population = population or self._default_population

        # NOTE: I looked into using a in-memory buffer to save data and caching only when they reached a threshold,
        # however on my computer it was actually slower than just calling file.write() each time. Likely the python
        # file writer is more efficent than what I could write. However still would like to benchmark on a NSF.
        self._buffer_handle.write('{} {} {}\n'.format(timestamp, population, node_id))

        if population not in self._pop_metadata:
            self._pop_metadata[population] = {'node_ids': set(), 'n_spikes': 0}
        self._pop_metadata[population]['node_ids'].add(node_id)
        self._pop_metadata[population]['n_spikes'] += 1
        self._spike_counts += 1

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

    def units(self, population=None):
        return self._units

    def set_units(self, u, population=None):
        self._units = u

    def node_ids(self, population=None):
        population = population if population is not None else self._default_population
        if population not in self._pop_metadata:
            return []
        return list(self._pop_metadata[population]['node_ids'])

    def n_spikes(self, population=None):
        population = population if population is not None else self._default_population
        if population not in self._pop_metadata:
            return 0
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

        sorting_cols = [col_population]
        if sort_order == SortOrder.by_time:
            sorting_cols = [col_population, col_timestamps]
        elif sort_order == SortOrder.by_id:
            sorting_cols = [col_population, col_node_ids]

        ret_df = pd.read_csv(
            self._buffer_filename, sep=' ', names=[col_timestamps, col_population, col_node_ids]
        ).sort_values(sorting_cols)

        # filter by population
        if np.isscalar(populations):
            ret_df = ret_df[ret_df[col_population] == populations]
        elif populations is not None:
            ret_df = ret_df[ret_df[col_population].isin(populations)]

        if not with_population_col:
            ret_df = ret_df.drop(col_population, axis=1)

        ret_df = ret_df.astype({col_timestamps: float, col_node_ids: np.int64})

        return ret_df

    def flush(self):
        self._buffer_handle.flush()

        # Found an issue with even after flushing the csv there can be a lag before data is actually cached to the disk.
        # this can have problems with other processes on a different rank tries open the file that hasn't been
        # completely saved. This hack should hopefully ensure that each rank has fully cached their spikes to disk.
        for i in range(10):
            with open(self._buffer_filename) as fh:
                fcount = len(fh.readlines())
                if fcount == self._spike_counts:
                    break
                time.sleep(0.5)
        else:
            print('Warning: spike counts on rank {} cache does not match total added.'.format(MPI_rank))


    def close(self):
        self._buffer_handle.close()
        if os.path.exists(self._buffer_filename):
            os.remove(self._buffer_filename)

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        self.flush()

        self._sort_buffer_file(self._buffer_filename, sort_order)
        filter_fnc = _create_filter(populations, time_window)
        with open(self._buffer_filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=' ')
            for row in csv_reader:
                t = float(row[0])
                p = row[1]
                if filter_fnc(p=p, t=t):
                    yield t, p, int(row[2])

        return

    def _sort_buffer_file(self, file_name, sort_order):
        # sort a spikes cache file
        # Currently we just read "file_name" into a dataframe, sort it, and resave it to the file
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
        self.mpi_rank = kwargs.get('MPI_rank', MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', MPI_size)
        self._cache_name = cache_name
        self._all_ranks_data = {}

        super(STCSVMPIBuffer, self).__init__(cache_dir, default_population=default_population, **kwargs)

    def _cache_fname(self, cache_dir):
        if self.mpi_rank == 0:
            if not os.path.exists(self._cache_dir):
                os.makedirs(self._cache_dir)
        comm_barrier()

        return os.path.join(self._cache_dir, '.bmtk.{}.cache.node{}.csv'.format(self._cache_name, self.mpi_rank))

    def _all_cached_files(self):
        return [os.path.join(self._cache_dir, '.bmtk.{}.cache.node{}.csv'.format(self._cache_name, r))
                for r in range(MPI_size)]

    def _gather(self):
        self._all_ranks_data = {}
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue
            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    pop = row[1]
                    if pop not in self._all_ranks_data:
                        self._all_ranks_data[pop] = {'n_spikes': 0, 'node_ids': set()}

                    self._all_ranks_data[pop]['n_spikes'] += 1  # self._all_ranks_data.get(pop, 0) + 1
                    self._all_ranks_data[pop]['node_ids'].add(int(row[2]))

    def _gather_times(self, node_id, population):
        timestamps = []
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue
            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    pop = row[1]
                    nid = int(row[2])
                    if nid == node_id and pop == population:
                        timestamps.append(float(row[0]))
        return timestamps

    @property
    def populations(self):
        return self.get_populations(on_rank='all')

    def get_populations(self, on_rank='all'):
        if on_rank == 'local':
            pops = super(STCSVMPIBuffer, self).populations
            pops.sort()  # import populations are in the same order on all ranks
            return pops

        self.flush()
        comm_barrier()
        pops = None
        if on_rank == 'all':
            self._gather()
            pops = list(self._all_ranks_data.keys())

        elif on_rank == 'root':
            if MPI_rank == 0:
                self._gather()
                pops = list(self._all_ranks_data.keys())

        if pops is not None:
            pops.sort()

        return pops

    def n_spikes(self, population=None, on_rank='all'):
        if on_rank == 'local':
            return super(STCSVMPIBuffer, self).n_spikes(population=population)

        population = population if population is not None else self._default_population
        self.flush()
        comm_barrier()

        if on_rank == 'all':
            self._gather()
            return self._all_ranks_data[population]['n_spikes'] if population in self._all_ranks_data else 0

        elif on_rank == 'root':
            if MPI_rank == 0:
                self._gather()
                return self._all_ranks_data[population]['n_spikes'] if population in self._all_ranks_data else 0
            else:
                return None

        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

    def node_ids(self, population=None, on_rank='all'):
        if on_rank == 'local':
            return super(STCSVMPIBuffer, self).node_ids(population=population)

        population = population if population is not None else self._default_population
        self.flush()
        comm_barrier()

        if on_rank == 'all':
            self._gather()
            return list(self._all_ranks_data[population]['node_ids']) if population in self._all_ranks_data else []

        elif on_rank == 'root':
            if MPI_rank == 0:
                self._gather()
                return list(self._all_ranks_data[population]['node_ids']) if population in self._all_ranks_data else []
            else:
                return None

        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

    def get_times(self, node_id, population=None, time_window=None, on_rank='all', **kwargs):
        # population = population if population is not None else self._default_population
        if on_rank == 'local':
            # calling super.get_times() will fail since it relies on spikes()
            return np.array([t[0] for t in super(STCSVMPIBuffer, self).spikes(
                populations=population, time_window=time_window) if t[2] == node_id])

        population = population if population is not None else self._default_population
        self.flush()
        comm_barrier()

        if on_rank == 'all':
            timestamps = self._gather_times(node_id=node_id, population=population)
        elif on_rank == 'root':
            timestamps = self._gather_times(node_id, population) if MPI_rank == 0 else None
        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

        if time_window is not None:
            timestamps = [t for t in timestamps if time_window[0] <= t <= time_window[1]]

        return timestamps

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, on_rank='all', **kwargs):
        if on_rank == 'local':
            return super(STCSVMPIBuffer, self).spikes(populations=populations, time_window=time_window,
                                                      sort_order=sort_order, **kwargs)
        self.flush()
        comm_barrier()

        if on_rank == 'all':
            return self._sort_helper(populations, time_window, sort_order)
        elif on_rank == 'root':
            if MPI_rank == 0:
                return self._sort_helper(populations, time_window, sort_order)
            else:
                return []

    def _sort_helper(self, populations, time_window, sort_order):
        filter_fnc = _create_filter(populations, time_window)
        if sort_order == SortOrder.by_time or sort_order == SortOrder.by_id:
            for file_name in self._all_cached_files():
                if not os.path.exists(file_name):
                    continue

                self._sort_buffer_file(file_name, sort_order)

            return self._sorted_itr(filter_fnc, 0 if sort_order == SortOrder.by_time else 2)
        else:
            return self._unsorted_itr(filter_fnc)

    def _unsorted_itr(self, filter_fnc):
        for fn in self._all_cached_files():
            if not os.path.exists(fn):
                continue

            with open(fn, 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=' ')
                for row in csv_reader:
                    t = float(row[0])
                    p = row[1]
                    if filter_fnc(p=p, t=t):
                        yield t, p, int(row[2])

        return

    def _sorted_itr(self, filter_fnc, sort_col):
        """Iterates through all the spikes on each rank, returning them in the specified order"""
        import heapq

        def next_row(csv_reader):
            try:
                rn = next(csv_reader)
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

            if filter_fnc(row[1], row[2]):
                yield row


class STCSVMPIBufferV2(STCSVMPIBuffer):
    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, on_rank='all',
                     **kwargs):
        if on_rank == 'local':
            return super(STCSVMPIBufferV2, self).to_dataframe(populations=populations, sort_order=populations,
                                                              with_population_col=with_population_col, **kwargs)

        if np.isscalar(populations):
            populations = [populations]  # so we can use dataframe.isin() later

        ret_df = None
        self.flush()
        comm_barrier()
        if on_rank == 'all':
            cached_files = self._all_cached_files()
        elif on_rank == 'root':
            cached_files = self._all_cached_files() if MPI_rank == 0 else []
        else:
            raise ValueError('Invalid option "{}" for mpi on_rank parameter'.format(on_rank))

        for file_name in cached_files:
            if not os.path.exists(file_name):
                continue

            df = pd.read_csv(file_name, sep=' ', names=[col_timestamps, col_population, col_node_ids])
            if populations is not None:
                df = df[df[col_population].isin(populations)]

            if not with_population_col:
                df.drop(col_population, axis=1)
            ret_df = df if ret_df is None else ret_df.append(df)

        if ret_df is not None:
            # pandas doesn't always do a good job of reading in the correct dtype for each column
            ret_df = ret_df.astype({col_timestamps: float, col_node_ids: np.int64})

            if sort_order == SortOrder.by_time:
                ret_df = ret_df.sort_values(col_timestamps)
            elif sort_order == SortOrder.by_id:
                ret_df = ret_df.sort_values(col_node_ids)

        comm_barrier()
        return ret_df
