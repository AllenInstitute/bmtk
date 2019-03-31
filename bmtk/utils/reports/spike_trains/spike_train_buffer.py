import os
import numpy as np
import six

from .core import pop_na, STReader

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

class STWriter(object):
    def add_spike(self, node_id, timestamp, population=None):
        pass

    def add_spikes(self, nodes, timestamps, population=None):
        pass

    def import_spikes(self, obj):
        pass

    def flush(self):
        pass


class DiskBuffer(object):
    def __init__(self, cache_dir, **kwargs):
        self.cache_dir = cache_dir
        self.mpi_rank = kwargs.get('MPI_rank', MPI_rank)
        self.mpi_size = kwargs.get('MPI_size', MPI_size)
        self.cached_fname = '.spikes.cache.node{}.csv'.format(self.mpi_rank)

        if self.mpi_rank == 0:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
        barrier()

        self.cached_fhandle = open(os.path.join(cache_dir, self.cached_fname), 'w')

    def write(self, timestamp, population, node_id):
        self.cached_fhandle.write('{} {} {}\n'.format(timestamp, population, node_id))


class MemoryBuffer(object):
    pass


class STBufferedWriter(STWriter, STReader):
    def __init__(self, buffer_dir=None, populations=None, **kwargs):
        if populations is None or isinstance(populations, six.string_types) or np.isscalar(populations):
            self.default_pop = pop_na
        elif len(populations) == 1:
            self.default_pop = populations[0]
        else:
            self.default_pop = Exception

        self._buffer = DiskBuffer('cache')

    def add_spike(self, node_id, timestamp, population=None):
        self._buffer.write(timestamp=timestamp, population=population, node_id=node_id)

    def add_spikes(self, node_ids, timestamps, population=None):
        if population is None:
            population = self.default_pop

        if np.isscalar(node_ids):
            for ts in timestamps:
                self.add_spike(node_ids, ts, population)
        else:
            for node_id, ts in zip(node_ids, timestamps):
                self.add_spike(node_id, ts, population)

    def import_spikes(self, obj):
        pass

    def flush(self):
        pass
