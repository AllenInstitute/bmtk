import numpy as np
from six import string_types

from .core import find_file_type
from .spike_train_readers import load_sonata_file, CSVSTReader, NWBSTReader
from .spike_train_buffer import STMemoryBuffer, STCSVBuffer, STMPIBuffer, STCSVMPIBufferV2

from bmtk.utils.sonata.utils import get_node_ids
from bmtk.utils.io import bmtk_world_comm


MPI_size = bmtk_world_comm.MPI_size


class SpikeTrains(object):
    """A class for creating and reading spike files.

    """
    def __init__(self, spikes_adaptor=None, **kwargs):
        # There are a number of strategies for reading and writing spike trains, depending on memory limitations, if
        #  MPI is being used, or if there if read-only from disk. I'm using a decorator/adaptor pattern and moving
        #  the actual functionality to the Buffered/ReadOnly classes that implement SpikeTrainsAPI.
        if spikes_adaptor is not None:
            self.adaptor = spikes_adaptor
        else:
            # TODO: Check that comm has gather, reduce, etc methods; if not can't use STMPIBuffer
            use_mpi = MPI_size > 1
            cache_to_disk = 'cache_dir' in kwargs

            if use_mpi and cache_to_disk:
                self.adaptor = STCSVMPIBufferV2(**kwargs)
            elif cache_to_disk:
                self.adaptor = STCSVBuffer(**kwargs)
            elif use_mpi:
                self.adaptor = STMPIBuffer(**kwargs)
            else:
                self.adaptor = STMemoryBuffer(**kwargs)

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(spikes_adaptor=CSVSTReader(path, **kwargs))

    @classmethod
    def from_sonata(cls, path, **kwargs):
        sonata_adaptor = load_sonata_file(path, **kwargs)
        return cls(spikes_adaptor=sonata_adaptor)

    @classmethod
    def from_nwb(cls, path, **kwargs):
        return cls(spikes_adaptor=NWBSTReader(path, **kwargs))

    @classmethod
    def load(cls, path, file_type=None, **kwargs):
        file_type = file_type.lower() if file_type else find_file_type(path)
        if file_type == 'h5' or file_type == 'sonata':
            return cls.from_sonata(path, **kwargs)
        elif file_type == 'nwb':
            return cls.from_nwb(path, **kwargs)
        elif file_type == 'csv':
            return cls.from_csv(path, **kwargs)

    def __getattr__(self, item):
        return getattr(self.adaptor, item)

    def __setattr__(self, key, value):
        if key == 'adaptor':
            self.__dict__[key] = value
        else:
            self.adaptor.__dict__[key] = value

    def __len__(self):
        return self.adaptor.__len__()

    def __eq__(self, other):
        return self.adaptor.__eq__(other)

    def __lt__(self, other):
        return self.adaptor.__lt__(other)

    def __le__(self, other):
        return self.adaptor.__le__(other)

    def __gt__(self, other):
        return self.adaptor.__gt__(other)

    def __ge__(self, other):
        return self.adaptor.__ge__(other)

    def __ne__(self, other):
        return self.adaptor.__ne__(other)


class PoissonSpikeGenerator(SpikeTrains):
    """ A Class for generating spike-trains with a homogeneous and inhomogeneous Poission distribution.

    Uses the methods describe in Dayan and Abbott, 2001.
    """
    max_spikes_per_node = 10000000

    def __init__(self, population=None, seed=None, **kwargs):
        if population is not None and 'default_population' not in kwargs:
            kwargs['default_population'] = population

        if seed:
            np.random.seed(seed)

        # if buffer_dir is not None:
        #     adaptor = STCSVBuffer(cache_dir=buffer_dir, **kwargs)
        # else:
        #     adaptor = STMemoryBuffer(**kwargs)

        super(PoissonSpikeGenerator, self).__init__(**kwargs)
        self.units = 's'

    def add(self, node_ids, firing_rate, population=None, times=(0.0, 1.0)):
        # TODO: Add refactory period
        if isinstance(node_ids, string_types):
            # if user passes in path to nodes.h5 file count number of nodes
            node_ids = get_node_ids(node_ids, population)
        if np.isscalar(node_ids):
            # In case user passes in single node_id
            node_ids = [node_ids]

        if np.isscalar(firing_rate):
            self._build_fixed_fr(node_ids, population, firing_rate, times)
        elif isinstance(firing_rate, (list, np.ndarray)):
            self._build_inhomegeous_fr(node_ids, population, firing_rate, times)

    def time_range(self, population=None):
        df = self.to_dataframe(populations=population, with_population_col=False)
        timestamps = df['timestamps']
        return np.min(timestamps), np.max(timestamps)

    def _build_fixed_fr(self, node_ids, population, fr, times):
        if np.isscalar(times) and times > 0.0:
            tstart = 0.0
            tstop = times
        else:
            tstart = times[0]
            tstop = times[-1]
            if tstart >= tstop:
                raise ValueError('Invalid start and stop times.')

        for node_id in node_ids:
            c_time = tstart
            while True:
                interval = -np.log(1.0 - np.random.uniform()) / fr
                c_time += interval
                if c_time > tstop:
                    break

                self.add_spike(node_id=node_id, population=population, timestamp=c_time)

    def _build_inhomegeous_fr(self, node_ids, population, fr, times):
        if np.min(fr) <= 0:
            raise Exception('Firing rates must not be negative')
        max_fr = np.max(fr)

        times = times
        tstart = times[0]
        tstop = times[-1]

        for node_id in node_ids:
            c_time = tstart
            time_indx = 0
            while True:
                # Using the prunning method, see Dayan and Abbott Ch 2
                interval = -np.log(1.0 - np.random.uniform()) / max_fr
                c_time += interval
                if c_time > tstop:
                    break

                # A spike occurs at t_i, find index j st times[j-1] < t_i < times[j], and interpolate the firing rates
                # using fr[j-1] and fr[j]
                while times[time_indx] <= c_time:
                    time_indx += 1

                fr_i = _interpolate_fr(c_time, times[time_indx-1], times[time_indx],
                                       fr[time_indx-1], fr[time_indx])

                if not fr_i/max_fr < np.random.uniform():
                    self.add_spike(node_id=node_id, population=population, timestamp=c_time)


def _interpolate_fr(t, t0, t1, fr0, fr1):
    # Used to interpolate the firing rate at time t from a discrete list of firing rates
    return fr0 + (fr1 - fr0)*(t - t0)/t1
