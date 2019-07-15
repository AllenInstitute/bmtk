import numpy as np
from six import string_types

from .core import SortOrder as sort_order
from .core import STBuffer
from .adaptors import CSVSTReader, write_csv
from .adaptors import load_sonata_file, write_sonata
from .adaptors import NWBSTReader
from .adaptors import find_file_type
from .spike_train_buffer import STMemoryBuffer, STCSVBuffer, STMPIBuffer

from bmtk.utils.sonata.utils import get_node_ids
from bmtk.utils.io import bmtk_world_comm


class SpikeTrains(object):
    def __init__(self, adaptor=None, **kwargs):
        if adaptor is None:
            if bmtk_world_comm.MPI_size > 1:
                self._adaptor = STMPIBuffer(**kwargs)
            else:
                self._adaptor = STCSVBuffer(**kwargs)
        else:
            self._adaptor = adaptor

        #self._read_adaptor = self._write_adaptor = self._adaptor

    @property
    def write_adaptor(self):
        return self._adaptor

    @property
    def read_adaptor(self):
        return self._adaptor

    @property
    def populations(self):
        return self.read_adaptor.populations

    @property
    def units(self):
        return self.read_adaptor.units

    @units.setter
    def units(self, v):
        self.read_adaptor.units = v

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(adaptor=CSVSTReader(path, **kwargs))

    @classmethod
    def from_sonata(cls, path, **kwargs):
        sonata_adaptor = load_sonata_file(path, **kwargs)
        return cls(adaptor=sonata_adaptor)

    @classmethod
    def from_nwb(cls, path, **kwargs):
        return cls(adaptor=NWBSTReader(path, **kwargs))
        # return NWBSTReader(path, **kwargs)

    @classmethod
    def load(cls, path, file_type=None, **kwargs):
        file_type = file_type.lower() if file_type else find_file_type(path)
        if file_type == 'h5' or file_type == 'sonata':
            return cls.from_sonata(path, **kwargs)
        elif file_type == 'nwb':
            return cls.from_nwb(path, **kwargs)
        elif file_type == 'csv':
            return cls.from_csv(path, **kwargs)

    def nodes(self, populations=None):
        return self.read_adaptor.nodes(populations=populations)

    def n_spikes(self, population=None):
        return self.read_adaptor.n_spikes(population=population)

    def time_range(self, populations=None):
        return self.read_adaptor.time_range(populations=populations)

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        return self.read_adaptor.get_times(node_id=node_id, population=population, time_window=time_window, **kwargs)

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=sort_order.none, **kwargs):
        return self.read_adaptor.to_dataframe(node_ids=node_ids, populations=populations, time_window=time_window,
                                              sort_order=sort_order, **kwargs)

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=sort_order.none, **kwargs):
        return self.read_adaptor.spikes(node_ids=node_ids, populations=populations, time_window=time_window,
                                        sort_order=sort_order, **kwargs)

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        return self.write_adaptor.add_spike(node_id=node_id, timestamp=timestamp, population=population, **kwargs)

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        return self.write_adaptor.add_spikes(node_ids=node_ids, timestamps=timestamps, population=population, **kwargs)

    def import_spikes(self, obj, **kwargs):
        raise NotImplementedError()

    def flush(self):
        self.write_adaptor.flush()

    def close(self):
        self.write_adaptor.close()

    def to_csv(self, path, mode='w', sort_order=sort_order.none, **kwargs):
        # self._write_adaptor.flush()
        if bmtk_world_comm.MPI_rank == 0:
            write_csv(path=path, spiketrain_reader=self.read_adaptor, mode=mode, sort_order=sort_order, **kwargs)

    def to_sonata(self, path, mode='a', sort_order=sort_order.none, **kwargs):
        if isinstance(self.write_adaptor, STBuffer):
            self.write_adaptor.flush()
        if bmtk_world_comm.MPI_rank == 0:
            write_sonata(path=path, spiketrain_reader=self.read_adaptor, mode=mode, sort_order=sort_order, **kwargs)
        bmtk_world_comm.barrier()

    def is_equal(self, other, populations=None, err=0.00001, time_window=None):
        if populations is None:
            # Both must contain the same populations
            populations = self.populations
            if set(other.populations) != set(populations):
                return False
        else:
            # Comparing only a subset of the node populations, make sure both files contains them (or both files don't
            # contain the populations
            populations = [populations] if np.isscalar(populations) else populations
            for p in populations:
                if (p in self.populations) != (p in other.populations):
                    return False

        for p in populations:
            if time_window is None:
                # check that each SpikeTrains contain the same number and ids of nodes so we don't have to iterate
                # through each spike. This won't always work if the user limits the time-window.
                self_nodes = sorted([n[1] for n in self.nodes(populations=p)])
                other_nodes = sorted([n[1] for n in other.nodes(populations=p)])
                if not np.all(self_nodes == other_nodes):
                    return False
            else:
                # If the time-window being checked is restricted
                self_nodes = set([n[1] for n in self.nodes(p)]) & set([n[1] for n in other.nodes(p)])

            for node_id in self_nodes:
                self_ts = self.get_times(node_id=node_id, population=p, time_window=time_window)
                other_ts = other.get_times(node_id=node_id, population=p, time_window=time_window)

                if len(self_ts) != len(other_ts):
                    return False

                for t0, t1 in zip(self_ts, other_ts):
                    if abs(t1 - t0) > err:
                        return False

        return True

    def to_nwb(self, path, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.read_adaptor)

    def __eq__(self, other):
        return self.is_equal(other)


class PoissonSpikeGenerator(SpikeTrains):
    """ A Class for generating spike-trains with a homogeneous and inhomogeneous Poission distribution.

    Uses the methods describe in Dayan and Abbott, 2001.
    """
    max_spikes_per_node = 10000000

    def __init__(self, buffer_dir=None, population=None, seed=None, **kwargs):
        if population is not None and 'default_population' not in kwargs:
            kwargs['default_population'] = population

        if seed:
            np.random.seed(seed)

        if buffer_dir is not None:
            adaptor = STCSVBuffer(cache_dir=buffer_dir, **kwargs)
        else:
            adaptor = STMemoryBuffer(**kwargs)

        super(PoissonSpikeGenerator, self).__init__(adaptor=adaptor)
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
