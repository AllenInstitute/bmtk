import numpy as np

from .core import SortOrder as sort_order
from .core import pop_na
from .nwb_adaptors import NWBSTReader
from .csv_adaptors import CSVSTReader, write_csv
from .spike_train_buffer import STMemoryBuffer, STCSVBuffer
from .sonata_adaptors import SonataSTReader, write_sonata


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


class SpikeTrains(object):
    def __init__(self, read_adaptor=None, write_adaptor=None, **kwargs):
        if write_adaptor is None:
            self._write_adaptor = STCSVBuffer(**kwargs)
        else:
            self._write_adaptor = write_adaptor

        if read_adaptor is None:
            self._read_adaptor = self._write_adaptor
        else:
            self._read_adaptor = read_adaptor

    @property
    def write_adaptor(self):
        return self._write_adaptor

    @property
    def read_adaptor(self):
        return self._read_adaptor

    @property
    def populations(self):
        return self.read_adaptor.populations

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(read_adaptor=CSVSTReader(path, **kwargs))

    @classmethod
    def from_sonata(cls, path, **kwargs):
        return cls(read_adaptor=SonataSTReader(path, **kwargs))
        # return SONATASTReader(path, **kwargs)

    @classmethod
    def from_nwb(cls, path, **kwargs):
        return cls(read_adaptor=NWBSTReader(path, **kwargs))
        # return NWBSTReader(path, **kwargs)

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
        self._write_adaptor.flush()

    def close(self):
        self._write_adaptor.close()

    def to_csv(self, path, mode='w', sort_order=sort_order.none, **kwargs):
        # self._write_adaptor.flush()
        if MPI_rank == 0:
            write_csv(path=path, spiketrain_reader=self.read_adaptor, mode=mode, sort_order=sort_order, **kwargs)

    def to_sonata(self, path, mode='w', sort_order=sort_order.none, **kwargs):
        self._write_adaptor.flush()
        if MPI_rank == 0:
            write_sonata(path=path, spiketrain_reader=self.read_adaptor, mode=mode, sort_order=sort_order, **kwargs)

    def to_nwb(self, path, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.read_adaptor)


class PoissonSpikeGenerator(SpikeTrains):
    max_spikes_per_node = 10000000

    def __init__(self, node_ids, population, firing_rate, times=(0.0, 1.0), buffer_dir=None):
        if buffer_dir is not None:
            adaptor = STCSVBuffer(cache_dir=buffer_dir, default_population=population)
        else:
            adaptor = STMemoryBuffer(default_population=population)

        self._node_ids = node_ids
        self._times = times

        super(PoissonSpikeGenerator, self).__init__(read_adaptor=adaptor, write_adaptor=adaptor)
        if np.isscalar(firing_rate):
            self._build_fixed_fr(firing_rate)

        elif isinstance(firing_rate, (list, np.ndarray)):
            self._build_inhomegeous_fr(firing_rate)

    def _build_fixed_fr(self, fr):
        if np.isscalar(self._times) and self._times > 0.0:
            tstart = 0
            tstop = self._times
        else:
            tstart = self._times[0]
            tstop = self._times[-1]
            if tstart >= tstop:
                raise ValueError('Invalid start and stop times.')

        for node_id in self._node_ids:
            c_time = tstart
            while True:
                interval = -np.log(1.0 - np.random.uniform()) / fr
                c_time += interval
                if c_time > tstop:
                    break

                self.add_spike(node_id=node_id, timestamp=c_time)

    def _build_inhomegeous_fr(self, fr):
        if np.min(fr) <= 0:
            raise Exception('Firing rates must not be negative')
        max_fr = np.max(fr)

        tstart = self._times[0]
        tstop = self._times[-1]
        time_indx = 0

        for node_id in self._node_ids:
            c_time = tstart
            while True:
                interval = -np.log(1.0 - np.random.uniform()) / max_fr
                c_time += interval
                if c_time > tstop:
                    break

                while self._times[time_indx] <= c_time:
                    time_indx += 1

                fr_i = _interpolate_fr(c_time, self._times[time_indx-1], self._times[time_indx],
                                       fr[time_indx-1], fr[time_indx])

                if not fr_i/max_fr < np.random.uniform():
                    self.add_spike(node_id=node_id, timestamp=c_time)


def _interpolate_fr(t, t0, t1, fr0, fr1):
    # print t, t0, t1, fr0, fr1
    return fr0 + (fr1 - fr0)*(t - t0)/t1

