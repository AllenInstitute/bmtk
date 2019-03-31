from .core import SortOrder as sort_order
from .core import pop_na
from .nwb_adaptors import NWBSTReader
from .csv_adaptors import CSVSTReader, write_csv
from .sonata_adaptors import SonataSTReader, write_sonata


class SpikeTrains(object):
    def __init__(self, read_adaptor=None, write_adaptor=None, **kwargs):
        self._read_adaptor = read_adaptor
        self._write_adaptor = write_adaptor

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

    def to_csv(self, path, mode='w', sort_order=sort_order.none, **kwargs):
        write_csv(path=path, spiketrain_reader=self.read_adaptor, mode=mode, sort_order=sort_order, **kwargs)

    def to_sonata(self, path, mode='w', sort_order=sort_order.none, **kwargs):
        write_sonata(path=path, spiketrain_reader=self.read_adaptor, mode=mode, sort_order=sort_order, **kwargs)

    def to_nwb(self, path, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.read_adaptor)
