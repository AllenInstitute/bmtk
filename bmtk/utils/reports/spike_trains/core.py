from enum import Enum


class SortOrder(Enum):
    none = 'none'
    by_id = 'by_id'
    by_time = 'by_time'
    unknown = 'unknown'


col_timestamps = 'timestamps'
col_node_ids = 'node_ids'
col_population = 'population'
csv_headers = [col_timestamps, col_population, col_node_ids]
pop_na = '<sonata:none>'


class STReader(object):
    @property
    def populations(self):
        raise NotImplementedError()

    def nodes(self, populations=None):
        raise NotImplementedError()

    def n_spikes(self, population=None):
        raise NotImplementedError()

    def time_range(self, populations=None):
        raise NotImplementedError()

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        raise NotImplementedError()

    def to_dataframe(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def spikes(self, node_ids=None, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        raise NotImplementedError()

    def __len__(self):
        return len(self.to_dataframe())


class STBuffer(object):
    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        raise NotImplementedError()

    def add_spikes(self, nodes, timestamps, population=None, **kwargs):
        raise NotImplementedError()

    def import_spikes(self, obj, **kwargs):
        raise NotImplementedError()

    def flush(self):
        raise NotImplementedError()




