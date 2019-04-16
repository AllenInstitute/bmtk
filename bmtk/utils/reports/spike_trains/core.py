from enum import Enum


class SortOrder(Enum):
    none = 'none'
    by_id = 'by_id'
    by_time = 'by_time'
    unknown = 'unknown'


# convient method for converting different string values
sort_order_lu = {
    'by_time': SortOrder.by_time,
    'time': SortOrder.by_time,
    'by_id': SortOrder.by_id,
    'id': SortOrder.by_id,
    'node_id': SortOrder.by_id,
    'gid': SortOrder.by_id,
    'none': SortOrder.none,
    'na': SortOrder.none
}


col_timestamps = 'timestamps'
col_node_ids = 'node_ids'
col_population = 'population'
csv_headers = [col_timestamps, col_population, col_node_ids]
pop_na = '<sonata:none>'


class STReader(object):
    @property
    def populations(self):
        raise NotImplementedError()

    @property
    def units(self):
        # for backwards comptability assume everything is in milliseconds unless overwritten
        return 'ms'

    @units.setter
    def units(self, v):
        raise NotImplementedError()

    def sort_order(self, population):
        return SortOrder.unknown

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


def find_conversion(units_old, units_new):
    if units_new is None or units_old is None:
        return 1.0

    if units_old == 's' and units_new == 'ms':
        return 1000.

    if units_old == 'ms' and units_new == 's':
        return 0.001

    return 1.0
