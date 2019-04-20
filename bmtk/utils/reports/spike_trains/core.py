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
    """The interface for reading a SONATA formatted spike-train files or objects.

    """

    @property
    def populations(self):
        """Get all available spike population names

        :return: A list of strings
        """
        raise NotImplementedError()

    @property
    def units(self):
        """Returns the units used in the timestamps.

        :return: str
        """

        # for backwards comptability assume everything is in milliseconds unless overwritten
        # TODO: Use an enum/struct to pre-define the avilable units.
        # TODO: Different populations may use different units.
        return 'ms'

    @units.setter
    def units(self, v):
        raise NotImplementedError()

    def sort_order(self, population):
        return SortOrder.unknown

    def nodes(self, populations=None):
        """ Returns a list of (node-ids, population_name).

        :param populations: Name of population
        :return:
        """
        raise NotImplementedError()

    def n_spikes(self, population=None):
        """Get the number of spikes for the given population.

        :param population: population name. If none None will use the default population (when possible).
        :return: unsigned integer, number of spikes.
        """
        raise NotImplementedError()

    def time_range(self, populations=None):
        """Get the time range of the firing, i.e the min and max spike time, for the specified populations. If more
        than one node population is specified will search across all and return values that encompasses all spikes
        across each population.

        :param populations: None, A string, or a list of strings. Default None will search across all populations.
        :return: tuple of integers, (min-spike-time, max-spike-time). If No values found returns None
        """
        raise NotImplementedError()

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        """Returns a list of spike-times for a given node.

        :param node_id: The id of the node
        :param population: Name of the node-population which the node belongs to. By default will try to use the
        default population (if possible).
        :param time_window: A tuple (min-time, max-time) to limit the returned spikes. By default returns all spikes.
        :param kwargs:
        :return: list of spike times [float]
        """

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

    def close(self):
        raise NotImplementedError()


def find_conversion(units_old, units_new):
    if units_new is None or units_old is None:
        return 1.0

    if units_old == 's' and units_new == 'ms':
        return 1000.

    if units_old == 'ms' and units_new == 's':
        return 0.001

    return 1.0
