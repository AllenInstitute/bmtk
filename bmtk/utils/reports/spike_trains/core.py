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


def find_conversion(units_old, units_new):
    if units_new is None or units_old is None:
        return 1.0

    if units_old == 's' and units_new == 'ms':
        return 1000.

    if units_old == 'ms' and units_new == 's':
        return 0.001

    return 1.0


def find_file_type(path):
    """Tries to find the input type (sonata/h5, NWB, CSV) from the file-name"""
    if path is None:
        return ''

    path = path.lower()
    if path.endswith('.hdf5') or path.endswith('.hdf') or path.endswith('h5') or path.endswith('.sonata'):
        return 'h5'

    elif path.endswith('.nwb'):
        return 'nwb'

    elif path.endswith('.csv'):
        return 'csv'
