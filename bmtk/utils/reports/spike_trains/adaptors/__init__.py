from .csv_adaptors import CSVSTReader, write_csv
from .sonata_adaptors import write_sonata, load_sonata_file
from .nwb_adaptors import NWBSTReader


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