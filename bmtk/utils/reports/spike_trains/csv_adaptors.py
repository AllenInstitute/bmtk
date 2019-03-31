import os
import csv
from .core import STReader, SortOrder
from .core import csv_headers


def write_csv(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, **kwargs):
    path_dir = os.path.dirname(path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(path, mode=mode) as f:
        csv_writer = csv.writer(f, delimiter=' ')
        csv_writer.writerow(csv_headers)
        for spk in spiketrain_reader.spikes(sort_order=sort_order):
            csv_writer.writerow([spk[0], spk[1], spk[2]])
