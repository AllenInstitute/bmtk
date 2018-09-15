import os
import csv

from .base import SimModule


class RecordRates(SimModule):
    def __init__(self, csv_file=None, h5_file=None, tmp_dir='output'):
        self._save_to_csv = csv_file is not None
        self._tmp_csv_file = csv_file if self._save_to_csv else os.path.join(tmp_dir, '__tmp_rates.csv')
        self._tmp_csv_fhandle = open(self._tmp_csv_file, 'w')
        self._tmp_csv_writer = csv.writer(self._tmp_csv_fhandle, delimiter=' ')

        self._save_to_h5 = h5_file is not None

    def save(self, sim, gid, times, rates):
        for t, r in zip(times, rates):
            self._tmp_csv_writer.writerow([gid, t, r])
        self._tmp_csv_fhandle.flush()

    def finalize(self, sim):
        if self._save_to_h5:
            raise NotImplementedError

        self._tmp_csv_fhandle.close()
        if not self._save_to_csv:
            os.remove(self._tmp_csv_file)
