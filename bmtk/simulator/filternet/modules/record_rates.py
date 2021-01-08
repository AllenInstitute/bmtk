import os
import csv
import pandas as pd
import h5py
import numpy as np

from .base import SimModule


class RecordRates(SimModule):
    def __init__(self, csv_file=None, h5_file=None, tmp_dir='output'):
        csv_file = csv_file if csv_file is None or os.path.isabs(csv_file) else os.path.join(tmp_dir, csv_file)
        self._save_to_csv = csv_file is not None
        self._tmp_csv_file = csv_file if self._save_to_csv else os.path.join(tmp_dir, '__tmp_rates.csv')

        self._tmp_csv_fhandle = open(self._tmp_csv_file, 'w')
        self._tmp_csv_writer = csv.writer(self._tmp_csv_fhandle, delimiter=' ')
        self._tmp_csv_writer.writerow(['node_id', 'population', 'timestamps', 'firing_rates'])

        h5_file = h5_file if h5_file is None or os.path.isabs(h5_file) else os.path.join(tmp_dir, h5_file)
        self._save_to_h5 = h5_file is not None
        self._h5_file = h5_file
        self._timestamps = None

    def save(self, sim, cell, times, rates):
        if self._timestamps is None:
            self._timestamps = times

        for t, r in zip(times, rates):
            self._tmp_csv_writer.writerow([cell.gid, cell.population, t, r])
        self._tmp_csv_fhandle.flush()

    def finalize(self, sim):
        self._tmp_csv_fhandle.flush()
        self._tmp_csv_fhandle.close()

        if self._save_to_h5:
            try:
                rates_df = pd.read_csv(self._tmp_csv_file, sep=' ')
                rates_h5 = h5py.File(self._h5_file, 'w')
                rates_grp = rates_h5.create_group('/firing_rates')

                for pop, pop_table in rates_df.groupby('population'):
                    pop_grp = rates_grp.create_group(pop)
                    node_ids = pop_table['node_id'].unique().astype(np.uint)
                    n_nodes = len(node_ids)
                    n_timestamps = len(self._timestamps)

                    pop_grp.create_dataset('mapping/node_ids', data=node_ids)
                    pop_grp.create_dataset('mapping/timestamps', data=self._timestamps)
                    pop_grp.create_dataset(
                        'data',
                        data=np.reshape(pop_table['firing_rates'].values.astype(np.float), (n_nodes, n_timestamps)).T
                    )

            except Exception as e:
                print(e)
                print('Unable to save rates to hdf5')

        if not self._save_to_csv:
            os.remove(self._tmp_csv_file)
