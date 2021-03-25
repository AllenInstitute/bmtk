import os
import csv
import pandas as pd
import h5py
import numpy as np

from .base import SimModule
from bmtk.utils.io.ioutils import bmtk_world_comm


class RecordRates(SimModule):
    def __init__(self, csv_file=None, h5_file=None, tmp_dir='output', sort_order='node_id'):
        self._tmp_dir = tmp_dir
        self._csv_file = csv_file if csv_file is None or os.path.isabs(csv_file) else os.path.join(tmp_dir, csv_file)
        self._save_to_csv = csv_file is not None
        self._tmp_csv_file = os.path.join(tmp_dir, '_tmp_rates.{}.csv'.format(bmtk_world_comm.MPI_rank))

        self._tmp_csv_fhandle = open(self._tmp_csv_file, 'w')
        self._tmp_csv_writer = csv.writer(self._tmp_csv_fhandle, delimiter=' ')
        self._tmp_csv_writer.writerow(['node_id', 'population', 'timestamps', 'firing_rates'])

        h5_file = h5_file if h5_file is None or os.path.isabs(h5_file) else os.path.join(tmp_dir, h5_file)
        self._save_to_h5 = h5_file is not None
        self._h5_file = h5_file
        self._timestamps = None
        self._sort_order = sort_order

    def save(self, sim, cell, times, rates):
        if self._timestamps is None:
            self._timestamps = times

        for t, r in zip(times, rates):
            self._tmp_csv_writer.writerow([cell.gid, cell.population, t, r])
        self._tmp_csv_fhandle.flush()

    def finalize(self, sim):
        self._tmp_csv_fhandle.flush()
        self._tmp_csv_fhandle.close()
        bmtk_world_comm.barrier()

        if bmtk_world_comm.MPI_rank == 0:
            # Combine rates across all ranks
            combined_rates_df = None
            for r in range(bmtk_world_comm.MPI_size):
                rank_tmp_rates_path = os.path.join(self._tmp_dir, '_tmp_rates.{}.csv'.format(r))
                rank_rates_df = pd.read_csv(rank_tmp_rates_path, sep=' ')
                combined_rates_df = rank_rates_df if combined_rates_df is None else pd.concat([combined_rates_df,
                                                                                               rank_rates_df])
            combined_rates_df = combined_rates_df.sort_values([self._sort_order, 'population'])

            if self._save_to_h5:
                try:
                    rates_h5 = h5py.File(self._h5_file, 'w')
                    rates_grp = rates_h5.create_group('/firing_rates')

                    for pop, pop_table in combined_rates_df.groupby('population'):
                        pop_grp = rates_grp.create_group(pop)
                        node_ids = pop_table['node_id'].unique().astype(np.uint)
                        n_nodes = len(node_ids)
                        n_timestamps = len(self._timestamps)

                        pop_grp.create_dataset('node_id', data=node_ids)
                        pop_grp.create_dataset('times', data=self._timestamps)
                        pop_grp.create_dataset(
                            'firing_rates_Hz',
                            data=np.reshape(pop_table['firing_rates'].values.astype(np.float), (n_nodes, n_timestamps)).T
                        )

                except Exception as e:
                    print(e)
                    print('Unable to save rates to hdf5')

            if self._save_to_csv:
                combined_rates_df.to_csv(self._csv_file, sep=' ', index=False)

        bmtk_world_comm.barrier()
        os.remove(self._tmp_csv_file)

        bmtk_world_comm.barrier()
