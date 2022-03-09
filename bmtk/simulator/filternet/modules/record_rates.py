import os
import csv
import pandas as pd
import h5py
import numpy as np
import glob

from .base import SimModule
from bmtk.utils.io.ioutils import bmtk_world_comm


class RecordRates(SimModule):
    def __init__(self, csv_file=None, h5_file=None, tmp_dir='output', sort_order='node_id'):
        self._tmp_dir = tmp_dir
        self._csv_file = csv_file if csv_file is None or os.path.isabs(csv_file) else os.path.join(tmp_dir, csv_file)
        self._save_to_csv = csv_file is not None
        self._tmp_rates_path = None

        h5_file = h5_file if h5_file is None or os.path.isabs(h5_file) else os.path.join(tmp_dir, h5_file)
        self._save_to_h5 = h5_file is not None
        self._h5_file = h5_file

        self._sort_order = sort_order
        self._n_nodes = 0
        self._n_timesteps = 0

        self._timestamps = None
        self._node_ids = {}
        self._firing_rates = {}
        self._node_counter = 0

    def initialize(self, sim):
        self._node_counter = 0
        self._n_nodes = len(sim.local_cells())
        # self._node_ids = {} # np.zeros(len(sim.local_cells()), dtype=np.uint)

    def save(self, sim, cell, times, rates):
        if self._timestamps is None:
            self._n_timesteps = len(times)
            self._timestamps = times

        if cell.population not in self._firing_rates:
            self._firing_rates[cell.population] = np.zeros((self._n_nodes, self._n_timesteps), dtype=np.float)
            self._node_ids[cell.population] = np.zeros(self._n_nodes, dtype=np.uint)

        self._firing_rates[cell.population][self._node_counter, :] = rates
        self._node_ids[cell.population][self._node_counter] = cell.node_id
        self._node_counter += 1

    def finalize(self, sim):
        if bmtk_world_comm.MPI_size > 1:
            self._tmp_rates_path = os.path.join(self._tmp_dir, '.rates.{}.h5'.format(bmtk_world_comm.MPI_rank))
            self._write_rates_on_rank()
            bmtk_world_comm.barrier()

            self._combine_rates()
            bmtk_world_comm.barrier()

        if bmtk_world_comm.MPI_rank == 0:
            if self._sort_order in ['node_id', 'node_ids']:
                for pop in self._firing_rates.keys():
                    index_order = np.argsort(self._node_ids[pop])
                    self._node_ids[pop] = self._node_ids[pop][index_order]
                    self._firing_rates[pop] = self._firing_rates[pop][index_order, :]

            if self._save_to_h5:
                try:
                    rates_h5 = h5py.File(self._h5_file, 'w')
                    rates_grp = rates_h5.create_group('/firing_rates')
                    for pop, pop_table in self._firing_rates.items():
                        pop_grp = rates_grp.create_group(pop)
                        pop_grp.create_dataset('node_id', data=self._node_ids[pop])
                        pop_grp.create_dataset('times', data=self._timestamps)
                        pop_grp.create_dataset('firing_rates_Hz', data=self._firing_rates[pop].T)

                except Exception as e:
                    print(e)
                    print('Unable to save rates to hdf5')

            if self._save_to_csv:
                csv_fhandle = open(self._csv_file, 'w')
                csv_writer = csv.writer(csv_fhandle, delimiter=' ')
                csv_writer.writerow(['node_id', 'population', 'timestamps', 'firing_rates'])

                for pop in self._firing_rates.keys():
                    for i, node_id in enumerate(self._node_ids[pop]):
                        for ts, fr in zip(self._timestamps, self._firing_rates[pop][i, :]):
                            csv_writer.writerow([node_id, pop, ts, fr])

        bmtk_world_comm.barrier()
        self._clean()

    def _write_rates_on_rank(self):
        with h5py.File(self._tmp_rates_path, 'w') as h5:
            for pop in self._firing_rates.keys():
                pop_grp = h5.create_group(pop)
                pop_grp.create_dataset('time', data=self._timestamps)
                pop_grp.create_dataset('node_id', data=self._node_ids[pop])
                pop_grp.create_dataset('firing_rates_Hz', data=self._firing_rates[pop])

    def _combine_rates(self):
        n_cells = {}
        if bmtk_world_comm.MPI_rank == 0:
            rates_paths = glob.glob(os.path.join(self._tmp_dir, '.rates.*.h5'))
            h5_handles = []
            timestamps = None

            for rp in rates_paths:
                rates_h5 = h5py.File(rp, 'r')
                h5_handles.append(rates_h5)
                for pop, pop_grp in rates_h5.items():
                    if pop not in n_cells:
                        n_cells[pop] = 0

                    n_cells[pop] += pop_grp['firing_rates_Hz'].shape[0]
                    if timestamps is None:
                        timestamps = pop_grp['time'][()]
                    else:
                        assert (np.allclose(timestamps, pop_grp['time'][()]))

            n_timestamps = len(timestamps)

            self._firing_rates = {pop: np.zeros((n_cells[pop], n_timestamps), dtype=np.float) for pop in n_cells.keys()}
            self._node_ids = {pop: np.zeros(n_cells[pop], dtype=np.uint32) for pop in n_cells.keys()}
            beg_indices = {pop: 0 for pop in n_cells.keys()}

            for h5 in h5_handles:
                for pop, pop_grp in h5.items():
                    beg_index = beg_indices[pop]
                    end_index = beg_index + pop_grp['node_id'].shape[0]
                    self._firing_rates[pop][beg_index:end_index, :] = pop_grp['firing_rates_Hz'] # [:, :]
                    self._node_ids[pop][beg_index:end_index] = pop_grp['node_id'][:]
                    beg_indices[pop] = end_index

    def _clean(self):
        if self._tmp_rates_path is not None and os.path.exists(self._tmp_rates_path):
            try:
                os.remove(self._tmp_rates_path)
            except Exception as e:
                pass
