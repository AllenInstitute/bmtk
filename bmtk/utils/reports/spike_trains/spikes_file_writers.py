import os
import csv
import h5py
import numpy as np

from .core import SortOrder, csv_headers, col_population, find_conversion
from bmtk.utils.io import bmtk_world_comm
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


def write_sonata(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, units='ms',
                 population_renames=None, **kwargs):
    rank = bmtk_world_comm.MPI_rank

    path_dir = os.path.dirname(path)
    if bmtk_world_comm == 0 and path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir)

    spiketrain_reader.flush()
    bmtk_world_comm.barrier()

    populations = spiketrain_reader.populations
    spikes_root = None
    if rank == 0:
        h5 = h5py.File(path, mode=mode)
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        spikes_root = h5.create_group('/spikes') if '/spikes' not in h5 else h5['/spikes']

    for pop_name in populations: # metrics.keys():
        if bmtk_world_comm.MPI_rank == 0 and pop_name in spikes_root:
            # Problem if file already contains /spikes/<pop_name> # TODO: append new data to old spikes?!?
            raise ValueError('sonata file {} already contains a spikes group {}, '.format(path, pop_name) +
                             'skiping(use option mode="w" to overwrite)')

        pop_df = spiketrain_reader.to_dataframe(populations=pop_name, with_population_col=False, sort_order=sort_order,
                                                on_rank='root')
        if rank == 0:
            spikes_pop_grp = spikes_root.create_group(pop_name)
            if sort_order != SortOrder.unknown:
                spikes_pop_grp.attrs['sorting'] = sort_order.value

            spikes_pop_grp.create_dataset('timestamps', data=pop_df['timestamps'])
            spikes_pop_grp.create_dataset('node_ids', data=pop_df['node_ids'])
    bmtk_world_comm.barrier()


def write_sonata_itr(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, units='ms', population_renames=None,
                     **kwargs):
    path_dir = os.path.dirname(path)
    if bmtk_world_comm.MPI_rank == 0 and path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir)

    spiketrain_reader.flush()
    bmtk_world_comm.barrier()

    conv_factor = find_conversion(spiketrain_reader.units, units)
    if bmtk_world_comm.MPI_rank == 0:
        h5 = h5py.File(path, mode=mode)
        add_hdf5_magic(h5)
        add_hdf5_version(h5)
        spikes_root = h5.create_group('/spikes') if '/spikes' not in h5 else h5['/spikes']

    population_renames = population_renames or {}
    for pop_name in spiketrain_reader.populations:
        n_spikes = spiketrain_reader.n_spikes(pop_name)
        if n_spikes <= 0:
            continue

        if bmtk_world_comm.MPI_rank == 0:
            spikes_grp = spikes_root.create_group('{}'.format(population_renames.get(pop_name, pop_name)))
            if sort_order != SortOrder.unknown:
                spikes_grp.attrs['sorting'] = sort_order.value

            timestamps_ds = spikes_grp.create_dataset('timestamps', shape=(n_spikes,), dtype=np.float64)
            timestamps_ds.attrs['units'] = units
            node_ids_ds = spikes_grp.create_dataset('node_ids', shape=(n_spikes,), dtype=np.uint64)

        for i, spk in enumerate(spiketrain_reader.spikes(populations=pop_name, sort_order=sort_order)):
            if bmtk_world_comm.MPI_rank == 0:
                timestamps_ds[i] = spk[0]*conv_factor
                node_ids_ds[i] = spk[2]

    bmtk_world_comm.barrier()


def write_csv(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, include_header=True,
              include_population=True, units='ms', **kwargs):
    path_dir = os.path.dirname(path)
    if bmtk_world_comm.MPI_rank == 0 and path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir)

    df = spiketrain_reader.to_dataframe(sort_order=sort_order, on_rank='root')

    if bmtk_world_comm.MPI_rank == 0:
        df[['timestamps', 'population', 'node_ids']].to_csv(path, header=include_header, index=False, sep=' ')

    bmtk_world_comm.barrier()


def write_csv_itr(path, spiketrain_reader, mode='w', sort_order=SortOrder.none, include_header=True,
                  include_population=True, units='ms', **kwargs):
    path_dir = os.path.dirname(path)
    if bmtk_world_comm.MPI_rank == 0 and path_dir and not os.path.exists(path_dir):
        os.makedirs(path_dir)

    conv_factor = find_conversion(spiketrain_reader.units, units)
    cols_to_print = csv_headers if include_population else [c for c in csv_headers if c != col_population]
    if bmtk_world_comm.MPI_rank == 0:
        f = open(path, mode=mode)
        csv_writer = csv.writer(f, delimiter=' ')
        if include_header:
            csv_writer.writerow(cols_to_print)

    for spk in spiketrain_reader.spikes(sort_order=sort_order):
        if bmtk_world_comm.MPI_rank == 0:
            ts = spk[0]*conv_factor
            c_data = [ts, spk[1], spk[2]] if include_population else [ts, spk[2]]
            csv_writer.writerow(c_data)

    bmtk_world_comm.barrier()
