import os
import csv

import h5py
import pandas as pd
import numpy as np


class SpikeTrainWriter(object):
    class TmpFileMetadata(object):
        def __init__(self, file_name, sort_order=None):
            self.file_name = file_name
            self.sort_order = sort_order

    def __init__(self, tmp_dir, mpi_rank=0, mpi_size=1):
        # For NEST/NEURON based simulations it is prefereable not to use mpi4py, so let the parent simulator determine
        # MPI rank and size
        self._mpi_rank = mpi_rank
        self._mpi_size = mpi_size

        # used to temporary save spike files since for large simulations saving spikes into memory can crash the
        # system. Requires the user to create the directory
        self._tmp_dir = tmp_dir
        if self._tmp_dir is None or not os.path.exists(self._tmp_dir):
            raise Exception('Directory path {} does not exists'.format(self._tmp_dir))
        self._all_tmp_files = [self.TmpFileMetadata(self._get_tmp_filename(r)) for r in range(mpi_size)]
        # TODO: Determine best buffer size.
        self._tmp_file_handle = open(self._all_tmp_files[mpi_rank].file_name, 'w')

        self._tmp_spikes_handles = []  # used when sorting mulitple file
        self._spike_count = -1

    def _get_tmp_filename(self, rank):
        return os.path.join(self._tmp_dir, '_bmtk_tmp_spikes_{}.csv'.format(rank))

    def _count_spikes(self):
        if self._mpi_rank == 0:
            if self._spike_count > -1:
                return self._spike_count

            self._spike_count = 0
            for tmp_file in self._all_tmp_files:
                with open(tmp_file.file_name, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=' ')
                    self._spike_count += sum(1 for _ in csv_reader)

    def _sort_tmp_file(self, filedata, sort_order):
        # For now load spikes into pandas, it's the fastest way but may be an issue with memory
        if sort_order is None or filedata.sort_order == sort_order:
            return

        file_name = filedata.file_name
        tmp_spikes_ds = pd.read_csv(file_name, sep=' ', names=['time', 'gid'])
        tmp_spikes_ds = tmp_spikes_ds.sort_values(by=sort_order)
        tmp_spikes_ds.to_csv(file_name, sep=' ', index=False, header=False)
        filedata.sort_order = sort_order

    def _next_spike(self, rank):
        try:
            val = self._tmp_spikes_handles[rank].next()
            return float(val[0]), int(val[1]), rank
        except StopIteration:
            return None

    def add_spike(self, time, gid):
        self._tmp_file_handle.write('{:.3f} {}\n'.format(time, gid))

    def _sort_files(self, sort_order, sort_column, file_write_fnc):
        self._tmp_spikes_handles = []
        for fdata in self._all_tmp_files:
            self._sort_tmp_file(fdata, sort_order)
            self._tmp_spikes_handles.append(csv.reader(open(fdata.file_name, 'r'), delimiter=' '))

        spikes = []
        for rank in range(self._mpi_size):
            spike = self._next_spike(rank)
            if spike is not None:
                spikes.append(spike)

        # Iterate through all the ranks and find the first spike. Write that spike/gid to the output, then
        # replace that data point with the next spike on the selected rank
        indx = 0
        while spikes:
            # find which rank has the first spike
            selected_index = 0
            selected_val = spikes[0][sort_column]
            for i, spike in enumerate(spikes[1:]):
                if spike[sort_column] < selected_val:
                    selected_index = i + 1
                    selected_val = spike[sort_column]

            # write the spike to the file
            t, gid, rank = spikes.pop(selected_index)
            file_write_fnc(t, gid, indx)
            indx += 1

            # get the next spike on that rank and replace in spikes table
            another_spike = self._next_spike(rank)
            if another_spike is not None:
                spikes.append(another_spike)

    def _merge_files(self, file_write_fnc):
        indx = 0
        for fdata in self._all_tmp_files:
            with open(fdata.file_name, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=' ')
                for row in csv_reader:
                    file_write_fnc(float(row[0]), int(row[1]), indx)
                    indx += 1

    def _to_file(self, file_name, sort_order, file_write_fnc):
        if sort_order is None:
            sort_column = 0
        elif sort_order == 'time':
            sort_column = 0
        elif sort_order == 'gid':
            sort_column = 1
        else:
            raise Exception('Unknown sort order {}'.format(sort_order))

        # TODO: Need to make sure an MPI_Barrier is called beforehand
        self._tmp_file_handle.close()
        if self._mpi_rank == 0:
            if sort_order is not None:
                self._sort_files(sort_order, sort_column, file_write_fnc)
            else:
                self._merge_files(file_write_fnc)

    def to_csv(self, csv_file, sort_order=None):
        if self._mpi_rank == 0:
            # For the single rank case don't just copy the tmp-csv to the new name. It will fail if user calls to_hdf5
            # or to_nwb after calling to_csv.
            csv_handle = open(csv_file, 'w')
            csv_writer = csv.writer(csv_handle, delimiter=' ')
            file_write_fnc = lambda time, gid, indx: csv_writer.writerow([time, gid])
            self._to_file(csv_file, sort_order, file_write_fnc)
            csv_handle.close()

        # TODO: Let user pass in in barrier and use it here

    def to_nwb(self, nwb_file):
        raise NotImplementedError

    def to_hdf5(self, hdf5_file, sort_order=None):
        if self._mpi_rank == 0:
            with h5py.File(hdf5_file, 'w') as h5:
                self._count_spikes()
                spikes_grp = h5.create_group('/spikes')
                spikes_grp.attrs['sorting'] = 'none' if sort_order is None else sort_order
                time_ds = spikes_grp.create_dataset('timestamps', shape=(self._spike_count,), dtype=np.float)
                gid_ds = spikes_grp.create_dataset('gids', shape=(self._spike_count+1,), dtype=np.uint64)

                def file_write_fnc(time, gid, indx):
                    time_ds[indx] = time
                    gid_ds[indx] = gid
                self._to_file(hdf5_file, sort_order, file_write_fnc)

        # TODO: Need to make sure a barrier is used here (before close is called)

    def close(self):
        fname = self._all_tmp_files[self._mpi_rank].file_name
        if os.path.exists(fname):
            os.remove(fname)


class SpikeTrainInput(object):
    def __init__(self, format='nwb', trial=None):
        pass


    def get_spikes(self, gid):
        pass
