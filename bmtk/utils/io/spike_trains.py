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

        # Nest gid files uses tab seperators and a different order for tmp spike files.
        self.delimiter = ' '  # delimiter for temporary file
        self.time_col = 0
        self.gid_col = 1

    def _get_tmp_filename(self, rank):
        return os.path.join(self._tmp_dir, '_bmtk_tmp_spikes_{}.csv'.format(rank))

    def _count_spikes(self):
        if self._mpi_rank == 0:
            if self._spike_count > -1:
                return self._spike_count

            self._spike_count = 0
            for tmp_file in self._all_tmp_files:
                with open(tmp_file.file_name, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile, delimiter=self.delimiter)
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
            val = next(self._tmp_spikes_handles[rank])
            return val[0], val[1], rank
        except StopIteration:
            return None

    def add_spike(self, time, gid):
        self._tmp_file_handle.write('{:.6f} {}\n'.format(time, gid))

    def add_spikes(self, times, gid):
        for t in times:
            self.add_spike(t, gid)

    def add_spikes_file(self, file_name, sort_order=None):
        self._all_tmp_files.append(self.TmpFileMetadata(file_name, sort_order))

    def _sort_files(self, sort_order, sort_column, file_write_fnc):
        self._tmp_spikes_handles = []
        for fdata in self._all_tmp_files:
            self._sort_tmp_file(fdata, sort_order)
            self._tmp_spikes_handles.append(csv.reader(open(fdata.file_name, 'r'), delimiter=self.delimiter))

        spikes = []
        for rank in range(len(self._tmp_spikes_handles)):  # range(self._mpi_size):
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
            row = spikes.pop(selected_index)
            file_write_fnc(float(row[self.time_col]), int(row[self.gid_col]), indx)
            indx += 1

            # get the next spike on that rank and replace in spikes table
            another_spike = self._next_spike(row[2])
            if another_spike is not None:
                spikes.append(another_spike)

    def _merge_files(self, file_write_fnc):
        indx = 0
        for fdata in self._all_tmp_files:
            if not os.path.exists(fdata.file_name):
                continue

            with open(fdata.file_name, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=self.delimiter)
                for row in csv_reader:
                    file_write_fnc(float(row[self.time_col]), int(row[self.gid_col]), indx)
                    indx += 1

    def _to_file(self, file_name, sort_order, file_write_fnc):
        if sort_order is None:
            sort_column = 0
        elif sort_order == 'time':
            sort_column = self.time_col
        elif sort_order == 'gid':
            sort_column = self.gid_col
        else:
            raise Exception('Unknown sort order {}'.format(sort_order))

        # TODO: Need to make sure an MPI_Barrier is called beforehand
        self._tmp_file_handle.close()
        if self._mpi_rank == 0:
            if sort_order is not None:
                self._sort_files(sort_order, sort_column, file_write_fnc)
            else:
                self._merge_files(file_write_fnc)

    def to_csv(self, csv_file, sort_order=None, gid_map=None):
        # TODO: Need to call flush and then barrier
        if self._mpi_rank == 0:
            # For the single rank case don't just copy the tmp-csv to the new name. It will fail if user calls to_hdf5
            # or to_nwb after calling to_csv.
            self._count_spikes()
            csv_handle = open(csv_file, 'w')
            csv_writer = csv.writer(csv_handle, delimiter=' ')

            def file_write_fnc_identity(time, gid, indx):
                csv_writer.writerow([time, gid])

            def file_write_fnc_transform(time, gid, indx):
                # For the case when NEURON/NEST ids don't match with the user's gid table
                csv_writer.writerow([time, gid_map[gid]])

            file_write_fnc = file_write_fnc_identity if gid_map is None else file_write_fnc_transform
            self._to_file(csv_file, sort_order, file_write_fnc)
            csv_handle.close()

        # TODO: Let user pass in in barrier and use it here

    def to_nwb(self, nwb_file):
        raise NotImplementedError

    def to_hdf5(self, hdf5_file, sort_order=None, gid_map=None):
        if self._mpi_rank == 0:
            with h5py.File(hdf5_file, 'w') as h5:
                self._count_spikes()
                spikes_grp = h5.create_group('/spikes')
                spikes_grp.attrs['sorting'] = 'none' if sort_order is None else sort_order
                time_ds = spikes_grp.create_dataset('timestamps', shape=(self._spike_count,), dtype=np.float)
                gid_ds = spikes_grp.create_dataset('gids', shape=(self._spike_count,), dtype=np.uint64)

                def file_write_fnc_identity(time, gid, indx):
                    time_ds[indx] = time
                    gid_ds[indx] = gid

                def file_write_fnc_transform(time, gid, indx):
                    time_ds[indx] = time
                    gid_ds[indx] = gid_map[gid]

                file_write_fnc = file_write_fnc_identity if gid_map is None else file_write_fnc_transform
                self._to_file(hdf5_file, sort_order, file_write_fnc)

        # TODO: Need to make sure a barrier is used here (before close is called)

    def flush(self):
        self._tmp_file_handle.flush()

    def close(self):
        if self._mpi_rank == 0:
            for tmp_file in self._all_tmp_files:
                if os.path.exists(tmp_file.file_name):
                    os.remove(tmp_file.file_name)


class PoissonSpikesGenerator(object):
    def __init__(self, gids, firing_rate, tstart=0.0, tstop=1000.0):
        self._gids = gids
        self._firing_rate = firing_rate / 1000.0
        self._tstart = tstart
        self._tstop = tstop

    def to_hdf5(self, file_name, sort_order='gid'):
        if sort_order == 'gid':
            gid_list = []
            times_list = []
            if sort_order == 'gid':
                for gid in self._gids:
                    c_time = self._tstart
                    while c_time < self._tstop:
                        interval = -np.log(1.0 - np.random.uniform()) / self._firing_rate
                        c_time += interval
                        gid_list.append(gid)
                        times_list.append(c_time)

            with h5py.File(file_name, 'w') as h5:
                h5.create_dataset('/spikes/gids', data=gid_list, dtype=np.uint)
                h5.create_dataset('/spikes/timestamps', data=times_list, dtype=np.float)
                h5['/spikes'].attrs['sorting'] = 'by_gid'

        else:
            raise NotImplementedError


class SpikesInput(object):
    def get_spikes(self, gid):
        raise NotImplementedError()

    @staticmethod
    def load(name, module, input_type, params):
        module_lc = module.lower()
        if module_lc == 'nwb':
            return SpikesInputNWBv1(name, module, input_type, params)
        elif module_lc == 'h5' or module_lc == 'hdf5':
            return SpikesInputH5(name, module, input_type, params)
        elif module_lc == 'csv':
            return SpikesInputCSV(name, module, input_type, params)
        else:
            raise Exception('Unable to load spikes for module type {}'.format(module))


class SpikesInputNWBv1(SpikesInput):
    def __init__(self, name, module, input_type, params):
        self.input_file = params['input_file']
        self._h5_handle = h5py.File(self.input_file, 'r')

        if 'trial' in params:
            self.trial = params['trial']
            self._spike_trains_handles = {}
            for node_id, h5grp in self._h5_handle['processing'][self.trial]['spike_train'].items():
                self._spike_trains_handles[int(node_id)] = h5grp['data']

        elif '/spikes' in self._h5_handle:
            raise Exception

    def get_spikes(self, gid):
        return np.array(self._spike_trains_handles[gid])


class SpikesInputH5(SpikesInput):
    def __init__(self, name, module, input_type, params):
        self._input_file = params['input_file']
        self._h5_handle = h5py.File(self._input_file, 'r')
        self._sort_order = self._h5_handle['/spikes'].attrs.get('sorting', None)
        self._gid_ds = self._h5_handle['/spikes/gids']
        self._timestamps_ds = self._h5_handle['/spikes/timestamps']

        self._gid_indicies = {}
        self._build_indicies()

    def _build_indicies(self):
        if self._sort_order == 'by_gid':
            indx_beg = 0
            c_gid = self._gid_ds[0]
            for indx, gid in enumerate(self._gid_ds):
                if gid != c_gid:
                    self._gid_indicies[c_gid] = slice(indx_beg, indx)
                    c_gid = gid
                    indx_beg = indx
            self._gid_indicies[c_gid] = slice(indx_beg, indx+1)

        else:
            raise NotImplementedError

    def get_spikes(self, gid):
        return self._timestamps_ds[self._gid_indicies[gid]]


class SpikesInputCSV(SpikesInput):
    def __init__(self, name, module, input_type, params):
        self._spikes_df = pd.read_csv(params['input_file'], index_col='gid', sep=' ')

    def get_spikes(self, gid):
        spike_times_str = self._spikes_df.iloc[gid]['spike-times']
        return np.array(spike_times_str.split(','), dtype=float)
