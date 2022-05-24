import os
import h5py
import numpy as np

from bmtk.utils import io
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


class CellVarRecorder(object):
    """Used to save cell membrane variables (V, Ca2+, etc) to the described hdf5 format.

    For parallel simulations this class will write to a seperate tmp file on each rank, then use the merge method to
    combine the results. This is less efficent, but doesn't require the user to install mpi4py and build h5py in
    parallel mode. For better performance use the CellVarRecorderParallel class instead.
    """
    _io = io

    class DataTable(object):
        """A small struct to keep track of different \*/data (and buffer) tables"""
        def __init__(self, var_name):
            self.var_name = var_name
            # If buffering data, buffer_block will be an in-memory array and will write to data_block during when
            # filled. If not buffering buffer_block is an hdf5 dataset and data_block is ignored
            self.data_block = None
            self.buffer_block = None

    def __init__(self, file_name, tmp_dir, variables, buffer_data=True, mpi_rank=0, mpi_size=1, **kwargs):
        self._file_name = file_name
        self._h5_handle = None
        self._tmp_dir = tmp_dir
        self._variables = variables if isinstance(variables, list) else [variables]
        self._n_vars = len(self._variables)  # Used later to keep track if more than one var is saved to the same file.

        self._mpi_rank = mpi_rank
        self._mpi_size = mpi_size
        self._tmp_files = []
        self._saved_file = file_name
        self._population = kwargs.get('population', None)
        self._units = kwargs.get('units', None)

        if mpi_size > 1:
            self._io.log_warning('Was unable to run h5py in parallel (mpi) mode.' +
                                 ' Saving of membrane variable(s) may slow down.')
            tmp_fname = os.path.basename(file_name)  # make sure file names don't clash if there are multiple reports
            self._tmp_files = [os.path.join(tmp_dir, '__bmtk_tmp_cellvars_{}_{}'.format(r, tmp_fname))
                               for r in range(self._mpi_size)]
            self._file_name = self._tmp_files[self._mpi_rank]

        self._mapping_gids = []  # list of gids in the order they appear in the data
        self._gid_map = {}  # table for looking up the gid offsets
        self._map_attrs = {}  # Used for additonal attributes in /mapping

        self._mapping_element_ids = []  # sections
        self._mapping_element_pos = []  # segments
        self._mapping_index = [0]  # index_pointer

        self._buffer_data = buffer_data
        self._data_blocks = {var_name: self.DataTable(var_name) for var_name in self._variables}
        self._last_save_indx = 0  # for buffering, used to keep track of last timestep data was saved to disk

        self._buffer_block_size = 0
        self._total_steps = 0

        # Keep track of gids across the different ranks
        self._n_gids_all = 0
        self._n_gids_local = 0
        self._gids_beg = 0
        self._gids_end = 0

        # Keep track of segment counts across the different ranks
        self._n_segments_all = 0
        self._n_segments_local = 0
        self._seg_offset_beg = 0
        self._seg_offset_end = 0

        self._tstart = 0.0
        self._tstop = 0.0
        self._dt = 0.01
        self._is_initialized = False

    @property
    def tstart(self):
        return self._tstart

    @tstart.setter
    def tstart(self, time_ms):
        self._tstart = time_ms

    @property
    def tstop(self):
        return self._tstop

    @tstop.setter
    def tstop(self, time_ms):
        self._tstop = time_ms

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, time_ms):
        self._dt = time_ms

    @property
    def is_initialized(self):
        return self._is_initialized

    def _calc_offset(self):
        self._n_segments_all = self._n_segments_local
        self._seg_offset_beg = 0
        self._seg_offset_end = self._n_segments_local

        self._n_gids_all = self._n_gids_local
        self._gids_beg = 0
        self._gids_end = self._n_gids_local

    def _create_h5_file(self):
        self._h5_handle = h5py.File(self._file_name, 'w')
        add_hdf5_version(self._h5_handle)
        add_hdf5_magic(self._h5_handle)

    def add_cell(self, gid, sec_list, seg_list, **map_attrs):
        assert(len(sec_list) == len(seg_list))
        # TODO: Check the same gid isn't added twice
        n_segs = len(seg_list)
        self._gid_map[gid] = (self._n_segments_local, self._n_segments_local + n_segs)
        self._mapping_gids.append(gid)
        self._mapping_element_ids.extend(sec_list)
        self._mapping_element_pos.extend(seg_list)
        self._mapping_index.append(self._mapping_index[-1] + n_segs)
        self._n_segments_local += n_segs
        self._n_gids_local += 1
        for k, v in map_attrs.items():
            if k not in self._map_attrs:
                self._map_attrs[k] = v
            else:
                self._map_attrs[k].extend(v)

    def initialize(self, n_steps, buffer_size=0):
        self._calc_offset()
        self._create_h5_file()

        #root_name = '/report/{}'.format(self._population) if self._population is not None else '/'
        var_grp = self._h5_handle.create_group('/mapping')
        # TODO: gids --> node_ids
        var_grp.create_dataset('gids', shape=(self._n_gids_all,), dtype=np.uint)
        # TODO: element_id --> element_ids
        var_grp.create_dataset('element_id', shape=(self._n_segments_all,), dtype=np.uint)
        var_grp.create_dataset('element_pos', shape=(self._n_segments_all,), dtype=np.float)
        var_grp.create_dataset('index_pointer', shape=(self._n_gids_all+1,), dtype=np.uint64)
        var_grp.create_dataset('time', data=[self.tstart, self.tstop, self.dt])
        # TODO: Let user determine value
        var_grp['time'].attrs['units'] = 'ms'
        for k, v in self._map_attrs.items():
            var_grp.create_dataset(k, shape=(self._n_segments_all,), dtype=type(v[0]))

        var_grp['gids'][self._gids_beg:self._gids_end] = self._mapping_gids
        var_grp['element_id'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_ids
        var_grp['element_pos'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_pos
        var_grp['index_pointer'][self._gids_beg:(self._gids_end+1)] = self._mapping_index
        for k, v in self._map_attrs.items():
            var_grp[k][self._seg_offset_beg:self._seg_offset_end] = v

        self._total_steps = n_steps
        self._buffer_block_size = buffer_size
        if not self._buffer_data:
            # If data is not being buffered and instead written to the main block, we have to add a rank offset
            # to the gid offset
            for gid, gid_offset in self._gid_map.items():
                self._gid_map[gid] = (gid_offset[0] + self._seg_offset_beg, gid_offset[1] + self._seg_offset_beg)

        for var_name, data_tables in self._data_blocks.items():
            # If users are trying to save multiple variables in the same file put data table in its own /{var} group
            # (not sonata compliant). Otherwise the data table is located at the root
            data_grp = self._h5_handle if self._n_vars == 1 else self._h5_handle.create_group('/{}'.format(var_name))
            if self._buffer_data:
                # Set up in-memory block to buffer recorded variables before writing to the dataset
                data_tables.buffer_block = np.zeros((buffer_size, self._n_segments_local), dtype=np.float)
                data_tables.data_block = data_grp.create_dataset('data', shape=(n_steps, self._n_segments_all),
                                                                 dtype=np.float, chunks=True)
                # TODO: Remove Variable name
                data_tables.data_block.attrs['variable_name'] = var_name
                if self._units is not None:
                    data_tables.data_block.attrs['units'] = self._units
            else:
                # Since we are not buffering data, we just write directly to the on-disk dataset
                data_tables.buffer_block = data_grp.create_dataset('data', shape=(n_steps, self._n_segments_all),
                                                                   dtype=np.float, chunks=True)
                data_tables.buffer_block.attrs['variable_name'] = var_name
                if self._units is not None:
                    data_tables.buffer_block.attrs['units'] = self._units


        self._is_initialized = True

    def record_cell(self, gid, var_name, seg_vals, tstep):
        """Record cell parameters.

        :param gid: gid of cell.
        :param var_name: name of variable being recorded.
        :param seg_vals: list of all segment values
        :param tstep: time step
        """
        gid_beg, gid_end = self._gid_map[gid]
        buffer_block = self._data_blocks[var_name].buffer_block
        update_index = (tstep - self._last_save_indx)
        buffer_block[update_index, gid_beg:gid_end] = seg_vals

    def record_cell_block(self, gid, var_name, seg_vals):
        """Save cell parameters one block at a time

        :param gid: gid of cell.
        :param var_name: name of variable being recorded.
        :param seg_vals: A vector/matrix of values being recorded
        """
        gid_beg, gid_end = self._gid_map[gid]
        buffer_block = self._data_blocks[var_name].buffer_block
        if gid_end - gid_beg == 1:
            buffer_block[:, gid_beg] = seg_vals
        else:
            buffer_block[:, gid_beg:gid_end] = seg_vals

    def flush(self):
        """Move data from memory to dataset"""
        if self._buffer_data:
            blk_beg = self._last_save_indx
            blk_end = blk_beg + self._buffer_block_size
            if blk_end > self._total_steps:
                # Need to handle the case that simulation doesn't end on a block step
                blk_end = blk_beg + self._total_steps - blk_beg

            block_size = blk_end - blk_beg
            self._last_save_indx += block_size

            for _, data_table in self._data_blocks.items():
                data_table.data_block[blk_beg:blk_end, :] = data_table.buffer_block[:block_size, :]

    def close(self):
        self._h5_handle.close()

    def merge(self):
        if self._mpi_size > 1 and self._mpi_rank == 0:
            h5final = h5py.File(self._saved_file, 'w')
            tmp_h5_handles = [h5py.File(name, 'r') for name in self._tmp_files]

            # Find the gid and segment offsets for each temp h5 file
            gid_ranges = []  # list of (gid-beg, gid-end)
            gid_offset = 0
            total_gid_count = 0  # total number of gids across all ranks

            seg_ranges = []
            seg_offset = 0
            total_seg_count = 0  # total number of segments across all ranks
            time_ds = None
            for h5_tmp in tmp_h5_handles:
                seg_count = len(h5_tmp['/mapping/element_pos'])
                seg_ranges.append((seg_offset, seg_offset+seg_count))
                seg_offset += seg_count
                total_seg_count += seg_count

                gid_count = len(h5_tmp['mapping/gids'])
                gid_ranges.append((gid_offset, gid_offset+gid_count))
                gid_offset += gid_count
                total_gid_count += gid_count

                time_ds = h5_tmp['mapping/time']

            mapping_grp = h5final.create_group('mapping')
            if time_ds:
                mapping_grp.create_dataset('time', data=time_ds)
            element_id_ds = mapping_grp.create_dataset('element_id', shape=(total_seg_count,), dtype=np.uint)
            el_pos_ds = mapping_grp.create_dataset('element_pos', shape=(total_seg_count,), dtype=np.float)
            gids_ds = mapping_grp.create_dataset('gids', shape=(total_gid_count,), dtype=np.uint)
            index_pointer_ds = mapping_grp.create_dataset('index_pointer', shape=(total_gid_count+1,), dtype=np.uint)
            for k, v in self._map_attrs.items():
                mapping_grp.create_dataset(k, shape=(total_seg_count,), dtype=type(v[0]))

            # combine the /mapping datasets
            for i, h5_tmp in enumerate(tmp_h5_handles):
                tmp_mapping_grp = h5_tmp['mapping']
                beg, end = seg_ranges[i]
                element_id_ds[beg:end] = tmp_mapping_grp['element_id']
                el_pos_ds[beg:end] = tmp_mapping_grp['element_pos']
                for k, v in self._map_attrs.items():
                    mapping_grp[k][beg:end] = v

                # shift the index pointer values
                index_pointer = np.array(tmp_mapping_grp['index_pointer'])
                update_index = beg + index_pointer

                beg, end = gid_ranges[i]
                gids_ds[beg:end] = tmp_mapping_grp['gids']
                index_pointer_ds[beg:(end+1)] = update_index


            # combine the /var/data datasets
            for var_name in self._variables:
                data_name = '/data' if self._n_vars == 1 else '/{}/data'.format(var_name)
                # data_name = '/{}/data'.format(var_name)
                var_data = h5final.create_dataset(data_name, shape=(self._total_steps, total_seg_count), dtype=np.float)
                var_data.attrs['variable_name'] = var_name
                for i, h5_tmp in enumerate(tmp_h5_handles):
                    beg, end = seg_ranges[i]
                    var_data[:, beg:end] = h5_tmp[data_name]

            for tmp_file in self._tmp_files:
                os.remove(tmp_file)


class CellVarRecorderParallel(CellVarRecorder):
    """
    Unlike the parent, this take advantage of parallel h5py to writting to the results file across different ranks.

    """
    def __init__(self, file_name, tmp_dir, variables, buffer_data=True, **kwargs):
        super(CellVarRecorder, self).__init__(file_name, tmp_dir, variables, buffer_data=buffer_data, mpi_rank=0,
                                              mpi_size=1, **kwargs)

    def _calc_offset(self):
        from mpi4py import MPI  # Just needed for UNSIGNED_INT
        comm = io.bmtk_world_comm.comm
        rank = comm.Get_rank()
        nhosts = comm.Get_size()

        # iterate through the ranks let rank r determine the offset from rank r-1
        for r in range(comm.Get_size()):
            if rank == r:
                if rank < (nhosts - 1):
                    # pass the num of segments and num of gids to the next rank
                    offsets = np.array([self._n_segments_local, self._n_gids_local], dtype=np.uint)
                    comm.Send([offsets, MPI.UNSIGNED_INT], dest=(rank+1))

                if rank > 0:
                    # get num of segments and gids from prev. rank and calculate offsets
                    offset = np.empty(2, dtype=np.uint)
                    comm.Recv([offsets, MPI.UNSIGNED_INT], source=(r-1))
                    self._seg_offset_beg = offsets[0]
                    self._seg_offset_end = self._seg_offset_beg + self._n_segments_local

                    self._gids_beg = offset[1]
                    self._gids_end = self._gids_beg + self._n_gids_local

            comm.Barrier()

        # broadcast the total num of gids/segments from the final rank to all the others
        if rank == (nhosts - 1):
            total_counts = np.array([self._seg_offset_end, self._gids_end], dtype=np.uint)
        else:
            total_counts = np.empty(2, dtype=np.uint)

        comm.Bcast(total_counts, root=(nhosts-1))
        self._n_segments_all = total_counts[0]
        self._n_gids_all = total_counts[1]

    def _create_h5_file(self):
        self._h5_handle = h5py.File(self._file_name, 'w', driver='mpio', comm=io.bmtk_world_comm.comm)
        add_hdf5_version(self._h5_handle)
        add_hdf5_magic(self._h5_handle)

    def merge(self):
        pass
