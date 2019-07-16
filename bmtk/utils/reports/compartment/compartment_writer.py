import os
import h5py
import numpy as np

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version
from .compartment_reader import CompartmentReaderVer01 as CompartmentReader
from .core import CompartmentWriterABC

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhosts = comm.Get_size()
    barrier = comm.Barrier

except Exception as exc:
    rank = 0
    nhosts = 1
    barrier = lambda: None


class PopulationWriterv01(CompartmentWriterABC, CompartmentReader):
    """Used to save cell membrane variables (V, Ca2+, etc) to the described hdf5 format.

    For parallel simulations this class will write to a seperate tmp file on each rank, then use the merge method to
    combine the results. This is less efficent, but doesn't require the user to install mpi4py and build h5py in
    parallel mode. For better performance use the CellVarRecorderParrallel class instead.
    """
    class DataTable(object):
        """A small struct to keep track of different */data (and buffer) tables"""
        def __init__(self, var_name):
            self.var_name = var_name
            # If buffering data, buffer_block will be an in-memory array and will write to data_block during when
            # filled. If not buffering buffer_block is an hdf5 dataset and data_block is ignored
            self.data_block = None
            self.buffer_block = None

    def __init__(self, parent, population, variable=None, units=None, tstart=0.0, tstop=1.0, dt=0.01, n_steps=None,
                 buffer_size=0, **kwargs):
        self._h5_base = None
        self._parent = parent

        self._population = population
        self._variable = variable
        self._units = units
        self._tstart = tstart
        self._tstop = tstop
        self._dt = dt

        self._n_steps = n_steps
        #if self._n_steps is None:
        #    self._n_steps = int((self._tstop - self._tstart)/self._dt)

        self._tmp_files = []

        self._mapping_gids = []  # list of gids in the order they appear in the data
        self._gid_map = {}  # table for looking up the gid offsets
        self._element_data = {}  # Used for additonal attributes in /mapping

        self._mapping_element_ids = []  # sections
        self._mapping_element_pos = []  # segments
        self._mapping_index = [0]  # index_pointer

        self._buffer_size = buffer_size
        self._buffer_data = buffer_size > 0
        self._data_block = self.DataTable(self._variable)
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

        self._is_initialized = False

    @property
    def h5_base(self):
        if self._h5_base is None:
            self._h5_base = self._parent.report_group.create_group(self._population)

        return self._h5_base

    def _calc_offset(self):
        self._n_segments_all = self._n_segments_local
        self._seg_offset_beg = 0
        self._seg_offset_end = self._n_segments_local

        self._n_gids_all = self._n_gids_local
        self._gids_beg = 0
        self._gids_end = self._n_gids_local

    def set_units(self, val, population=None):
        self._units = val

    def units(self, population=None):
        return self._units

    def set_variable(self, val, population=None):
        self._variable = val

    def variable(self, population=None):
        return self._variable

    def set_tstart(self, val, population=None):
        self._tstart = val

    def tstart(self, population=None):
        return self._tstart

    def set_tstop(self, val, population=None):
        self._tstop = val

    def tstop(self, population=None):
        return self._tstop

    def set_dt(self, val, population=None):
        self._dt = val

    def dt(self, population=None):
        return self._dt

    def n_steps(self, population=None):
        if self._n_steps is None:
            self._n_steps = int((self._tstop - self._tstart) / self._dt)
        return self._n_steps

    def set_time_trace(self, val, population=None):
        raise NotImplementedError()

    def time_trace(self, population=None):
        raise NotImplementedError()

    def add_cell(self, node_id, element_ids, element_pos, **map_attrs):
        assert(len(element_ids) == len(element_pos))
        # TODO: Check the same gid isn't added twice
        n_segs = len(element_pos)
        self._gid_map[node_id] = (self._n_segments_local, self._n_segments_local + n_segs)
        self._mapping_gids.append(node_id)
        self._mapping_element_ids.extend(element_ids)
        self._mapping_element_pos.extend(element_pos)
        self._mapping_index.append(self._mapping_index[-1] + n_segs)
        self._n_segments_local += n_segs
        self._n_gids_local += 1
        for k, v in map_attrs.items():
            if k not in self._element_data:
                self._element_data[k] = v
            else:
                self._element_data[k].extend(v)

    def initialize(self, **kwargs):
        if self._is_initialized:
            return

        n_steps = self.n_steps()
        if n_steps <= 0:
            raise Exception('A non-zero positive integer num-of-steps is required to initialize the compartment report.'
                            'Please specify report length using the n_steps parameters (or using appropiate tstop,'
                            'tstart, and dt).')

        self._calc_offset()
        base_grp = self.h5_base

        var_grp = base_grp.create_group('mapping')
        var_grp.create_dataset('node_ids', shape=(self._n_gids_all,), dtype=np.uint)
        var_grp.create_dataset('element_ids', shape=(self._n_segments_all,), dtype=np.uint)
        var_grp.create_dataset('element_pos', shape=(self._n_segments_all,), dtype=np.float)
        var_grp.create_dataset('index_pointer', shape=(self._n_gids_all+1,), dtype=np.uint64)
        var_grp.create_dataset('time', data=[self.tstart(), self.tstop(), self.dt()])
        for k, v in self._element_data.items():
            var_grp.create_dataset(k, shape=(self._n_segments_all,), dtype=type(v[0]))

        var_grp['node_ids'][self._gids_beg:self._gids_end] = self._mapping_gids
        var_grp['element_ids'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_ids
        var_grp['element_pos'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_pos
        var_grp['index_pointer'][self._gids_beg:(self._gids_end+1)] = self._mapping_index
        for k, v in self._element_data.items():
            var_grp[k][self._seg_offset_beg:self._seg_offset_end] = v

        self._total_steps = n_steps
        self._buffer_block_size = self._buffer_size

        if not self._buffer_data:
            # If data is not being buffered and instead written to the main block, we have to add a rank offset
            # to the gid offset
            for gid, gid_offset in self._gid_map.items():
                self._gid_map[gid] = (gid_offset[0] + self._seg_offset_beg, gid_offset[1] + self._seg_offset_beg)

        
        if self._buffer_data:
            # Set up in-memory block to buffer recorded variables before writing to the dataset
            self._data_block.buffer_block = np.zeros((self._buffer_size, self._n_segments_local), dtype=np.float)

            self._data_block.data_block = base_grp.create_dataset('data', shape=(self.n_steps(), self._n_segments_all),
                                                                  dtype=np.float, chunks=True)
            if self._variable is not None:
                self._data_block.data_block.attrs['variable'] = self._variable

            if self._units is not None:
                self._data_block.data_block.attrs['units'] = self._units

        else:
            # Since we are not buffering data, we just write directly to the on-disk dataset
            self._data_block.buffer_block = base_grp.create_dataset('data', shape=(self.n_steps(), self._n_segments_all),
                                                               dtype=np.float, chunks=True)
            if self._variable is not None:
                self._data_block.buffer_block.attrs['variable'] = self._variable

            if self._units is not None:
                self._data_block.buffer_block.attrs['units'] = self._units

        self._is_initialized = True

    def record_cell(self, node_id, vals, tstep, population=None):
        """Record cell parameters.

        :param gid: gid of cell.
        :param var_name: name of variable being recorded.
        :param seg_vals: list of all segment values
        :param tstep: time step
        """
        self.initialize()
        gid_beg, gid_end = self._gid_map[node_id]
        buffer_block = self._data_block.buffer_block
        update_index = (tstep - self._last_save_indx)
        buffer_block[update_index, gid_beg:gid_end] = vals

    def record_cell_block(self, node_id, vals, beg_step, end_step, population=None):
        """Save cell parameters one block at a time

        :param gid: gid of cell.
        :param var_name: name of variable being recorded.
        :param seg_vals: A vector/matrix of values being recorded
        """
        self.initialize()
        gid_beg, gid_end = self._gid_map[node_id]
        buffer_block = self._data_block.buffer_block
        if isinstance(vals, list) or vals.ndim == 1:
            buffer_block[:, gid_beg] = vals
            #buffer_block[beg_step:end_step, gid_beg:gid_end] = vals
        else:
            buffer_block[:, gid_beg:gid_end] = vals

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

            self._data_block.data_block[blk_beg:blk_end, :] = self._data_block.buffer_block[:block_size, :]

    def close(self):
        # Let the parent take care of this
        pass

    def merge(self):
        # Let the parent take care of this
        pass


class CompartmentWriterv01(CompartmentWriterABC):
    def __init__(self, file_path, mode='w', default_population=None, cache_dir=None, variable=None, units=None,
                 buffer_size=0, tstart=0.0, tstop=0.0, dt=0.0, n_steps=None, **kwargs):
        self._mode = mode
        self._variable = variable
        self._units = units
        self._pop_tables = {}
        self._default_pop = default_population
        self._buffer_size = buffer_size

        self._tstart = tstart
        self._tstop = tstop
        self._dt = dt
        self._n_steps = n_steps
        self._kwargs = kwargs

        self._h5_handle = None
        self._h5report_grp = None

        self._mpi_rank = kwargs.get('mpi_rank', rank)
        self._mpi_size = kwargs.get('mpi_size', nhosts)

        self._final_fpath = file_path  # name of file being writen too.
        self._cache_dir = cache_dir or os.path.dirname(os.path.abspath(file_path))  # used for mulitple ranks
        self._base_name = os.path.basename(file_path)  # make sure file names don't clash if there are multiple reports
        self._interm_fpath = self._get_iterm_fpath() # In certain cases (parallelized simulation) split the final file by rank.

    def _get_iterm_fpath(self):
        if self._mpi_size > 1:
            return self.temp_files[self._mpi_rank]
        else:
            return self._final_fpath

    def _create_h5file(self):
        fdir = os.path.dirname(os.path.abspath(self._interm_fpath))
        if not os.path.exists(fdir):
            os.mkdir(fdir)

        self._h5_handle = h5py.File(self._interm_fpath, self._mode)
        add_hdf5_version(self._h5_handle)
        add_hdf5_magic(self._h5_handle)

    @property
    def report_group(self):
        if self._h5report_grp is None:
            self._create_h5file()
            if 'report' in self._h5_handle.keys():
                self._h5report_grp = self._h5_handle['report']
            else:
                self._h5report_grp = self._h5_handle.create_group('report')
        return self._h5report_grp

    @property
    def temp_files(self):
        return [os.path.join(self._cache_dir, '.bmtk_tmp_cellvars_{}_{}'.format(r, self._base_name))
                for r in range(self._mpi_size)]

    def set_units(self, val, population=None):
        self[population].set_units(val)

    def set_variable(self, val, population=None):
        self[population].set_variable(val)

    def set_tstart(self, val, population=None):
        self[population].set_tstart(val)

    def set_tstop(self, val, population=None):
        self[population].set_tstop(val)

    def set_dt(self, val, population=None):
        self[population].set_dt(val)

    def n_steps(self, population=None):
        self[population].n_steps()

    def set_time_trace(self, val, population=None):
        self[population].set_time_trace(val)

    def add_cell(self, node_id, element_ids, element_pos, population=None, **element_data):
        pop_str = population or self._default_pop
        pop_grp = self._build_or_fetch_pop(pop_str)
        pop_grp.add_cell(node_id=node_id, element_ids=element_ids, element_pos=element_pos, **element_data)

    def initialize(self):
        for pop_grp in self._pop_tables.values():
            pop_grp.initialize()

    def record_cell(self, node_id, vals, tstep, population=None):
        pop_str = population or self._default_pop
        pop_grp = self._build_or_fetch_pop(pop_str)
        pop_grp.record_cell(node_id=node_id, vals=vals, tstep=tstep)

    def record_cell_block(self, node_id, vals, beg_step, end_step, population=None):
        self[population].record_cell_block(node_id=node_id, vals=vals, beg_step=beg_step, end_step=end_step)

    def flush(self):
        for pop_grp in self._pop_tables.values():
            pop_grp.flush()

    def close(self):
        for pop_grp in self._pop_tables.values():
            pop_grp.close()
        self._h5_handle.close()
        if self._mpi_size > 1:
            self.merge()

    def merge(self):
        barrier()
        if self._mpi_size > 1 and self._mpi_rank == 0:
            h5final = h5py.File(self._final_fpath, 'w')
            tmp_reports = [CompartmentReader(name) for name in self.temp_files]
            populations = set()
            for r in tmp_reports:
                populations.update(r.populations)

            for pop in populations:
                # Find the gid and segment offsets for each temp h5 file
                gid_ranges = []  # list of (gid-beg, gid-end)
                gid_offset = 0
                total_gid_count = 0  # total number of gids across all ranks

                seg_ranges = []
                seg_offset = 0
                total_seg_count = 0  # total number of segments across all ranks
                times = None

                n_steps = 0
                variable = None
                units = None

                for rpt in tmp_reports:
                    if pop not in rpt.populations:
                        continue
                    report = rpt[pop]

                    seg_count = len(report.element_pos())  # ['/mapping/element_pos'])
                    seg_ranges.append((seg_offset, seg_offset + seg_count))
                    seg_offset += seg_count
                    total_seg_count += seg_count

                    gid_count = len(report.node_ids())  # h5_tmp['mapping/node_ids'])
                    gid_ranges.append((gid_offset, gid_offset + gid_count))
                    gid_offset += gid_count
                    total_gid_count += gid_count

                    times = report.time()  # h5_tmp['mapping/time']

                    n_steps = report.n_steps()
                    variable = report.variable()
                    units = report.units()

                mapping_grp = h5final.create_group('/report/{}/mapping'.format(pop))
                if times is not None and len(times) > 0:
                    mapping_grp.create_dataset('time', data=times)
                element_id_ds = mapping_grp.create_dataset('element_ids', shape=(total_seg_count,), dtype=np.uint)
                el_pos_ds = mapping_grp.create_dataset('element_pos', shape=(total_seg_count,), dtype=np.float)
                gids_ds = mapping_grp.create_dataset('node_ids', shape=(total_gid_count,), dtype=np.uint)
                index_pointer_ds = mapping_grp.create_dataset('index_pointer', shape=(total_gid_count + 1,),
                                                              dtype=np.uint)
                for rpt in tmp_reports:
                    if pop not in rpt.populations:
                        continue
                    report = rpt[pop]
                    for k, v in report.custom_columns().items():
                        if k not in mapping_grp.keys():
                            mapping_grp.create_dataset(k, shape=(total_seg_count,), dtype=type(v[0]))

                # combine the /mapping datasets
                i = 0
                for rpt in tmp_reports:
                    if pop not in rpt.populations:
                        continue

                    report = rpt[pop]

                    # tmp_mapping_grp = h5_tmp['mapping']
                    beg, end = seg_ranges[i]

                    element_id_ds[beg:end] = report.element_ids()  # tmp_mapping_grp['element_id']
                    el_pos_ds[beg:end] = report.element_pos()  # tmp_mapping_grp['element_pos']
                    for k, v in report.custom_columns().items():
                        mapping_grp[k][beg:end] = v

                    # shift the index pointer values
                    index_pointer = np.array(report.index_pointer())  # tmp_mapping_grp['index_pointer'])
                    update_index = beg + index_pointer

                    beg, end = gid_ranges[i]
                    gids_ds[beg:end] = report.node_ids()  # tmp_mapping_grp['node_ids']
                    index_pointer_ds[beg:(end + 1)] = update_index
                    i += 1

                # combine the /var/data datasets
                data_name = '/report/{}/data'.format(pop)
                # data_name = '/{}/data'.format(var_name)
                var_data = h5final.create_dataset(data_name, shape=(n_steps, total_seg_count), dtype=np.float)
                # var_data.attrs['variable_name'] = var_name
                i = 0
                for rpt in tmp_reports:
                    if pop not in rpt.populations:
                        continue
                    report = rpt[pop]

                    beg, end = seg_ranges[i]
                    var_data[:, beg:end] = report.data()
                    i += 1

                if variable is not None:
                    var_data.attrs['variable'] = variable

                if units is not None:
                    var_data.attrs['units'] = units

            for tmp_file in self.temp_files:
                os.remove(tmp_file)
        barrier()

    def _build_or_fetch_pop(self, population):
        if population is None:
            raise Exception('Please specify a valid node population (or use default_population parameter in constructor).')
        if population in self._pop_tables:
            pop_grp = self._pop_tables[population]
        else:
            pop_grp = PopulationWriterv01(self, population, variable=self._variable, units=self._units,
                                          tstart=self._tstart, tstop=self._tstop, dt=self._dt,
                                          buffer_size=self._buffer_size, n_steps=self._n_steps)
            self._pop_tables[population] = pop_grp
        return pop_grp

    def __getitem__(self, population):
        pop_str = population or self._default_pop
        return self._build_or_fetch_pop(pop_str)
