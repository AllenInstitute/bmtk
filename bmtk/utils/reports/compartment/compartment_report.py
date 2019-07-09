import os
import h5py
import numpy as np

from bmtk.utils import io
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version
from .compartment_reader import CompartmentReaderVer01 as SonataReaderDefault
from .core import CompartmentReaderABC


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nhosts = comm.Get_size()

except Exception as exc:
    pass


class CompartmentReport(object):
    def __init__(self, path, mode='r', adaptor=None, **kwargs):
        if adaptor is not None:
            self._adaptor = adaptor
        else:
            if mode == 'r':
                self._adaptor = SonataReaderDefault(path, **kwargs)
            else:
                #self._adaptor = CompartmentReportOLD(path, **kwargs)
                pass

    def initialize(self):
        pass

    def add_cell(self, node_id, sections, segments, population=None, **attrs):
        pass

    def record_cell(self, node_id, segment_vals, tstep, population=None):
        pass

    def record_cell_block(self, node_ids, segment_vals, tbegin, tend, population=None):
        pass

    def close(self):
        pass

    def flush(self):
        pass

    def from_sonata(self, path):
        pass

    def from_nwb(self, path):
        pass

    def from_path(self, path):
        pass

    @property
    def populations(self):
        return self._adaptor.populations

    def __getitem__(self, population):
        return self._adaptor[population]



'''
class PopulationReport(object):
    def __init__(self, filepath, mode='w', population=None, buffer_size=0, cache_dir=None, report_name=None,
                 units=None, variable=None, **kwargs):
        pass
'''

class PopulationReport(object):
    """Used to save cell membrane variables (V, Ca2+, etc) to the described hdf5 format.

    For parallel simulations this class will write to a seperate tmp file on each rank, then use the merge method to
    combine the results. This is less efficent, but doesn't require the user to install mpi4py and build h5py in
    parallel mode. For better performance use the CellVarRecorderParrallel class instead.
    """
    _io = io

    class DataTable(object):
        """A small struct to keep track of different */data (and buffer) tables"""
        def __init__(self, var_name):
            self.var_name = var_name
            # If buffering data, buffer_block will be an in-memory array and will write to data_block during when
            # filled. If not buffering buffer_block is an hdf5 dataset and data_block is ignored
            self.data_block = None
            self.buffer_block = None

    def __init__(self, file_name, population, tmp_dir, variable=None, units=None, buffer_data=True, **kwargs):
        self._file_name = file_name
        self._h5_handle = None
        self._tmp_dir = tmp_dir
        self._population = population
        self._variable = variable
        self._units = units

        #self._variables = variables if isinstance(variables, list) else [variables]
        #self._n_vars = len(self._variables)  # Used later to keep track if more than one var is saved to the same file.

        self._mpi_rank = kwargs.get('mpi_rank', rank)
        self._mpi_size = kwargs.get('mpi_size', nhosts)
        self._tmp_files = []
        self._saved_file = file_name

        if self._mpi_size > 1:
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
        self._data_block = self.DataTable(self._variable)
        #self._data_blocks = {var_name: self.DataTable(var_name) for var_name in self._variables}
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

    def add_cell(self, node_id, sec_list, seg_list, **map_attrs):
        assert(len(sec_list) == len(seg_list))
        # TODO: Check the same gid isn't added twice
        n_segs = len(seg_list)
        self._gid_map[node_id] = (self._n_segments_local, self._n_segments_local + n_segs)
        self._mapping_gids.append(node_id)
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

        base_grp = self._h5_handle.create_group('report/{}'.format(self._population))

        var_grp = base_grp.create_group('mapping')
        var_grp.create_dataset('node_ids', shape=(self._n_gids_all,), dtype=np.uint)
        var_grp.create_dataset('element_ids', shape=(self._n_segments_all,), dtype=np.uint)
        var_grp.create_dataset('element_pos', shape=(self._n_segments_all,), dtype=np.float)
        var_grp.create_dataset('index_pointer', shape=(self._n_gids_all+1,), dtype=np.uint64)
        var_grp.create_dataset('time', data=[self.tstart, self.tstop, self.dt])
        for k, v in self._map_attrs.items():
            var_grp.create_dataset(k, shape=(self._n_segments_all,), dtype=type(v[0]))

        var_grp['node_ids'][self._gids_beg:self._gids_end] = self._mapping_gids
        var_grp['element_ids'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_ids
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

        if self._buffer_data:
            # Set up in-memory block to buffer recorded variables before writing to the dataset
            self._data_block.buffer_block = np.zeros((buffer_size, self._n_segments_local), dtype=np.float)
            self._data_block.data_block = base_grp.create_dataset('data', shape=(n_steps, self._n_segments_all),
                                                             dtype=np.float, chunks=True)
            if self._variable is not None:
                self._data_block.data_block.attrs['variable'] = self._variable

            if self._units is not None:
                self._data_block.data_block.attrs['units'] = self._units

        else:
            # Since we are not buffering data, we just write directly to the on-disk dataset
            self._data_block.buffer_block = base_grp.create_dataset('data', shape=(n_steps, self._n_segments_all),
                                                               dtype=np.float, chunks=True)
            if self._variable is not None:
                self._data_block.buffer_block.attrs['variable'] = self._variable

            if self._units is not None:
                self._data_block.buffer_block.attrs['units'] = self._units

        #if self._units is not None:
        #    self._data_block

        self._is_initialized = True

    def record_cell(self, gid, seg_vals, tstep):
        """Record cell parameters.

        :param gid: gid of cell.
        :param var_name: name of variable being recorded.
        :param seg_vals: list of all segment values
        :param tstep: time step
        """
        gid_beg, gid_end = self._gid_map[gid]
        buffer_block = self._data_block.buffer_block
        update_index = (tstep - self._last_save_indx)
        buffer_block[update_index, gid_beg:gid_end] = seg_vals

    def record_cell_block(self, gid, seg_vals):
        """Save cell parameters one block at a time

        :param gid: gid of cell.
        :param var_name: name of variable being recorded.
        :param seg_vals: A vector/matrix of values being recorded
        """
        gid_beg, gid_end = self._gid_map[gid]
        buffer_block = self._data_block.buffer_block
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

            #for _, data_table in self._data_blocks.items():
            #    data_table.data_block[blk_beg:blk_end, :] = data_table.buffer_block[:block_size, :]
            self._data_block.data_block[blk_beg:blk_end, :] = self._data_block.buffer_block[:block_size, :]

    def close(self):
        self._h5_handle.close()

    def merge(self):
        if self._mpi_size > 1 and self._mpi_rank == 0:
            h5final = h5py.File(self._saved_file, 'w')
            tmp_reports = [CompartmentReader(name) for name in self._tmp_files]
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

                    seg_count = len(report.element_pos)  # ['/mapping/element_pos'])
                    seg_ranges.append((seg_offset, seg_offset + seg_count))
                    seg_offset += seg_count
                    total_seg_count += seg_count

                    gid_count = len(report.node_ids)  # h5_tmp['mapping/node_ids'])
                    gid_ranges.append((gid_offset, gid_offset + gid_count))
                    gid_offset += gid_count
                    total_gid_count += gid_count

                    times = report.time  # h5_tmp['mapping/time']

                    n_steps = report.n_steps
                    variable = report.variable
                    units = report.units

                mapping_grp = h5final.create_group('/report/{}/mapping'.format(pop))
                if times is not None and len(times) > 0:
                    mapping_grp.create_dataset('time', data=times)
                element_id_ds = mapping_grp.create_dataset('element_id', shape=(total_seg_count,), dtype=np.uint)
                el_pos_ds = mapping_grp.create_dataset('element_pos', shape=(total_seg_count,), dtype=np.float)
                gids_ds = mapping_grp.create_dataset('node_ids', shape=(total_gid_count,), dtype=np.uint)
                index_pointer_ds = mapping_grp.create_dataset('index_pointer', shape=(total_gid_count + 1,),
                                                              dtype=np.uint)
                for rpt in tmp_reports:
                    if pop not in rpt.populations:
                        continue
                    report = rpt[pop]
                    for k, v in report.custom_columns.items():
                        print(k, v)
                        mapping_grp.create_dataset(k, shape=(total_seg_count,), dtype=type(v[0]))

                # combine the /mapping datasets
                for i, rpt in enumerate(tmp_reports):
                    if pop not in rpt.populations:
                        continue

                    report = rpt[pop]

                    # tmp_mapping_grp = h5_tmp['mapping']
                    beg, end = seg_ranges[i]
                    element_id_ds[beg:end] = report.element_ids  # tmp_mapping_grp['element_id']
                    el_pos_ds[beg:end] = report.element_pos  # tmp_mapping_grp['element_pos']
                    for k, v in report.custom_columns.items():
                        mapping_grp[k][beg:end] = v

                    # shift the index pointer values
                    index_pointer = np.array(report.index_pointer)  # tmp_mapping_grp['index_pointer'])
                    update_index = beg + index_pointer

                    beg, end = gid_ranges[i]
                    gids_ds[beg:end] = report.node_ids  # tmp_mapping_grp['node_ids']
                    index_pointer_ds[beg:(end + 1)] = update_index

                # combine the /var/data datasets
                data_name = '/report/{}/data'.format(pop)
                # data_name = '/{}/data'.format(var_name)
                var_data = h5final.create_dataset(data_name, shape=(n_steps, total_seg_count), dtype=np.float)
                # var_data.attrs['variable_name'] = var_name
                for i, rpt in enumerate(tmp_reports):
                    if pop not in rpt.populations:
                        continue
                    report = rpt[pop]

                    beg, end = seg_ranges[i]
                    var_data[:, beg:end] = report.data

                if variable is not None:
                    var_data.attrs['variable'] = variable

                if units is not None:
                    var_data.attrs['units'] = units

            for tmp_file in self._tmp_files:
                os.remove(tmp_file)

    def from_sonata(self, path, **kwargs):
        pass

    def from_path(self, path, **kwargs):
        pass

    def from_nwb(self, path, **kwargs):
        pass


class CompartmentReportOLD(object):
    def __init__(self, file_name, tmp_dir, variable=None, units=None, default_population='pop_na', buffer_data=True):
        self._file_name = file_name
        self._tmp_dir = tmp_dir
        self._variable = variable
        self._units = units
        self._pop_tables = {}
        self._default_pop = default_population

        self._t_stop = 0.0
        self._t_start = 0.0
        self._dt = 0.1

    @property
    def t_start(self):
        return self._t_start

    @t_start.setter
    def t_start(self, time):
        self._t_start = time

    @property
    def t_stop(self):
        return self._t_stop

    @t_stop.setter
    def t_stop(self, time):
        self._t_stop = time

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = val

    def add_cell(self, node_id, sec_list, seg_list, population=None, **map_attrs):
        pop_str = population or self._default_pop
        pop_grp = self._build_or_fetch_pop(pop_str)
        pop_grp.add_cell(node_id=node_id, sec_list=sec_list, seg_list=seg_list, **map_attrs)

    def initialize(self, n_steps, buffer_size=0):
        for pop_grp in self._pop_tables.values():
            pop_grp.initialize(n_steps=n_steps, buffer_size=buffer_size)

    def record_cell(self, node_id, seg_vals, tstep, population=None):
        pop_str = population or self._default_pop
        pop_grp = self._pop_tables[pop_str]
        pop_grp.record_cell(gid=node_id, seg_vals=seg_vals, tstep=tstep)

    def record_cell_block(self, node_id, seg_vals, population=None):
        pass

    def flush(self):
        for pop_grp in self._pop_tables.values():
            pop_grp.flush()

    def close(self):
        for pop_grp in self._pop_tables.values():
            pop_grp.close()

    def merge(self):
        for pop_grp in self._pop_tables.values():
            pop_grp.merge()

    def _build_or_fetch_pop(self, population):
        if population in self._pop_tables:
            pop_grp = self._pop_tables[population]
        else:
            pop_grp = PopulationReport(self._file_name, population, self._tmp_dir, variable=self._variable,
                                       units=self._units)
            pop_grp.tstart = self.t_start
            pop_grp.tstop = self.t_stop
            pop_grp.dt = self.dt
            self._pop_tables[population] = pop_grp
        return pop_grp

    def __getitem__(self, item):
        return self._pop_tables(item)


class CompartmentReportParallel(CompartmentReportOLD):
    """
    Unlike the parent, this take advantage of parallel h5py to writting to the results file across different ranks.

    """
    def __init__(self, file_name, tmp_dir, variables, buffer_data=True):
        super(CompartmentReportParallel, self).__init__(file_name, tmp_dir, variables, buffer_data=buffer_data, mpi_rank=0,
                                              mpi_size=1)

    def _calc_offset(self):
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
        self._h5_handle = h5py.File(self._file_name, 'w', driver='mpio', comm=MPI.COMM_WORLD)
        add_hdf5_version(self._h5_handle)
        add_hdf5_magic(self._h5_handle)

    def merge(self):
        pass

'''
class CompartmentReport(object):
    def __init__(self, path, mode='r', adaptor=None, **kwargs):
        if adaptor is not None:
            self._adaptor = adaptor
        else:
            if mode == 'r':
                self._adaptor = CompartmentReader(path, **kwargs)
            else:
                self._adaptor = CompartmentReportOLD(path, **kwargs)

    def initialize(self):
        pass

    def add_cell(self, node_id, sections, segments, population=None, **attrs):
        pass

    def record_cell(self, node_id, segment_vals, tstep, population=None):
        pass

    def record_cell_block(self, node_ids, segment_vals, tbegin, tend, population=None):
        pass

    def close(self):
        pass

    def flush(self):
        pass

    def from_sonata(self, path):
        pass

    def from_nwb(self, path):
        pass

    def from_path(self, path):
        pass

    @property
    def populations(self):
        return self._adaptor.populations

    def __getitem__(self, population):
        return self._adaptor[population]
'''

'''
class PopulationReader(CompartmentReaderABC):
    mapping_ds = ['element_ids', 'element_pos', 'index_pointer', 'node_ids', 'time']

    def __init__(self, grp, population):
        self._data_grp = grp['data']
        self._mapping = grp['mapping']

        self._gid2data_table = {}
        if self._mapping is None:
            raise Exception('could not find /mapping group')
        else:
            gids_ds = self._mapping['node_ids']
            index_pointer_ds = self._mapping['index_pointer']
            for indx, gid in enumerate(gids_ds):
                self._gid2data_table[gid] = (index_pointer_ds[indx], index_pointer_ds[
                    indx + 1])  # slice(index_pointer_ds[indx], index_pointer_ds[indx+1])

            time_ds = self._mapping['time']
            self._t_start = time_ds[0]
            self._t_stop = time_ds[1]
            self._dt = time_ds[2]
            self._n_steps = int((self._t_stop - self._t_start) / self._dt)

        self._custom_cols = {col: grp for col, grp in self._mapping.items() if
                             col not in self.mapping_ds and isinstance(grp, h5py.Dataset)}

    @property
    def element_pos(self):
        return self._mapping['element_pos'][()]

    @property
    def element_ids(self):
        return self._mapping['element_ids'][()]

    @property
    def node_ids(self):
        return self._mapping['node_ids'][()]

    @property
    def time(self):
        return self._mapping['time'][()]

    @property
    def index_pointer(self):
        return self._mapping['index_pointer'][()]

    @property
    def custom_columns(self):
        return self._custom_cols

    @property
    def n_steps(self):
        return len(self._data_grp)

    @property
    def data(self):
        return self._data_grp

    @property
    def variable(self):
        return self.data.attrs.get('variable', None)

    @property
    def units(self):
        return self.data.attrs.get('units', None)


class CompartmentReader(CompartmentReaderABC):
    VAR_UNKNOWN = 'Unknown'
    UNITS_UNKNOWN = 'NA'

    def __init__(self, path, mode='r', **params):
        self._h5_handle = h5py.File(path, 'r')
        self._h5_root = self._h5_handle[params['h5_root']] if 'h5_root' in params else self._h5_handle['/']
        #self._var_data = {}
        #self._var_units = {}
        self._popgrps = {}

        self._mapping = None

        if 'report' in self._h5_handle.keys():
            report_grp = self._h5_handle['report']
            for pop_name, pop_grp in report_grp.items():
                self._popgrps[pop_name] = PopulationReader(pop_grp, pop_name)

    @property
    def populations(self):
        return self._popgrps.keys()

    def __getitem__(self, population):
        return self._popgrps[population]

    @property
    def variables(self):
        return list(self._var_data.keys())

    @property
    def gids(self):
        return list(self._gid2data_table.keys())

    @property
    def t_start(self):
        return self._t_start

    @property
    def t_stop(self):
        return self._t_stop

    @property
    def dt(self):
        return self._dt

    @property
    def time_trace(self):
        return np.linspace(self.t_start, self.t_stop, num=self._n_steps, endpoint=True)

    @property
    def h5(self):
        return self._h5_root

    def _find_units(self, data_set):
        return data_set.attrs.get('units', CompartmentReader.UNITS_UNKNOWN)

    def units(self, var_name=VAR_UNKNOWN):
        return self._var_units[var_name]

    def n_compartments(self, gid):
        bounds = self._gid2data_table[gid]
        return bounds[1] - bounds[0]

    def compartment_ids(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_id'][bounds[0]:bounds[1]]

    def compartment_positions(self, gid):
        bounds = self._gid2data_table[gid]
        return self._mapping['element_pos'][bounds[0]:bounds[1]]

    def data(self, gid, var_name=VAR_UNKNOWN, time_window=None, compartments='origin'):
        if var_name not in self.variables:
            raise Exception('Unknown variable {}'.format(var_name))

        if time_window is None:
            time_slice = slice(0, self._n_steps)
        else:
            if len(time_window) != 2:
                raise Exception('Invalid time_window, expecting tuple [being, end].')

            window_beg = max(int((time_window[0] - self.t_start) / self.dt), 0)
            window_end = min(int((time_window[1] - self.t_start) / self.dt), self._n_steps / self.dt)
            time_slice = slice(window_beg, window_end)

        multi_compartments = True
        if compartments == 'origin' or self.n_compartments(gid) == 1:
            # Return the first (and possibly only) compartment for said gid
            gid_slice = self._gid2data_table[gid][0]
            multi_compartments = False
        elif compartments == 'all':
            # Return all compartments
            gid_slice = slice(self._gid2data_table[gid][0], self._gid2data_table[gid][1])
        else:
            # return all compartments with corresponding element id
            compartment_list = list(compartments) if isinstance(compartments, (long, int)) else compartments
            begin = self._gid2data_table[gid][0]
            end = self._gid2data_table[gid][1]
            gid_slice = [i for i in range(begin, end) if self._mapping[i] in compartment_list]

        data = np.array(self._var_data[var_name][time_slice, gid_slice])
        return data.T if multi_compartments else data

    def close(self):
        self._h5_handle.close()
'''
