import os
import h5py
import numpy as np

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nhosts = comm.Get_size()




class CellVarRecorder(object):
    def __init__(self, file_name, tmp_dir, variable, buffer_data=True, mpi_rank=0, mpi_size=1):
        self._file_name = file_name
        self._h5_handle = None
        self._tmp_dir = tmp_dir
        self._variable = variable

        #
        self._mpi_rank = mpi_rank
        self._mpi_size = mpi_size
        self._tmp_files = []
        self._saved_file = file_name
        if mpi_size > 1:
            # io.log_warning('Cell Var Recorder')
            self._tmp_files = [os.path.join(tmp_dir, '__bmtk_tmp_cellvars_{}.h5'.format(r))
                               for r in range(self._mpi_size)]
            self._file_name = self._tmp_files[self._mpi_rank]

        self._mpi_offset = 0
        self._total_segments_mpi = 0

        self._mapping_gids = []
        self._gid_map = {}
        #self._gid_map_counter = 0

        self._mapping_element_ids = []
        self._mapping_element_pos = []
        self._mapping_index = [0]
        #self._mapping_index_ptr = 0
        #self._total_segments = 0

        self._buffer_data = buffer_data
        self._data_block = None
        self._buffer_block = None
        self._last_save_indx = 0
        self._buffer_block_size = 0
        #self._index_modifer = 0
        self._total_steps = 0

        self._n_gids_all = 0
        self._n_gids_local = 0
        self._gids_beg = 0
        self._gids_end = 0

        self._n_segments_all = 0
        self._n_segments_local = 0
        self._seg_offset_beg = 0
        self._seg_offset_end = 0

    def _calc_offset(self):
        self._n_segments_all = self._n_segments_local
        self._seg_offset_beg = 0
        self._seg_offset_end = self._n_segments_local

        self._n_gids_all = self._n_gids_local
        self._gids_beg = 0
        self._gids_end = self._n_segments_local

    def add_cell(self, gid, sec_list, seg_list):
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


    def initialize(self, n_steps, buffer_size=0):
        self._calc_offset()

        # print self._file_name
        self._h5_handle = h5file = h5py.File(self._file_name, 'w') #, driver='mpio', comm=MPI.COMM_WORLD)
        var_grp = h5file.create_group(self._variable)

        var_grp.create_dataset('mapping/gids', shape=(self._n_gids_all,), dtype=np.uint)
        var_grp.create_dataset('mapping/element_id', shape=(self._n_segments_all,), dtype=np.uint)
        var_grp.create_dataset('mapping/element_pos', shape=(self._n_segments_all,), dtype=np.float)
        var_grp.create_dataset('mapping/index_pointer', shape=(self._n_gids_all+1,), dtype=np.uint64)

        var_grp['mapping/gids'][self._gids_beg:self._gids_end] = self._mapping_gids
        var_grp['mapping/element_id'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_ids
        var_grp['mapping/element_pos'][self._seg_offset_beg:self._seg_offset_end] = self._mapping_element_pos
        var_grp['mapping/index_pointer'][self._gids_beg:self._gids_end] = self._mapping_index

        '''
        var_grp.create_dataset('mapping/gids', data=self._mapping_gids)
        var_grp.create_dataset('mapping/element_id', data=self._mapping_element_ids)
        var_grp.create_dataset('mapping/element_pos', data=self._mapping_element_pos)
        var_grp.create_dataset('mapping/index_pointer', data=self._mapping_index)
        '''


        self._total_steps = n_steps
        if self._buffer_data:
            self._buffer_block = np.zeros((buffer_size, self._n_segments_local), dtype=np.float)
            self._data_block = var_grp.create_dataset('data', shape=(n_steps, self._n_segments_all), dtype=np.float,
                                                      chunks=True)
            self._buffer_block_num = 0
            self._buffer_block_size = buffer_size
        else:
            for gid, gid_offset in self._gid_map.items():
                self._gid_map[gid] = (gid_offset[0] + self._seg_offset_beg, gid_offset[1] + self._seg_offset_beg)

            self._buffer_block = var_grp.create_dataset('data', shape=(n_steps, self._n_segments_all), dtype=np.float,
                                                        chunks=True)

    def record_cell(self, gid, seg_vals, tstep):
        gid_beg, gid_end = self._gid_map[gid]
        update_index = (tstep - self._last_save_indx)
        self._buffer_block[update_index, gid_beg:gid_end] = seg_vals


    def add_val(self, gid, element_id, element_pos, value, tstep):

        self._index_modifer += self._buffer_block_size


        #gid_beg, gid_end = self._gid_map[gid]
        #self._data_ds[tstep, gid_beg:gid_end] = value
        pass


    def flush(self):
        if self._buffer_data:
            blk_beg = self._last_save_indx
            blk_end = blk_beg + self._buffer_block_size
            # print blk_beg, blk_end

            self._data_block[blk_beg:blk_end, :] = self._buffer_block
            self._last_save_indx += self._buffer_block_size

    def close(self):
        # print 'close'
        self._h5_handle.close()

    def merge(self):
        if self._mpi_size > 1 and self._mpi_rank == 0:
            print 'blah'
            h5final = h5py.File(self._saved_file, 'w')
            print self._tmp_files

            tmp_h5_handles = [h5py.File(name, 'r') for name in self._tmp_files]
            tmp_file_seg_counts = []
            offsets = [0]
            total_seg_count = 0
            #gid_offsets = [0]
            gids_tracker = []

            gid_offset = 0


            for h5handle in tmp_h5_handles:
                var_grp = h5handle[self._variable]
                seg_count = var_grp['data'].shape[1]
                tmp_file_seg_counts.append(seg_count)
                offsets.append(offsets[-1] + seg_count)

                gid_count = var_grp['mapping/gids'].shape[0]
                #offset = gid[]
                gids_tracker.append((gid_offset, gid_offset+gid_count))
                gid_offset += gid_count

                #total_seg_count += h5handle['/v/data'].shape[1]

            total_seg_count = sum(tmp_file_seg_counts)
            print total_seg_count
            print self._total_steps
            gid_total = gids_tracker[-1][1]
            output_var_grp = h5final.create_dataset(self._variable)
            data_ds = output_var_grp.create_dataset('data', shape=(self._total_steps, total_seg_count), dtype=float)
            #h5final.create_dataset('/v/mapping/gids', shape=(total_seg_count,), dtype=)
            element_id_ds = output_var_grp.create_dataset('mapping/element_id', shape=(total_seg_count,), dtype=np.uint)
            el_pos_ds = output_var_grp.create_dataset('mapping/element_pos', shape=(total_seg_count,), dtype=np.float)
            gids_ds = output_var_grp.create_dataset('mapping/gids', shape=(gid_total,), dtype=np.uint)
            index_pointer_ds = output_var_grp.create_dataset('mapping/index_pointer', shape=(gid_total+1,), dtype=np.uint)
            for i, h5handle in enumerate(tmp_h5_handles):
                var_grp = h5handle[self._variable]

                beg = offsets[i]
                end = beg + tmp_file_seg_counts[i]
                print beg, end
                #print h5handle['/v/mapping/element_id'].shape
                element_id_ds[beg:end] = var_grp['mapping/element_id']
                el_pos_ds[beg:end] = var_grp['mapping/element_pos']
                #print h5handle['/v/data'].shape

                data_ds[:, beg:end] = var_grp['data']

                gid_beg, gid_end = gids_tracker[i]
                gids_ds[gid_beg:gid_end] = var_grp['mapping/gids']
                index_pointer = np.array(var_grp['mapping/index_pointer'])

                update_index = beg + index_pointer
                print gid_beg, gid_end
                index_pointer_ds[gid_beg:(gid_end+1)] = update_index
                #exit()


                # data_ds[:, beg:end] = h5handle['/v/data']

            #h5final.create_



class CellVarRecorderParallel(CellVarRecorder):
    def __init__(self, file_name, tmp_dir, variable, buffer_data=True, mpi_rank=0, mpi_size=1):
        pass

    def _calc_offset(self):
        for r in range(comm.Get_size()):
            if rank == r:
                if rank < (nhosts - 1):
                    offset = np.array([self._total_segments], dtype=np.uint)
                    comm.Send([offset, MPI.UNSIGNED_INT], dest=(rank+1))

                if rank > 0:
                    offset = np.empty(1, dtype=np.uint)
                    comm.Recv([offset, MPI.UNSIGNED_INT], source=(r-1))
                    self._offset = offset[0]
                    #offset = comm.Recv(source=(r-1))
                    #print r, offset[0]

            comm.Barrier()
