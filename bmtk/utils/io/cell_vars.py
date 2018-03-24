import h5py
import numpy as np


class CellVarRecorder(object):
    def __init__(self, file_name, tmp_dir, variable, buffer_data=True):
        self._file_name = file_name
        self._tmp_dir = tmp_dir
        self._variable = variable

        #self._mpi_rank = mpi_rank
        #self._mpi_size = mpi_size

        self._mapping_gids = []
        self._gid_map = {}
        #self._gid_map_counter = 0

        self._mapping_element_ids = []
        self._mapping_element_pos = []
        self._mapping_index = [0]
        #self._mapping_index_ptr = 0
        self._total_segments = 0

        self._buffer_data = buffer_data
        self._data_block = None
        self._buffer_block = None
        self._last_save_indx = 0
        self._buffer_block_size = 0
        #self._index_modifer = 0

    def add_cell(self, gid, sec_list, seg_list):
        assert(len(sec_list) == len(seg_list))
        n_segs = len(seg_list)
        self._gid_map[gid] = (self._total_segments, self._total_segments + n_segs)
        self._mapping_gids.append(gid)
        self._mapping_element_ids.extend(sec_list)
        self._mapping_element_pos.extend(seg_list)
        self._mapping_index.append(self._mapping_index[-1] + n_segs)
        self._total_segments += n_segs

    def initialize(self, n_steps, buffer_size=0):
        h5file = h5py.File(self._file_name, 'w')
        var_grp = h5file.create_group(self._variable)
        var_grp.create_dataset('mapping/gids', data=self._mapping_gids)
        var_grp.create_dataset('mapping/element_id', data=self._mapping_element_ids)
        var_grp.create_dataset('mapping/element_pos', data=self._mapping_element_pos)

        if self._buffer_data:
            self._buffer_block = np.zeros((buffer_size, self._total_segments), dtype=np.float)
            self._data_block = var_grp.create_dataset('data', shape=(buffer_size, self._total_segments), dtype=np.float,
                                                      chunks=True)
            self._buffer_block_num = 0
            self._buffer_block_size = buffer_size
        else:
            self._buffer_block = var_grp.create_dataset('data', shape=(n_steps, self._total_segments), dtype=np.float,
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
            print blk_beg, blk_end

            self._data_block[blk_beg:blk_end, :] = self._buffer_block
            self._last_save_indx += self._buffer_block_size
