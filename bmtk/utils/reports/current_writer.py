# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import os
import h5py
import numpy as np

from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version

class CurrentWriterv01(object):
    """Used to save current currents to the described hdf5 format.
    """

    class DataTable(object):
        """A small struct to keep track of different data (and buffer) tables"""
        def __init__(self):
            # If buffering data, buffer_block will be an in-memory array and will write to data_block during when
            # filled. If not buffering buffer_block is an hdf5 dataset and data_block is ignored
            self.data_block = None
            self.buffer_block = None

    def __init__(self, file_name, num_currents, units=None, tstart=0.0, tstop=1.0, dt=0.01, n_steps=None,
                 buffer_size=0, buffer_data=True, **kwargs):
        self._units = units
        self._tstart = tstart
        self._tstop = tstop
        self._dt = dt

        self._num_currents = num_currents

        self._n_steps = n_steps

        self._buffer_size = buffer_size
        self._buffer_data = buffer_data and buffer_size>0
        self._data_block = self.DataTable()
        self._last_save_indx = 0  # for buffering, used to keep track of last timestep data was saved to disk

        self._file_name = file_name
        self._create_h5file()

        self._buffer_block_size = 0
        self._total_steps = 0

        self._is_initialized = False

    def set_units(self, val):
        self._units = val

    def units(self):
        return self._units

    def set_tstart(self, val):
        self._tstart = val

    def tstart(self):
        return self._tstart

    def set_tstop(self, val):
        self._tstop = val

    def tstop(self):
        return self._tstop

    def set_dt(self, val):
        self._dt = val

    def dt(self):
        return self._dt

    def n_steps(self):
        if self._n_steps is None:
            self._n_steps = int((self._tstop - self._tstart) / self._dt)
        return self._n_steps

    def _create_h5file(self):
        fdir = os.path.dirname(os.path.abspath(self._file_name))
        if not os.path.exists(fdir):
            os.mkdir(fdir)
        self.h5_base = h5py.File(self._file_name, 'w')
        add_hdf5_version(self.h5_base)
        add_hdf5_magic(self.h5_base)

    def initialize(self, **kwargs):
        if self._is_initialized:
            return

        n_steps = self.n_steps()
        if n_steps <= 0:
            raise Exception('A non-zero positive integer num-of-steps is required to initialize the current report.'
                            'Please specify report length using the n_steps parameters (or using appropiate tstop,'
                            'tstart, and dt).')

        #self._calc_offset()
        base_grp = self.h5_base

        self._total_steps = n_steps
        self._buffer_block_size = self._buffer_size

        if self._buffer_data:
            # Set up in-memory block to buffer recorded variables before writing to the dataset.
            self._data_block.buffer_block = np.zeros((self._buffer_size, self._num_currents), dtype=np.float)

            self._data_block.data_block = base_grp.create_dataset('data', shape=(self.n_steps(), self._num_currents),
                                                                  dtype=np.float, chunks=True)

            if self._units is not None:
                self._data_block.data_block.attrs['units'] = self._units

        else:
            # Since we are not buffering data, we just write directly to the on-disk dataset.
            self._data_block.buffer_block = base_grp.create_dataset('data', shape=(self.n_steps(), self._n_segments_all),
                                                               dtype=np.float, chunks=True)

            if self._units is not None:
                self._data_block.buffer_block.attrs['units'] = self._units

        self._is_initialized = True

    def record_clamps(self, vals, tstep):
        """Record clamp currents.

        :param vals: list of currents for each clamp
        :param tstep: time step
        """
        self.initialize()
        buffer_block = self._data_block.buffer_block
        update_index = (tstep - self._last_save_indx)
        buffer_block[update_index, :] = vals

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
        self.h5_base.close()

    def merge(self):
        pass
