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
import glob
from bmtk.utils.io.spike_trains import SpikeTrainWriter
from bmtk.simulator.pointnet.io_tools import io

import nest


MPI_RANK = nest.Rank()
N_HOSTS = nest.NumProcesses()


class SpikesMod(object):
    """Module use for saving spikes

    """

    def __init__(self, tmp_dir, spikes_file_csv=None, spikes_file=None, spikes_file_nwb=None, spikes_sort_order=None):
        self._csv_fname = spikes_file_csv
        self._h5_fname = spikes_file
        self._nwb_fname = spikes_file_nwb

        self._tmp_dir = tmp_dir
        self._tmp_file_base = 'tmp_spike_times'
        self._spike_labels = os.path.join(self._tmp_dir, self._tmp_file_base)

        self._spike_writer = SpikeTrainWriter(tmp_dir=tmp_dir, mpi_rank=MPI_RANK, mpi_size=N_HOSTS)
        self._spike_writer.delimiter = '\t'
        self._spike_writer.gid_col = 0
        self._spike_writer.time_col = 1
        self._sort_order = spikes_sort_order

        self._spike_detector = None

    def initialize(self, sim):
        self._spike_detector = nest.Create("spike_detector", 1, {'label': self._spike_labels, 'withtime': True,
                                                                 'withgid': True, 'to_file': True})

        for pop_name, pop in sim._graph._nestid2nodeid_map.items():
            nest.Connect(list(pop.keys()), self._spike_detector)

    def finalize(self, sim):
        if MPI_RANK == 0:
            for gdf_file in glob.glob(self._spike_labels + '*.gdf'):
                self._spike_writer.add_spikes_file(gdf_file)
        io.barrier()

        gid_map = sim._graph._nestid2gid

        if self._csv_fname is not None:
            self._spike_writer.to_csv(self._csv_fname, sort_order=self._sort_order, gid_map=gid_map)
            io.barrier()

        if self._h5_fname is not None:
            self._spike_writer.to_hdf5(self._h5_fname, sort_order=self._sort_order, gid_map=gid_map)
            io.barrier()

        if self._nwb_fname is not None:
            self._spike_writer.to_nwb(self._nwb_fname, sort_order=self._sort_order, gid_map=gid_map)
            io.barrier()

        self._spike_writer.close()
