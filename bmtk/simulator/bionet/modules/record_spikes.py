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
import csv
import h5py
import numpy as np
from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.utils.io.spike_trains import SpikeTrainWriter

from neuron import h


pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class SpikesMod(SimulatorMod):
    """Module use for saving spikes

    """

    def __init__(self, tmp_dir, spikes_file_csv=None, spikes_file=None, spikes_file_nwb=None, spikes_sort_order=None):
        # TODO: Have option to turn off caching spikes to csv.
        self._csv_fname = spikes_file_csv
        self._save_csv = spikes_file_csv is not None

        self._h5_fname = spikes_file
        self._save_h5 = spikes_file is not None

        self._nwb_fname = spikes_file_nwb
        self._save_nwb = spikes_file_nwb is not None

        self._tmpdir = tmp_dir
        self._sort_order = spikes_sort_order

        self._spike_writer = SpikeTrainWriter(tmp_dir=tmp_dir, mpi_rank=MPI_RANK, mpi_size=N_HOSTS)

    def initialize(self, sim):
        # TODO: since it's possible that other modules may need to access spikes, set_spikes_recordings() should
        # probably be called in the simulator itself.
        sim.set_spikes_recording()

    def block(self, sim, block_interval):
        # take spikes from Simulator spikes vector and save to the tmp file
        for gid, tVec in sim.spikes_table.items():
            for t in tVec:
                self._spike_writer.add_spike(time=t, gid=gid)

        pc.barrier()  # wait until all ranks have been saved
        sim.set_spikes_recording()  # reset recording vector

    def finalize(self, sim):
        self._spike_writer.flush()
        pc.barrier()

        if self._save_csv:
            self._spike_writer.to_csv(self._csv_fname, sort_order=self._sort_order)
            pc.barrier()

        if self._save_h5:
            self._spike_writer.to_hdf5(self._h5_fname, sort_order=self._sort_order)
            pc.barrier()

        if self._save_nwb:
            self._spike_writer.to_nwb(self._nwb_fname, sort_order=self._sort_order)
            pc.barrier()

        self._spike_writer.close()
