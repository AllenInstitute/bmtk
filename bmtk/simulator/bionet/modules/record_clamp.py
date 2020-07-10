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
from neuron import h
import h5py
import numpy as np

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.utils.reports.current_writer import CurrentWriterv01

pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class ClampReport(SimulatorMod):
    def __init__(self, tmp_dir, file_name, variable_name, buffer_data=True, **kwargs):
        """Module used for saving NEURON clamp currents at each given step of the simulation.

        :param variable_name: which type of clamp it is a report of. As of now options are se, ic, and f_ic.
        :param clamps: indexes of the clamp list to make a report of.
        :param tmp_dir:
        :param file_name: name of h5 file to save variable to.
        :param buffer_data: Set to true then data will be saved to memory until written to disk during each block, reqs.
        more memory but faster. Set to false and data will be written to disk on each step (default: True)
        """

        self._tmp_dir = tmp_dir

        self._file_name = file_name if os.path.isabs(file_name) else os.path.join(tmp_dir, file_name)

        if N_HOSTS > 1:
            self._tmp_files = []
            for i in range(N_HOSTS):
                tmp_name = variable_name + str(i)
                self._tmp_files.append(os.path.join(self._tmp_dir, tmp_name))
            self._rank_file = self._tmp_files[MPI_RANK]
        else:
            self._rank_file = self._file_name

        self._var_recorder = None
        self._variable_name = variable_name

        self._buffer_data = buffer_data
    
    @property
    def variable(self):
        return self._variable_name

    def initialize(self, sim, clamps):
        self._clamps = clamps

        self._var_recorder = CurrentWriterv01(self._rank_file, num_currents=len(self._clamps),
                                              buffer_size=sim.nsteps_block, buffer_data=self._buffer_data, tstart=0.0,
                                              tstop=sim.tstop, dt=sim.dt)

        self._var_recorder.initialize()

    def step(self, sim, tstep):
        # save the current of each clamp at the current time-step.
        vals = []
        for clamp in self._clamps:
            vals.append(clamp._stim.i)
        self._var_recorder.record_clamps(vals, tstep)

    def block(self, sim, block_interval):
        # write variables in memory to file
        self._var_recorder.flush()

    def finalize(self, sim):
        # TODO: Build in mpi signaling into var_recorder
        pc.barrier()
        self._var_recorder.close()

        if MPI_RANK == 0:
            self.merge()

    def merge(self):
        if N_HOSTS > 1:
            h5final = h5py.File(self._file_name, 'w')
            all_currents = []
            for i in range(N_HOSTS):
                tmp_file = h5py.File(self._tmp_files[i], 'r')
                data = tmp_file['data'].value
                for current in data:
                    if len(current) > 0:
                        all_currents.append(current)
                tmp_file.close()
                os.remove(self._tmp_files[i])
            h5final.create_dataset('data', data=np.array(all_currents))
            h5final.close()

