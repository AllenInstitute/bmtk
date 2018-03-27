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

import h5py
from neuron import h

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet import io

from bmtk.utils.io import cell_vars
try:
    # Check to see if h5py is built to run in parallel
    if h5py.get_config().mpi:
        MembraneRecorder = cell_vars.CellVarRecorderParallel
    else:
        MembraneRecorder = cell_vars.CellVarRecorder

except Exception as e:
    MembraneRecorder = cell_vars.CellVarRecorder

MembraneRecorder._io = io

pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class MembraneReport(SimulatorMod):
    def __init__(self, tmp_dir, file_name, variables, cells, sections='all', buffer_data=True):
        """Module used for saving NEURON cell properities at each given step of the simulation.

        :param tmp_dir:
        :param file_name: name of h5 file to save variable.
        :param variables: list of cell variables to record
        :param gids: list of gids to to record
        :param sections:
        :param buffer_data: Set to true then data will be saved to memory until written to disk during each block, reqs.
        more memory but faster. Set to false and data will be written to disk on each step (default: True)
        """
        self._variables = variables
        self._tmp_dir = tmp_dir
        self._file_name = file_name
        self._all_gids = cells
        self._local_gids = []
        self._sections = sections

        self._var_recorder = MembraneRecorder(file_name, tmp_dir, self._variables, buffer_data=buffer_data,
                                              mpi_rank=MPI_RANK, mpi_size=N_HOSTS)

        self._gid_list = []  # list of all gids that will have their variables saved
        self._data_block = {}  # table of variable data indexed by [gid][variable]
        self._block_step = 0  # time step within a given block

    def _get_gids(self, sim):
        # get list of gids to save. Will only work for biophysical cells saved on the current MPI rank
        self._local_gids = list(set(sim.gids['biophysical']) & set(self._all_gids))

    def initialize(self, sim):
        self._get_gids(sim)

        # TODO: get section by name and/or list of section ids
        # Build segment/section list
        for gid in self._local_gids:
            sec_list = []
            seg_list = []
            cell = sim.net.get_local_cell(gid)
            cell.store_segments()
            for sec_id, sec in enumerate(cell.get_sections()):
                for seg in sec:
                    # TODO: Make sure the seg has the recorded variable(s)
                    sec_list.append(sec_id)
                    seg_list.append(seg.x)

            self._var_recorder.add_cell(gid, sec_list, seg_list)

        self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)

    def step(self, sim, tstep, rel_time=0.0):
        # save all necessary cells/variables at the current time-step into memory
        for gid in self._local_gids:
            cell = sim.net.get_local_cell(gid)
            for var_name in self._variables:
                seg_vals = [getattr(seg, var_name) for seg in cell.get_segments()]
                self._var_recorder.record_cell(gid, var_name, seg_vals, tstep)

        self._block_step += 1

    def block(self, sim, block_interval):
        # write variables in memory to file
        self._var_recorder.flush()

    def finalize(self, sim):
        if self._block_step > 0:
            # TODO: Write partial block
            # just in case the simulation doesn't end on a block step
            self.block(sim, (sim.n_steps - self._block_step, sim.n_steps))

        pc.barrier()
        self._var_recorder.close()

        pc.barrier()
        self._var_recorder.merge()


class SomaReport(MembraneReport):
    """Special case for when only needing to save the soma variable"""
    def __init__(self, tmp_dir, file_name, variable_name, cells, sections='soma', buffer_data=True):
        super(SomaReport, self).__init__(tmp_dir=tmp_dir, file_name=file_name, variables=variable_name, cells=cells,
                                         sections=sections, buffer_data=buffer_data)

    def initialize(self, sim):
        self._get_gids(sim)

        for gid in self._local_gids:
            self._var_recorder.add_cell(gid, [0], [0.5])
        self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)

    def step(self, sim, tstep, rel_time=0.0):
        # save all necessary cells/variables at the current time-step into memory
        for gid in self._local_gids:
            cell = sim.net.get_local_cell(gid)
            for var_name in self._variables:
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                self._var_recorder.record_cell(gid, var_name, [var_val], tstep)

        self._block_step += 1
