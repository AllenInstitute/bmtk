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
import numpy as np
import h5py
from neuron import h
from neuron import nrn

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.io_tools import io
from bmtk.utils.reports import CompartmentReport

try:
    # Check to see if h5py is built to run in parallel
    if h5py.get_config().mpi:
        MembraneRecorder = CompartmentReport
    else:
        MembraneRecorder = CompartmentReport

except Exception as e:
    MembraneRecorder = CompartmentReport

MembraneRecorder._io = io

pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class RecordMechanisms(SimulatorMod):
    def __init__(self, output_dir, cells=None, sections='all', **kwargs):
        # self._all_variables = list(variable_name)
        # self._variables = list(variable_name)
        # self._transforms = {}
        # self._special_variables = []
        # for var_name, fnc_name in transform.items():
        #     if fnc_name is None or len(fnc_name) == 0:
        #         del self._transforms[var_name]
        #         continue
        #
        #     fnc = transforms_table[fnc_name]
        #     self._transforms[var_name] = fnc
        #     self._variables.remove(var_name)

        self._output_dir = output_dir
        self._cells = cells
        self._sections = sections
        self._gids = None
        self._gid_map = None

        self._mechanisms_map = {}

        # # TODO: Full path should be determined by config/simulation_reports module
        # self._file_name = file_name if os.path.isabs(file_name) else os.path.join(tmp_dir, file_name)
        # self._all_gids = cells
        # self._local_gids = []
        # self._sections = sections
        #
        # self._var_recorder = None
        # #self._var_recorder = MembraneRecorder(self._file_name, self._tmp_dir, self._all_variables,
        # #                                      buffer_data=buffer_data, mpi_rank=MPI_RANK, mpi_size=N_HOSTS)
        #
        # self._gid_list = gids  # list of all gids that will have their variables saved
        # self._data_block = {}  # table of variable data indexed by [gid][variable]
        # self._block_step = 0  # time step within a given block
        # self._gid_map = None

    def _get_gids(self, sim):
        gid_list = list(sim.net.get_node_set(self._cells).gids())

    def _save_sim_data(self, sim):
        #self._var_recorder.tstart = 0.0
        #self._var_recorder.tstop = sim.tstop
        #self._var_recorder.dt = sim.dt
        pass

    def initialize(self, sim):
        self._gid_map = sim.net.gid_pool
        self._gids = list(sim.net.get_node_set(self._cells).gids())

        for gid in self._gids:
            self._mechanisms_map[gid] = {}

            pop_id = self._gid_map.get_pool_id(gid)
            cell = sim.net.get_cell_gid(gid)
            cell_seg = cell.hobj.soma[0](0.5)
            for vname in dir(cell_seg):
                vobj = getattr(cell_seg, vname)
                if isinstance(vobj, nrn.Mechanism):
                    mech_vars = [n for n in dir(vobj) if isinstance(getattr(vobj, n), (int, float))]
                    if not mech_vars:
                        continue
                    else:
                        csv_fname = os.path.join(self._output_dir, 'cell{}.{}.csv'.format(gid, vname))
                        csv_writer = csv.writer(open(csv_fname, 'w'), delimiter=' ')
                        csv_writer.writerow(['time'] + mech_vars)
                        self._mechanisms_map[gid][vname] = {
                            'vars': mech_vars,
                            'csv_fname': csv_fname,
                            'csv_writer': csv_writer
                        }

            mechanisms = list(self._mechanisms_map[gid].keys())
            if mechanisms:
                io.log_info('Cell {} found mechanisms: {}'.format(gid, mechanisms))
            else:
                io.log_info('Could not find any neuron mechanisms for cell {}'.format(gid))

    def step(self, sim, tstep):
        time = (tstep+1)*sim.dt
        # exit()
        for gid, gid_mechs in self._mechanisms_map.items():
            cell = sim.net.get_cell_gid(gid)
            cell_seg = cell.hobj.soma[0](0.5)

            for mech_name, mech in gid_mechs.items():
                mech_obj = getattr(cell_seg, mech_name)
                var_vals = [getattr(mech_obj, vn) for vn in mech['vars']]
                mech['csv_writer'].writerow([time] + var_vals)

    def block(self, sim, block_interval):
        pass

    def finalize(self, sim):
        pass


class RecordSoma(RecordMechanisms):
    """Special case for when only needing to save the soma variable"""
    def __init__(self, tmp_dir, file_name, variable_name, cells, sections='soma', buffer_data=True, transform={}, **kwargs):
        super(RecordSoma, self).__init__(tmp_dir=tmp_dir, file_name=file_name, variable_name=variable_name, cells=cells,
                                         sections=sections, buffer_data=buffer_data, transform=transform, **kwargs)

    def initialize(self, sim):
        self._var_recorder = MembraneRecorder(self._file_name, mode='w', variable=self._variables[0],
                                              buffer_size=sim.nsteps_block, tstart=0.0, tstop=sim.tstop, dt=sim.dt)
        self._gid_map = sim.net.gid_pool

        self._get_gids(sim)
        # self._save_sim_data(sim)

        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            # self._var_recorder.add_cell(gid, [0], [0.5])
            self._var_recorder.add_cell(pop_id.node_id, population=pop_id.population, element_ids=[0],
                                        element_pos=[0.5])

        # self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)
        self._var_recorder.initialize()

    def step(self, sim, tstep, rel_time=0.0):
        # save all necessary cells/variables at the current time-step into memory
        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            cell = sim.net.get_cell_gid(gid)
            for var_name in self._variables:
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                # self._var_recorder.record_cell(gid, var_name, [var_val], tstep)
                self._var_recorder.record_cell(pop_id.node_id, population=pop_id.population, vals=[var_val],
                                               tstep=tstep)

            for var_name, fnc in self._transforms.items():
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                new_val = fnc(var_val)
                # self._var_recorder.record_cell(gid, var_name, [new_val], tstep)
                self._var_recorder.record_cell(pop_id.node_id, population=pop_id.population, vals=[new_val],
                                               tstep=tstep)

        self._block_step += 1
