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
from neuron import h
import decimal

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


def first_element(lst):
    return lst[0]


transforms_table = {
    'first_element': first_element,
}


swc_type_maps = {
    'undefined': 0,
    'soma': 1, 'somatic': 1,
    'axon': 2, 'axonal': 2,
    'dend': 3, 'basal': 3, 'basal dendrite': 3, '(basal) dendrite': 3,
    'apic': 4, 'apical': 4, 'apical denrite': 4,
    'custom': 5, 'other': 5
}


class MembraneReport(SimulatorMod):
    def __init__(self, tmp_dir, file_name, variable_name, cells=None, gids=None, sections='all', buffer_data=True,
                 transform={}, **kwargs):
        """Module used for saving NEURON cell properities at each given step of the simulation.

        :param tmp_dir:
        :param file_name: name of h5 file to save variable.
        :param variables: list of cell variables to record
        :param gids: list of gids to to record
        :param sections:
        :param buffer_data: Set to true then data will be saved to memory until written to disk during each block, reqs.
        more memory but faster. Set to false and data will be written to disk on each step (default: True)
        """
        self._all_variables = list(variable_name)
        self._variables = list(variable_name)
        self._report_name = self._variables[0]
        self._transforms = {}
        # self._special_variables = []
        for var_name, fnc_name in transform.items():
            if fnc_name is None or len(fnc_name) == 0:
                del self._transforms[var_name]
                continue

            fnc = transforms_table[fnc_name]
            self._transforms[var_name] = fnc
            self._variables.remove(var_name)

        self._tmp_dir = tmp_dir

        # TODO: Full path should be determined by config/simulation_reports module
        self._file_name = file_name if os.path.isabs(file_name) else os.path.join(tmp_dir, file_name)
        self._all_gids = cells
        self._local_gids = []
        self._var_recorder = None
        self._gid_list = gids  # list of all gids that will have their variables saved
        self._data_block = {}  # table of variable data indexed by [gid][variable]
        self._block_step = 0  # time step within a given block
        self._curr_step = 0
        self._gid_map = None

        # In the use-case that users passes in "dt", "start_time", or "stop_time" parameters manually. Otherwise
        # set to None and get values from corresponding simulation values in initialize() method
        self._dt = kwargs.get('dt', None)
        self._start_time = kwargs.get('start_time', None)
        self._end_time = kwargs.get('end_time', None)

        self._dt_step = None
        self._start_step = None
        self._end_step = None

        self._sections = sections if isinstance(sections, (list, tuple)) else [sections]
        self._sections = [s.lower() for s in self._sections]
        if len(self._sections) == 0 or 'all' in self._sections:
            self._section_types = {v for v in swc_type_maps.values()}
        else:
            self._section_types = {v for k, v in swc_type_maps.items() if k in self._sections}

    def _get_gids(self, sim):
        # get list of gids to save. Will only work for biophysical cells saved on the current MPI rank
        if self._gid_list is not None:
            selected_gids = set(self._gid_list)
        else:
            selected_gids = set(sim.net.get_node_set(self._all_gids).gids())
        self._local_gids = list(set(sim.biophysical_gids) & selected_gids)

    def _is_multiple(self, dividend, divisor, rtol=1.0e-4):
        if np.isclose(dividend, divisor, rtol=rtol):
            return True
        else:
            val = np.float64(dividend/divisor)
            return val.is_integer()

    def _set_valid_steps(self, sim):
        # For dt, start_time and stop_time; if not explicity set by the users then default back to the "run" values
        # in the config. If user is setting their own values, make sure they are valid/within the simulation range,
        # and make sure they are all multiples of dt.
        if self._dt is None:
            self._dt = sim.dt
        elif self._dt < sim.dt:
            raise ValueError('report dt cannot be less than simulation dt ({} < {}).'.format(
                self._dt, sim.dt
            ))
        elif not self._is_multiple(self._dt, sim.dt):
            # Users can only sample at a rate that in a whole-number multiple of the simulation dt
            raise ValueError('report dt must be a integer multiple of simulation dt ({} != {}*m).'.format(
                self._dt, sim.dt
            ))

        if self._start_time is None:
            self._start_time = sim.tstart
        elif self._start_time < sim.tstart:
            raise ValueError('start_time cannot be less than simulation tstart.')
        elif not self._is_multiple(self._start_time, self._dt):
            # Make sure the start-time occurs at a self._dt multiple
            raise ValueError('report start_time ({}) must be a integer multiple of dt ({}).'.format(
                self._start_time, self._dt
            ))

        if self._end_time is None:
            self._end_time = sim.tstop
        elif self._end_time > sim.tstop:
            raise ValueError('end_time value cannot be greater than simulation tstop ({} > {}).'.format(
                self._end_time, sim.tstop
            ))
        elif not self._is_multiple(self._end_time, self._dt):
            # Make sure the start-time occurs at a dt multiple
            raise ValueError('report end_time ({}) must be a integer multiple of dt ({}).'.format(
                self._end_time, self._dt
            ))

        self._dt_step = int(self._dt/sim.dt)
        self._start_step = int(self._start_time/sim.dt)
        self._end_step = (self._end_time/sim.dt)

    def _record_on_step(self, tstep):
        # return tstep >= self._start_step and tstep <= self._end_step and tstep % self._dt_step == 0
        return self._start_step <= tstep < self._end_step and tstep % self._dt_step == 0

    def _save_sim_data(self, sim):
        pass

    def initialize(self, sim):
        self._set_valid_steps(sim)
        self._var_recorder = MembraneRecorder(
            self._file_name,
            mode='w',
            variable=self._report_name,
            buffer_size=sim.nsteps_block,
            tstart=self._start_time,
            tstop=self._end_time,
            dt=self._dt
        )
        self._gid_map = sim.net.gid_pool

        self._get_gids(sim)
        #self._save_sim_data(sim)

        # Build segment/section list
        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            sec_list = []
            seg_list = []
            swc_ids_beg = []
            swc_ids_end = []
            seg_types = []

            cell = sim.net.get_cell_gid(gid)
            morph = cell.morphology
            segs = morph.seg_props
            for sec_id, x0, x, x1, stype in zip(segs.sec_id, segs.x0, segs.x, segs.x1, segs.type):
                if stype in self._section_types:
                    swc_id_beg, _ = morph.get_swc_id(sec_id, x0)
                    swc_id_end, _ = morph.get_swc_id(sec_id, x1)
                    sec_list.append(sec_id)
                    seg_list.append(x)

                    # TODO: Have option to turn on/off saving SWC data
                    swc_ids_beg.append(swc_id_beg)
                    swc_ids_end.append(swc_id_end)
                    seg_types.append(stype)

            self._var_recorder.add_cell(
                node_id=pop_id.node_id, population=pop_id.population,
                element_ids=sec_list, element_pos=seg_list,
                swc_ids_beg=swc_ids_beg, swc_ids_end=swc_ids_end,
                seg_types=seg_types
            )

        self._var_recorder.initialize()

    def step(self, sim, tstep):
        # save all necessary cells/variables at the current time-step into memory
        if not self._record_on_step(tstep):
            return

        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            cell = sim.net.get_cell_gid(gid)
            segs = cell.morphology.segments
            stypes = cell.morphology.seg_props.type
            for var_name in self._variables:
                seg_vals = [getattr(seg, var_name) for seg, stype in zip(segs, stypes) if stype in self._section_types]
                self._var_recorder.record_cell(
                    pop_id.node_id,
                    population=pop_id.population,
                    vals=seg_vals,
                    tstep=self._curr_step
                )

            for var_name, fnc in self._transforms.items():
                seg_vals = [fnc(getattr(seg, var_name)) for seg, stype in zip(segs, stypes) if stype in self._section_types]
                self._var_recorder.record_cell(
                    pop_id.node_id,
                    population=pop_id.population,
                    val=seg_vals,
                    tstep=self._curr_step
                )

        # self._block_step += 1
        self._curr_step += 1

    def block(self, sim, block_interval):
        # write variables in memory to file
        self._var_recorder.flush()

    def finalize(self, sim):
        # TODO: Build in mpi signaling into var_recorder
        pc.barrier()
        self._var_recorder.close()


class SomaReport(MembraneReport):
    """Special case for when only needing to save the soma variable"""
    def __init__(self, tmp_dir, file_name, variable_name, cells, sections='soma', buffer_data=True, transform={}, **kwargs):
        super(SomaReport, self).__init__(tmp_dir=tmp_dir, file_name=file_name, variable_name=variable_name, cells=cells,
                                         sections=sections, buffer_data=buffer_data, transform=transform, **kwargs)

    def initialize(self, sim):
        self._set_valid_steps(sim)
        self._var_recorder = MembraneRecorder(
            self._file_name,
            mode='w',
            variable=self._report_name,
            buffer_size=sim.nsteps_block,
            tstart=self._start_time,
            tstop=self._end_time,
            dt=self._dt
        )
        self._gid_map = sim.net.gid_pool
        self._get_gids(sim)

        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            self._var_recorder.add_cell(pop_id.node_id, population=pop_id.population, element_ids=[0],
                                        element_pos=[0.5])

        # self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)
        self._var_recorder.initialize()

    def step(self, sim, tstep, rel_time=0.0):
        # save all necessary cells/variables at the current time-step into memory
        if not self._record_on_step(tstep):
            return

        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            cell = sim.net.get_cell_gid(gid)
            for var_name in self._variables:
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                self._var_recorder.record_cell(
                    pop_id.node_id,
                    population=pop_id.population,
                    vals=[var_val],
                    tstep=self._curr_step
                )

            for var_name, fnc in self._transforms.items():
                var_val = getattr(cell.hobj.soma[0](0.5), var_name)
                new_val = fnc(var_val)
                self._var_recorder.record_cell(
                    pop_id.node_id,
                    population=pop_id.population,
                    vals=[new_val],
                    tstep=self._curr_step)

        # self._block_step += 1
        self._curr_step += 1
