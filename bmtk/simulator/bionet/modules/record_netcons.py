import os
import csv
import h5py
import numpy as np
from neuron import h

from .sim_module import SimulatorMod
from bmtk.simulator.bionet.biocell import BioCell
# from bmtk.simulator.bionet.io_tools import io
# from bmtk.simulator.bionet.pointprocesscell import PointProcessCell
from bmtk.utils.reports import CompartmentReport

try:
    # Check to see if h5py is built to run in parallel
    if h5py.get_config().mpi:
        MembraneRecorder = CompartmentReport  # cell_vars.CellVarRecorderParallel
    else:
        MembraneRecorder = CompartmentReport  # cell_vars.CellVarRecorder
except Exception as e:
    MembraneRecorder = CompartmentReport  # cell_vars.CellVarRecorder

pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class NetconReport(SimulatorMod):
    def __init__(self, tmp_dir, file_name, variable_name, cells, sections='all', syn_type='Exp2Syn', buffer_data=True,
                 transform={}, **kwargs):
        """Module used for saving NEURON cell properties at each given step of the simulation.

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

        self._tmp_dir = tmp_dir

        self._file_name = file_name if os.path.isabs(file_name) else os.path.join(tmp_dir, file_name)
        self._all_gids = cells
        self._local_gids = []
        self._sections = sections

        self._var_recorder = None
        # self._var_recorder = MembraneRecorder(self._file_name, self._tmp_dir, self._all_variables,
        #                                       buffer_data=buffer_data, mpi_rank=MPI_RANK, mpi_size=N_HOSTS)

        self._virt_lookup = {}
        self._gid_lookup = {}
        self._sec_lookup = {}

        self._gid_list = []  # list of all gids that will have their variables saved
        self._data_block = {}  # table of variable data indexed by [gid][variable]
        self._block_step = 0  # time step within a given block
        self._curr_step = 0
        self._object_lookup = {}
        self._syn_type = syn_type
        self._gid_map = None

        # In the use-case that users passes in "dt", "start_time", or "stop_time" parameters manually. Otherwise
        # set to None and get values from corresponding simulation values in initialize() method
        self._dt = kwargs.get('dt', None)
        self._start_time = kwargs.get('start_time', None)
        self._end_time = kwargs.get('end_time', None)

        self._dt_step = None
        self._start_step = None
        self._end_step = None

    def _get_gids(self, sim):
        selected_gids = set(sim.net.get_node_set(self._all_gids).gids())
        self._local_gids = list(set(sim.local_gids) & selected_gids)

    def _save_sim_data(self, sim):
        self._var_recorder.tstart = 0.0
        self._var_recorder.tstop = sim.tstop
        self._var_recorder.dt = sim.dt

    def _get_syn_location(self, nc, cell):
        if isinstance(cell, BioCell):
            sec_x = nc.postloc()
            sec = h.cas()
            sec_id = self._sec_lookup[cell.gid][sec]  # cell.get_section_id(sec)
            h.pop_section()
            return sec_id, sec_x
        else:
            return -1, -1

    def _is_multiple(self, dividend, divisor, rtol=1.0e-4):
        if np.isclose(dividend, divisor, rtol=rtol):
            return True
        else:
            val = np.float64(dividend/divisor)
            return val.is_integer()

    def _set_valid_steps(self, sim):
        # For dt, start_time and stop_time; if not explicitly set by the users then default back to the "run" values
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
        return self._start_step <= tstep < self._end_step and tstep % self._dt_step == 0

    def initialize(self, sim):
        self._gid_map = sim.net.gid_pool
        self._get_gids(sim)

        self._set_valid_steps(sim)
        self._var_recorder = MembraneRecorder(
            self._file_name,
            mode='w',
            variable=self._variables[0],
            buffer_size=sim.nsteps_block,
            tstart=self._start_time,
            tstop=self._end_time,
            dt=self._dt,
            # n_steps=sim.n_steps
        )

        for node_pop in sim.net.node_populations:
            pop_name = node_pop.name
            for node in node_pop[0::1]:
                if node.model_type != 'virtual':
                    self._gid_lookup[node.gid] = (pop_name, node.node_id)

        for gid, cell in sim.net.get_local_cells().items():
            trg_pop, trg_id = self._gid_lookup[gid]
            if isinstance(cell, BioCell):
                self._sec_lookup[gid] = {sec_name: sec_id for sec_id, sec_name in enumerate(cell.get_sections_id())}

        for gid in self._local_gids:
            pop_id = self._gid_map.get_pool_id(gid)
            sec_list = []
            seg_list = []
            src_list = []
            syn_objects = []

            cell = sim.net.get_cell_gid(gid)
            for nc in cell.netcons:
                synapse = nc.syn()
                if self._syn_type is None or synapse.hname().startswith(self._syn_type):
                    sec_id, seg_x = self._get_syn_location(nc, cell)
                    src_gid = int(nc.srcgid())
                    sec_list.append(sec_id)
                    seg_list.append(seg_x)
                    src_list.append(src_gid)
                    syn_objects.append(nc.syn())
                elif self._syn_type == 'netcon':
                    syn_objects.append(nc)

            if syn_objects:
                # self._var_recorder.add_cell(gid, sec_list, seg_list, src_ids=src_list, trg_ids=[gid]*len(src_list))
                self._var_recorder.add_cell(
                    node_id=pop_id.node_id,
                    population=pop_id.population,
                    element_ids=sec_list,
                    element_pos=seg_list,
                    src_ids=src_list,
                    trg_ids=[gid]*len(src_list)
                )

                self._object_lookup[gid] = syn_objects

        # self._var_recorder.initialize(sim.n_steps, sim.nsteps_block)
        self._var_recorder.initialize()

    def step(self, sim, tstep):
        # save all necessary cells/variables at the current time-step into memory
        if not self._record_on_step(tstep):
            return

        for gid, netcon_objs in self._object_lookup.items():
            pop_id = self._gid_map.get_pool_id(gid)
            for var_name in self._variables:
                syn_values = [getattr(syn, var_name) for syn in netcon_objs]
                if syn_values:
                    self._var_recorder.record_cell(
                        pop_id.node_id,
                        population=pop_id.population,
                        vals=syn_values,
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
