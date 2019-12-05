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
import time
from six import string_types
from neuron import h
from bmtk.simulator.core.simulator import Simulator
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet.iclamp import IClamp
from bmtk.simulator.bionet import modules as mods
from bmtk.simulator.core.node_sets import NodeSet
import bmtk.simulator.utils.simulation_reports as reports
import bmtk.simulator.utils.simulation_inputs as inputs
from bmtk.utils.io import spike_trains

from bmtk.utils.reports.spike_trains import SpikeTrains


pc = h.ParallelContext()    # object to access MPI methods


class BioSimulator(Simulator):
    """Includes methods to run and control the simulation"""

    def __init__(self, network, dt, tstop, v_init, celsius, nsteps_block, start_from_state=False):
        self.net = network

        self._start_from_state = start_from_state
        self.dt = dt
        self.tstop = tstop
        self.tstart = 0.0

        self._v_init = v_init
        self._celsius = celsius
        self._h = h

        self.tstep = int(round(h.t / h.dt))
        self.tstep_start_block = self.tstep
        self.nsteps = int(round(h.tstop/h.dt))

        # make sure the block size isn't small than the total number of steps
        # TODO: should we send a warning that block-step size is being reset?
        self._nsteps_block = nsteps_block if self.nsteps > nsteps_block else self.nsteps

        self.__tstep_end_block = 0
        self.__tstep_start_block = 0

        h.runStopAt = h.tstop
        h.steps_per_ms = 1/h.dt

        self._set_init_conditions()  # call to save state
        h.cvode.cache_efficient(1)
               
        h.pysim = self  # use this objref to be able to call postFadvance from proc advance in advance.hoc
        self._iclamps = []

        self._output_dir = 'output'
        self._log_file = 'output/log.txt'

        self._spikes = {}  # for keeping track of different spike times, key of cell gids

        self._cell_variables = []  # location of saved cell variables
        self._cell_vars_dir = 'output/cellvars'

        self._sim_mods = []  # list of modules.SimulatorMod's

    @property
    def dt(self):
        return h.dt

    @dt.setter
    def dt(self, ms):
        h.dt = ms

    @property
    def tstop(self):
        return h.tstop

    @tstop.setter
    def tstop(self, ms):
        h.tstop = ms

    @property
    def v_init(self):
        return self._v_init

    @v_init.setter
    def v_init(self, voltage):
        self._v_init = voltage

    @property
    def celsius(self):
        return self._celsius

    @celsius.setter
    def celsius(self, c):
        self._celsius = c

    @property
    def n_steps(self):
        return int(round(self.tstop/self.dt))

    @property
    def cell_variables(self):
        return self._cell_variables

    @property
    def cell_var_output(self):
        return self._cell_vars_dir

    @property
    def spikes_table(self):
        return self._spikes

    @property
    def nsteps_block(self):
        return self._nsteps_block

    @property
    def h(self):
        return self._h

    @property
    def biophysical_gids(self):
        return self.net.cell_type_maps('biophysical').keys()

    @property
    def local_gids(self):
        # return self.net.get
        return self.net.local_gids

    def simulation_time(self, units='ms'):
        units_lc = units.lower()
        time_ms = self.tstop - self.tstart
        if units_lc == 'ms':
            return time_ms
        elif units_lc == 's':
            return time_ms/1000.0

        else:
            raise AttributeError('Uknown unit type {}'.format(units))

    def __elapsed_time(self, time_s):
        if time_s < 120:
            return '{:.4} seconds'.format(time_s)
        elif time_s < 7200:
            mins, secs = divmod(time_s, 60)
            return '{} minutes, {:.4} seconds'.format(mins, secs)
        else:
            mins, secs = divmod(time_s, 60)
            hours, mins = divmod(mins, 60)
            return '{} hours, {} minutes and {:.4} seconds'.format(hours, mins, secs)

    def _set_init_conditions(self):
        """Set up the initial conditions: either read from the h.SaveState or from config["condidtions"]"""
        pc.set_maxstep(10)
        h.stdinit()
        self.tstep = int(round(h.t/h.dt))
        self.tstep_start_block = self.tstep

        if self._start_from_state:
            # io.read_state()
            io.log_info('Read the initial state saved at t_sim: {} ms'.format(h.t))
        else:
            h.v_init = self.v_init

        h.celsius = self.celsius
                
    def set_spikes_recording(self):
        for gid, _ in self.net.get_local_cells().items():
            tvec = self.h.Vector()
            gidvec = self.h.Vector()
            pc.spike_record(gid, tvec, gidvec)
            self._spikes[gid] = tvec

    def attach_current_clamp(self, amplitude, delay, duration, gids=None):

        # TODO: Create appropiate module
        if gids is None or gids=='all':
            gids = self.biophysical_gids

        if isinstance(gids, int):
            gids = [gids]
        elif isinstance(gids, string_types):
            gids = [int(gids)]
        elif isinstance(gids, NodeSet):
            gids = gids.gids()


        gids = list(set(self.local_gids) & set(gids))

        for idx,gid in enumerate(gids):
            cell = self.net.get_cell_gid(gid)
            Ic = IClamp(amplitude[idx], delay, duration)

            Ic.attach_current(cell)
            self._iclamps.append(Ic)

    def add_mod(self, module):
        self._sim_mods.append(module)

    def run(self):
        """Run the simulation:
        if beginning from a blank state, then will use h.run(),
        if continuing from the saved state, then will use h.continuerun() 
        """
        for mod in self._sim_mods:
            mod.initialize(self)

        self.start_time = h.startsw()
        s_time = time.time()
        pc.timeout(0)
         
        pc.barrier()  # wait for all hosts to get to this point
        io.log_info('Running simulation for {:.3f} ms with the time step {:.3f} ms'.format(self.tstop, self.dt))
        io.log_info('Starting timestep: {} at t_sim: {:.3f} ms'.format(self.tstep, h.t))
        io.log_info('Block save every {} steps'.format(self.nsteps_block))

        if self._start_from_state:
            h.continuerun(h.tstop)
        else:
            h.run(h.tstop)        # <- runs simuation: works in parallel
                    
        pc.barrier()

        for mod in self._sim_mods:
            mod.finalize(self)
        pc.barrier()

        end_time = time.time()

        sim_time = self.__elapsed_time(end_time - s_time)
        io.log_info('Simulation completed in {} '.format(sim_time))

    def report_load_balance(self):
        comptime = pc.step_time()
        avgcomp = pc.allreduce(comptime, 1)/pc.nhost()
        maxcomp = pc.allreduce(comptime, 2)
        io.log_info('Maximum compute time is {} seconds.'.format(maxcomp))
        io.log_info('Approximate exchange time is {} seconds.'.format(comptime - maxcomp))
        if maxcomp != 0.0:
            io.log_info('Load balance is {}.'.format(avgcomp/maxcomp))

    def post_fadvance(self): 
        """
        Runs after every execution of fadvance (see advance.hoc)
        Called after every time step to perform computation and save data to memory block or to disk.
        The initial condition tstep=0 is not being saved 
        """
        for mod in self._sim_mods:
            mod.step(self, self.tstep)

        self.tstep += 1

        if (self.tstep % self.nsteps_block == 0) or self.tstep == self.nsteps:
            io.log_info('    step:{} t_sim:{:.2f} ms'.format(self.tstep, h.t))
            self.__tstep_end_block = self.tstep
            time_step_interval = (self.__tstep_start_block, self.__tstep_end_block)

            for mod in self._sim_mods:
                mod.block(self, time_step_interval)

            self.__tstep_start_block = self.tstep   # starting point for the next block

    @classmethod
    def from_config(cls, config, network, set_recordings=True):
        # TODO: convert from json to sonata config if necessary

        sim = cls(network=network,
                  dt=config.dt,
                  tstop=config.tstop,
                  v_init=config.v_init,
                  celsius=config.celsius,
                  nsteps_block=config.block_step)

        network.io.log_info('Building cells.')
        network.build_nodes()

        network.io.log_info('Building recurrent connections')
        network.build_recurrent_edges()

        # TODO: Need to create a gid selector
        for sim_input in inputs.from_config(config):
            node_set = network.get_node_set(sim_input.node_set)
            if sim_input.input_type == 'spikes':
                io.log_info('Building virtual cell stimulations for {}'.format(sim_input.name))
                path = sim_input.params['input_file']
                spikes = SpikeTrains.load(path=path, file_type=sim_input.module, **sim_input.params)
                #exit()
                #spikes = spike_trains.SpikesInput.load(name=sim_input.name, module=sim_input.module,
                #                                       input_type=sim_input.input_type, params=sim_input.params)
                network.add_spike_trains(spikes, node_set)

            elif sim_input.module == 'IClamp':
                # TODO: Parse from csv file
                amplitude = sim_input.params['amp']
                delay = sim_input.params['delay']
                duration = sim_input.params['duration']
                gids = sim_input.params['node_set']
                sim.attach_current_clamp(amplitude, delay, duration, node_set)

            elif sim_input.module == 'xstim':
                sim.add_mod(mods.XStimMod(**sim_input.params))

            else:
                io.log_exception('Can not parse input format {}'.format(sim_input.name))

        # Parse the "reports" section of the config and load an associated output module for each report
        sim_reports = reports.from_config(config)
        for report in sim_reports:
            if isinstance(report, reports.SpikesReport):
                mod = mods.SpikesMod(**report.params)

            elif report.module == 'netcon_report':
                mod = mods.NetconReport(**report.params)

            elif isinstance(report, reports.MembraneReport):
                if report.params['sections'] == 'soma':
                    mod = mods.SomaReport(**report.params)

                else:
                    mod = mods.MembraneReport(**report.params)

            elif isinstance(report, reports.ECPReport):
                mod = mods.EcpMod(**report.params)
                # Set up the ability for ecp on all relevant cells
                # TODO: According to spec we need to allow a different subset other than only biophysical cells
                for gid, cell in network.cell_type_maps('biophysical').items():
                    cell.setup_ecp()

            elif report.module == 'save_synapses':
                mod = mods.SaveSynapses(**report.params)



            else:
                # TODO: Allow users to register customized modules using pymodules
                io.log_warning('Unrecognized module {}, skipping.'.format(report.module))
                continue

            sim.add_mod(mod)

        return sim
