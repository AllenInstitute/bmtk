# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 2017. Allen Institute. All rights reserved.
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
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
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
from neuron import h
import numpy as np
from bmtk.simulator.bionet import io
#from bmtk.simulator.bionet.recxelectrode import RecXElectrode
from bmtk.simulator.bionet.iclamp import IClamp
from bmtk.simulator.bionet import modules as mods
#from bmtk.simulator.bionet.modules.ecp import EcpMod
#from bmtk.simulator.bionet.modules.record_spikes import SpikesMod
#from bmtk.simulator.bionet.modules.record_cellvars import CellVarsMod


pc = h.ParallelContext()    # object to access MPI methods


class Simulation(object):
    """Includes methods to run and control the simulation"""

    def __init__(self, conf, network, dt, tstop, v_init, celsius, nsteps_block, start_from_state=False):
        self.net = network
        self.conf = conf
        self.gids = {'save_cell_vars': self.net.saved_gids, 'biophysical': self.net.biopyhys_gids}

        self._start_from_state = start_from_state
        self.dt = dt
        self.tstop = tstop
        self._v_init = v_init
        self._celsius = celsius
        # self.rel = None
        self.h = h

        self._nsteps_block = nsteps_block
        # self._rel_nsites = 0
        self.data_block = {}

        # self._has_cell_vars = False
        self.tstep = int(round(h.t / h.dt))
        self.tstep_start_block = self.tstep
        self.nsteps = int(round(h.tstop/h.dt))

        h.runStopAt = h.tstop
        h.steps_per_ms = 1/h.dt

        self.set_init_conditions()  # call to save state
        h.cvode.cache_efficient(1)
               
        h.pysim = self  # use this objref to be able to call postFadvance from proc advance in advance.hoc
        self._iclamps = []

        self._output_dir = 'output'
        self._log_file = 'output/log.txt'
        # self._spikes_ascii_file = 'output/spikes.txt'
        # self._spikes_hdf5_file = 'output/spikes.h5'

        self._spikes = {}

        self._cell_variables = []
        self._cell_vars_dir = 'output/cellvars'
        # self._calc_ecp = False
        # self._ecp_file = 'output/ecp.h5'

        self._sim_mods = []


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

    '''
    @property
    def nsites(self):
        return self._rel_nsites
    '''

    @property
    def n_steps(self):
        return int(round(self.tstop/self.dt))

    '''
    def set_save_variables(self, variable_list, output_dir):
        self._cell_variables = variable_list
        self._cell_vars_dir = output_dir
    '''

    @property
    def cell_variables(self):
        return self._cell_variables

    @property
    def cell_var_output(self):
        return self._cell_vars_dir

    '''
    @property
    def spikes_hdf5_file(self):
        return self._spikes_hdf5_file

    @spikes_hdf5_file.setter
    def spikes_hdf5_file(self, file_name):
        self._spikes_hdf5_file = file_name
    '''

    @property
    def spikes_table(self):
        return self._spikes


    '''
    @property
    def spikes_ascii_file(self):
        return self._spikes_ascii_file

    @spikes_ascii_file.setter
    def spikes_ascii_file(self, file_name):
        self._spikes_ascii_file = file_name
    '''

    @property
    def nsteps_block(self):
        return self._nsteps_block

    def set_init_conditions(self):
        """Set up the initial conditions: either read from the h.SaveState or from config["condidtions"]"""
        pc.set_maxstep(10)
        h.stdinit()
        self.tstep = int(round(h.t/h.dt))
        self.tstep_start_block = self.tstep

        if self._start_from_state:
            # io.read_state()
            io.print2log0('Read the initial state saved at t_sim: {} ms'.format(h.t))
        else:
            h.v_init = self.v_init  # self.conf['conditions']['v_init']

        h.celsius = self.celsius  # self.conf['conditions']['celsius']
                
    def set_spikes_recording(self):
        for gid in self.net.cells:
            tvec = self.h.Vector()
            gidvec = self.h.Vector()
            pc.spike_record(gid, tvec, gidvec)
            self._spikes[gid] = tvec

    def set_recordings(self):
        """Set recordings of ECP, spikes and somatic traces"""
        io.print2log0('Setting up recordings...')
        if not self._start_from_state:
            # if starting from a new initial state
            io.create_output_files(self, self.gids)
        else:
            io.extend_output_files(self.gids)

        # self.create_data_block()
        # self.set_spike_recording()
        io.print2log0('Recordings are set!')
        pc.barrier()

    def attach_current_clamp(self, amplitude, delay, duration, gids=None):
        # TODO: verify current clamp works with MPI
        if gids is None:
            gids = self.gids['biophysical']
        if isinstance(gids, int):
            gids = [gids]
        elif isinstance(gids, basestring):
            gids = [int(gids)]

        for gid in gids:
            if gid not in self.gids['biophysical']:
                print("Warning: attempting to attach current clamp to non-biophysical gid {}.".format(gid))

            cell = self.net.cells[gid]
            Ic = IClamp(amplitude, delay, duration)
            Ic.attach_current(cell)
            self._iclamps.append(Ic)

    def add_mod(self, module):
        self._sim_mods.append(module)

    '''
    def create_data_block(self):
        """Create block in memory to store ouputs from each time step"""
        nt_block = self.nsteps_block
            
        self.data_block = {}
        self.data_block["tsave"] = round(h.t, 3)
        self.data_block['cells'] = {}

        if self.cell_variables:
            for gid in self.gids["save_cell_vars"]:   # only includes gids on this rank

                self.data_block['cells'][gid] = {}        
                for var in self.cell_variables:
                    self.data_block['cells'][gid][var] = np.zeros(nt_block)
    '''

    '''
    def set_spike_recording(self):
        """Set dictionary of hocVectors for spike recordings"""
        spikes = {}
        for gid in self.net.cells:
            tVec = h.Vector()
            gidVec = h.Vector()
            pc.spike_record(gid, tVec, gidVec)
            spikes[gid] = tVec
            
        self.data_block["spikes"] = spikes
    '''

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

    def run(self):
        """
        Run the simulation: 
        if beginning from a blank state, then will use h.run(),
        if continuing from the saved state, then will use h.continuerun() 
        """
        for mod in self._sim_mods:
            mod.initialize(self)

        self.start_time = h.startsw()
        s_time = time.time()
        pc.timeout(0)
         
        pc.barrier()  # wait for all hosts to get to this point
        io.print2log0('Running simulation until tstop: %.3f ms with the time step %.3f ms' % (self.tstop, self.dt))

        io.print2log0('Starting timestep: %d at t_sim: %.3f ms' % (self.tstep, h.t))
        io.print2log0('Block save every %d steps' % (self.nsteps_block))

        if self._start_from_state:
            h.continuerun(h.tstop)
        else:
            h.run(h.tstop)        # <- runs simuation: works in parallel
                    
        pc.barrier()  #

        for mod in self._sim_mods:
            mod.finalize(self)
        pc.barrier()


        end_time = time.time()

        sim_time = self.__elapsed_time(end_time - s_time)
        io.print2log0now('Simulation completed in {} '.format(sim_time))

    def report_load_balance(self):
        comptime = pc.step_time()
        print('comptime: ', comptime, pc.allreduce(comptime, 1))
        avgcomp = pc.allreduce(comptime, 1)/pc.nhost()
        maxcomp = pc.allreduce(comptime, 2)
        io.print2log0('Maximum compute time is {} seconds.'.format(maxcomp))
        io.print2log0('Approximate exchange time is {} seconds.'.format(comptime - maxcomp))
        if maxcomp != 0.0:
            io.print2log0('Load balance is {}.'.format(avgcomp/maxcomp))

    def post_fadvance(self): 
        """
        Runs after every execution of fadvance (see advance.hoc)
        Called after every time step to perform computation and save data to memory block or to disk.
        The initial condition tstep=0 is not being saved 
        """
        self.tstep += 1
        tstep_block = self.tstep - self.tstep_start_block  # time step within a block

        for mod in self._sim_mods:
            mod.step(self, self.tstep)

        self.save_data_to_block(tstep_block)
        if (self.tstep % self.nsteps_block == 0) or self.tstep == self.nsteps:
            io.print2log0('    step:%d t_sim:%.3f ms' % (self.tstep, h.t))
            self.tstep_end_block = self.tstep
           
            time_step_interval = (self.tstep_start_block, self.tstep_end_block)
            io.save_block_to_disk(self.conf, self.data_block, time_step_interval)  # block save data
            # self.set_spike_recording()

            for mod in self._sim_mods:
                mod.block(self, time_step_interval)

            self.tstep_start_block = self.tstep   # starting point for the next block

    def save_data_to_block(self, tstep_block):
        """Compute data and save to a memory block"""
        self.data_block["tsave"] = round(h.t, 3)

        '''
        # save to block the intracellular variables
        if self.cell_variables:
            for gid in list(set(self.gids['save_cell_vars']) & set(self.gids['biophysical'])):
                 
                cell_data_block = self.data_block['cells'][gid] 
                cell = self.net.cells[gid] 

                for var in self.cell_variables:
                    # subtract 1 because indexes start at 0 while the time step starts at 1
                    cell_data_block[var][tstep_block-1] = getattr(cell.hobj.soma[0](0.5), var)
        '''

    @classmethod
    def from_config(cls, config, network, set_recordings=True):
        sim = cls(conf=config, network=network,
                  dt=config['run']['dt'],
                  tstop=config['run']['tstop'],
                  v_init=config['conditions']['v_init'],
                  celsius=config['conditions']['celsius'],
                  nsteps_block=config['run']['nsteps_block'])

        if config['run']['save_cell_vars']:
            cell_vars = config['run']['save_cell_vars']
            cell_vars_output = config['output']['cell_vars_dir']
            cellvars_mod = mods.CellVarsMod(outputdir=cell_vars_output, variables=cell_vars)
            sim.add_mod(cellvars_mod)

            # sim.set_save_variables(config['run']['save_cell_vars'],
            #                        config['output']['cell_vars_dir'])
            # sim.cell_variables = config['run']['save_cell_vars']

        if set_recordings:
            # sim.spikes_hdf5_file = config['output']['spikes_hdf5_file']
            # sim.spikes_ascii_file = config['output']['spikes_ascii_file']

            config_output = config['output']
            output_dir = config_output['output_dir']

            spikes_csv_file = config_output.get('spikes_ascii_file', None)
            spikes_h5_file = config_output.get('spikes_hdf5_file', None)
            if spikes_csv_file is not None or spikes_h5_file is not None:
                spikes_mod = mods.SpikesMod(tmpdir=output_dir,csv_filename=spikes_csv_file, h5_filename=spikes_h5_file)
                sim.add_mod(spikes_mod)

            if config['run']['calc_ecp']:
                ecp_mod = mods.EcpMod(ecp_file=config['output']['ecp_file'],
                                      positions_file=config['recXelectrode']['positions'])
                sim.add_mod(ecp_mod)
                # sim.calculate_ecp(config['recXelectrode']['positions'],
                #                   config['output']['ecp_file'])
            sim.set_recordings()

        if 'input' in config:
            for input_dict in config['input']:
                in_type = input_dict['type'].lower()
                if in_type == 'iclamp':
                    amplitude = input_dict['amp']
                    delay = input_dict.get('del', 0.0)
                    duration = input_dict['dur']
                    gids = input_dict.get('gids', None)
                    sim.attach_current_clamp(amplitude, delay, duration, gids)

        return sim
