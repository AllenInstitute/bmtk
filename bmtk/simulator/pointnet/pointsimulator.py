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
import nest
from six import string_types
from six import moves

from bmtk.simulator.core.simulator import Simulator
from bmtk.simulator.pointnet.config import Config
from bmtk.simulator.pointnet.io_tools import io
import bmtk.simulator.utils.simulation_reports as reports
import bmtk.simulator.utils.simulation_inputs as inputs
from bmtk.utils.reports.spike_trains import SpikeTrains
from . import modules as mods
from bmtk.simulator.core.node_sets import NodeSet


class PointSimulator(Simulator):
    def __init__(self, graph, dt=0.001, overwrite=True, print_time=False, n_thread=1):
        self._tstop = 0.0  # simulation time
        self._dt = dt  # time step
        self._output_dir = './output/'  # directory where log and temporary output will be stored
        self._overwrite = overwrite
        self._block_run = False
        self._block_size = -1

        self._cells_built = False
        self._internal_connections_built = False

        self._graph = graph
        self._external_cells = {}  # dict-of-dict of external pointnet cells with keys [network_name][cell_id]
        self._internal_cells = {}  # dictionary of internal pointnet cells with cell_id as key
        self._nest_id_map = {}  # a map between NEST IDs and Node-IDs

        self._spikedetector = None
        self._spikes_file = None  # File where all output spikes will be collected and saved
        self._tmp_spikes_file = None  # temporary gdf files of spike-trains
        self._spike_trains_ds = {}  # used to temporary store NWB datasets containing spike trains

        self._spike_detector = None

        self._mods = []

        self._inputs = {}  # Used to hold references to nest input objects (current_generators, etc)

        # TODO: Make this a parameter in the config file
        # TODO: Allow different inputs to have different options
        self._sg_params = {'precise_times': True}

        # Reset the NEST kernel for a new simualtion
        # TODO: move this into it's own function and make sure it is called before network is built
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": self._dt, "overwrite_files": self._overwrite, "print_time": print_time, "local_num_threads": n_thread})

    @property
    def tstart(self):
        return 0.0

    @property
    def dt(self):
        return self._dt

    @property
    def tstop(self):
        return self._tstop

    def simulation_time(self, units='ms'):
        units_lc = units.lower()
        time_ms = self.tstop - self.tstart
        if units_lc == 'ms':
            return time_ms
        elif units_lc == 's':
            return time_ms/1000.0

    @tstop.setter
    def tstop(self, val):
        self._tstop = val

    @property
    def n_steps(self):
        return int((self.tstop-self.tstart)/self.dt)

    @property
    def net(self):
        return self._graph

    @property
    def gid_map(self):
        return self._graph._nestid2gid

    def set_spike_generator_params(self, **params):
        self._sg_params = params

    def get_spike_generator_params(self):
        return self._sg_params

    def _get_block_trial(self, duration):
        """
        Compute necessary number of block trials, the length of block simulation and the simulation length of the last
        block run, if necessary.
        """
        if self._block_run:
            data_res = self._block_size * self._dt
            fn = duration / data_res
            n = int(fn)
            res = fn - n
        else:
            n = -1
            res = -1
            data_res = -1
        return n, res, data_res

    '''
    def set_spikes_recordings(self):
        # TODO: Pass in output-dir and file name to save to
        # TODO: Allow for sorting - overwrite bionet module
        self._spike_detector = nest.Create("spike_detector", 1, {'label': os.path.join(self.output_dir, 'tmp_spike_times'),
                                             'withtime': True, 'withgid': True, 'to_file': True})
        # print self._spike_detector

        for pop_name, pop in self._graph._nestid2nodeid_map.items():
            # print pop.keys()

            nest.Connect(pop.keys(), self._spike_detector)
        # exit()
    '''
            
    def run(self, tstop=None):
        if tstop is None:
            tstop = self._tstop

        for mod in self._mods:
            mod.initialize(self)

        io.barrier()

        io.log_info('Starting Simulation')
        n, res, data_res = self._get_block_trial(tstop)
        if n > 0:
            for r in moves.range(n):
                nest.Simulate(data_res)
        if res > 0:
            nest.Simulate(res * self.dt)
        if n < 0:
            nest.Simulate(tstop)

        io.barrier()
        io.log_info('Simulation finished, finalizing results.')
        for mod in self._mods:
            mod.finalize(self)
        io.barrier()
        io.log_info('Done.')

    def add_mod(self, mod):
        self._mods.append(mod)

    @classmethod
    def from_config(cls, configure, graph, n_thread=None):
        # load the json file or object
        if isinstance(configure, string_types):
            config = Config.from_json(configure, validate=True)
        elif isinstance(configure, dict):
            config = configure
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(configure, type(configure)))

        if 'run' not in config:
            raise Exception('Json file is missing "run" entry. Unable to build PointNetwork.')
        run_dict = config['run']
        
        # override the n_thread setting from the config file
        if 'n_thread' in run_dict:
            if n_thread is not None and n_thread != run_dict['n_thread']:
                # give a warinig that the user is overriding the argument.
                io.log_warning(
                    f'Overriding n_thread setting in an argument ({n_thread}) with the value in config file ({run_dict["n_thread"]}).'
                )
            n_thread = run_dict['n_thread']
        else:
            if n_thread is None:
                n_thread = 1  # default to 1 thread if not set in config or argument


        # Get network parameters
        # step time (dt) is set in the kernel and should be passed
        overwrite = run_dict['overwrite_output_dir'] if 'overwrite_output_dir' in run_dict else True
        print_time = run_dict['print_time'] if 'print_time' in run_dict else False
        dt = run_dict['dt']  # TODO: make sure dt exists
        network = cls(graph, dt=dt, overwrite=overwrite, n_thread=n_thread)

        if 'output_dir' in config['output']:
            network.output_dir = config['output']['output_dir']

        if 'block_run' in run_dict and run_dict['block_run']:
            if 'block_size' not in run_dict:
                raise Exception('"block_run" is set to True but "block_size" not found.')
            network._block_size = run_dict['block_size']

        if 'duration' in run_dict:
            network.tstop = run_dict['duration']
        elif 'tstop' in run_dict:
            network.tstop = run_dict['tstop']

        if 'precise_times' in run_dict:
            network.set_spike_generator_params(precise_times=run_dict['precise_times'])

        if run_dict.get('allow_offgrid_spikes', False):
            network.set_spike_generator_params(allow_offgrid_spikes=True)

        # Create the output-directory, or delete existing files if it already exists
        graph.io.log_info('Setting up output directory')
        if not os.path.exists(config['output']['output_dir']):
            os.mkdir(config['output']['output_dir'])
        elif overwrite:
            for gfile in glob.glob(os.path.join(config['output']['output_dir'], '*.gdf')):
                os.remove(gfile)

        for sim_input in inputs.from_config(config):
            if sim_input.input_type == 'spikes' and sim_input.module in ['nwb', 'csv', 'sonata', 'h5', 'hdf5']:
                network.add_mod(mods.SpikesInputsMod(
                    name=sim_input.name,
                    input_type=sim_input.input_type,
                    module=sim_input.module,
                    # node_set=sim_input.node_set,
                    **sim_input.params
                ))

            elif sim_input.module == 'IClamp':
                network.add_mod(mods.IClampMod(input_type=sim_input.input_type, **sim_input.params))

            elif sim_input.module == 'ecephys_probe':
                network.add_mod(mods.PointECEphysUnitsModule(name=sim_input.name, **sim_input.params))
            
            else:
                graph.io.log_warning('Unknown input type {}'.format(sim_input.input_type))

        sim_reports = reports.from_config(config)
        for report in sim_reports:           
            if report.module == 'spikes_report':
                mod = mods.SpikesMod(**report.params)

            elif isinstance(report, reports.MembraneReport):
                # For convience and for compliance with SONATA format. "membrane_report" and "multimeter_report is the
                # same in pointnet.
                mod = mods.MultimeterMod(**report.params)
            
            elif isinstance(report, reports.WeightRecorder):
                mod = mods.WeightRecorder(name=report.report_name, **report.params)

            else:
                graph.io.log_exception('Unknown report type {}'.format(report.module))

            mod.preload(sim=graph)
            network.add_mod(mod)

        graph.io.log_info('Building cells.')
        graph.build_nodes()

        graph.io.log_info('Building recurrent connections')
        graph.build_recurrent_edges()

        io.log_info('Network created.')
        return network
