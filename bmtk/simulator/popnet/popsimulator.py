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
import logging
from six import string_types

from dipde.internals.internalpopulation import InternalPopulation
from dipde.internals.externalpopulation import ExternalPopulation
from dipde.internals.connection import Connection
import dipde

from bmtk.simulator.core.simulator import Simulator
from . import config as cfg
from . import utils as poputils
import bmtk.simulator.utils.simulation_inputs as inputs
from bmtk.utils.reports.spike_trains import SpikeTrains
from bmtk.utils.io import firing_rates


class PopSimulator(Simulator):
    def __init__(self, graph, dt=0.0001, tstop=0.0, overwrite=True):
        self._graph = graph

        self._tstop = tstop
        self._dt = dt
        self._rates_file = None  # name of file where the output is saved

        self.__population_list = []  # list of all populations, internal and external
        self.__connection_list = []  # list of all connections
        self._dipde_network = None  # reference to dipde.Network object

        self.io = self._graph.io

    @property
    def tstop(self):
        return self._tstop

    @tstop.setter
    def tstop(self, value):
        self._tstop = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def rates_file(self):
        return self._rates_file

    @rates_file.setter
    def rates_file(self, value):
        self._rates_file = value

    @property
    def populations(self):
        return self.__population_list

    @property
    def connections(self):
        return self.__connection_list

    def add_rates_nwb(self, network, nwb_file, trial, force=False):
        """Creates external population firing rates from an NWB file.

        Will iterate through a processing trial of an NWB file by assigning gids the population it belongs too and
        taking the average firing rate.

        This should be done before calling build_cells(). If a population has already been assigned a firing rate an
        error will occur unless force=True.

        :param network: Name of network with external populations.
        :param nwb_file: NWB file with spike rates.
        :param trial: trial id in NWB file
        :param force: will overwrite existing firing rates
        """
        existing_rates = self._rates[network]  # TODO: validate network exists
        # Get all unset, external populations in a network.
        network_pops = self._graph.get_populations(network)
        selected_pops = []
        for pop in network_pops:
            if pop.is_internal:
                continue
            elif not force and pop.pop_id in existing_rates:
                self.io.log_info('Firing rate for {}/{} has already been set, skipping.'.format(network, pop.pop_id))
            else:
                selected_pops.append(pop)

        if selected_pops:
            # assign firing rates from NWB file
            # TODO:
            rates_dict = poputils.get_firing_rate_from_nwb(selected_pops, nwb_file, trial)
            self._rates[network].update(rates_dict)

    def add_rate_hz(self, network, pop_id, rate, force=False):
        """Set the firing rate of an external population.

        This should be done before calling build_cells(). If a population has already been assigned a firing rate an
        error will occur unless force=True.

        :param network: name of network with wanted exteranl population
        :param pop_id: name/id of external population
        :param rate: firing rate in Hz.
        :param force: will overwrite existing firing rates
        """
        self.__add_rates_validator(network, pop_id, force)
        self._rates[network][pop_id] = rate

    def __add_rates_validator(self, network, pop_id, force):
        if network not in self._graph.networks:
            raise Exception('No network {} found in PopGraph.'.format(network))

        pop = self._graph.get_population(network, pop_id)
        if pop is None:
            raise Exception('No population with id {} found in {}.'.format(pop_id, network))
        if pop.is_internal:
            raise Exception('Population {} in {} is not an external population.'.format(pop_id, network))
        if not force and pop_id in self._rates[network]:
            raise Exception('The firing rate for {}/{} already set and force=False.'.format(network, pop_id))

    def _get_rate(self, network, pop):
        """Gets the firing rate for a given population"""
        return self._rates[network][pop.pop_id]

    def build_populations(self):
        """Build dipde Population objects from graph nodes.

        To calculate external populations firing rates, it first see if a population's firing rate has been manually
        set in the graph. Otherwise it attempts to calulate the firing rate from the call to add_rate_hz, add_rates_NWB,
        etc. (which should be called first).
        """
        for network in self._graph.networks:
            for pop in self._graph.get_populations(network):
                if pop.is_internal:
                    dipde_pop = self.__create_internal_pop(pop)

                else:
                    dipde_pop = self.__create_external_pop(pop, self._get_rate(network, pop))

                self.__population_list.append(dipde_pop)
                self.__population_table[network][pop.pop_id] = dipde_pop

    def set_logging(self, log_file):
        # TODO: move this out of the function, put in io class
        if os.path.exists(log_file):
            os.remove(log_file)

        # get root logger
        logger = logging.getLogger()
        for h in list(logger.handlers):
            # remove existing handlers that will write to console.
            logger.removeHandler(h)

        # creates handler that write to log_file
        logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)

    def set_external_connections(self, network_name):
        """Sets the external connections for populations in a given network.

        :param network_name: name of external network with External Populations to connect to internal pops.
        """
        for edge in self._graph.get_edges(network_name):
            # Get source and target populations
            src = edge.source
            source_pop = self.__population_table[src.network][src.pop_id]
            trg = edge.target
            target_pop = self.__population_table[trg.network][trg.pop_id]

            # build a connection.
            self.__connection_list.append(self.__create_connection(source_pop, target_pop, edge))

    def set_recurrent_connections(self):
        """Initialize internal connections."""
        for network in self._graph.internal_networks():
            for edge in self._graph.get_edges(network):
                src = edge.source
                source_pop = self.__population_table[src.network][src.pop_id]
                trg = edge.target
                target_pop = self.__population_table[trg.network][trg.pop_id]
                self.__connection_list.append(self.__create_connection(source_pop, target_pop, edge))

    def run(self, tstop=None):
        # TODO: Check if cells/connections need to be rebuilt.

        # Create the network
        dipde_pops = [p.dipde_obj for p in self._graph.populations]
        dipde_conns = [c.dipde_obj for c in self._graph.connections]
        self._dipde_network = dipde.Network(population_list=dipde_pops, connection_list=dipde_conns)

        if tstop is None:
            tstop = self.tstop

        self.io.log_info("Running simulation.")
        self._dipde_network.run(t0=0.0, tf=tstop, dt=self.dt)
        # TODO: make record_rates optional?
        self.__record_rates()
        self.io.log_info("Finished simulation.")

    def __create_internal_pop(self, params):
        # TODO: use getter methods directly in case arguments are not stored in dynamics params
        # pop = InternalPopulation(**params.dynamics_params)
        pop = InternalPopulation(**params.model_params)
        return pop

    def __create_external_pop(self, params, rates):
        pop = ExternalPopulation(rates, record=False)
        return pop

    def __create_connection(self, source, target, params):
        return Connection(source, target, nsyn=params.nsyns, delays=params.delay, weights=params.weight)

    def __record_rates(self):
        with open(self._rates_file, 'w') as f:
            for pop in self._graph.internal_populations:
                if pop.record:
                    for time, rate in zip(pop.dipde_obj.t_record, pop.dipde_obj.firing_rate_record):
                        f.write('{} {} {}\n'.format(pop.pop_id, time, rate))

    @classmethod
    def from_config(cls, configure, graph):
        # load the json file or object
        if isinstance(configure, string_types):
            config = cfg.from_json(configure, validate=True)
        elif isinstance(configure, dict):
            config = configure
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(configure, type(configure)))

        if 'run' not in config:
            raise Exception('Json file is missing "run" entry. Unable to build Bionetwork.')
        run_dict = config['run']

        # Get network parameters
        # step time (dt) is set in the kernel and should be passed
        overwrite = run_dict['overwrite_output_dir'] if 'overwrite_output_dir' in run_dict else True
        print_time = run_dict['print_time'] if 'print_time' in run_dict else False
        dt = run_dict['dt']  # TODO: make sure dt exists
        tstop = float(config.tstop) / 1000.0
        network = cls(graph, dt=config.dt, tstop=tstop, overwrite=overwrite)

        if 'output_dir' in config['output']:
            network.output_dir = config['output']['output_dir']

        # network.spikes_file = config['output']['spikes_ascii']

        if 'block_run' in run_dict and run_dict['block_run']:
            if 'block_size' not in run_dict:
                raise Exception('"block_run" is set to True but "block_size" not found.')
            network._block_size = run_dict['block_size']

        if 'duration' in run_dict:
            network.duration = run_dict['duration']

        graph.io.log_info('Building cells.')
        graph.build_nodes()

        graph.io.log_info('Building recurrent connections')
        graph.build_recurrent_edges()

        for sim_input in inputs.from_config(config):
            node_set = graph.get_node_set(sim_input.node_set)
            if sim_input.input_type == 'spikes':
                path = sim_input.params['input_file']
                spikes = SpikeTrains.load(path=path, file_type=sim_input.module, **sim_input.params)
                graph.io.log_info('Build virtual cell stimulations for {}'.format(sim_input.name))
                graph.add_spike_trains(spikes, node_set)
            else:
                graph.io.log_info('Build virtual cell stimulations for {}'.format(sim_input.name))
                rates = firing_rates.RatesInput(sim_input.params)
                graph.add_rates(rates, node_set)

        # Create the output file
        if 'output' in config:
            out_dict = config['output']

            rates_file = out_dict.get('rates_file', None)
            if rates_file is not None:
                rates_file = rates_file if os.path.isabs(rates_file) else os.path.join(config.output_dir, rates_file)
                # create directory if required
                network.rates_file = rates_file
                parent_dir = os.path.dirname(rates_file)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

            if 'log_file' in out_dict:
                log_file = out_dict['log_file']
                network.set_logging(log_file)

        graph.io.log_info('Network created.')
        return network
