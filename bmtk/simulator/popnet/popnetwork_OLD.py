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

from dipde.internals.internalpopulation import InternalPopulation
from dipde.internals.externalpopulation import ExternalPopulation
from dipde.internals.connection import Connection
import dipde

import bmtk.simulator.popnet.config as cfg
import bmtk.simulator.popnet.utils as poputils


class PopNetwork (object):
    def __init__(self, graph):
        self._graph = graph

        self._duration = 0.0
        self._dt = 0.0001
        self._rates_file = None  # name of file where the output is saved

        self.__population_list = []  # list of all populations, internal and external
        self.__population_table = {graph: {} for graph in self._graph.networks}  # population lookup by [network][id]
        self.__connection_list = []  # list of all connections
        self._dipde_network = None  # reference to dipde.Network object

        # diction of rates for every external network/pop_id. Prepopulate dictionary with populations whose rates
        # have already been manually set, otherwise they should use one of the add_rates_* function.
        self._rates = {network: {pop.pop_id: pop.firing_rate for pop in self._graph.get_populations(network)
                                 if not pop.is_internal and pop.is_firing_rate_set}
                       for network in self._graph.networks}

        """
        for network in self._graph.networks:
            for pop in self._graph.get_populations(network):

                if pop.is_internal:
                    dipde_pop = self.__create_internal_pop(pop)

                else:
                    if pop.is_firing_rate_set:
                        rates = pop.firing_rate
        """

    @property
    def duration(self):
        return self._duration

    @duration.setter
    def duration(self, value):
        self._duration = value

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
                print('Firing rate for {}/{} has already been set, skipping.'.format(network, pop.pop_id))
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

    def run(self, duration=None):
        # TODO: Check if cells/connections need to be rebuilt.

        # Create the networ
        self._dipde_network = dipde.Network(population_list=self.populations, connection_list=self.__connection_list)

        if duration is None:
            duration = self.duration

        print("running simulation...")
        self._dipde_network.run(t0=0.0, tf=duration, dt=self.dt)
        # TODO: make record_rates optional?
        self.__record_rates()
        print("done simulation.")

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
            # TODO: store internal populations separately, unless there is a reason to save external populations
            #      (there isn't and it will be problematic)
            for network, pop_list in self.__population_table.items():
                for pop_id, pop in pop_list.items():
                    if pop.record:
                        for time, rate in zip(pop.t_record, pop.firing_rate_record):
                            f.write('{} {} {}\n'.format(pop_id, time, rate))

    @classmethod
    def from_config(cls, configure, graph):
        # load the json file or object
        if isinstance(configure, basestring):
            config = cfg.from_json(configure, validate=True)
        elif isinstance(configure, dict):
            config = configure
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(configure, type(configure)))
        network = cls(graph)

        if 'run' not in config:
            raise Exception('Json file is missing "run" entry. Unable to build Bionetwork.')
        run_dict = config['run']

        # Create the output file
        if 'output' in config:
            out_dict = config['output']

            rates_file = out_dict.get('rates_file', None)
            if rates_file is not None:
                # create directory if required
                network.rates_file = rates_file
                parent_dir = os.path.dirname(rates_file)
                if not os.path.exists(parent_dir):
                    os.makedirs(parent_dir)

            if 'log_file' in out_dict:
                log_file = out_dict['log_file']
                network.set_logging(log_file)

        # get network parameters
        if 'duration' in run_dict:
            network.duration = run_dict['duration']

        if 'dt' in run_dict:
            network.dt = run_dict['dt']

        # TODO: need to get firing rates before building populations
        if 'input' in config:
            for netinput in config['input']:
                if netinput['type'] == 'external_spikes' and netinput['format'] == 'nwb' and netinput['active']:
                    # Load external network spike trains from an NWB file.
                    print('Setting firing rates for {} from {}.'.format(netinput['source_nodes'], netinput['file']))
                    network.add_rates_nwb(netinput['source_nodes'], netinput['file'], netinput['trial'])

                if netinput['type'] == 'pop_rate':
                    print('Setting {}/{} to fire at {} Hz.'.format(netinput['source_nodes'], netinput['pop_id'], netinput['rate']))
                    network.add_rate_hz(netinput['source_nodes'], netinput['pop_id'], netinput['rate'])

                # TODO: take input as function with Population argument

        # Build populations
        print('Building Populations')
        network.build_populations()

        # Build recurrent connections
        if run_dict['connect_internal']:
            print('Building recurrention connections')
            network.set_recurrent_connections()

        # Build external connections. Set connection to default True and turn off only if explicitly stated.
        # NOTE: It might be better to set to default off?!?! Need to dicuss what would be more intuitive for the users.
        # TODO: ignore case of network name
        external_network_settings = {name: True for name in graph.external_networks()}
        if 'connect_external' in run_dict:
            external_network_settings.update(run_dict['connect_external'])
        for netname, connect in external_network_settings.items():
            if connect:
                print('Setting external connections for {}'.format(netname))
                network.set_external_connections(netname)

        return network
