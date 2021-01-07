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
import json
import ast
import numpy as np

# from bmtk.simulator.core.config import ConfigDict
from bmtk.simulator.core.simulation_config import SimulationConfig as ConfigDict
### from bmtk.simulator.utils.property_maps import NodePropertyMap, EdgePropertyMap
from bmtk.utils import sonata
from bmtk.simulator.core.io_tools import io

from bmtk.simulator.core.node_sets import NodeSet, NodeSetAll


"""
Creates a graph of nodes and edges from multiple network files for all simulators.

Consists of edges and nodes. All classes are abstract and should be reimplemented by a specific simulator. Also
contains base factor methods for building a network from a config file (or other).
"""

class SimGraph(object):
    model_type_col = 'model_type'

    def __init__(self):
        self._components = {}  # components table, i.e. paths to model files.
        self._io = io

        self._node_populations = {}
        self._internal_populations_map = {}
        self._virtual_populations_map = {}

        self._virtual_cells_nid = {}

        self._recurrent_edges = {}
        self._external_edges = {}

        self._node_sets = {}
        self._using_gids = False

    @property
    def io(self):
        return self._io

    @property
    def node_populations(self):
        return list(self._node_populations.keys())

    def get_node_set(self, node_set):
        if node_set in self._node_sets.keys():
            return self._node_sets[node_set]

        elif isinstance(node_set, (dict, list)):
            return NodeSet(node_set, self)

        else:
            self.io.log_exception('Unable to load or find node_set "{}"'.format(node_set))

    def get_node_populations(self):
        return self._node_populations.values()

    def get_node_population(self, population_name):
        return self._node_populations[population_name]

    def get_component(self, key):
        """Get the value of item in the components dictionary.

        :param key: name of component
        :return: value assigned to component
        """
        return self._components[key]

    def add_component(self, key, value):
        """Add a component key-value pair

        :param key: name of component
        :param value: value
        """
        self._components[key] = value

    def _validate_components(self):
        """Make sure various components (i.e. paths) exists before attempting to build the graph."""
        return True

    def __avail_model_types(self, population):
        model_types = set()
        for grp in population.groups:
            if self.model_type_col not in grp.all_columns:
                self.io.log_exception('model_type is missing from nodes.')

            model_types.update(set(np.unique(grp.get_values(self.model_type_col))))
        return model_types

    def _preprocess_node_types(self, node_population):
        # TODO: The following figures out the actually used node-type-ids. For mem and speed may be better to just
        # process them all
        node_type_ids = node_population.type_ids
        # TODO: Verify all the node_type_ids are in the table
        node_types_table = node_population.types_table

        # TODO: Convert model_type to a enum
        morph_dir = self.get_component('morphologies_dir')
        if morph_dir is not None and 'morphology' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                if node_type['morphology'] is None:
                    continue
                # TODO: Check the file exits
                # TODO: See if absolute path is stored in csv
                node_type['morphology'] = os.path.join(morph_dir, node_type['morphology'])

        if 'dynamics_params' in node_types_table.columns and 'model_type' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                dynamics_params = node_type['dynamics_params']
                if isinstance(dynamics_params, dict):
                    continue

                model_type = node_type['model_type']
                if model_type == 'biophysical':
                    params_dir = self.get_component('biophysical_neuron_models_dir')
                elif model_type == 'point_process':
                    params_dir = self.get_component('point_neuron_models_dir')
                elif model_type == 'point_soma':
                    params_dir = self.get_component('point_neuron_models_dir')
                else:
                    # Not sure what to do in this case, throw Exception?
                    params_dir = self.get_component('custom_neuron_models')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    node_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    self.io.log_exception('Could not find node dynamics_params file {}.'.format(params_path))

    def _preprocess_edge_types(self, edge_pop):
        edge_types_table = edge_pop.types_table
        edge_type_ids = np.unique(edge_pop.type_ids)

        for et_id in edge_type_ids:
            edge_type = edge_types_table[et_id]
            if 'dynamics_params' in edge_types_table.columns:
                dynamics_params = edge_type['dynamics_params']
                params_dir = self.get_component('synaptic_models_dir')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    edge_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    self.io.log_exception('Could not find edge dynamics_params file {}.'.format(params_path))

            # Split target_sections
            if 'target_sections' in edge_type:
                trg_sec = edge_type['target_sections']
                if trg_sec is not None:
                    try:
                        edge_type['target_sections'] = ast.literal_eval(trg_sec)
                    except Exception as exc:
                        self.io.log_warning('Unable to split target_sections list {}'.format(trg_sec))
                        edge_type['target_sections'] = None

            # Split target distances
            if 'distance_range' in edge_type:
                dist_range = edge_type['distance_range']
                if dist_range is not None:
                    try:
                        # TODO: Make the distance range has at most two values
                        edge_type['distance_range'] = json.loads(dist_range)
                    except Exception as e:
                        try:
                            edge_type['distance_range'] = [0.0, float(dist_range)]
                        except Exception as e:
                            self.io.log_warning('Unable to parse distance_range {}'.format(dist_range))
                            edge_type['distance_range'] = None

    def external_edge_populations(self, src_pop, trg_pop):
        return self._external_edges.get((src_pop, trg_pop), [])

    def add_nodes(self, sonata_file, populations=None):
        """Add nodes from a network to the graph.

        :param sonata_file: A NodesFormat type object containing list of nodes.
        :param populations: name/identifier of network. If none will attempt to retrieve from nodes object
        """
        nodes = sonata_file.nodes

        selected_populations = nodes.population_names if populations is None else populations
        for pop_name in selected_populations:
            if pop_name not in nodes:
                # when user wants to simulation only a few populations in the file
                print('HERE')
                continue

            if pop_name in self.node_populations:
                # Make sure their aren't any collisions
                self.io.log_exception('There are multiple node populations with name {}.'.format(pop_name))

            node_pop = nodes[pop_name]
            self._preprocess_node_types(node_pop)
            self._node_populations[pop_name] = node_pop

            # Segregate into virtual populations and non-virtual populations
            model_types = self.__avail_model_types(node_pop)
            if 'virtual' in model_types:
                self._virtual_populations_map[pop_name] = node_pop
                self._virtual_cells_nid[pop_name] = {}
                model_types -= set(['virtual'])
                if model_types:
                    # We'll allow a population to have virtual and non-virtual nodes but it is not ideal
                    self.io.log_warning('Node population {} contains both virtual and non-virtual nodes which can '.format(pop_name) +
                                        'cause memory and build-time inefficency. Consider separating virtual nodes ' +
                                        'into their own population')

            if model_types:
                self._internal_populations_map[pop_name] = node_pop

            self._node_sets[pop_name] = NodeSet({'population': pop_name}, self)

    def build_nodes(self):
        raise NotImplementedError

    def build_recurrent_edges(self):
        raise NotImplementedError

    def add_edges(self, sonata_file, populations=None, source_pop=None, target_pop=None):
        """

        :param sonata_file:
        :param populations:
        :param source_pop:
        :param target_pop:
        :return:
        """
        edges = sonata_file.edges
        selected_populations = edges.population_names if populations is None else populations

        for pop_name in selected_populations:
            if pop_name not in edges:
                continue

            edge_pop = edges[pop_name]
            self._preprocess_edge_types(edge_pop)

            # Check the source nodes exists
            src_pop = source_pop if source_pop is not None else edge_pop.source_population
            is_internal_src = src_pop in self._internal_populations_map.keys()
            is_external_src = src_pop in self._virtual_populations_map.keys()

            trg_pop = target_pop if target_pop is not None else edge_pop.target_population
            is_internal_trg = trg_pop in self._internal_populations_map.keys()

            if not is_internal_trg:
                self.io.log_exception(('Node population {} does not exists (or consists of only virtual nodes). ' +
                                      '{} edges cannot create connections.').format(trg_pop, pop_name))

            if not (is_internal_src or is_external_src):
                self.io.log_exception('Source node population {} not found. Please update {} edges'.format(src_pop,
                                                                                                           pop_name))
            if is_internal_src:
                if trg_pop not in self._recurrent_edges:
                    self._recurrent_edges[trg_pop] = []
                self._recurrent_edges[trg_pop].append(edge_pop)

            if is_external_src:
                if trg_pop not in self._external_edges:
                    self._external_edges[(src_pop, trg_pop)] = []
                self._external_edges[(src_pop, trg_pop)].append(edge_pop)

    @classmethod
    def from_config(cls, conf, **properties):
        """Generates a graph structure from a json config file or dictionary.

        :param conf: name of json config file, or a dictionary with config parameters
        :param properties: optional properties.
        :return: A graph object of type cls
        """
        graph = cls(**properties)

        # The simulation run script should create a config-dict since it's likely to vary based on the simulator engine,
        # however in the case the user doesn't we will try a generic conversion from dict/json to ConfigDict
        if isinstance(conf, ConfigDict):
            config = conf
        else:
            try:
                config = ConfigDict.load(conf)
            except Exception as e:
                graph.io.log_exception('Could not convert {} (type "{}") to json.'.format(conf, type(conf)))

        if not config.with_networks:
            graph.io.log_exception('Could not find any network files. Unable to build network.')

        # TODO: These are simulator specific
        graph.spike_threshold = config.spike_threshold
        graph.dL = config.dL

        # load components
        for name, value in config.components.items():
            graph.add_component(name, value)
        graph._validate_components()

        # load nodes
        gid_map = config.gid_mappings
        for node_dict in config.nodes:
            nodes_net = sonata.File(data_files=node_dict['nodes_file'], data_type_files=node_dict['node_types_file'],
                                    gid_table=gid_map)
            graph.add_nodes(nodes_net)

        # load edges
        for edge_dict in config.edges:
            target_network = edge_dict['target'] if 'target' in edge_dict else None
            source_network = edge_dict['source'] if 'source' in edge_dict else None
            edge_net = sonata.File(data_files=edge_dict['edges_file'], data_type_files=edge_dict['edge_types_file'])
            graph.add_edges(edge_net, source_pop=target_network, target_pop=source_network)

        graph._node_sets['all'] = NodeSetAll(graph)
        for ns_name, ns_filter in conf.node_sets.items():
            graph._node_sets[ns_name] = NodeSet(ns_filter, graph)

        return graph
