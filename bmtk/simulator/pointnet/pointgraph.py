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
import functools

from collections import Counter

import numpy as np

from bmtk.simulator.utils.graph import SimGraph, SimEdge, SimNode
from property_schemas import CellTypes, DefaultPropertySchema
from . import io

import nest


class PointEdge(SimEdge):
    def __init__(self, edge, dynamics_params, graph):
        super(PointEdge, self).__init__(edge, dynamics_params)
        self._graph = graph
        self._nsyns = graph.property_schema.nsyns(edge)
        self._delay = edge['delay']

    @property
    def nsyns(self):
        return self._nsyns

    @property
    def delay(self):
        return self._delay

    def weight(self, source, target):
        return self._graph.property_schema.get_edge_weight(source, target, self)


class PointNode(SimNode):
    def __init__(self, node_id, graph, network, node_params):
        super(PointNode, self).__init__(node_id, graph, network, node_params)
        # self.__nest_id = -1
        self._dynamics_params = None
        self._model_type = node_params[graph.property_schema.get_model_type_column()]
        self._model_class = graph.property_schema.get_cell_type(node_params)

    @property
    def model_type(self):
        return self._model_type

    @property
    def model_class(self):
        return self._model_class


class PointGraph(SimGraph):
    def __init__(self, **properties):
        super(PointGraph, self).__init__(**properties)
        self._io = io

        self.__weight_functions = {}
        self._params_cache = {}

        self._batch_nodes = True

        # self._params_column = self._property_schema.get_params_column()

        # TODO: create a discovery function, assign ColumnLookups based on columns and not format
        #self._network_format = network_format
        #if self._network_format == TabularNetwork_AI:
        #    self._MT = AIModelClass
        #else:
        #    self._MT = ModelClass

    def __get_params(self, node_params):
        if node_params.with_dynamics_params:
            # TODO: use property, not name
            return node_params['dynamics_params']

        params_file = node_params[self._params_column]
        # params_file = self._MT.params_column(node_params) #node_params['dynamics_params']
        if params_file in self._params_cache:
            return self._params_cache[params_file]
        else:
            params_dir = self.get_component('models_dir')
            params_path = os.path.join(params_dir, params_file)
            params_dict = json.load(open(params_path, 'r'))
            self._params_cache[params_file] = params_dict
            return params_dict

    def _add_node(self, node_params, network):
        node = PointNode(node_params.gid, self, network, node_params)
        if node.model_class == CellTypes.Point:
            node.model_params = self.__get_params(node_params)
            # node.dynamics_params = self.__get_params(node_params)
            self._add_internal_node(node, network)

        elif node.model_class == CellTypes.Virtual:
            self._add_external_node(node, network)

        else:
            raise Exception('Unknown model type {}'.format(node_params['model_type']))

    # TODO: reimplement with py_modules like in bionet
    def add_weight_function(self, function, name=None):
        fnc_name = name if name is not None else function.__name__
        self.__weight_functions[fnc_name] = functools.partial(function)

    def get_weight_function(self, name):
        return self.__weight_functions[name]

    def _create_edge(self, edge, dynamics_params):
        return PointEdge(edge, dynamics_params, self)

    def _to_node_type(self, node_type_id, node_type_params, network='__internal__'):
        nt = {}
        nt['type'] = node_type_params['model_type']
        nt['node_type_id'] = node_type_id

        model_params = {}
        params_file = os.path.join(self.get_component('models_dir'), node_type_params['params_file'])
        for key, value in json.load(open(params_file, 'r')).iteritems():
            model_params[key] = value
        nt['params'] = model_params

        return nt

    def _to_node(self, node_id, node_type_id, node_params, network='__internal__'):
        node = self.Node(node_id, node_type_id, node_params, self.get_node_type(node_type_id, network))
        return node

    def _preprocess_node_types(self, node_population):
        node_type_ids = np.unique(node_population.type_ids)
        # TODO: Verify all the node_type_ids are in the table
        node_types_table = node_population.types_table

        if 'dynamics_params' in node_types_table.columns and 'model_type' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                dynamics_params = node_type['dynamics_params']
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

        if 'model_processing' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                if node_type['model_processing'] is not None:
                    self._batch_nodes = False
                    break


    def _parse_model_template(self, template_str):
        return template_str.split(':')[1]


    _nest_id_map = {}
    _nestid2nodeid_map = {}


    def build_nodes(self):
        for node_pop in self._internal_populations_map.values():
            # BATCH MODE
            id_map = {}
            nest_id_map = {}

            node_ids = node_pop.node_ids
            node_type_ids = node_pop.type_ids

            ntids_counter = Counter(node_type_ids)
            node_groups = {nt_id: np.zeros(ntids_counter[nt_id], dtype=np.uint32) for nt_id in ntids_counter}
            node_groups_counter = {nt_id: 0 for nt_id in ntids_counter}
            for node_id, node_type_id in zip(node_ids, node_type_ids):
                grp_indx = node_groups_counter[node_type_id]
                node_groups[node_type_id][grp_indx] = node_id
                node_groups_counter[node_type_id] += 1

            node_types_table = node_pop.types_table
            for nt_id in ntids_counter:
                nest_model = self._parse_model_template(node_types_table[nt_id]['model_template'])
                n_nodes = ntids_counter[nt_id]
                params = node_types_table[nt_id]['dynamics_params']

                # TODO: Save a list of all internal nest_id objects so it can be accessed for setting recordings, etc.
                nest_ids = nest.Create(nest_model, n_nodes, params)
                for node_id, nest_id in zip(node_groups[nt_id], nest_ids):
                    id_map[node_id] = nest_id
                    nest_id_map[nest_id] = node_id

            self._nest_id_map[node_pop.name] = id_map
            self._nestid2nodeid_map[node_pop.name] = nest_id_map

    def build_recurrent_edges(self):
        if not self._recurrent_edges:
            return 0

        recurrent_edges = [edge_pop for _, edge_list in self._recurrent_edges.items() for edge_pop in edge_list]
        for edge_pop in recurrent_edges:
            n_edges = len(edge_pop)
            trg_ids = np.zeros(n_edges, dtype=np.uint32)
            src_ids = np.zeros(n_edges, dtype=np.uint32)
            syn_weights = np.zeros(n_edges, dtype=np.float)
            syn_delays = np.zeros(n_edges, dtype=np.float)

            src_pop_name = edge_pop.source_population
            src_nest_ids = self._nest_id_map[src_pop_name]

            trg_pop_name = edge_pop.target_population
            trg_nest_ids = self._nest_id_map[trg_pop_name]

            for i, edge in enumerate(edge_pop):
                trg_ids[i] = trg_nest_ids[edge.target_node_id]
                src_ids[i] = src_nest_ids[edge.source_node_id]
                syn_weights[i] = edge['syn_weight']
                syn_delays[i] = edge['delay']

            nest.Connect(src_ids, trg_ids, conn_spec='one_to_one',
                         syn_spec={'weight': syn_weights, 'delay': syn_delays})

    _virtual_ids_map = {}

    def add_spike_trains(self, spike_trains):
        for pop_name in self._virtual_populations_map.keys():
            if pop_name not in spike_trains.populations:
                continue

            if pop_name not in self._virtual_ids_map:
                # virt_pop_map = {}
                virt_node_pop = self._virtual_populations_map[pop_name]
                node_ids = virt_node_pop.node_ids
                n_nodes = len(node_ids)
                nest_ids = nest.Create('spike_generator', n_nodes, {})
                self._virtual_ids_map[pop_name] = {node_id: nest_id for node_id, nest_id in zip(node_ids, nest_ids)}

            src_nodes_map = self._virtual_ids_map[pop_name]
            for node_id, nest_id in src_nodes_map.items():
                # TODO: Look into unrolling this into one set_status command
                nest.SetStatus([nest_id], {'spike_times': spike_trains.get_spikes(node_id)})

            for trg_pop_name in self._nest_id_map.keys():
                trg_nodes_map = self._nest_id_map[trg_pop_name]
                for edge_pop in self.external_edge_populations(src_pop=pop_name, trg_pop=trg_pop_name):
                    n_edges = len(edge_pop)
                    trg_ids = np.zeros(n_edges, dtype=np.uint32)
                    src_ids = np.zeros(n_edges, dtype=np.uint32)
                    syn_weights = np.zeros(n_edges, dtype=np.float)
                    syn_delays = np.zeros(n_edges, dtype=np.float)

                    for i, edge in enumerate(edge_pop):
                        trg_ids[i] = trg_nodes_map[edge.target_node_id]
                        src_ids[i] = src_nodes_map[edge.source_node_id]
                        syn_weights[i] = edge['syn_weight']
                        syn_delays[i] = edge['delay']

                    # print n_edges

                    nest.Connect(src_ids, trg_ids, conn_spec='one_to_one',
                                 syn_spec={'weight': syn_weights, 'delay': syn_delays})

                    #syn_weights = None


        '''
        # TODO: Check the order, I believe this can be built faster
        for trg_pop_name, nid_table in self._local_cells_nid.items():
            for edge_pop in self._recurrent_edges[trg_pop_name]:
                src_pop_name = edge_pop.source_population
                prop_maps = self._edge_property_maps[edge_pop.name]
                for trg_nid, trg_cell in nid_table.items():
                    for edge in edge_pop.get_target(trg_nid):
                        # Create edge object
                        bioedge = BioEdge(edge, self, prop_maps[edge.group_id])
                        src_node = self.get_node(src_pop_name, edge.source_node_id)
                        syn_count += trg_cell.set_syn_connection(bioedge, src_node)

        '''

    '''
    def add_nodes(self, sonata_file, populations=None):
        nodes = sonata_file.nodes

        selected_populations = nodes.population_names if populations is None else populations
        for pop_name in selected_populations:
            node_pop = nodes[pop_name]
            for grp in node_pop.groups:
                if grp.has_dynamics_params:
                    self._batch_nodes = False


                print node_pop.group_ids

        exit()
    '''



