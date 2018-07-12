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
import numpy as np

from bmtk.simulator.core.simulator_network import SimNetwork
#from bmtk.simulator.core.graph import SimGraph
#from property_schemas import PopTypes, DefaultPropertySchema
#from popnode import InternalNode, ExternalPopulation
#from popedge import PopEdge
from bmtk.simulator.popnet import utils as poputils
from bmtk.simulator.popnet.sonata_adaptors import PopEdgeAdaptor

from dipde.internals.internalpopulation import InternalPopulation
from dipde.internals.externalpopulation import ExternalPopulation
from dipde.internals.connection import Connection

'''
class PopNode(object):
    def __init__(self, node, property_map, graph):
        self._node = node
        self._property_map = property_map
        self._graph = graph

    @property
    def dynamics_params(self):
        # TODO: Use propert map
        return self._node['dynamics_params']

    @property
    def node_id(self):
        # TODO: Use property map
        return self._node.node_id
'''


class Population(object):
    def __init__(self, pop_id):
        self._pop_id = pop_id
        self._nodes = []
        self._params = None

        self._dipde_obj = None

    def add_node(self, pnode):
        self._nodes.append(pnode)
        if self._params is None and pnode.dynamics_params is not None:
            self._params = pnode.dynamics_params.copy()

    @property
    def pop_id(self):
        return self._pop_id

    @property
    def dipde_obj(self):
        return self._dipde_obj

    @property
    def record(self):
        return True

    def build(self):
        params = self._nodes[0].dynamics_params
        self._dipde_obj = InternalPopulation(**params)

    def get_gids(self):
        for node in self._nodes:
            yield node.node_id

    def __getitem__(self, item):
        return self._params[item]

    def __setitem__(self, key, value):
        self._params[key] = value

    def __repr__(self):
        return str(self._pop_id)


class ExtPopulation(Population):
    def __init__(self, pop_id):
        super(ExtPopulation, self).__init__(pop_id)
        self._firing_rate = None

    @property
    def record(self):
        return False

    @property
    def firing_rate(self):
        return self._firing_rate

    @firing_rate.setter
    def firing_rate(self, value):
        self.build(value)

    def build(self, firing_rate):
        if firing_rate is not None:
            self._firing_rate = firing_rate

        self._dipde_obj = ExternalPopulation(firing_rate)


class PopEdge(object):
    def __init__(self, edge, property_map, graph):
        self._edge = edge
        self._prop_map = property_map
        self._graph = graph

    @property
    def nsyns(self):
        # TODO: Use property map
        return self._edge['nsyns']

    @property
    def delay(self):
        return self._edge['delay']

    @property
    def weight(self):
        return self._edge['syn_weight']


class PopConnection(object):
    def __init__(self, src_pop, trg_pop):
        self._src_pop = src_pop
        self._trg_pop = trg_pop
        self._edges = []

        self._dipde_conn = None

    def add_edge(self, edge):
        self._edges.append(edge)

    def build(self):
        edge = self._edges[0]
        self._dipde_conn = Connection(self._src_pop._dipde_obj, self._trg_pop._dipde_obj, edge.nsyns, edge.delay,
                                      edge.syn_weight)

    @property
    def dipde_obj(self):
        return self._dipde_conn


class PopNetwork(SimNetwork):
    def __init__(self, group_by='node_type_id', **properties):
        super(PopNetwork, self).__init__()

        self.__all_edges = []
        self._group_key = group_by
        self._gid_table = {}
        self._edges = {}
        self._target_edges = {}
        self._source_edges = {}

        self._params_cache = {}
        #self._params_column = property_schema.get_params_column()
        self._dipde_pops = {}
        self._external_pop = {}
        self._all_populations = []
        # self._loaded_external_pops = {}

        self._nodeid2pop_map = {}

        self._connections = {}
        self._external_connections = {}
        self._all_connections = []

    @property
    def populations(self):
        return self._all_populations

    @property
    def connections(self):
        return self._all_connections

    @property
    def internal_populations(self):
        return self._dipde_pops.values()

    def _register_adaptors(self):
        super(PopNetwork, self)._register_adaptors()
        self._edge_adaptors['sonata'] = PopEdgeAdaptor

    def build_nodes(self):
        if self._group_key == 'node_id' or self._group_key is None:
            self._build_nodes()
        else:
            self._build_nodes_grouped()

    def _build_nodes(self):
        for node_pop in self.node_populations:
            if node_pop.internal_nodes_only:
                nid2pop_map = {}
                for node in node_pop.get_nodes():
                    #pnode = PopNode(node, prop_maps[node.group_id], self)
                    pop = Population(node.node_id)
                    pop.add_node(node)
                    pop.build()

                    self._dipde_pops[node.node_id] = pop
                    self._all_populations.append(pop)
                    nid2pop_map[node.node_id] = pop

                self._nodeid2pop_map[node_pop.name] = nid2pop_map

        """
        for node_pop in self._internal_populations_map.values():
            prop_maps = self._node_property_maps[node_pop.name]
            nid2pop_map = {}
            for node in node_pop:
                pnode = PopNode(node, prop_maps[node.group_id], self)
                pop = Population(node.node_id)
                pop.add_node(pnode)
                pop.build()

                self._dipde_pops[node.node_id] = pop
                self._all_populations.append(pop)
                nid2pop_map[node.node_id] = pop

            self._nodeid2pop_map[node_pop.name] = nid2pop_map
        """

    def _build_nodes_grouped(self):
        # Organize every single sonata-node into a given population.
        for node_pop in self.node_populations:
            nid2pop_map = {}
            if node_pop.internal_nodes_only:
                for node in node_pop.get_nodes():
                    pop_key = node[self._group_key]
                    if pop_key not in self._dipde_pops:
                        pop = Population(pop_key)
                        self._dipde_pops[pop_key] = pop
                        self._all_populations.append(pop)

                    pop = self._dipde_pops[pop_key]
                    pop.add_node(node)
                    nid2pop_map[node.node_id] = pop

                self._nodeid2pop_map[node_pop.name] = nid2pop_map

            for dpop in self._dipde_pops.values():
                dpop.build()

        """
        for node_pop in self._internal_populations_map.values():
            prop_maps = self._node_property_maps[node_pop.name]
            nid2pop_map = {}
            for node in node_pop:
                pop_key = node[self._group_key]
                pnode = PopNode(node, prop_maps[node.group_id], self)
                if pop_key not in self._dipde_pops:
                    pop = Population(pop_key)
                    self._dipde_pops[pop_key] = pop
                    self._all_populations.append(pop)

                pop = self._dipde_pops[pop_key]
                pop.add_node(pnode)
                nid2pop_map[node.node_id] = pop

            self._nodeid2pop_map[node_pop.name] = nid2pop_map

        for dpop in self._dipde_pops.values():
            dpop.build()
        """

    def build_recurrent_edges(self):
        recurrent_edge_pops = [ep for ep in self._edge_populations if not ep.virtual_connections]

        for edge_pop in recurrent_edge_pops:
            if edge_pop.recurrent_connections:
                src_pop_maps = self._nodeid2pop_map[edge_pop.source_nodes]
                trg_pop_maps = self._nodeid2pop_map[edge_pop.target_nodes]
                for edge in edge_pop.get_edges():
                    src_pop = src_pop_maps[edge.source_node_id]
                    trg_pop = trg_pop_maps[edge.target_node_id]
                    conn_key = (src_pop, trg_pop)
                    if conn_key not in self._connections:
                        conn = PopConnection(src_pop, trg_pop)
                        self._connections[conn_key] = conn
                        self._all_connections.append(conn)

                    self._connections[conn_key].add_edge(edge)

            elif edge_pop.mixed_connections:
                raise NotImplementedError()

        for conn in self._connections.values():
            conn.build()

        """
        recurrent_edges = [edge_pop for _, edge_list in self._recurrent_edges.items() for edge_pop in edge_list]
        for edge_pop in recurrent_edges:
            prop_maps = self._edge_property_maps[edge_pop.name]
            src_pop_maps = self._nodeid2pop_map[edge_pop.source_population]
            trg_pop_maps = self._nodeid2pop_map[edge_pop.target_population]
            for edge in edge_pop:
                src_pop = src_pop_maps[edge.source_node_id]
                trg_pop = trg_pop_maps[edge.target_node_id]
                conn_key = (src_pop, trg_pop)
                if conn_key not in self._connections:
                    conn = PopConnection(src_pop, trg_pop)
                    self._connections[conn_key] = conn
                    self._all_connections.append(conn)

                pop_edge = PopEdge(edge, prop_maps[edge.group_id], self)
                self._connections[conn_key].add_edge(pop_edge)

        for conn in self._connections.values():
            conn.build()
        # print len(self._connections)
        """

    def find_edges(self, source_nodes=None, target_nodes=None):
        # TODO: Move to parent
        selected_edges = self._edge_populations[:]

        if source_nodes is not None:
            selected_edges = [edge_pop for edge_pop in selected_edges if edge_pop.source_nodes == source_nodes]

        if target_nodes is not None:
            selected_edges = [edge_pop for edge_pop in selected_edges if edge_pop.target_nodes == target_nodes]

        return selected_edges

    def add_spike_trains(self, spike_trains, node_set):
        # Build external node populations
        src_nodes = [node_pop for node_pop in self.node_populations if node_pop.name in node_set.population_names()]
        for node_pop in src_nodes:
            pop_name = node_pop.name
            if node_pop.name not in self._external_pop:
                external_pop_map = {}
                src_pop_map = {}
                for node in node_pop.get_nodes():
                    pop_key = node[self._group_key]
                    if pop_key not in external_pop_map:
                        pop = ExtPopulation(pop_key)
                        external_pop_map[pop_key] = pop
                        self._all_populations.append(pop)

                    pop = external_pop_map[pop_key]
                    pop.add_node(node)
                    src_pop_map[node.node_id] = pop

                self._nodeid2pop_map[pop_name] = src_pop_map

                firing_rates = poputils.get_firing_rates(external_pop_map.values(), spike_trains)
                self._external_pop[pop_name] = external_pop_map
                for dpop in external_pop_map.values():
                    dpop.build(firing_rates[dpop.pop_id])

            else:
                # TODO: Throw error spike trains should only be called once per source population
                # external_pop_map = self._external_pop[pop_name]
                src_pop_map = self._nodeid2pop_map[pop_name]

            unbuilt_connections = []
            for source_reader in src_nodes:
                for edge_pop in self.find_edges(source_nodes=source_reader.name):
                    trg_pop_map = self._nodeid2pop_map[edge_pop.target_nodes]
                    for edge in edge_pop.get_edges():
                        src_pop = src_pop_map[edge.source_node_id]
                        trg_pop = trg_pop_map[edge.target_node_id]
                        conn_key = (src_pop, trg_pop)
                        if conn_key not in self._external_connections:
                            pconn = PopConnection(src_pop, trg_pop)
                            self._external_connections[conn_key] = pconn
                            unbuilt_connections.append(pconn)
                            self._all_connections.append(pconn)

                        #pop_edge = PopEdge(edge, prop_maps[edge.group_id], self)
                        self._external_connections[conn_key].add_edge(edge)

            for pedge in unbuilt_connections:
                pedge.build()
        #exit()

        """
            print node_pop.name


            exit()
            if node_pop.name in self._virtual_ids_map:
                 continue

            virt_node_map = {}
            if node_pop.virtual_nodes_only:
                print 'HERE'
                exit()


        for pop_name, node_pop in self._virtual_populations_map.items():
            if pop_name not in spike_trains.populations:
                continue

            # Build external population if it already hasn't been built
            if pop_name not in self._external_pop:
                prop_maps = self._node_property_maps[pop_name]
                external_pop_map = {}
                src_pop_map = {}
                for node in node_pop:
                    pop_key = node[self._group_key]
                    pnode = PopNode(node, prop_maps[node.group_id], self)
                    if pop_key not in external_pop_map:
                        pop = ExtPopulation(pop_key)
                        external_pop_map[pop_key] = pop
                        self._all_populations.append(pop)

                    pop = external_pop_map[pop_key]
                    pop.add_node(pnode)
                    src_pop_map[node.node_id] = pop

                self._nodeid2pop_map[pop_name] = src_pop_map

                firing_rates = poputils.get_firing_rates(external_pop_map.values(), spike_trains)
                self._external_pop[pop_name] = external_pop_map
                for dpop in external_pop_map.values():
                    dpop.build(firing_rates[dpop.pop_id])

            else:
                # TODO: Throw error spike trains should only be called once per source population
                # external_pop_map = self._external_pop[pop_name]
                src_pop_map = self._nodeid2pop_map[pop_name]

            unbuilt_connections = []
            for node_pop in self._internal_populations_map.values():
                trg_pop_map = self._nodeid2pop_map[node_pop.name]
                for edge_pop in self.external_edge_populations(src_pop=pop_name, trg_pop=node_pop.name):
                    for edge in edge_pop:
                        src_pop = src_pop_map[edge.source_node_id]
                        trg_pop = trg_pop_map[edge.target_node_id]
                        conn_key = (src_pop, trg_pop)
                        if conn_key not in self._external_connections:
                            pconn = PopConnection(src_pop, trg_pop)
                            self._external_connections[conn_key] = pconn
                            unbuilt_connections.append(pconn)
                            self._all_connections.append(pconn)

                        pop_edge = PopEdge(edge, prop_maps[edge.group_id], self)
                        self._external_connections[conn_key].add_edge(pop_edge)

            for pedge in unbuilt_connections:
                pedge.build()
        """


    def add_rates(self, rates, node_set):
        if self._group_key == 'node_id':
            id_lookup = lambda n: n.node_id
        else:
            id_lookup = lambda n: n[self._group_key]

        src_nodes = [node_pop for node_pop in self.node_populations if node_pop.name in node_set.population_names()]
        for node_pop in src_nodes:
            pop_name = node_pop.name
            if node_pop.name not in self._external_pop:
                external_pop_map = {}
                src_pop_map = {}
                for node in node_pop.get_nodes():
                    pop_key = id_lookup(node)
                    if pop_key not in external_pop_map:
                        pop = ExtPopulation(pop_key)
                        external_pop_map[pop_key] = pop
                        self._all_populations.append(pop)

                    pop = external_pop_map[pop_key]
                    pop.add_node(node)
                    src_pop_map[node.node_id] = pop

                self._nodeid2pop_map[pop_name] = src_pop_map

                self._external_pop[pop_name] = external_pop_map
                for dpop in external_pop_map.values():
                    firing_rates = rates.get_rate(dpop.pop_id)
                    dpop.build(firing_rates)

            else:
                # TODO: Throw error spike trains should only be called once per source population
                # external_pop_map = self._external_pop[pop_name]
                src_pop_map = self._nodeid2pop_map[pop_name]

            unbuilt_connections = []
            for source_reader in src_nodes:
                for edge_pop in self.find_edges(source_nodes=source_reader.name):
                    trg_pop_map = self._nodeid2pop_map[edge_pop.target_nodes]
                    for edge in edge_pop.get_edges():
                        src_pop = src_pop_map[edge.source_node_id]
                        trg_pop = trg_pop_map[edge.target_node_id]
                        conn_key = (src_pop, trg_pop)
                        if conn_key not in self._external_connections:
                            pconn = PopConnection(src_pop, trg_pop)
                            self._external_connections[conn_key] = pconn
                            unbuilt_connections.append(pconn)
                            self._all_connections.append(pconn)

                        #pop_edge = PopEdge(edge, prop_maps[edge.group_id], self)
                        self._external_connections[conn_key].add_edge(edge)

            for pedge in unbuilt_connections:
                pedge.build()

        """
        for pop_name, node_pop in self._virtual_populations_map.items():
            if pop_name not in rates.populations:
                continue

            # Build external population if it already hasn't been built
            if pop_name not in self._external_pop:
                prop_maps = self._node_property_maps[pop_name]
                external_pop_map = {}
                src_pop_map = {}
                for node in node_pop:
                    pop_key = id_lookup(node)
                    #pop_key = node[self._group_key]
                    pnode = PopNode(node, prop_maps[node.group_id], self)
                    if pop_key not in external_pop_map:
                        pop = ExtPopulation(pop_key)
                        external_pop_map[pop_key] = pop
                        self._all_populations.append(pop)

                    pop = external_pop_map[pop_key]
                    pop.add_node(pnode)
                    src_pop_map[node.node_id] = pop

                self._nodeid2pop_map[pop_name] = src_pop_map

                firing_rate = rates.get_rate(pop_key)
                self._external_pop[pop_name] = external_pop_map
                for dpop in external_pop_map.values():
                    dpop.build(firing_rate)

            else:
                # TODO: Throw error spike trains should only be called once per source population
                # external_pop_map = self._external_pop[pop_name]
                src_pop_map = self._nodeid2pop_map[pop_name]
        """

    '''
    def _add_node(self, node, network):
        pops = self._networks[network]
        pop_key = node[self._group_key]
        if pop_key in pops:
            pop = pops[pop_key]
            pop.add_gid(node.gid)
            self._gid_table[network][node.gid] = pop
        else:
            model_class = self.property_schema.get_pop_type(node)
            if model_class == PopTypes.Internal:
                pop = InternalNode(pop_key, self, network, node)
                pop.add_gid(node.gid)
                pop.model_params = self.__get_params(node)
                self._add_internal_node(pop, network)

            elif model_class == PopTypes.External:
                # TODO: See if we can get firing rate from dynamics_params
                pop = ExternalPopulation(pop_key, self, network, node)
                pop.add_gid(node.gid)
                self._add_external_node(pop, network)

            else:
                raise Exception('Unknown model type')

            if network not in self._gid_table:
                self._gid_table[network] = {}
            self._gid_table[network][node.gid] = pop
    '''

    def __get_params(self, node_params):
        if node_params.with_dynamics_params:
            return node_params['dynamics_params']

        params_file = node_params[self._params_column]
        if params_file in self._params_cache:
            return self._params_cache[params_file]
        else:
            params_dir = self.get_component('models_dir')
            params_path = os.path.join(params_dir, params_file)
            params_dict = json.load(open(params_path, 'r'))
            self._params_cache[params_file] = params_dict
            return params_dict

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
                elif model_type == 'population':
                    params_dir = self.get_component('population_models_dir')
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


    '''
    def add_edges(self, edges, target_network=None, source_network=None):
        # super(PopGraph, self).add_edges(edges)

        target_network = target_network if target_network is not None else edges.target_network
        if target_network not in self._target_edges:
            self._target_edges[target_network] = []

        source_network = source_network if source_network is not None else edges.source_network
        if source_network not in self._source_edges:
            self._source_edges[source_network] = []

        target_pops = self.get_populations(target_network)
        source_pops = self.get_populations(source_network)
        source_gid_table = self._gid_table[source_network]

        for target_pop in target_pops:
            for target_gid in target_pop.get_gids():
                for edge in edges.edges_itr(target_gid):
                    source_pop = source_gid_table[edge.source_gid]
                    self._add_edge(source_pop, target_pop, edge)
    '''

    def _add_edge(self, source_pop, target_pop, edge):
        src_id = source_pop.node_id
        trg_id = target_pop.node_id
        edge_type_id = edge['edge_type_id']
        edge_key = (src_id, source_pop.network, trg_id, target_pop.network, edge_type_id)

        if edge_key in self._edges:
            return
        else:
            # TODO: implement dynamics params
            dynamics_params = self._get_edge_params(edge)
            pop_edge = PopEdge(source_pop, target_pop, edge, dynamics_params)
            self._edges[edge_key] = pop_edge
            self._source_edges[source_pop.network].append(pop_edge)
            self._target_edges[target_pop.network].append(pop_edge)

    def get_edges(self, source_network):
        return self._source_edges[source_network]

    def edges_table(self, target_network, source_network):
        return self._edges_table[(target_network, source_network)]

    def get_populations(self, network):
        return super(PopNetwork, self).get_nodes(network)

    def get_population(self, node_set, gid):
        return self._nodeid2pop_map[node_set][gid]

    def rebuild(self):
        for _, ns in self._nodeid2pop_map.items():
            for _, pop in ns.items():
                pop.build()

        for pc in self._all_connections:
            pc.build()