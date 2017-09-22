import os
import json

from bmtk.simulator.utils.graph import SimGraph
from property_schemas import PopTypes, DefaultPropertySchema
from popnode import InternalNode, ExternalPopulation
from popedge import PopEdge


class PopGraph(SimGraph):
    def __init__(self, property_schema=None, group_by='node_type_id', **properties):
        property_schema = property_schema if property_schema is not None else DefaultPropertySchema
        super(PopGraph, self).__init__(property_schema)

        self.__all_edges = []
        self._group_key = group_by
        self._gid_table = {}
        self._edges = {}
        self._target_edges = {}
        self._source_edges = {}

        self._params_cache = {}
        self._params_column = property_schema.get_params_column()

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
        return super(PopGraph, self).get_nodes(network)

    def get_population(self, network, pop_id):
        return self.get_node(pop_id, network)
