# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 20XX. Allen Institute. All rights reserved.
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
import os
import json

import config as cfg
from bmtk.utils.io import TabularNetwork
from bmtk.utils.property_schema import PropertySchema


"""Creates a graph of nodes and edges from multiple network files for all simulators.

Consists of edges and nodes. All classes are abstract and should be reimplemented by a specific simulator. Also
contains base factor methods for building a network from a config file (or other).
"""


class SimEdge(object):
    def __init__(self, original_params, dynamics_params):
        self._orig_params = original_params
        self._dynamics_params = dynamics_params
        self._updated_params = {'dynamics_params': self._dynamics_params}

    def __getitem__(self, item):
        if item in self._updated_params:
            return self._updated_params[item]
        else:
            return self._orig_params[item]


class SimNode(object):
    def __init__(self, node_id, graph, network, params):
        self._node_id = node_id
        self._graph = graph
        self._graph_params = params
        self._node_type_id = params['node_type_id']
        self._network = network
        self._updated_params = {}

        self._model_params = {}

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type_id(self):
        return self._node_type_id

    @property
    def network(self):
        """Name of network node belongs too."""
        return self._network

    @property
    def model_params(self):
        """Parameters (json file, nml, dictionary) that describe a specific node"""
        return self._model_params

    @model_params.setter
    def model_params(self, value):
        self._model_params = value

    def __contains__(self, item):
        return item in self._updated_params or item in self._graph_params

    def __getitem__(self, item):
        if item in self._updated_params:
            return self._updated_params[item]
        else:
            return self._graph_params[item]


class SimGraph(object):
    def __init__(self, property_schema=PropertySchema):
        self._networks = {}  # table of networks/node_ids
        self._internal_nodes_table = {}  # table of just the internal nodes by node_id
        self._internal_networks = set()  # list of internal network names
        self._external_networks = set()  # list of external network names

        self._components = {}  # components table, i.e. paths to model files.

        self._edges_table = {}  # table of edges, organized by (target_network,source_network)
        self.__edge_params_cache = {}  # dynamics_params for edges

        self._property_schema = property_schema

    @property
    def networks(self):
        """Returns list of all network names, external and internal"""
        return self._networks.keys()

    @property
    def property_schema(self):
        return self._property_schema

    @property_schema.setter
    def property_schema(self, value):
        self._property_schema = value

    def external_networks(self):
        """List of all external network names"""
        return list(self._external_networks)

    def internal_networks(self):
        """List of all internal network names"""
        return list(self._internal_networks)

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

    def get_nodes(self, network):
        return self._networks[network].values()

    def get_node(self, node_id, network):
        return self._networks[network][node_id]

    def get_internal_nodes(self):
        return self._internal_nodes_table.values()

    def _from_json(self, file_name):
        return cfg.from_json(file_name)

    def _validate_components(self):
        """Make sure various components (i.e. paths) exists before attempting to build the graph."""
        # TODO: need to implement
        return True

    def add_nodes(self, nodes, network=None):
        """Add nodes from a network to the graph.

        :param nodes: A NodesFormat type object containing list of nodes.
        :param network: name/identifier of network. If none will attempt to retrieve from nodes object
        """
        if network is None:
            # TODO: Throw an error if network name cannot be resolved.
            network = nodes.name

        if network in self._networks:
            raise Exception('Network {} already exists.'.format(network))
        else:
            # TODO: Is it beneficial to preallocate list and use insert instead of append?
            self._networks[network] = {}

        # go through each node in the network and add them to the graph
        for node in nodes:
            self._add_node(node, network)

    def _add_node(self, node, network):
        raise NotImplementedError()

    def _add_internal_node(self, node_params, network):
        """Add node from network into graph

        :param node_params: A node object to add to the graph
        :param network: Name of network
        """
        gid = node_params.node_id
        if gid in self._internal_nodes_table:
            raise Exception('Found multiple nodes with gid {}. Please fix.'.format(gid))

        self._networks[network][gid] = node_params
        self._internal_nodes_table[gid] = node_params
        self._internal_networks.add(network)  # indicate that 'network' contains internal nodes

    def _add_external_node(self, node_params, network):
        """Add a virtual node to the graph.

        :param node_params: node object
        :param network: name of network
        """
        # TODO: There must be a separate external network table, in the case the network is mixed then
        #       self._networks[network] will als be mixed.
        self._networks[network][node_params.node_id] = node_params
        self._external_networks.add(network)

    def add_edges(self, edges, source_network=None, target_network=None):
        src_network = source_network if source_network is not None else edges.source_network
        trg_network = target_network if target_network is not None else edges.target_network
        if trg_network is None:
            raise Exception('Unable to resolve name for target network in edges file.')
        if src_network is None:
            raise Exception('Unable to resolve name for source network in edges file.')

        # TODO: reimplement ability to set weight with functions. Scan each group properities to see 'syn_weight' exists
        #       or if 'weight_function'. Create edge template that passes back weight accordingly
        for net in [src_network, trg_network]:
            if net not in self.networks:
                raise Exception('Network {} has not been added to the graph, can not make connections'.format(net))

        self._edges_table[(trg_network, src_network)] = edges

    def edges_table(self, target_network, source_network):
        return self._edges_table[(target_network, source_network)]

    def edges_iterator(self, target_gid, source_network):
        target_node = self._internal_nodes_table[target_gid]
        target_network = target_node.network
        source_network_table = self._networks[source_network]

        edges = self.edges_table(target_network, source_network)
        for e in edges.edges_itr(target_gid):
            dynamics_params = self._get_edge_params(e)
            edge_wrapper = self._create_edge(e, dynamics_params)
            source_node = source_network_table[e.source_gid]

            yield target_node, source_node, edge_wrapper

    def _create_edge(self, edge, dynamics_params):
        return SimEdge(edge, dynamics_params)

    def _get_edge_params(self, edge):
        if edge.with_dynamics_params:
            return edge['dynamics_params']

        # TODO: need to find a way to dynamically determine params columns once
        if 'dynamics_params' in edge:
            params_file = edge['dynamics_params']
        elif 'params_file' in edge:
            params_file = edge['params_file']
        else:
            raise Exception('No params file for edge model.')

        if params_file in self.__edge_params_cache:
            return self.__edge_params_cache[params_file]
        else:
            params_dir = self.get_component('synaptic_models')
            params_path = os.path.join(params_dir, params_file)
            params_dict = json.load(open(params_path, 'r'))
            self.__edge_params_cache[params_file] = params_dict
            return params_dict

    @classmethod
    def from_config(cls, conf, network_format=TabularNetwork, property_schema=None, **properties):
        """Generates a graph structure from a json config file or dictionary.

        :param conf: name of json config file, or a dictionary with config parameters
        :param network_format: storage representation of networks
        :param property_schema: column schema used to build graph
        :param properties: optional properties.
        :return: A graph object of type cls
        """
        graph = cls(property_schema) if not properties else cls(property_schema, **properties)
        if isinstance(conf, basestring):
            config = graph._from_json(conf)
        elif isinstance(conf, dict):
            config = conf
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(conf, type(conf)))

        if 'components' in config:
            for name, value in config['components'].iteritems():
                graph.add_component(name, value)
            graph._validate_components()

        if 'networks' not in config:
            raise Exception('Could not find any network files. Unable to build network.')

        network_files = config['networks']
        if 'node_files' not in network_files:
            raise Exception("Could not find any node files. Unable to build network.")

        for nodes_config in config['networks']['node_files']:
            nodes_file = nodes_config['nodes']
            node_types_file = nodes_config['node_types']
            network_name = None if 'name' not in nodes_config else nodes_config['name']
            nf = network_format.load_nodes(nodes_file, node_types_file)
            graph.add_nodes(nf, network_name)

        if 'edges_files' in config['networks']:
            for edges_config in config['networks']['edges_files']:
                edges_file = edges_config['edges']
                edge_types_file = edges_config['edge_types']

                ef = network_format.load_edges(edges_file, edge_types_file)
                target_network = edges_config['target'] if 'target' in edges_config else None
                source_network = edges_config['source'] if 'source' in edges_config else None
                graph.add_edges(ef, target_network=target_network, source_network=source_network)

        return graph
