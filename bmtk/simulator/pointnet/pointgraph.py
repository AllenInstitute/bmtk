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
import os
import json
import functools

from bmtk.simulator.utils.graph import SimGraph, SimEdge, SimNode
from property_schemas import CellTypes, DefaultPropertySchema


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
    def __init__(self, property_schema=None):
        property_schema = property_schema if property_schema is not None else DefaultPropertySchema
        super(PointGraph, self).__init__(property_schema)
        self.__weight_functions = {}
        self._params_cache = {}

        self._params_column = self._property_schema.get_params_column()

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
