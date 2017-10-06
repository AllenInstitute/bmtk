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

from bmtk.simulator.utils.graph import SimGraph, SimEdge, SimNode
import bmtk.simulator.bionet.config as cfg
from bmtk.simulator.bionet.property_schemas import DefaultPropertySchema, CellTypes


class BioEdge(SimEdge):
    def __init__(self, original_params, dynamics_params, graph):
        super(BioEdge, self).__init__(original_params, dynamics_params)
        self._graph = graph
        self._preselected_targets = graph.property_schema.preselected_targets()
        self._target_sections = graph.property_schema.target_sections(original_params)
        self._target_distance = graph.property_schema.target_distance(original_params)
        self._nsyns = graph.property_schema.nsyns(original_params)

    @property
    def preselected_targets(self):
        return self._preselected_targets

    def weight(self, source, target):
        return self._graph.property_schema.get_edge_weight(source, target, self)

    @property
    def target_sections(self):
        return self._target_sections

    @property
    def target_distance(self):
        return self._target_distance

    @property
    def nsyns(self):
        return self._nsyns

    def load_synapses(self, section_x, section_id):
        return self._graph.property_schema.load_synapse_obj(self, section_x, section_id)


class BioNode(SimNode):
    def __init__(self, node_id, graph, network, node_params):
        super(BioNode, self).__init__(node_id, graph, network, node_params)

        self._cell_type = graph.property_schema.get_cell_type(node_params)
        self._morphology_file = None
        self._positions = graph.property_schema.get_positions(node_params)  # TODO: implement with lazy evaluation

    @property
    def cell_type(self):
        return self._cell_type

    @property
    def positions(self):
        return self._positions

    @property
    def morphology_file(self):
        return self._morphology_file

    @morphology_file.setter
    def morphology_file(self, value):
        self._morphology_file = value
        self._updated_params['morphology_file'] = value

    def load_hobj(self):
        return self._graph.property_schema.load_cell_hobj(self)


class BioGraph(SimGraph):
    def __init__(self, property_schema=None):
        property_schema = property_schema if property_schema is not None else DefaultPropertySchema
        super(BioGraph, self).__init__(property_schema)

        self.__local_nodes_table = {}
        self.__virtual_nodes_table = {}
        self.__morphology_cache = {}

        self._params_cache = {}
        self._params_column = self.property_schema.get_params_column()

    @staticmethod
    def get_default_property_schema():
        raise NotImplementedError()

    def _from_json(self, file_name):
        return cfg.from_json(file_name, validate=True)

    def __get_morphology(self, node):
        morphology_file = node['morphology_file']
        if morphology_file in self.__morphology_cache:
            return self.__morphology_cache[morphology_file]
        else:
            full_path = os.path.join(self.get_component('morphologies_dir'), morphology_file)
            self.__morphology_cache[morphology_file] = full_path
            return full_path

    def __get_params(self, node, cell_type):
        if node.with_dynamics_params:
            return node[self._params_column]

        params_file = node[self._params_column]

        if params_file in self._params_cache:
            return self._params_cache[params_file]

        else:
            # find the full path of the parameters file from the config file
            if cell_type == CellTypes.Biophysical:
                params_dir = self.get_component('biophysical_neuron_models_dir')
            elif cell_type == CellTypes.Point:
                params_dir = self.get_component('point_neuron_models_dir')
            else:
                # Not sure what to do in this case, throw Exception?
                params_dir = self.get_component('custom_neuron_models')

            params_path = os.path.join(params_dir, params_file)

            # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
            # cell_model loader function handle the extension.
            try:
                params_val = json.load(open(params_path, 'r'))
            except Exception:
                params_val = params_path

            # cache and return the value
            self._params_cache[params_file] = params_val
            return params_val

    def _create_edge(self, edge, dynamics_params):
        return BioEdge(edge, dynamics_params, self)

    def _add_node(self, node_params, network):
        node = BioNode(node_params.gid, self, network, node_params)
        if node.cell_type == CellTypes.Biophysical:
            node.morphology_file = self.__get_morphology(node_params)
            node.model_params = self.__get_params(node_params, node.cell_type)
            self._add_internal_node(node, network)

        elif node.cell_type == CellTypes.Point:
            node.model_params = self.__get_params(node_params, CellTypes.Point)
            self._add_internal_node(node, network)

        elif node.cell_type == CellTypes.Virtual:
            self.__virtual_nodes_table[node.node_id] = node
            self._add_external_node(node, network)

        else:
            raise Exception('Unknown model type {}'.format(node_params['model_type']))
