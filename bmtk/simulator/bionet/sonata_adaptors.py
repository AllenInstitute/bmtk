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
import types
import numpy as np

from bmtk.simulator.core.sonata_reader import NodeAdaptor, SonataBaseNode, EdgeAdaptor, SonataBaseEdge
from bmtk.simulator.bionet import nrn


class BioNode(SonataBaseNode):
    @property
    def position(self):
        return self._prop_adaptor.position(self._node)

    @property
    def morphology_file(self):
        return self._node['morphology']

    @property
    def rotation_angle_xaxis(self):
        return self._prop_adaptor.rotation_angle_xaxis(self._node)

    @property
    def rotation_angle_yaxis(self):
        # TODO: Combine rotation alnges into a single property
        return self._prop_adaptor.rotation_angle_yaxis(self._node)

    @property
    def rotation_angle_zaxis(self):
        return self._prop_adaptor.rotation_angle_zaxis(self._node)

    @property
    def rotations(self):
        return self._prop_adaptor.rotations(self._node)

    @property
    def rotations_quaternion(self):
        return self._prop_adaptor.rotations(self._node)

    def load_cell(self):
        model_template = self.model_template
        template_name = model_template[1]
        model_type = self.model_type
        if nrn.py_modules.has_cell_model(self['model_template'], model_type):
            cell_fnc = nrn.py_modules.cell_model(self['model_template'], model_type)
        else:
            cell_fnc = nrn.py_modules.cell_model(model_template[0], model_type)

        dynamics_params = self.dynamics_params
        hobj = cell_fnc(self, template_name, dynamics_params)

        for model_processing_str in self.model_processing:
            processing_fnc = nrn.py_modules.cell_processor(model_processing_str)
            hobj = processing_fnc(hobj, self, dynamics_params)

        return hobj


class BioNodeAdaptor(NodeAdaptor):
    def get_node(self, sonata_node):
        return BioNode(sonata_node, self)

    @classmethod
    def patch_adaptor(cls, adaptor, node_group, network):
        node_adaptor = NodeAdaptor.patch_adaptor(adaptor, node_group, network)

        # Position
        if 'positions' in node_group.all_columns:
            node_adaptor.position = types.MethodType(positions, adaptor)
        elif 'position' in node_group.all_columns:
            node_adaptor.position = types.MethodType(position, adaptor)
        elif 'x' in node_group.all_columns:
            if 'z' in node_group.all_columns and 'y' in node_group.all_columns:
                node_adaptor.position = types.MethodType(position_xyz, adaptor)
            elif 'y' in node_group.all_columns:
                node_adaptor.position = types.MethodType(position_xy, adaptor)
            else:
                node_adaptor.position = types.MethodType(position_x, adaptor)
        else:
            node_adaptor.position = types.MethodType(positions_default, adaptor)

        # Rotation angles
        if 'rotation_angle_xaxis' in node_group.all_columns:
            node_adaptor.rotation_angle_xaxis = types.MethodType(rotation_angle_x, node_adaptor)
        else:
            node_adaptor.rotation_angle_xaxis = types.MethodType(rotation_angle_default, node_adaptor)

        if 'rotation_angle_yaxis' in node_group.all_columns:
            node_adaptor.rotation_angle_yaxis = types.MethodType(rotation_angle_y, node_adaptor)
        else:
            node_adaptor.rotation_angle_yaxis = types.MethodType(rotation_angle_default, node_adaptor)

        if 'rotation_angle_zaxis' in node_group.all_columns:
            node_adaptor.rotation_angle_zaxis = types.MethodType(rotation_angle_z, node_adaptor)
        else:
            node_adaptor.rotation_angle_zaxis = types.MethodType(rotation_angle_default, node_adaptor)

        if 'rotations' in node_group.all_columns:
            node_adaptor.rotations = types.MethodType(rotations, node_adaptor)
        else:
            node_adaptor.rotations = types.MethodType(value_none, node_adaptor)

        return node_adaptor


def rotations(self, node):
    return node['rotations']

def value_none(self, node):
    return None

def positions_default(self, node):
    return np.array([0.0, 0.0, 0.0])


def positions(self, node):
    return node['positions']


def position(self, node):
    return node['position']


def rotation_angle_default(self, node):
    return 0.0


def rotation_angle_x(self, node):
    return node['rotation_angle_xaxis']


def rotation_angle_y(self, node):
    return node['rotation_angle_yaxis']


def rotation_angle_z(self, node):
    return node['rotation_angle_zaxis']


def position_xyz(self, node):
    return np.array([node['x'], node['y'], node['z']])

def position_xy(self, node):
    return np.array([node['x'], node['y'], 0.0])

def position_x(self, node):
    return np.array([node['x'], 0.0, 0.0])

class BioEdge(SonataBaseEdge):
    def load_synapses(self, section_x, section_id):
        synapse_fnc = nrn.py_modules.synapse_model(self.model_template)
        return synapse_fnc(self.dynamics_params, section_x, section_id)


class BioEdgeAdaptor(EdgeAdaptor):
    def get_edge(self, sonata_edge):
        return BioEdge(sonata_edge, self)
