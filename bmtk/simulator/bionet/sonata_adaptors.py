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
        return self._node['morphology_file']

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

    def load_cell(self):
        model_template = self.model_template
        model_type = self.model_type
        if nrn.py_modules.has_cell_model(self['model_template'], model_type):
            cell_fnc = nrn.py_modules.cell_model(self['model_template'], model_type)
            template_name = None
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

    @staticmethod
    def patch_adaptor(adaptor, node_group):
        node_adaptor = NodeAdaptor.patch_adaptor(adaptor, node_group)

        # Position
        if 'positions' in node_group.all_columns:
            node_adaptor.position = types.MethodType(positions, adaptor)
        elif 'position' in node_group.all_columns:
            node_adaptor.position = types.MethodType(position, adaptor)
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

        return node_adaptor


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


class BioEdge(SonataBaseEdge):
    def load_synapses(self, section_x, section_id):
        synapse_fnc = nrn.py_modules.synapse_model(self.model_template)
        return synapse_fnc(self.dynamics_params, section_x, section_id)


class BioEdgeAdaptor(EdgeAdaptor):
    def get_edge(self, sonata_edge):
        return BioEdge(sonata_edge, self)
