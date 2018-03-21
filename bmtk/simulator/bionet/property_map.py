import types

import nrn


# TODO: Consider using partial functions
class PropertyMap(object):
    def __init__(self, graph):
        self._graph = graph

    def _parse_model_template(self, model_template):
        if model_template in self._template_cache:
            return self._template_cache[model_template]
        else:
            template_parts = model_template.split(':')
            assert(len(template_parts) == 2)
            directive, template = template_parts[0], template_parts[1]
            self._template_cache[model_template] = (directive, template)
            return directive, template

    def load_cell(self, node):
        model_template = node['model_template']
        model_type = node['model_type']
        if nrn.py_modules.has_cell_model(model_template, model_type):
            cell_fnc = nrn.py_modules.cell_model(model_template, node['model_type'])
            template_name = None
        else:
            directive, template_name = self._parse_model_template(node['model_template'])
            cell_fnc = nrn.py_modules.cell_model(directive, node['model_type'])

        dynamics_params = self.dynamics_params(node)
        return cell_fnc(node, template_name, dynamics_params)

    @classmethod
    def build_map(cls, node_group, biograph):
        prop_map = cls(biograph)

        if 'positions' in node_group.all_columns:
            prop_map.positions = types.MethodType(positions, prop_map)
        elif 'position' in node_group.all_columns:
            prop_map.positions = types.MethodType(position, prop_map)
        else:
            prop_map.positions = types.MethodType(positions_default, prop_map)

        # Use gids or node_ids
        if node_group.has_gids:
            prop_map.gid = types.MethodType(gid, prop_map)
        else:
            prop_map.gid = types.MethodType(node_id, prop_map)

        # dynamics_params
        if node_group.has_dynamics_params:
            prop_map.dynamics_params = types.MethodType(group_dynamics_params, prop_map)
        else:
            prop_map.dynamics_params = types.MethodType(types_dynamics_params, prop_map)

        # Rotation angles
        if 'rotation_angle_xaxis' in node_group.all_columns:
            prop_map.rotation_angle_xaxis = types.MethodType(rotation_angle_x, prop_map)
        else:
            prop_map.rotation_angle_xaxis = types.MethodType(rotation_angle_default, prop_map)

        if 'rotation_angle_yaxis' in node_group.all_columns:
            prop_map.rotation_angle_yaxis = types.MethodType(rotation_angle_y, prop_map)
        else:
            prop_map.rotation_angle_yaxis = types.MethodType(rotation_angle_default, prop_map)

        if 'rotation_angle_zaxis' in node_group.all_columns:
            prop_map.rotation_angle_zaxis = types.MethodType(rotation_angle_z, prop_map)
        else:
            prop_map.rotation_angle_zaxis = types.MethodType(rotation_angle_default, prop_map)

        return prop_map


def positions_default(self, node):
    return [0.0, 0.0, 0.0]


def positions(self, node):
    return node['positions']


def position(self, node):
    return node['position']


def gid(self, node):
    return node['gid']


def node_id(self, node):
    return node.node_id


def rotation_angle_default(self, node):
    return 0.0

def rotation_angle_x(self, node):
    return node['rotation_angle_xaxis']

def rotation_angle_y(self, node):
    return node['rotation_angle_yaxis']

def rotation_angle_z(self, node):
    return node['rotation_angle_zaxis']


def types_dynamics_params(self, node):
    return node['dynamics_params']


def group_dynamics_params(self, node):
    return node.dynamics_params



class EdgePropertyMap(object):
    def __init__(self, graph):
        self._graph = graph

    @classmethod
    def build_map(cls, edge_group, biograph):
        prop_map = cls(biograph)

        if 'syn_weight' in edge_group.columns:
            prop_map.weight = types.MethodType(syn_weight, prop_map)
            prop_map.preselected_targets = True
            prop_map.nsyns = types.MethodType(no_nsyns, prop_map)
        else:
            prop_map.preselected_targets = False


        return prop_map


def syn_weight(self, edge, source, target):
    return edge['syn_weight']

def nsyns(self, edge):
    return edge['nsyns']

def no_nsyns(self, edge):
    return 1