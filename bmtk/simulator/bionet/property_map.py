import types
import ast

import nrn

import io


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
        else:  # 'dynamics_params' in node_group.all_columns:
            prop_map.dynamics_params = types.MethodType(types_dynamics_params, prop_map)
        #else:
        #    io.log_exception('No dynamics_params column or group for /nodes/{}/{}'.format(node_group.parent.name,
        #                                                                                 node_group.group_id))

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

    def load_synapse_obj(self, edge, section_x, section_id):
        synapse_fnc = nrn.py_modules.synapse_model(edge['model_template'])
        return synapse_fnc(edge['dynamics_params'], section_x, section_id)

    @classmethod
    def build_map(cls, edge_group, biograph):
        prop_map = cls(biograph)

        # For fetching/calculating synaptic weights
        if 'weight_function' in edge_group.all_columns:
            # Customized function for user to calculate the synaptic weight
            prop_map.syn_weight = types.MethodType(weight_function, prop_map)
        elif 'syn_weight' in edge_group.all_columns:
            # Just return the synaptic weight
            prop_map.syn_weight = types.MethodType(syn_weight, prop_map)
        else:
            io.log_exception('Could not find syn_weight or weight_function properties. Cannot create connections.')

        # For determining the synapse placement
        if 'sec_id' in edge_group.all_columns:
            prop_map.preselected_targets = True
            prop_map.nsyns = types.MethodType(no_nsyns, prop_map)
        elif 'nsyns' in edge_group.all_columns:
            prop_map.preselected_targets = False
            prop_map.nsyns = types.MethodType(nsyns, prop_map)
        else:
            # I'd like to put a warning or exception however we have to consider point neuron synapses
            pass

        # For target sections
        '''
        if 'syn_weight' not in edge_group.all_columns:
            io.log_exception('Edges {} missing syn_weight property for connections.'.format(edge_group.parent.name))
        else:
            prop_map.syn_weight = types.MethodType(syn_weight, prop_map)



        if 'syn_weight' in edge_group.columns:
            prop_map.weight = types.MethodType(syn_weight, prop_map)
            prop_map.preselected_targets = True
            prop_map.nsyns = types.MethodType(no_nsyns, prop_map)
        else:
            prop_map.preselected_targets = False
        '''
        return prop_map


def syn_weight(self, edge, source, target):
    return edge['syn_weight']


def weight_function(self, edge, source, target):
    weight_fnc_name = edge['weight_function']
    if weight_fnc_name is None:
        weight_fnc = nrn.py_modules.synaptic_weight('default_weight_fnc')
        return weight_fnc(edge, source, target)

    elif nrn.py_modules.has_synaptic_weight(weight_fnc_name):
        weight_fnc = nrn.py_modules.synaptic_weight(weight_fnc_name)
        return weight_fnc(edge, source, target)

    else:
        io.log_exception('weight_function {} is not defined.'.format(weight_fnc_name))


def nsyns(self, edge):
    return edge['nsyns']


def no_nsyns(self, edge):
    return 1


def target_sections(self, edge):
    # TODO: use biograph.__preprocess_edge_types to save
    print edge['distance_range']
    exit()
    return edge['distance_range']

