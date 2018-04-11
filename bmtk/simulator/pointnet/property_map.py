import types
import numpy as np

import nest

from bmtk.simulator.pointnet.pyfunction_cache import py_modules
from bmtk.simulator.pointnet.io_tools import io

class NodePropertyMap(object):
    def __init__(self, graph):
        self._graph = graph
        # TODO: Move template_cache to parent graph so it can be shared across diff populations.
        self._template_cache = {}
        self.node_types_table = None

        self.batch = True


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
        model_type = self._parse_model_template(node['model_template'])[1]
        dynamics_params = self.dynamics_params(node)
        fnc_name = node['model_processing']
        if fnc_name is None:
            return nest.Create(model_type, 1, dynamics_params)
        else:
            cell_fnc = py_modules.cell_processor(fnc_name)
            return cell_fnc(model_type, node, dynamics_params)

    @classmethod
    def build_map(cls, node_group, graph):
        prop_map = cls(graph)

        node_types_table = node_group.parent.node_types_table
        prop_map.node_types_table = node_types_table

        if 'model_processing' in node_group.columns:
            prop_map.batch = False
        elif 'model_processing' in node_group.all_columns:
            model_fncs = [node_types_table[ntid]['model_processing'] for ntid in np.unique(node_group.node_type_ids)
                          if node_types_table[ntid]['model_processing'] is not None]

            if model_fncs:
                prop_map.batch = False

        if node_group.has_dynamics_params:
            prop_map.batch = False
            prop_map.dynamics_params = types.MethodType(group_dynamics_params, prop_map)
        else:  # 'dynamics_params' in node_group.all_columns:
            prop_map.dynamics_params = types.MethodType(types_dynamics_params, prop_map)

        if prop_map.batch:
            prop_map.model_type = types.MethodType(model_type_batched, prop_map)
            prop_map.model_params = types.MethodType(model_params_batched, prop_map)
        else:
            prop_map.model_type = types.MethodType(model_type, prop_map)
            prop_map.model_params = types.MethodType(model_params, prop_map)

        if node_group.has_gids:
            prop_map.gid = types.MethodType(gid, prop_map)
        else:
            prop_map.gid = types.MethodType(node_id, prop_map)

        return prop_map


def gid(self, node):
    return node['gid']


def node_id(self, node):
    return node.node_id


def model_type(self, node):
    return self._parse_model_template(node['model_template'])


def model_type_batched(self, node_type_id):
    return self._parse_model_template(self.node_types_table[node_type_id]['model_template'])


def model_params(self, node):
    return {}


def model_params_batched(self, node_type_id):
    return self.node_types_table[node_type_id]['dynamics_params']


def types_dynamics_params(self, node):
    return node['dynamics_params']


def group_dynamics_params(self, node):
    return node.dynamics_params


class EdgePropertyMap(object):
    def __init__(self, graph, source_population, target_population):
        self._graph = graph
        self._source_population = source_population
        self._target_population = target_population

        self.batch = True
        self.synpatic_models = []


    def synaptic_model(self, edge):
        return edge['model_template']


    def synpatic_params(self, edge):
        params_dict = {'weight': self.syn_weight(edge), 'delay': edge['delay']}
        params_dict.update(edge['dynamics_params'])
        return params_dict

    @classmethod
    def build_map(cls, edge_group, biograph):
        prop_map = cls(biograph, edge_group.parent.source_population, edge_group.parent.source_population)
        if 'model_template' in edge_group.columns:
            prop_map.batch = False
        elif 'model_template' in edge_group.all_columns:
            edge_types_table = edge_group.parent.edge_types_table
            syn_models = set(edge_types_table[etid]['model_template']
                             for etid in np.unique(edge_types_table.edge_type_ids))
            prop_map.synpatic_models = list(syn_models)
        else:
            prop_map.synpatic_models = ['static_synapse']
            #s = [edge_types_table[ntid]['model_template'] for ntid in np.unique(edge_types_table.node_type_ids)
            #              if edge_types_table[ntid]['model_template'] is not None]


        # For fetching/calculating synaptic weights
        edge_types_weight_fncs = set()
        edge_types_table = edge_group.parent.edge_types_table
        for etid in edge_types_table.edge_type_ids:
            weight_fnc = edge_types_table[etid].get('weight_function', None)
            if weight_fnc is not None:
                edge_types_weight_fncs.add(weight_fnc)

        if 'weight_function' in edge_group.group_columns or edge_types_weight_fncs:
            # Customized function for user to calculate the synaptic weight
            prop_map.syn_weight = types.MethodType(weight_function, prop_map)

        elif 'syn_weight' in edge_group.all_columns:
            # Just return the synaptic weight
            prop_map.syn_weight = types.MethodType(syn_weight, prop_map)
        else:
            io.log_exception('Could not find syn_weight or weight_function properties. Cannot create connections.')

        # For determining the synapse placement
        if 'nsyns' in edge_group.all_columns:
            prop_map.nsyns = types.MethodType(nsyns, prop_map)
        else:
            # It will get here for connections onto point neurons
            prop_map.nsyns = types.MethodType(no_syns, prop_map)

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


def syn_weight(self, edge):
    return edge['syn_weight']*self.nsyns(edge)


def weight_function(self, edge):
    weight_fnc_name = edge['weight_function']
    src_node = self._graph.get_node(self._source_population, edge.source_node_id)
    trg_node = self._graph.get_node(self._target_population, edge.target_node_id)

    if weight_fnc_name is None:
        weight_fnc = py_modules.synaptic_weight('default_weight_fnc')
        return weight_fnc(edge, src_node, trg_node)# *self.nsyns(edge)

    elif py_modules.has_synaptic_weight(weight_fnc_name):
        weight_fnc = py_modules.synaptic_weight(weight_fnc_name)
        return weight_fnc(edge, src_node, trg_node)

    else:
        io.log_exception('weight_function {} is not defined.'.format(weight_fnc_name))


def nsyns(self, edge):
    return edge['nsyns']


def no_syns(self, edge):
    return 1