import os
import ast
import json
import types

import numpy as np


class SonataBaseEdge(object):
    def __init__(self, sonata_edge, edge_adaptor):
        self._edge = sonata_edge
        self._prop_adaptor = edge_adaptor

    @property
    def source_node_id(self):
        return self._edge.source_node_id

    @property
    def source_population(self):
        return self._edge.source_population

    @property
    def target_node_id(self):
        return self._edge.target_node_id

    @property
    def target_population(self):
        return self._edge.target_population

    @property
    def dynamics_params(self):
        return self._prop_adaptor.dynamics_params(self._edge)

    @property
    def delay(self):
        return self._edge['delay']

    @property
    def weight_function(self):
        return self._prop_adaptor.weight_function(self._edge)

    @property
    def preselected_targets(self):
        return self._prop_adaptor.preselected_targets

    @property
    def target_sections(self):
        return self._edge['target_sections']

    @property
    def target_distance(self):
        return self._edge['distance_range']

    @property
    def is_gap_junction(self):
        try:
            return self._edge['is_gap_junction']
        except:
            return False

    @property
    def edge_type_id(self):
        return self._edge.edge_type_id

    @property
    def nsyns(self):
        return self._prop_adaptor.nsyns(self._edge)

    @property
    def model_template(self):
        return self._edge['model_template']

    def syn_weight(self, src_node, trg_node):
        return self._prop_adaptor.syn_weight(self, src_node=src_node, trg_node=trg_node)

    def __getitem__(self, item):
        return self._edge[item]


class EdgeAdaptor(object):
    def __init__(self, network):
        self._network = network
        self._func_caches = self._network.py_function_caches

    @property
    def batch_process(self):
        return False

    @batch_process.setter
    def batch_process(self, flag):
        pass

    def get_edge(self, sonata_node):
        return SonataBaseEdge(sonata_node, self)

    @staticmethod
    def preprocess_edge_types(network, edge_population):
        edge_types_table = edge_population.types_table
        edge_type_ids = np.unique(edge_population.type_ids)

        for et_id in edge_type_ids:
            edge_type = edge_types_table[et_id]
            if 'dynamics_params' in edge_types_table.columns and edge_type['dynamics_params'] != None:
                dynamics_params = edge_type['dynamics_params']
                params_dir = network.get_component('synaptic_models_dir')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    edge_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    network.io.log_exception('Could not find edge dynamics_params file {}.'.format(params_path))

            # Split target_sections
            if 'target_sections' in edge_type:
                trg_sec = edge_type['target_sections']
                if trg_sec is not None:
                    try:
                        edge_type['target_sections'] = ast.literal_eval(trg_sec)
                    except Exception as exc:
                        network.io.log_warning('Unable to split target_sections list {}'.format(trg_sec))
                        edge_type['target_sections'] = None

            # Split target distances
            if 'distance_range' in edge_type:
                dist_range = edge_type['distance_range']
                if dist_range is not None:
                    try:
                        # TODO: Make the distance range has at most two values
                        edge_type['distance_range'] = json.loads(dist_range)
                    except Exception as e:
                        try:
                            edge_type['distance_range'] = [0.0, float(dist_range)]
                        except Exception as e:
                            network.io.log_warning('Unable to parse distance_range {}'.format(dist_range))
                            edge_type['distance_range'] = None

    @classmethod
    def create_adaptor(cls, edge_group, network):
        prop_map = cls(network)
        return cls.patch_adaptor(prop_map, edge_group)

    @staticmethod
    def patch_adaptor(adaptor, edge_group):
        # dynamics_params
        if edge_group.has_dynamics_params:
            adaptor.dynamics_params = types.MethodType(group_dynamics_params, adaptor)
        else:  # 'dynamics_params' in node_group.all_columns:
            adaptor.dynamics_params = types.MethodType(types_dynamics_params, adaptor)

        # For fetching/calculating synaptic weights
        if 'weight_function' in edge_group.all_columns:
            # Customized function for user to calculate the synaptic weight
            adaptor.weight_function = types.MethodType(weight_function, adaptor)
            adaptor.syn_weight = types.MethodType(syn_weight_function, adaptor)
        elif 'syn_weight' in edge_group.all_columns:
            # Just return the synaptic weight
            adaptor.weight_function = types.MethodType(ret_none_function, adaptor)
            adaptor.syn_weight = types.MethodType(syn_weight, adaptor)
        else:
            raise Exception('Could not find syn_weight or weight_function properties. Cannot create connections.')

        # For determining the synapse placement
        if 'sec_id' in edge_group.all_columns:
            adaptor.preselected_targets = True
            adaptor.nsyns = types.MethodType(no_nsyns, adaptor)
        elif 'nsyns' in edge_group.all_columns:
            adaptor.preselected_targets = False
            adaptor.nsyns = types.MethodType(nsyns, adaptor)
        else:
            # It will get here for connections onto point neurons
            adaptor.preselected_targets = True
            adaptor.nsyns = types.MethodType(no_nsyns, adaptor)

        return adaptor


def ret_none_function(self, edge):
    return None


def weight_function(self, edge):
    return edge['weight_function']


def syn_weight(self, edge, src_node, trg_node):
    return edge['syn_weight']


def syn_weight_function(self, edge, src_node, trg_node):
    weight_fnc_name = edge.weight_function
    if weight_fnc_name is None:
        weight_fnc = self._func_caches.py_modules.synaptic_weight('default_weight_fnc')
        return weight_fnc(edge, src_node, trg_node)

    elif self._func_caches.py_modules.has_synaptic_weight(weight_fnc_name):
        weight_fnc = self._func_caches.py_modules.synaptic_weight(weight_fnc_name)
        return weight_fnc(edge, src_node, trg_node)

    else:
        self._network.io.log_exception('weight_function {} is not defined.'.format(weight_fnc_name))


def nsyns(self, edge):
    return edge['nsyns']


def no_nsyns(self, edge):
    return 1


def types_dynamics_params(self, node):
    return node['dynamics_params']


def group_dynamics_params(self, node):
    return node.dynamics_params