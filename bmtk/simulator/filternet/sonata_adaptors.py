import os
import json
import types
import numpy as np

from bmtk.simulator.core.sonata_reader import NodeAdaptor, SonataBaseNode, EdgeAdaptor, SonataBaseEdge


class FilterNode(SonataBaseNode):
    def __init__(self, node, prop_adaptor):
        super(FilterNode, self).__init__(node, prop_adaptor)
        self._jitter = (0.0, 0.0)

    @property
    def non_dom_params(self):
        return self._prop_adaptor.non_dom_params(self._node)

    @property
    def tuning_angle(self):
        return self._prop_adaptor.tuning_angle(self._node)

    @property
    def predefined_jitter(self):
        return self._prop_adaptor.predefined_jitter

    @property
    def jitter(self):
        if self.predefined_jitter:
            return (self._node['jitter_lower'], self._node['jitter_upper'])
        else:
            return self._jitter

    @jitter.setter
    def jitter(self, val):
        self._jitter = val

    @property
    def sf_sep(self):
        return self._node['sf_sep']

    @property
    def weights(self):
        return self._prop_adaptor.weights(self._node)

    @property
    def kpeaks(self):
        return self._prop_adaptor.kpeaks(self._node)

    @property
    def delays(self):
        return self._prop_adaptor.delays(self._node)

    @property
    def weights_non_dom(self):
        return self._prop_adaptor.weights_non_dom(self._node)

    @property
    def kpeaks_non_dom(self):
        return self._prop_adaptor.kpeaks_non_dom(self._node)

    @property
    def delays_non_dom(self):
        return self._prop_adaptor.delays_non_dom(self._node)


class FilterNodeAdaptor(NodeAdaptor):
    def get_node(self, sonata_node):
        return FilterNode(sonata_node, self)

    @staticmethod
    def preprocess_node_types(network, node_population):
        NodeAdaptor.preprocess_node_types(network, node_population)

        node_type_ids = np.unique(node_population.type_ids)
        node_types_table = node_population.types_table
        if 'non_dom_params' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                dynamics_params = node_type['non_dom_params']
                if isinstance(dynamics_params, dict):
                    continue
    
                if dynamics_params is None:
                    continue

                params_dir = network.get_component('filter_models_dir')
                params_path = os.path.join(params_dir, dynamics_params)
                try:
                    params_val = json.load(open(params_path, 'r'))
                    node_type['non_dom_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    network.io.log_exception('Could not find node dynamics_params file {}.'.format(params_path))

    @classmethod
    def patch_adaptor(cls, adaptor, node_group, network):
        node_adaptor = NodeAdaptor.patch_adaptor(adaptor, node_group, network)

        if 'non_dom_params' in node_group.all_columns:
            node_adaptor.non_dom_params = types.MethodType(non_dom_params, node_adaptor)
        else:
            node_adaptor.non_dom_params = types.MethodType(return_none, node_adaptor)

        jitter_lower = 'jitter_lower' in node_group.all_columns
        jitter_upper = 'jitter_upper' in node_group.all_columns
        if jitter_lower and jitter_upper:
            node_adaptor.predefined_jitter = True
        elif jitter_upper ^ jitter_lower:
            raise Exception('Need to define both jitter_lower and jitter_upper (or leave both empty)')
        else:
            node_adaptor.predefined_jitter = False

        if 'tuning_angle' in node_group.all_columns:
            node_adaptor.tuning_angle = types.MethodType(tuning_angle_preset, node_adaptor)
        else:
            node_adaptor.tuning_angle = types.MethodType(tuning_angle_rand, node_adaptor)

        find_weight_params(node_group, node_adaptor)
        find_kpeaks_params(node_group, node_adaptor)
        find_delays_params(node_group, node_adaptor)

        find_nondom_weight_params(node_group, node_adaptor)
        find_nondom_kpeaks_params(node_group, node_adaptor)
        find_nondom_delays_params(node_group, node_adaptor)

        return node_adaptor

def non_dom_params(self, node):
    return node['non_dom_params']


def return_none(self, node):
    return None


def tuning_angle_preset(self, node):
    return node['tuning_angle']


def tuning_angle_rand(self, node):
    return np.random.uniform(0.0, 360.0)

def weights(self, node):
    return node['weights']


def find_weight_params(node_group, node_adaptor):
    if 'weights' in node_group.all_columns:
        node_adaptor.weights = types.MethodType(lambda self, node: node['weights'], node_adaptor)

    elif 'opt_wts' in node_group.all_columns:
        node_adaptor.weights = types.MethodType(lambda self, node: node['opt_wts'], node_adaptor)

    elif 'weights_dom' in node_group.all_columns:
        node_adaptor.weights = types.MethodType(lambda self, node: node['weights_dom'], node_adaptor)

    elif 'weight_dom_0' in node_group.all_columns and 'weight_dom_1' in node_group.all_columns:
        node_adaptor.weights = types.MethodType(lambda self, node: [node['weight_dom_0'], node['weight_dom_1']],
                                                node_adaptor)
    else:
        node_adaptor.weights = types.MethodType(return_none, node_adaptor)


def find_kpeaks_params(node_group, node_adaptor):
    if 'kpeaks' in node_group.all_columns:
        node_adaptor.kpeaks = types.MethodType(lambda self, node: node['kpeaks'], node_adaptor)

    elif 'opt_kpeaks' in node_group.all_columns:
        node_adaptor.kpeaks = types.MethodType(lambda self, node: node['opt_kpeaks'], node_adaptor)

    elif 'kpeaks_dom' in node_group.all_columns:
        node_adaptor.kpeaks = types.MethodType(lambda self, node: node['kpeaks_dom'], node_adaptor)

    elif 'kpeaks_dom_0' in node_group.all_columns and 'kpeaks_dom_1' in node_group.all_columns:
        node_adaptor.kpeaks = types.MethodType(lambda self, node: [node['kpeaks_dom_0'], node['kpeaks_dom_1']],
                                                node_adaptor)
    else:
        node_adaptor.kpeaks = types.MethodType(return_none, node_adaptor)


def find_delays_params(node_group, node_adaptor):
    if 'delays' in node_group.all_columns:
        node_adaptor.delays = types.MethodType(lambda self, node: node['delays'], node_adaptor)

    elif 'opt_delays' in node_group.all_columns:
        node_adaptor.delays = types.MethodType(lambda self, node: node['opt_delays'], node_adaptor)

    elif 'delays_dom' in node_group.all_columns:
        node_adaptor.delays = types.MethodType(lambda self, node: node['delays_dom'], node_adaptor)

    elif 'delays_dom_0' in node_group.all_columns and 'delays_dom_1' in node_group.all_columns:
        node_adaptor.delays = types.MethodType(lambda self, node: [node['delays_dom_0'], node['delays_dom_1']],
                                                node_adaptor)
    else:
        node_adaptor.delays = types.MethodType(return_none, node_adaptor)


def find_nondom_weight_params(node_group, node_adaptor):
    if 'weights_non_dom' in node_group.all_columns:
        node_adaptor.weights_non_dom = types.MethodType(lambda self, node: node['weights_non_dom'], node_adaptor)

    elif 'weight_non_dom_0' in node_group.all_columns and 'weight_non_dom_1' in node_group.all_columns:
        node_adaptor.weights_non_dom = types.MethodType(
            lambda self, node: [node['weight_non_dom_0'], node['weight_non_dom_1']], node_adaptor
        )
    else:
        node_adaptor.weights_non_dom = types.MethodType(return_none, node_adaptor)


def find_nondom_kpeaks_params(node_group, node_adaptor):
    if 'kpeaks_non_dom' in node_group.all_columns:
        node_adaptor.kpeaks_non_dom = types.MethodType(lambda self, node: node['kpeaks_non_dom'], node_adaptor)

    elif 'kpeaks_non_dom_0' in node_group.all_columns and 'kpeaks_non_dom_1' in node_group.all_columns:
        node_adaptor.kpeaks_non_dom = types.MethodType(
            lambda self, node: [node['kpeaks_non_dom_0'], node['kpeaks_non_dom_1']], node_adaptor
        )
    else:
        node_adaptor.kpeaks_non_dom = types.MethodType(return_none, node_adaptor)


def find_nondom_delays_params(node_group, node_adaptor):
    if 'delays_non_dom' in node_group.all_columns:
        node_adaptor.delays_non_dom = types.MethodType(lambda self, node: node['delays_non_dom'], node_adaptor)

    elif 'delay_non_dom_0' in node_group.all_columns and 'delay_non_dom_1' in node_group.all_columns:
        node_adaptor.delays_non_dom = types.MethodType(
            lambda self, node: [node['delay_non_dom_0'], node['delay_non_dom_1']], node_adaptor
        )
    else:
        node_adaptor.delays_non_dom = types.MethodType(return_none, node_adaptor)
