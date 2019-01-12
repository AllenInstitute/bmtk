import os
import json
import types
import numpy as np

from bmtk.simulator.core.sonata_reader import NodeAdaptor, SonataBaseNode, EdgeAdaptor, SonataBaseEdge


class FilterNode(SonataBaseNode):
    @property
    def non_dom_params(self):
        return self._prop_adaptor.non_dom_params(self._node)


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

        return node_adaptor


def non_dom_params(self, node):
    return node['non_dom_params']


def return_none(self, node):
    return None
