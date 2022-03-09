import os
import json
import types
import numpy as np


class SonataBaseNode(object):
    def __init__(self, node, prop_adaptor):
        self._node = node
        self._prop_adaptor = prop_adaptor

    @property
    def node_id(self):
        return self._prop_adaptor.node_id(self._node)

    @property
    def population_name(self):
        return self._node.population_name

    @property
    def gid(self):
        return self._prop_adaptor.gid(self._node)

    @property
    def dynamics_params(self):
        return self._prop_adaptor.dynamics_params(self._node)

    @property
    def model_type(self):
        return self._prop_adaptor.model_type(self._node)

    @property
    def model_template(self):
        return self._prop_adaptor.model_template(self._node)

    @property
    def model_processing(self):
        return self._prop_adaptor.model_processing(self._node)

    @property
    def network(self):
        return self._prop_adaptor.network

    @property
    def population(self):
        return self._prop_adaptor.network

    def __getitem__(self, prop_key):
        return self._node[prop_key]

    def __contains__(self, item):
        return item in self._node


class NodeAdaptor(object):
    COL_MODEL_TYPE = 'model_type'
    COL_GID = 'gid'
    COL_DYNAMICS_PARAM = 'dynamics_params'
    COL_MODEL_TEMPLATE = 'model_template'
    COL_MODEL_PROCESSING = 'model_processing'

    def __init__(self, network):
        self._network = network
        self._model_template_cache = {}
        self._model_processing_cache = {}

    @property
    def batch_process(self):
        return False

    @batch_process.setter
    def batch_process(self, flag):
        pass

    def node_id(self, node):
        return node.node_id

    def model_type(self, node):
        return node[self.COL_MODEL_TYPE]

    def model_template(self, node):
        # TODO: If model-template comes from the types table we should split it in _preprocess_types
        model_template_str = node[self.COL_MODEL_TEMPLATE]
        if model_template_str is None:
            return None
        elif model_template_str in self._model_template_cache:
            return self._model_template_cache[model_template_str]
        else:
            template_parts = model_template_str.split(':')
            directive, template = template_parts[0], template_parts[1]
            self._model_template_cache[model_template_str] = (directive, template)
            return directive, template

    def model_processing(self, node):
        model_processing_str = node[self.COL_MODEL_PROCESSING]
        if model_processing_str is None:
            return []
        else:
            # TODO: Split in the node_types_table when possible
            return model_processing_str.split(',')

    @staticmethod
    def preprocess_node_types(network, node_population):
        # TODO: The following figures out the actually used node-type-ids. For mem and speed may be better to just
        # process them all
        #node_type_ids = node_population.type_ids
        node_type_ids = np.unique(node_population.type_ids)
        # TODO: Verify all the node_type_ids are in the table
        node_types_table = node_population.types_table

        # TODO: Convert model_type to a enum
        if network.has_component('morphologies_dir'):
            morph_dir = network.get_component('morphologies_dir')
            if morph_dir is not None and 'morphology' in node_types_table.columns:
                for nt_id in node_type_ids:
                    node_type = node_types_table[nt_id]
                    if node_type['morphology'] is None:
                        continue

                    # TODO: See if absolute path is stored in csv
                    swc_path = os.path.join(morph_dir, node_type['morphology'])

                    # According to Sonata format, the .swc extension is not needed. Thus we need to add it if req.
                    if not os.path.exists(swc_path) and not swc_path.endswith('.swc'):
                        swc_path += '.swc'
                        if not os.path.exists(swc_path):
                            network.io.log_exception('Could not find node morphology file {}.'.format(swc_path))

                    node_type['morphology'] = swc_path

        if 'dynamics_params' in node_types_table.columns and 'model_type' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                dynamics_params = node_type['dynamics_params']
                if isinstance(dynamics_params, dict):
                    continue

                if dynamics_params is None:
                    continue

                model_type = node_type['model_type']
                if model_type == 'biophysical':
                    params_dir = network.get_component('biophysical_neuron_models_dir')
                elif model_type in ['point_process', 'point_neuron']:
                    params_dir = network.get_component('point_neuron_models_dir')
                elif model_type == 'point_soma':
                    params_dir = network.get_component('point_neuron_models_dir')
                elif model_type == 'population':
                    params_dir = network.get_component('population_models_dir')
                elif model_type == 'lgnmodel' or model_type == 'virtual':
                    params_dir = network.get_component('filter_models_dir')
                else:
                    # Not sure what to do in this case, throw Exception?
                    params_dir = network.get_component('custom_neuron_models')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    node_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    network.io.log_exception('Could not find node dynamics_params file {}.'.format(params_path))

        # TODO: Use adaptor to validate model_type and model_template values

    @classmethod
    def create_adaptor(cls, node_group, network):
        prop_map = cls(network)
        return cls.patch_adaptor(prop_map, node_group, network)

    @classmethod
    def patch_adaptor(cls, adaptor, node_group, network):
        adaptor.network = network

        # Use node_id if the user hasn't specified a gid table
        if not node_group.has_gids:
            adaptor.gid = types.MethodType(NodeAdaptor.node_id, adaptor)

        # dynamics_params
        if node_group.has_dynamics_params:
            adaptor.dynamics_params = types.MethodType(group_dynamics_params, adaptor)
        elif 'dynamics_params' in node_group.all_columns:
            adaptor.dynamics_params = types.MethodType(types_dynamics_params, adaptor)
        else:
            adaptor.dynamics_params = types.MethodType(none_function, adaptor)

        if 'model_template' not in node_group.all_columns:
            adaptor.model_template = types.MethodType(none_function, adaptor)

        if 'model_processing' not in node_group.all_columns:
            adaptor.model_processing = types.MethodType(empty_list, adaptor)

        return adaptor

    def get_node(self, sonata_node):
        return SonataBaseNode(sonata_node, self)


def none_function(self, node):
    return None


def empty_list(self, node):
    return []


def types_dynamics_params(self, node):
    return node['dynamics_params']


def group_dynamics_params(self, node):
    return node.dynamics_params

