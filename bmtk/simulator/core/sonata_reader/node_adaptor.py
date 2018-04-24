import types


class SonataBaseNode(object):
    def __init__(self, node, prop_adaptor):
        self._node = node
        self._prop_adaptor = prop_adaptor

    @property
    def node_id(self):
        return self._prop_adaptor.node_id(self._node)

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

    def __getitem__(self, prop_key):
        return self._node[prop_key]


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

    @classmethod
    def create_adaptor(cls, node_group, network):
        prop_map = cls(network)
        return cls.patch_adaptor(prop_map, node_group)

    @staticmethod
    def patch_adaptor(adaptor, node_group):
        # Use node_id if the user hasn't specified a gid table
        if not node_group.has_gids:
            adaptor.gid = types.MethodType(NodeAdaptor.node_id, adaptor)

        # dynamics_params
        if node_group.has_dynamics_params:
            adaptor.dynamics_params = types.MethodType(group_dynamics_params, adaptor)
        else:  # 'dynamics_params' in node_group.all_columns:
            adaptor.dynamics_params = types.MethodType(types_dynamics_params, adaptor)

        if 'model_template' not in node_group.all_columns:
            adaptor.model_template = types.MethodType(none_function, adaptor)

        if 'model_processing' not in node_group.all_columns:
            adaptor.model_processing = types.MethodType(none_function, adaptor)

        return adaptor

    def get_node(self, sonata_node):
        return SonataBaseNode(sonata_node, self)


def none_function(self, node):
    return None


def types_dynamics_params(self, node):
    return node['dynamics_params']


def group_dynamics_params(self, node):
    return node.dynamics_params

