class Node(dict):
    def __init__(self, node_id, node_params, node_type_properties, params_hash=-1):
        super(Node, self).__init__({})

        self._node_params = node_params
        self._node_params['node_id'] = node_id
        self._node_type_properties = node_type_properties
        self._params_hash = params_hash
        self._node_id = node_id

    @property
    def node_id(self):
        return self._node_id

    @property
    def node_type_id(self):
        return self._node_type_properties['node_type_id']

    @property
    def params(self):
        return self._node_params

    @property
    def node_type_properties(self):
        return self._node_type_properties

    @property
    def params_hash(self):
        return self._params_hash

    def get(self, key, default=None):
        if key in self._node_params:
            return self._node_params[key]
        elif key in self._node_type_properties:
            return self._node_type_properties[key]
        else:
            return default

    def __contains__(self, item):
        return item in self._node_type_properties or item in self._node_params

    def __getitem__(self, item):
        if item in self._node_params:
            return self._node_params[item]
        else:
            return self._node_type_properties[item]

    def __hash__(self):
        return hash(self.node_id)

    def __repr__(self):
        tmp_dict = dict(self._node_type_properties)
        tmp_dict.update(self._node_params)
        return tmp_dict.__repr__()
