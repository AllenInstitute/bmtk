from node import Node


class NodeSet(object):
    def __init__(self, N, node_params, node_type_properties):
        self.__N = N
        self.__node_params = node_params
        self.__node_type_properties = node_type_properties

        assert('node_type_id' in node_type_properties)
        self.__node_type_id = node_type_properties['node_type_id']

        # Used for determining which node_sets share the same params columns
        columns = self.__node_params.keys()
        columns.sort()
        self.__params_col_hash = hash(str(columns))

    @property
    def N(self):
        return self.__N

    @property
    def node_type_id(self):
        return self.__node_type_id

    @property
    def params_keys(self):
        return self.__node_params.keys()

    @property
    def params_hash(self):
        return self.__params_col_hash

    def build(self, nid_generator):
        # fetch existing node ids or create new ones
        node_ids = self.__node_params.get('node_id', None)
        if node_ids is None:
            node_ids = [nid for nid in nid_generator(self.N)]

        # turn node_params from dictionary of lists to a list of dictionaries.
        ap_flat = [{} for _ in xrange(self.N)]
        for key, plist in self.__node_params.iteritems():
            for i, val in enumerate(plist):
                ap_flat[i][key] = val

        # create node objects
        return [Node(nid, params, self.__node_type_properties, self.__params_col_hash)
                for (nid, params) in zip(node_ids, ap_flat)]
