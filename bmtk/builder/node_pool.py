from ast import literal_eval


class NodePool(object):
    """Stores a collection of nodes based off some query of the network.

    Returns the results of a query of nodes from a network using the nodes() method. Nodes are still generated and
    saved by the network, this just stores the query information and provides iterator methods for accessing different
    nodes.

    TODO:
    * Implement a collection-set algebra including | and not operators. ie.
        nodes = net.nodes(type=1) | net.nodes(type=2)
    * Implement operators on properties
        nodes = net.nodes(val) > 100
        nodes = 100 in net.nodes(val)
    """

    def __init__(self, network, **properties):
        self.__network = network
        self.__properties = properties
        self.__filter_str = None

    def __len__(self):
        return sum(1 for _ in self)

    def __iter__(self):
        return (n for n in self.__network._nodes_iter() if self.__query_object_properties(n, self.__properties))

    @property
    def network(self):
        return self.__network

    @property
    def network_name(self):
        return self.__network.name

    @property
    def filter_str(self):
        if self.__filter_str is None:
            if len(self.__properties) == 0:
                self.__filter_str = '*'
            else:
                self.__filter_str = ''
                for k, v in self.__properties.iteritems():
                    conditional = "{}=='{}'".format(k, v)
                    self.__filter_str += conditional + '&'
                if self.__filter_str.endswith('&'):
                    self.__filter_str = self.__filter_str[0:-1]

        return self.__filter_str

    @classmethod
    def from_filter(cls, network, filter_str):
        assert(isinstance(filter_str, basestring))
        if len(filter_str) == 0 or filter_str == '*':
            return cls(network, position=None)

        properties = {}
        for condtional in filter_str.split('&'):
            var, val = condtional.split('==')
            properties[var] = literal_eval(val)
        return cls(network, position=None, **properties)

    def __query_object_properties(self, obj, props):
        if props is None:
            return True

        for k, v in props.iteritems():
            ov = obj.get(k, None)
            if ov is None:
                return False

            if hasattr(v, '__call__'):
                if not v(ov):
                    return False
            elif isinstance(v, list):
                if ov not in v:
                    return False
            elif ov != v:
                return False

        return True
