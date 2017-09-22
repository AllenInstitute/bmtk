class Edge(object):
    def __init__(self, edge_id, sources, targets, edge_params):
        self.__edge_id = edge_id
        self.__sources = sources
        self.__targets = targets
        self.__edge_params = edge_params

    @property
    def id(self):
        return self.__edge_id

    @property
    def sources(self):
        return self.__sources

    @property
    def targets(self):
        return self.__targets

    @property
    def parameters(self):
        return self.__edge_params

    def __repr__(self):
        rstring = 'EdgeType(edge_type:{}, sources:"{}", targets:"{}", params={})'.format(self.__edge_id,
                                                                                         self.__sources.filter_str,
                                                                                         self.__targets.filter_str,
                                                                                         self.__edge_params)
        return rstring