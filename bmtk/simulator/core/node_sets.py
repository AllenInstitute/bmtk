from .io_tools import io


class NodeSet(object):
    def __init__(self, filter_params, network):
        self._network = network
        self._populations = []
        self._preselected_gids = None

        if isinstance(filter_params, list):
            self._preselected_gids = filter_params
        elif isinstance(filter_params, dict):
            self._filter = filter_params.copy()
            self._populations = self._find_populations()
        else:
            io.log_exception('Unknown node set params type {}'.format(type(filter_params)))

    def _find_populations(self):
        for k in ['population', 'populations']:
            if k in self._filter:
                node_pops = []
                for pop_name in to_list(self._filter[k]):
                    node_pops.append(self._network.get_node_population(pop_name))
                del self._filter[k]
                return node_pops

        return self._network.get_node_populations()

    def populations(self):
        return self._populations

    def population_names(self):
        return [p.name for p in self._populations]

    def gids(self):
        if self._preselected_gids is not None:
            for gid in self._preselected_gids:
                yield gid
        else:
            for pop in self._populations:
                for node in pop.filter(self._filter):
                    yield self._network.gid_pool.get_gid(name=pop.name, node_id=node.node_id)

    def nodes(self):
        return None


class NodeSetAll(NodeSet):
    def __init__(self, network):
        super(NodeSetAll, self).__init__({}, network)


def to_list(val):
    if isinstance(val, list):
        return val
    else:
        return [val]
