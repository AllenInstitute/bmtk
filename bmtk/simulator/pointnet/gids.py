from collections import namedtuple

PopulationID = namedtuple('PopulationID', 'node_id population')


class GidPool(object):
    def __init__(self):
        self._popid2gid = {}  # (pop_name, node_id) --> nest_id
        self._gid2pop_id = {}  # nest_id --> (pop_name, node_id)

    @property
    def gids(self):
        return list(self._gid2pop_id.keys())

    def create_pool(self, name):
        self._popid2gid[name] = {}

    def add(self, name, node_id, gid):
        self._popid2gid[name][node_id] = gid
        self._gid2pop_id[gid] = PopulationID(population=name, node_id=node_id)

    def get_gid(self, name, node_id):
        return self._popid2gid[name][node_id]

    def get_pool_id(self, gid):
        return self._gid2pop_id[gid]
