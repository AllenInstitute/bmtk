import numpy as np
from collections import namedtuple

PopulationID = namedtuple('PopulationID', 'node_id population')


class GidPool(object):
    def __init__(self):
        # map from pool-id --> gid
        self._accumulated_offset = 0
        self._pool_offsets = {}

        # map from gid --> pop, node_id
        self._offsets = np.array([0], dtype=np.uint64)
        self._offset2pool_map = {}

    def add_pool(self, name, n_nodes):
        offset_index = len(self._offsets)
        self._offset2pool_map[offset_index] = name
        self._offsets = np.append(self._offsets, np.array([self._accumulated_offset + n_nodes], dtype=np.uint64))

        self._pool_offsets[name] = self._accumulated_offset
        self._accumulated_offset += n_nodes

    def get_gid(self, name, node_id):
        return self._pool_offsets[name] + node_id

    def get_pool_id(self, gid):
        offset_indx = np.searchsorted(self._offsets, gid, 'right')
        node_id = gid - self._offsets[offset_indx-1]
        pool_name = self._offset2pool_map[offset_indx]
        return PopulationID(int(node_id), pool_name)
