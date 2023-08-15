from collections import namedtuple
import pandas as pd
import nest

from .nest_utils import nest_version

PopulationID = namedtuple('PopulationID', 'node_id population')


def ids2list_nest2(nest_ids):
    return nest_ids


def ids2list_nest3(nest_ids):
    if isinstance(nest_ids, nest.NodeCollection):
        return nest_ids.tolist()
    else:
        return nest_ids


ids2list = ids2list_nest3 if nest_version[0] >= 3 else ids2list_nest2


class GidPool(object):
    def __init__(self):
        # self._popid2gid = {}  # (pop_name, node_id) --> nest_id
        self._gid2pop_id = {}  # nest_id --> (pop_name, node_id)
        self._nestid_lu = {}

    @property
    def gids(self):
        return list(self._gid2pop_id.keys())

    @property
    def populations(self):
        return list(self._nestid_lu.keys())

    def add(self, name, node_id, gid):
        raise NotImplementedError()

    def get_gid(self, name, node_id):
        return self.get_nestids(name=name, node_ids=[node_id])[0]

    def get_pool_id(self, gid):
        return self._gid2pop_id[gid]

    def create_pool(self, name):
        pass

    def add_nestids(self, name, node_ids, nest_ids):
        # in NEST 3.0+ nest.Create() returns a NodeCollection instead of a list of ids, need to convert
        nest_ids = ids2list(nest_ids)

        if name not in self._nestid_lu:
            lu_table = pd.DataFrame({'nest_ids': nest_ids, 'node_ids': node_ids})
            lu_table = lu_table.set_index('node_ids')
        else:
            new_df = pd.DataFrame({'nest_ids': nest_ids, 'node_ids': node_ids})
            new_df = new_df.set_index('node_ids')
            lu_table = self._nestid_lu[name]
            lu_table = pd.concat((lu_table, new_df))
            # lu_table = lu_table.reindex(lu_table.index.values)

        self._nestid_lu[name] = lu_table

        for node_id, nest_id in zip(node_ids, nest_ids):
            self._gid2pop_id[nest_id] = PopulationID(population=name, node_id=node_id)

    def add_gids(self, name, node_ids, gids):
        self.add_nestids(name=name, node_ids=node_ids, nest_ids=gids)

    def get_nestids(self, name, node_ids):
        nestids_table = self._nestid_lu[name]
        return nestids_table.loc[node_ids]['nest_ids'].values

    def get_gids(self, name, node_ids):
        return self.get_nestids(name=name, node_ids=node_ids)

    def __len__(self):
        return len(self.gids)
