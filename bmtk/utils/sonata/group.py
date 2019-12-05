# Copyright 2017. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import pandas as pd

from .column_property import ColumnProperty
from .node import Node, NodeSet
from .edge import Edge, EdgeSet


class Group(object):
    """A container containig a node/edge population groups.

    A node or edge population will have one or more groups, each having a unique identifier. Each group shared the same
    columns and datatypes, thus each group is essentially a different model.
    """

    def __init__(self, group_id, h5_group, parent):
        self._group_id = int(group_id)
        self._parent = parent
        self._types_table = parent.types_table
        self._h5_group = h5_group
        self._types_index_col = self._types_table.index_column_name

        self._group_columns = ColumnProperty.from_h5(h5_group)
        # TODO: combine group_columns, group_column_names and group_columns_map, doesn't need to be 3 structures
        self._group_column_map = {col.name: col for col in self._group_columns}
        self._group_column_names = set(col.name for col in self._group_columns)
        self._group_table = {prop: h5_group[prop.name] for prop in self._group_columns}
        self._ncolumns = len(self._group_columns)

        self._all_columns = self._group_columns + self._types_table.columns
        self._all_column_names = set(col.name for col in self._all_columns)

        self._nrows = 0  # number of group members

        # For storing dynamics_params subgroup (if it exists)
        self._has_dynamics_params = 'dynamics_params' in self._h5_group and len(self._h5_group['dynamics_params']) > 0
        self._dynamics_params_columns = []

        # An index of all the rows in parent population that map onto a member of this group
        self._parent_indicies = None  # A list of parent rows indicies
        self._parent_indicies_built = False

        self.check_format()

    @property
    def group_id(self):
        return self._group_id

    @property
    def has_dynamics_params(self):
        return False

    @property
    def columns(self):
        return self._group_columns

    @property
    def group_columns(self):
        return self._group_columns

    @property
    def all_columns(self):
        return self._all_columns

    @property
    def has_gids(self):
        return self._parent.has_gids

    @property
    def parent(self):
        return self._parent

    def get_dataset(self, column_name):
        return self._group_table[column_name]

    def column(self, column_name, group_only=False):
        if column_name in self._group_column_map:
            return self._group_column_map[column_name]
        elif not group_only and column_name in self._types_table.columns:
            return self._types_table.column(column_name)
        else:
            return KeyError

    def check_format(self):
        # Check that all the properties have the same number of rows
        col_counts = [col.nrows for col in self._group_columns + self._dynamics_params_columns]
        if len(set(col_counts)) > 1:
            # TODO: Would be nice to warn user which dataset have different size
            raise Exception('properties in {}/{} have different ranks'.format(self._parent.name, self._group_id))
        elif len(set(col_counts)) == 1:
            self._nrows = col_counts[0]

    def build_indicies(self, force=False):
        raise NotImplementedError

    def to_dataframe(self):
        raise NotImplementedError

    def get_values(self, property_name, all_rows=False):
        """Returns all values for a group property.

        Note that a row within a group may not have a corresponding node/edge, or they may have a different order or
        multiple node/edges may share the same group row. Setting all_rows=False will return all the values as you
        see if you iterated through all the population's items. Setting all_rows=True just returns the data as a
        list as they appear in the dataset (will be faster).

        :param property_name: Name of dataset property/column to fetch.
        :param all_rows: Set false to return order in which they appear in population, false to return entire dataset
        :return: A list of values for the given column name.
        """
        raise NotImplementedError

    def __len__(self):
        return self._nrows

    def __getitem__(self, group_index):
        group_props = {}
        for cname, h5_obj in self._group_table.items():
            group_props[cname] = h5_obj[group_index]
        return group_props

    def __contains__(self, prop_name):
        """Search that a column name exists in this group"""
        return prop_name in self._group_column_names


class NodeGroup(Group):
    def __init__(self, group_id, h5_group, parent):
        super(NodeGroup, self).__init__(group_id, h5_group, parent)
        # Note: Don't call build_indicies right away so uses can call __getitem__ without having to load all the
        # node_ids

    @property
    def node_ids(self):
        self.build_indicies()
        # print self._parent_indicies
        return self._parent.inode_ids(self._parent_indicies)

    @property
    def node_type_ids(self):
        self.build_indicies()
        return self._parent.inode_type_ids(self._parent_indicies)

    @property
    def gids(self):
        self.build_indicies()
        return self._parent.igids(self._parent_indicies)

    def build_indicies(self, force=False):
        if self._parent_indicies_built and not force:
            return

        # TODO: Check for the special case where there is only one group
        # TODO: If memory becomes an issue on very larget nodes (10's of millions) consider using a generator
        # I've pushed the actual building of the population->group indicies onto the parent population
        self._parent_indicies = self._parent.group_indicies(self.group_id, build_cache=True)
        self._parent_indicies_built = True

    def get_values(self, property_name, filtered_indicies=True):
        self.build_indicies()
        # TODO: Check if property_name is node_id, node_type, or gid

        if property_name in self._group_columns:
            if not filtered_indicies:
                # Just return all values in dataset
                return np.array(self._group_table[property_name])
            else:
                # Return only those values for group indicies with associated nodes
                grp_indicies = self._parent.igroup_indicies(self._parent_indicies)
                # It is possible that the group_index is unorderd or contains duplicates which will cause h5py slicing
                # to fail. Thus convert to a numpy array
                # TODO: loading the entire table is not good if the filtered nodes is small, consider building.
                tmp_array = np.array(self._group_table[property_name])
                return tmp_array[grp_indicies]

        elif property_name in self._parent.node_types_table.columns:
            # For properties that come from node-types table we need to build the results from scratch
            # TODO: Need to performance test, I think this code could be optimized.
            node_types_table = self._parent.node_types_table
            nt_col = node_types_table.column(property_name)
            tmp_array = np.empty(shape=len(self._parent_indicies), dtype=nt_col.dtype)
            for i, ntid in enumerate(self.node_type_ids):
                tmp_array[i] = node_types_table[ntid][property_name]

            return tmp_array

    def to_dataframe(self):
        self.build_indicies()

        # Build a dataframe of group properties
        # TODO: Include dynamics_params?
        properties_df = pd.DataFrame()
        for col in self._group_columns:
            if col.dimension > 1:
                for i in range(col.dimension):
                    # TODO: see if column name exists in the attributes
                    col_name = '{}.{}'.format(col.name, i)
                    properties_df[col_name] = pd.Series(self._h5_group[col.name][:, i])
            else:
                properties_df[col.name] = pd.Series(self._h5_group[col.name])

        # Build a dataframe of parent node (node_id, gid, node_types, etc)
        root_df = pd.DataFrame()
        root_df['node_type_id'] = pd.Series(self.node_type_ids)
        root_df['node_id'] = pd.Series(self.node_ids)
        root_df['node_group_index'] = pd.Series(self._parent.igroup_indicies(self._parent_indicies))  # used as pivot
        if self._parent.has_gids:
            root_df['gid'] = self.gids

        # merge group props df with parent df
        results_df = root_df.merge(properties_df, how='left', left_on='node_group_index', right_index=True)
        results_df = results_df.drop('node_group_index', axis=1)

        # Build node_types dataframe and merge
        node_types_df = self._parent.node_types_table.to_dataframe()
        # remove properties that exist in the group
        node_types_cols = [c.name for c in self._parent.node_types_table.columns if c not in self._group_columns]
        node_types_df = node_types_df[node_types_cols]

        # TODO: consider caching these results
        return results_df.merge(node_types_df, how='left', left_on='node_type_id', right_index=True)

    def filter(self, **filter_props):
        """Filter all nodes in the group by key=value pairs.

        The filter specifications may apply to either node_type or group column properties. Currently at the moment
        it only supports equivlency. An intersection (and operator) is done for every different filter pair. This will
        produce a generator of all nodes matching the the filters.

        for node in filter(pop_name='VIp', depth=10.0):
           assert(node['pop_name'] == 'VIp' and node['depth'] == 10.0)

        :param filter_props: keys and their values to filter nodes on.
        :return: A generator that produces all valid nodes within the group with matching key==value pairs.
        """
        # TODO: Integrate this with NodeSet.
        self.build_indicies()
        node_types_table = self._parent.node_types_table
        node_type_filter = set(node_types_table.node_type_ids)  # list of valid node_type_ids
        type_filter = False
        group_prop_filter = {}  # list of 'prop_name'==prov_val for group datasets
        group_filter = False
        node_id_filter = []

        # Build key==value lists
        for filter_key, filter_val in filter_props.items():
            # TODO: Check if node_type_id is an input
            if filter_key in self._group_columns:
                # keep of list of group_popertiess to filter
                group_prop_filter[filter_key] = filter_val
                group_filter = True

            elif filter_key in node_types_table.columns:
                # for node_types we just keep a list of all node_type_ids with matching key==value pairs
                node_type_filter &= set(node_types_table.find(filter_key, filter_val))
                type_filter = True

            elif filter_key in ['node_id', 'node_ids']:
                node_id_filter += filter_val

            else:
                # TODO: should we raise an exception?
                # TODO: User logger
                print('Could not find property {} in either group or types table. Ignoring.'.format(filter_key))

        # iterate through all nodes, skipping ones that don't have matching key==value pairs
        for indx in self._parent_indicies:
            # TODO: Don't build the node until you filter out node_type_id
            node = self._parent.get_row(indx)
            if type_filter and node.node_type_id not in node_type_filter:
                # confirm node_type_id is a correct one
                continue

            if node_id_filter and node.node_id not in node_id_filter:
                continue

            if group_filter:
                # Filter by group property values
                # TODO: Allow group properties to handle lists
                src_failed = True
                for k, v in group_prop_filter.items():
                    if node[k] != v:
                        break
                else:
                    src_failed = False

                if src_failed:
                    continue

            yield node

    def __iter__(self):
        self.build_indicies()
        # Pass a list of indicies into the NodeSet, the NodeSet will take care of the iteration
        return NodeSet(self._parent_indicies, self._parent).__iter__()


class EdgeGroup(Group):
    def __init__(self, group_id, h5_group, parent):
        super(EdgeGroup, self).__init__(group_id, h5_group, parent)
        self._indicies_count = 0  # Used to keep track of number of indicies (since it contains multple ranges)

        self.__itr_index = 0
        self.__itr_range = []
        self.__itr_range_idx = 0
        self.__itr_range_max = 0

    def build_indicies(self, force=False):
        if self._parent_indicies_built and not force:
            return

        # Saves indicies as a (potentially empty) list of ranges
        # TODO: Turn index into generator, allows for cheaper iteration over the group
        self._indicies_count, self._parent_indicies = self._parent.group_indicies(self.group_id, build_cache=False)
        self._parent_indicies_built = True

    def to_dataframe(self):
        self.build_indicies()

        # Build a dataframe of group properties
        # TODO: Include dynamics_params?
        properties_df = pd.DataFrame()
        for col in self._group_columns:
            if col.dimension > 1:
                for i in range(col.dimension):
                    # TODO: see if column name exists in the attributes
                    col_name = '{}.{}'.format(col.name, i)
                    properties_df[col_name] = pd.Series(self._h5_group[col.name][:, i])
            else:
                properties_df[col.name] = pd.Series(self._h5_group[col.name])

        # Build a dataframe of parent node
        root_df = pd.DataFrame()
        root_df['edge_type_id'] = pd.Series(self.edge_type_ids)
        root_df['source_node_id'] = pd.Series(self.src_node_ids)
        root_df['target_node_id'] = pd.Series(self.trg_node_ids)
        root_df['edge_group_index'] = pd.Series(self._parent.group_indicies(self.group_id, as_list=True))  # pivot col

        # merge group props df with parent df
        results_df = root_df.merge(properties_df, how='left', left_on='edge_group_index', right_index=True)
        results_df = results_df.drop('edge_group_index', axis=1)

        # Build node_types dataframe and merge
        edge_types_df = self._parent.edge_types_table.to_dataframe()
        # remove properties that exist in the group
        edge_types_cols = [c.name for c in self._parent.edge_types_table.columns if c not in self._group_columns]
        edge_types_df = edge_types_df[edge_types_cols]

        # TODO: consider caching these results
        return results_df.merge(edge_types_df, how='left', left_on='edge_type_id', right_index=True)

    def _get_parent_ds(self, parent_ds):
        self.build_indicies()
        ds_vals = np.zeros(self._indicies_count, dtype=parent_ds.dtype)
        c_indx = 0
        for indx_range in self._parent_indicies:
            indx_beg, indx_end = indx_range[0], indx_range[1]
            n_indx = c_indx + (indx_end - indx_beg)
            ds_vals[c_indx:n_indx] = parent_ds[indx_beg:indx_end]
            c_indx = n_indx

        return ds_vals

    @property
    def src_node_ids(self):
        return self._get_parent_ds(self.parent._source_node_id_ds)

    @property
    def trg_node_ids(self):
        return self._get_parent_ds(self.parent._target_node_id_ds)

    @property
    def edge_type_ids(self):
        return self._get_parent_ds(self.parent._type_id_ds)

    def get_values(self, property_name, all_rows=False):
        # TODO: Need to take into account if property_name is in the edge-types
        if property_name not in self.columns:
            raise KeyError

        if all_rows:
            return np.array(self._h5_group[property_name])
        else:
            self.build_indicies()
            # Go through all ranges and build the return list
            dataset = self._h5_group[property_name]
            return_list = np.empty(self._indicies_count, self._h5_group[property_name].dtype)
            i = 0
            for r_beg, r_end in self._parent_indicies:
                r_len = r_end - r_beg
                return_list[i:(i+r_len)] = dataset[r_beg:r_end]
                i += r_len
            return return_list

    def filter(self, **filter_props):
        # TODO: I'm not sure If I want to do this? Need to check on a larger dataset than I currently have.
        raise NotImplementedError

    def __iter__(self):
        self.build_indicies()
        # TODO: Implement using an EdgeSet
        if len(self._parent_indicies) == 0:
            self.__itr_max_range = 0
            self.__itr_range = []
            self.__itr_index = 0
        else:
            # Stop at the largest range end (I'm not sure if the indicies are ordered, if we can make it ordered then
            # in the future just use self_parent_indicies[-1][1]
            self.__itr_range_max = len(self._parent_indicies)
            self.__itr_range_idx = 0
            self.__itr_range = self._parent_indicies[0]
            self.__itr_index = self.__itr_range[0]

        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.__itr_range_idx >= self.__itr_range_max:
            raise StopIteration

        nxt_edge = self._parent.get_row(self.__itr_index)
        self.__itr_index += 1
        if self.__itr_index >= self.__itr_range[1]:
            # iterator has moved past the current range
            self.__itr_range_idx += 1
            if self.__itr_range_idx < self.__itr_range_max:
                # move the iterator onto next range
                self.__itr_range = self._parent_indicies[self.__itr_range_idx]  # update range
                self.__itr_index = self.__itr_range[0]  # update iterator to start and the beginning of new range
            else:
                self.__itr_range = []

        return nxt_edge
