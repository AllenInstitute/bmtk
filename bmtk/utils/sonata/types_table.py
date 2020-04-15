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
import numbers
import math

from .column_property import ColumnProperty


def remove_nans(types_dict):
    """Convert nan values to None in type row (dict)"""
    for k, v in types_dict.items():
        if isinstance(v, numbers.Real) and math.isnan(v):
            types_dict[k] = None


class TypesTable(object):
    def __init__(self, parent=None):
        self._parent = None  # Used to keep track of FileRoot object this table belongs to
        self._columns = {}
        self._index_typeid2df = {}  # map from node(edge)_type_id --> csv Row
        self._column_map = {}  # TODO: Use defaultdict
        # self._id_table = self.IDSearcher(self)
        self._dataframes = []  # list of all pandas dataframe (types tables)

        self._cached_node_types = {}
        self._df_cache = None

        self._itr_indx = 0
        self._itr_end = 0

    @property
    def index_column_name(self):
        raise NotImplementedError

    @property
    def type_ids(self):
        return list(self._index_typeid2df.keys())

    @property
    def columns(self):
        return list(self._columns.values())

    def column(self, column_name):
        return self._columns[column_name]

    def add_table(self, nt_df):
        # TODO: Just saving the entire dataframe currently because we don't expect the node-types table to get too large
        # (few hundred rows at the most). If that changes consider to loading the csv until explicity called by user.
        self._dataframes.append(nt_df)

        # Check that the type ids are unique and build id --> dataframe map
        nt_df.set_index(keys=self.index_column_name, inplace=True)
        for type_id in list(nt_df.index):
            if type_id in self._index_typeid2df:
                raise Exception('Multiple {}s with value {}.'.format(self.index_column_name, type_id))
            self._index_typeid2df[type_id] = nt_df

        columns = ColumnProperty.from_csv(nt_df)
        for col in columns:
            self._columns[col.name] = col
            if col in self._column_map:
                # TODO: make sure dtype matches. Bad things can happen if the same col has heterogeneous dtypes
                self._column_map[col.name].append(nt_df)
            else:
                self._column_map[col.name] = [nt_df]

    def find(self, column_key, column_val, silent=False):
        """Returns a list of type_ids that contain column property column_key==column_val

        :param column_key: Name of column to search
        :param column_val: Value of column to select for
        :param silent: Set to true to prevent KeyError if column_key doesn't exist (default=False)
        :return: A (potentially empty) list of type_ids
        """
        if not silent and column_key not in self.columns:
            raise KeyError

        is_list = isinstance(column_val, list)
        selected_ids = []  # running list of valid type-ids
        column_dtype = self.column(column_key).dtype
        for df in self._column_map[column_key]:
            # if a csv column has all NONE values, pandas will load the values as float(NaN)'s. Thus for str/object
            # columns we need to check dtype otherwise we'll get an invalid comparisson.
            if df[column_key].dtype == column_dtype:
                if is_list:
                    indicies = df[df[column_key].isin(column_val)].index
                else:
                    indicies = df[df[column_key] == column_val].index

                if len(indicies) > 0:
                    selected_ids.extend(list(indicies))

        return selected_ids

    def to_dataframe(self, cache=False):
        if self._df_cache is not None:
            return self._df_cache

        if len(self._dataframes) == 0:
            return None
        elif len(self._dataframes) == 1:
            merged_table = self._dataframes[0]
        else:
            # merge all dataframes together
            merged_table = self._dataframes[0].reset_index()  # TODO: just merge on the indicies rather than reset
            for df in self._dataframes[1:]:
                try:
                    merged_table = merged_table.merge(df.reset_index(), how='outer')
                except ValueError as ve:
                    # There is a potential issue if merging where one dtype is different from another (ex, if all
                    # model_template's are NONE pandas will load column as float64). First solution is to find columns
                    # that differ and upcast columns as object's (TODO: look for better solution)
                    right_df = df.reset_index()
                    for col in set(merged_table.columns) & set(right_df.columns):
                        # find all shared columns whose dtype differs
                        if merged_table[col].dtype != right_df[col].dtype:
                            # change column(s) dtype to object
                            merged_table[col] = merged_table[col] if merged_table[col].dtype == object \
                                else merged_table[col].astype(object)
                            right_df[col] = right_df[col] if right_df[col].dtype == object \
                                else right_df[col].astype(object)

                    merged_table = merged_table.merge(right_df, how='outer')

            merged_table.set_index(self.index_column_name, inplace=True)

        if cache:
            self._df_cache = merged_table

        return merged_table

    def __iter__(self):
        self._itr_indx = 0
        self._itr_end = len(self.type_ids)
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        if self._itr_indx >= self._itr_end:
            raise StopIteration

        ntid = self.type_ids[self._itr_indx]
        self._itr_indx += 1
        return self[ntid]

    def __getitem__(self, type_id):
        if isinstance(type_id, tuple):
            return [self[ntid] for ntid in type_id]

        elif isinstance(type_id, numbers.Integral):
            if type_id not in self._index_typeid2df:
                raise Exception('{} {} not found'.format(self.index_column_name, type_id))

            if type_id in self._cached_node_types:
                return self._cached_node_types[type_id]
            else:
                nt_dict = self._index_typeid2df[type_id].loc[type_id].to_dict()
                # TODO: consider just removing key from dict if value is None/NaN
                remove_nans(nt_dict)  # pd turns None into np.nan's. Temp soln is to just convert them back.
                self._cached_node_types[type_id] = nt_dict
                self._cached_node_types[type_id][self.index_column_name] = type_id  # include node/edge_type_id
                return nt_dict
        else:
            raise Exception('Unsupported search on node-type-id')

    def __contains__(self, type_id):
        return type_id in self._index_typeid2df

    def __repr__(self):
        return repr(self.to_dataframe())


class NodeTypesTable(TypesTable):
    def __init__(self, parent=None):
        super(NodeTypesTable, self).__init__(parent)

    @property
    def index_column_name(self):
        return 'node_type_id'

    @property
    def node_type_ids(self):
        return self.type_ids


class EdgeTypesTable(TypesTable):
    def __init__(self, parent=None):
        super(EdgeTypesTable, self).__init__(parent)

    @property
    def index_column_name(self):
        return 'edge_type_id'

    @property
    def edge_type_ids(self):
        return self.type_ids
