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
import pandas as pd
import h5py


"""
An interface for reading network files.

We are continuing to develop network file format this interface is a way to provide backward compatibility. This
namespace should not be instantiated directly, and updates to the network standard should be given their own. The
class TabularNetwork, NodeRow, NodesFile, EdgeRow and EdgesFile are abstract and should be overridden.

In general the developed formats have all take schema:
 * Networks are split between nodes (NodesFile) and edges (EdgesFile)
 * Each type is made up of rows (NodeRow, EdgeRow)
 * Each row has its own set column properties (ColumnProperty), depending on the file/group it belongs too.
 * Each row also has properties from (edge/node)-type metadata.
"""


##########################################
# Interface files
##########################################
class TabularNetwork(object):
    """Factory for loading nodes and edges files."""
    @staticmethod
    def load_nodes(nodes_file, node_types_file):
        raise NotImplementedError()

    @staticmethod
    def load_edges(edges_file, edge_types_files):
        raise NotImplementedError()


class NodeRow(object):
    """Node file row.

    Each row represents node/cell/population in a network and can include edge-type metadata and dynamics_params when
    applicable. The only mandatory for a NodeRow is a unique gid (i.e cell_id, node_id). Properties can be accessed
    like a dictionary.
    """
    def __init__(self, gid, node_props, types_props):
        self._gid = gid
        self._node_props = node_props  # properties from the csv/hdf5 file
        self._node_type_props = types_props  # properties from the edge_types metadata file

    @property
    def gid(self):
        return self._gid

    @property
    def with_dynamics_params(self):
        """Set to true if dynamics_params subgroup attached to HDF5 properities"""
        raise NotImplementedError()

    @property
    def dynamics_params(self):
        raise NotImplementedError()

    @property
    def columns(self):
        return self._node_props.keys() + self._node_type_props.keys()

    @property
    def node_props(self):
        return self._node_props

    @property
    def node_type_props(self):
        return self._node_type_props

    def get(self, prop_key, default=None):
        # First see if property existing in node file, then check node-types
        if prop_key in self._node_props:
            return self._node_props[prop_key]
        elif prop_key in self._node_type_props:
            return self._node_type_props[prop_key]
        else:
            return default

    def __contains__(self, prop_key):
        return prop_key in self._node_props.keys() or prop_key in self._node_type_props.keys()

    def __getitem__(self, prop_key):
        val = self.get(prop_key)
        if val is None:
            raise Exception('Invalid property key {}.'.format(prop_key))
        return val

    def __repr__(self):
        return build_row_repr(self)


class EdgeRow(object):
    """Representation of a edge.

    Edges must include a source and target node gid. Other properties, from the edges or edge-types files, can be
    directly accessed like a dictionary.
    """
    def __init__(self, trg_gid, src_gid, edge_props={}, edge_type_props={}):
        self._trg_gid = trg_gid
        self._src_gid = src_gid
        self._edge_props = edge_props
        self._edge_type_props = edge_type_props

    @property
    def target_gid(self):
        return self._trg_gid

    @property
    def source_gid(self):
        return self._src_gid

    @property
    def with_dynamics_params(self):
        raise NotImplementedError()

    @property
    def dynamics_params(self):
        raise NotImplementedError()

    @property
    def columns(self):
        return self._edge_props.keys() + self._edge_type_props.keys()

    @property
    def edge_props(self):
        return self._edge_props

    def __contains__(self, prop_key):
        return prop_key in self._edge_props.keys() or prop_key in self._edge_type_props.keys()

    def __getitem__(self, prop_key):
        if prop_key in self._edge_props:
            return self._edge_props[prop_key]
        elif prop_key in self._edge_type_props:
            return self._edge_type_props[prop_key]
        else:
            raise Exception('Invalid property name {}.'.format(prop_key))

    def __repr__(self):
        return build_row_repr(self)


class NodesFile(object):
    """Class for reading and iterating properties of each node in a nodes/node-types file.

    Use the load method to load in the necessary node files. Nodes can be accessed using an interator:
      nodes = NodesFile()
      nodes.load(nodes_file.h5, node_types.csv)
      for node in nodes:
         print node['prop']
         ...
    Or indivdually by gid:
       node = nodes[101]
       print node['prop']
    """
    def __init__(self):
        self._network_name = None
        self._version = None
        self._iter_index = 0
        self._nrows = 0
        self._node_types_table = None

    @property
    def name(self):
        """name of network containing these nodes"""
        return self._network_name

    @property
    def version(self):
        return self._version

    @property
    def gids(self):
        raise NotImplementedError()

    @property
    def node_types_table(self):
        return self._node_types_table

    def load(self, nodes_file, node_types_file):
        raise NotImplementedError()

    def get_node(self, gid, cache=False):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __iter__(self):
        self._iter_index = 0
        return self

    def next(self):
        raise NotImplementedError()

    def __getitem__(self, gid):
        return self.get_node(gid)


class EdgesFile(object):
    """Class for reading and iterating over edge files.

    Use the load() method to instantiate from the file. Edges can be accessed for any given edge with a target-gid
    using the edges_itr() method:
      edges = EdgesFile()
      edges.load(edge_file.h5, edge_types.csv)
      for edge_prop in edges.edges_itr(101):
          assert(edge_prop.target_gid == 101)
          source_node = nodes[edge_prop.source_gid]
          print edge_prop['prop_name']
    """
    @property
    def source_network(self):
        """Name of network containing the source gids"""
        raise NotImplementedError()

    @property
    def target_network(self):
        """Name of network containing the target gids"""
        raise NotImplementedError()

    def load(self, edges_file, edge_types_file):
        raise NotImplementedError()

    def edges_itr(self, target_gid):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


##########################################
# Helper functions
##########################################
class ColumnProperty(object):
    """Representation of a column name and metadata from a hdf5 dataset, csv column, etc.

    """
    def __init__(self, name, dtype, dimension, attrs={}):
        self._name = name
        self._dtype = dtype
        self._dim = dimension
        self._attrs = attrs

    @property
    def name(self):
        return self._name

    @property
    def dtype(self):
        return self._dtype

    @property
    def dimension(self):
        return self._dim

    @property
    def attributes(self):
        return self._attrs

    @classmethod
    def from_h5(cls, hf_obj, name=None):
        if isinstance(hf_obj, h5py.Dataset):
            ds_name = name if name is not None else hf_obj.name.split('/')[-1]
            ds_dtype = hf_obj.dtype

            # If the dataset shape is in the form "(N, M)" then the dimension is M. If the shape is just "(N)" then the
            # dimension is just 1
            dim = 1 if len(hf_obj.shape) < 2 else hf_obj.shape[1]
            return cls(ds_name, ds_dtype, dim, attrs=hf_obj.attrs)

        elif isinstance(hf_obj, h5py.Group):
            columns = []
            for name, ds in hf_obj.items():
                if isinstance(ds, h5py.Dataset):
                    columns.append(ColumnProperty.from_h5(ds, name))
            return columns

        else:
            raise Exception('Unable to convert hdf5 object {} to a property or list of properties.'.format(hf_obj))

    @classmethod
    def from_csv(cls, pd_obj, name=None):
        if isinstance(pd_obj, pd.Series):
            c_name = name if name is not None else pd_obj.name
            c_dtype = pd_obj.dtype
            return cls(c_name, c_dtype, 1)

        elif isinstance(pd_obj, pd.DataFrame):
            return [cls(name, pd_obj[name].dtype, 1) for name in pd_obj.columns]

        else:
            raise Exception('Unable to convert pandas object {} to a property or list of properties.'.format(pd_obj))

    def __hash__(self):
        return hash(self._name)

    def __repr__(self):
        return '{} ({})'.format(self.name, self.dtype)


class TypesTable(dict):
    def __init__(self, types_file, index_column, seperator=' ', comment='#'):
        super(TypesTable, self).__init__()

        types_df = pd.read_csv(types_file, sep=seperator, comment=comment)
        self._columns = ColumnProperty.from_csv(types_df)
        for _, row in types_df.iterrows():
            # TODO: iterrows does not preserve dtype and should be replaced with itertuples
            type_id = row[index_column]
            row = {col.name: row[col.name] for col in self._columns}
            self.update({type_id: row})

    @property
    def columns(self):
        return self._columns


def build_row_repr(row):
    columns = row.columns
    if columns > 0:
        rstr = "{"
        for c in columns:
            rstr += "'{}': {}, ".format(c, row[c])
        return rstr[:-2] + "}"
    else:
        return "{}"
