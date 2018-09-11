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
from . import utils
from .file_root import NodesRoot, EdgesRoot


class File(object):
    def __init__(self, data_files, data_type_files, mode='r', gid_table=None, require_magic=True):
        if mode != 'r':
            raise Exception('Currently only read mode is supported.')

        self._data_files = utils.listify(data_files)
        self._data_type_files = utils.listify(data_type_files)

        # Open and check HDF5 file(s)
        self._h5_file_handles = [utils.load_h5(f, mode) for f in self._data_files]
        if require_magic:
            map(utils.check_magic, self._h5_file_handles)  # Check magic attribute in h5 files

        # Check version number
        avail_versions = set(map(utils.get_version, self._h5_file_handles))
        if len(avail_versions) == 1:
            self._version = list(avail_versions)[0]
        elif len(avail_versions) > 1:
            # TODO: log as warning
            print('Warning: Passing in multiple hdf5 files of different version')
            self._version = ','.join(avail_versions)
        else:
            self._version = utils.VERSION_NA

        self._csv_file_handles = [(f, utils.load_csv(f)) for f in self._data_type_files]

        self._has_nodes = False
        self._nodes = None  # /nodes object
        self._nodes_groups = []  # list of all hdf5 /nodes group
        self._node_types_dataframes = []  # list of all csv node-types dataframe

        self._has_edges = False
        self._edges = None  # /edges object
        self._edges_groups = []  # list of all hdf5 /edges group
        self._edge_types_dataframes = []  # list of csv edge-types dataframes

        # for multiple inputs sort into edge files and node files
        self._sort_types_file()
        self._sort_h5_files()

        if not (self._has_nodes or self._has_edges):
            raise Exception('Could not find neither nodes nor edges for the given file(s).')

        if self._has_nodes:
            self._nodes = NodesRoot(nodes=self._nodes_groups, node_types=self._node_types_dataframes, gid_table=gid_table)

        if self._has_edges:
            self._edges = EdgesRoot(edges=self._edges_groups, edge_types=self._edge_types_dataframes)

    @property
    def nodes(self):
        return self._nodes

    @property
    def has_nodes(self):
        return self._has_nodes

    @property
    def edges(self):
        return self._edges

    @property
    def has_edges(self):
        return self._has_edges

    @property
    def version(self):
        return self._version

    def _sort_types_file(self):
        # TODO: node/edge type_id columnn names should not be hardcoded
        for filename, df in self._csv_file_handles:
            has_node_type_id = 'node_type_id' in df.columns
            has_edge_type_id = 'edge_type_id' in df.columns
            if has_node_type_id and has_edge_type_id:
                # TODO: users may be creating their own dataframe and thus not have a filename
                raise Exception('types file {} has both node_types_id and edge_types_id column.'.format(filename))
            elif has_node_type_id:
                self._node_types_dataframes.append(df)
            elif has_edge_type_id:
                self._edge_types_dataframes.append(df)
            else:
                # TODO: if strict this should fail immedietely
                print('Warning: Could not determine if file {} was an edge-types or node-types file. Ignoring'.format(filename))

    def _sort_h5_files(self):
        for h5 in self._h5_file_handles:
            has_nodes = '/nodes' in h5
            has_edges = '/edges' in h5
            if not (has_nodes or has_edges):
                print('File {} contains neither nodes nor edges. Ignoring'.format(h5.filename))
            else:
                if has_nodes:
                    self._nodes_groups.append(h5)
                    self._has_nodes = True
                if has_edges:
                    self._edges_groups.append(h5)
                    self._has_edges = True
