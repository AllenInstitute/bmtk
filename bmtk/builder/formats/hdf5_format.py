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
import os
import csv
import json
import math
import h5py
import pandas as pd
from ast import literal_eval

import bmtk
from .iformats import IFormat
from bmtk.builder.node_pool import NodePool
from time import gmtime, strftime


class HDF5Format(IFormat):
    """
    Format prior to Blue-brain project collaboration.
    Saves as:
    nodes (csv)
    node_types (csv)
    edge_types (csv)
    edges (h5)
    """

    CSV_DELIMITER = ' '
    COL_NODE_TYPE_ID = 'node_type_id'
    COL_EDGE_TYPE_ID = 'edge_type_id'
    COL_TARGET_QUERY = 'target_query'
    COL_SOURCE_QUERY = 'source_query'
    COL_NODE_ID = 'node_id'
    BASE_DIR = 'network'

    @property
    def format(self):
        return 'msdk.HDF5Format'

    def save(self, directory, **kwargs):
        """ saves nodes.csv, node_types.csv, edges.h5, edge_types.csv and .metadata.json. Will overwrite existing files.

        :param directory: Directory where all the files will be saved, creating dir if it doesn't exists.
        :param kwargs:
        """
        if directory is None:
            base_path = os.path.join(self.BASE_DIR, self._network.name)
        else:
            base_path = directory

        metadata = {
            'version': bmtk.__version__,
            'name': self._network.name,
            'date_created': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            'file_format': self.format,
            'network_class': self._network.__class__.__name__
        }

        # save node-types.
        node_types_path = os.path.join(base_path, 'node_types.csv')
        self.save_node_types(node_types_path, **kwargs)
        metadata['node_types_file'] = 'node_types.csv'

        # save individual nodes.
        if self._network.nodes_built:
            # make sure nodes have been built
            nodes_path = os.path.join(base_path, 'nodes.csv')
            self.save_nodes(nodes_path, **kwargs)
            metadata['nodes_file'] = 'nodes.csv'
        else:
            print('Nodes not built. Unable to save to nodes.csv.')

        # save edge-types.
        edge_types_path = os.path.join(base_path, 'edge_types.csv')
        self.save_edge_types(edge_types_path, **kwargs)
        metadata['edge_types_file'] = 'edge_types.csv'

        # save edges if they have been built
        if self._network.edges_built:
            edges_path = os.path.join(base_path, 'edges.h5')
            self.save_edges(edges_path, **kwargs)
            metadata['edges_file'] = 'edges.h5'
        else:
            print('Edges not built. Unable to save to edges.h5.')

        # save the metadata file
        metadata_path = os.path.join(base_path, '.metadata.json')
        with open(metadata_path, 'w') as mdfile:
            json.dump(metadata, mdfile, indent=2)

    def save_node_types(self, file_name, columns=None, **kwargs):
        """Write node_types to csv.

        :param file_name: path to csv file. Will be overwritten if it exists
        :param columns: optional columns (not incl. manditory ones). If None then will use all node properties.
        :param kwargs: optional
        """
        self.__checkpath(file_name, **kwargs)

        # csv should always start with node_type_id
        manditory_cols = [self.COL_NODE_TYPE_ID]

        # Determine which columns are in the node_types file and their order
        nt_properties = self._network.node_type_properties
        opt_cols = []
        if columns is None:
            # use all node type properties
            opt_cols = list(nt_properties)
        else:
            # check that columns specified by user exists
            for col_name in columns:
                if col_name not in nt_properties:
                    raise Exception('No node property {} found in network, cannot save {}.'.format(col_name, file_name))
                else:
                    opt_cols.append(col_name)

        # write to csv iteratively
        cols = manditory_cols + opt_cols
        with open(file_name, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=self.CSV_DELIMITER)
            csvw.writerow(cols)
            for node_set in self._network._node_sets:
                props = node_set.properties
                row = []
                for cname in cols:
                    # TODO: determine dtype of parameters so we can use the appropiate none value
                    row.append(props.get(cname, 'NA')) # get column name or NA if it doesn't exists for this node
                csvw.writerow(row)

    def save_nodes(self, file_name, columns=None, **kwargs):
        """Write nodes to csv.

        :param file_name: path to csv file. Will be overwritten if it exists
        :param columns: optional columns (not incl. manditory ones). If None then will use all node properties.
        :param kwargs: optional
        """
        self.__checkpath(file_name, **kwargs)

        # csv will start with node_id and node_type_id
        manditory_columns = [self.COL_NODE_ID, self.COL_NODE_TYPE_ID]

        # optional columns from either node params or node-type properties
        opt_columns = []
        if columns is None:
            opt_columns = list(self._network.node_params)
        else:
            all_cols = self._network.node_params | self._network.node_type_properties
            for col_name in columns:
                if col_name not in all_cols:
                    # verify params/properties exist
                    raise Exception('No edge property {} found in network, cannot save {}.'.format(col_name, file_name))
                else:
                    opt_columns.append(col_name)

        # write to csv
        with open(file_name, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=self.CSV_DELIMITER)
            csvw.writerow(manditory_columns + opt_columns)
            for nid, node in self._network.nodes():
                row = [node.node_id, node.node_type_id]
                for cname in opt_columns:
                    row.append(node.get(cname, 'NA'))
                csvw.writerow(row)

    def save_edge_types(self, file_name, columns=None, **kwargs):
        """Write edge-types to csv.

        :param file_name: path to csv file. Will be overwritten if it exists
        :param columns: optional columns (not incl. manditory ones). If None then will use all node properties.
        :param kwargs: optional
        """
        self.__checkpath(file_name, **kwargs)

        # start with edge_type_id, target_query and source_query
        manditory_cols = [self.COL_EDGE_TYPE_ID, self.COL_TARGET_QUERY, self.COL_SOURCE_QUERY]

        # optional columns
        edge_props = self._network.edge_type_properties
        opt_cols = []
        if columns is None:
            opt_cols = list(edge_props)
        else:
            for col_name in columns:
                if col_name not in edge_props:
                    raise Exception('No edge property {} found in network, cannot save {}.'.format(col_name, file_name))
                else:
                    opt_cols.append(col_name)

        # write to csv by iteratively going through all edge-types
        with open(file_name, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=self.CSV_DELIMITER)
            csvw.writerow(manditory_cols + opt_cols)
            for et in self._network._edge_sets:
                edge = et['edge']
                targetnodes = edge.targets  # get source as NodePools to get the source_query strings
                sourcenodes = edge.sources  # same with target
                row_array = [edge.id, targetnodes.filter_str, sourcenodes.filter_str]
                edge_params = edge.parameters
                for col in opt_cols:
                    row_array.append(edge_params.get(col, 'NA'))
                csvw.writerow(row_array)

    def save_edges(self, file_name, **kwargs):
        """Saves edges to edges.h5

        :param file_name: path to hdf5 file. Will be overwritten if it exists
        :param kwargs: optional
        """
        self.__checkpath(file_name, **kwargs)

        # Get sources, targets, nsyns and edge_type_id for all edges.
        print("> building tables with %d nodes and %d edges" % (self._network.nnodes, self._network.nedges))
        indptr_table = [0]
        nsyns_table = []
        src_gids_table = []
        edge_types_table = []
        for trg in self._network.nodes():
            # the targets have to be ordered.
            tid = trg[1].node_id
            for edges in self._network.edges([tid], rank=1):
                src_gids_table.append(edges[0])
                nsyns_table.append(edges[2])
                edge_types_table.append(edges[3])

            indptr_table.append(len(src_gids_table))

        # save to h5
        print("> saving tables to %s" % (file_name))
        with h5py.File(file_name, 'w') as hf:
            hf.create_dataset('edge_ptr', data=indptr_table)
            hf.create_dataset('num_syns', data=nsyns_table)
            hf.create_dataset('src_gids', data=src_gids_table)
            hf.create_dataset('edge_types', data=edge_types_table)

    def __checkpath(self, file_name, **kwargs):
        """Makes sure file_name is a valid file path and can be written."""
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            # create file's directory if it doesn't exist
            os.makedirs(dir_path)

    def __load_nodes(self, nodes_file, node_types_file):
        """Loads nodes and node_types from exists files

        :param nodes_file: path to nodes csv
        :param node_types_file: path to node_types csv
        """
        def eval(val):
            # Helper function that can convert csv to an appropiate type. Helpful for cells of lists (positions, etc)
            # TODO: keep column dtypes in metadata and use that for converting each column
            if isinstance(val, float) and math.isnan(val):
                return None
            elif isinstance(val, basestring):
                try:
                    # this will be helpful for turning strings into lists where appropiate "(0, 1, 2)" --> (0, 1, 2)
                    return literal_eval(val)
                except ValueError:
                    return val
            return val

        if nodes_file is None and node_types_file is None:
            return None

        elif nodes_file is not None and node_types_file is not None:
            # Get the array_params from nodes_file and properties from nodes_types_file, combine them to call
            # add_nodes() function and rebuilt the nodes.
            nt_df = pd.read_csv(node_types_file, self.CSV_DELIMITER)  #, index_col=self.COL_NODE_TYPE_ID)
            n_df = pd.read_csv(nodes_file, self.CSV_DELIMITER)

            for _, row in nt_df.iterrows():
                # iterate through the node_types, find all nodes with matching node_type_id and get those node's
                # parameters as a dictionary of lists
                node_type_props = {l: eval(row[l]) for l in nt_df.columns if eval(row[l]) is not None}
                selected_nodes = n_df[n_df[self.COL_NODE_TYPE_ID] == row[self.COL_NODE_TYPE_ID]]
                N = len(selected_nodes.axes[0])
                array_params = {l: list(selected_nodes[l]) for l in selected_nodes.columns
                                if l not in ['node_type_id', 'position']}

                # Special function for position_params
                position = None
                position_params = None
                if 'position' in selected_nodes.columns:
                    position_params = {'location': [eval(p) for p in selected_nodes['position']]}
                    position = 'points'

                self._network.add_nodes(N, position=position, position_params=position_params,
                                        array_params=array_params, **node_type_props)

            self._network._build_nodes()

        elif node_types_file is not None:
            # nodes_types exists but nodes doesn't. We convert each row (node_type) in the csv to a collection
            # of nodes with N=1, no array_params.
            nt_df = pd.read_csv(node_types_file, self.CSV_DELIMITER)
            for _, row in nt_df.iterrows():
                node_type_props = {l: eval(row[l]) for l in nt_df.columns if eval(row[l]) is not None}
                self._network.add_nodes(N=1, **node_type_props)
            self._network._build_nodes()

        elif nodes_file is not None:
            # nodes exists but node_types doesn't. In this case group together all nodes by node_type_id and add them
            # as a single population (with no node_params)
            n_df = pd.read_csv(nodes_file, self.CSV_DELIMITER)
            for nt_id, df in n_df.groupby(self.COL_NODE_TYPE_ID):
                N = len(df.axes[0])
                array_params = {l: list(df[l]) for l in df.columns
                                if l not in ['node_type_id', 'position']}

                position = None
                position_params = None
                if 'position' in df.columns:
                    position_params = {'location': [eval(p) for p in df['position']]}
                    position = 'points'

                self._network.add_nodes(N, position=position, position_params=position_params,
                                        array_params=array_params, node_type_id=nt_id)
            self._network._build_nodes()

    def __load_edge_types(self, edges_file, edge_types_file):
        """Loads edges and edge_types

        :param edges_file: path to edges hdf5
        :param edge_types_file: path to edge_types csv
        """
        if edge_types_file is None and edges_file is None:
            return

        if edge_types_file is not None:
            # load in the edge-types. iterate through all the rows of edge_types.csv and call connect() function.
            et_pd = pd.read_csv(edge_types_file, self.CSV_DELIMITER)
            prop_cols = [label for label in et_pd.columns
                         if label not in [self.COL_SOURCE_QUERY, self.COL_TARGET_QUERY]]

            for _, row in et_pd.iterrows():
                # the connect function requires a Pool of nodes (like net.nodes()) or a dictionary filter.
                source_nodes = NodePool.from_filter(self._network, row[self.COL_SOURCE_QUERY])
                target_nodes = NodePool.from_filter(self._network, row[self.COL_TARGET_QUERY])
                # TODO: evaluate edge-properties and exclude any that are None.
                edge_params = {label: row[label] for label in prop_cols}

                # don't try to guess connection rule
                self._network.connect(source=source_nodes, target=target_nodes, edge_params=edge_params)

        if edges_file is not None:
            # Create edges from h5.
            if not self._network.nodes_built:
                print('The nodes have not been built. Cannot load edges file.')
                return

            # load h5 tables
            edges_h5 = h5py.File(edges_file, 'r')
            edge_types_ds = edges_h5['edge_types']
            num_syns_ds = edges_h5['num_syns']
            src_gids_ds = edges_h5['src_gids']
            edge_ptr_ds = edges_h5['edge_ptr']
            n_edge_ptr = len(edge_ptr_ds)

            # the network needs edge-types objects while building the edges. If the edge_types_file exists then they
            # would have been added in the previous section of code. If edge_types_file is missing we will create
            # filler edge types based on the edge_type_id's found in edge_ptr dataset
            if edge_types_file is None:
                for et_id in set(edges_h5['edge_types'][:]):
                    self._network.connect(edge_params={self.COL_NODE_TYPE_ID: et_id})

            # TODO: if edge_types.csv does exists we should check it has matching edge_type_ids with edges.h5/edge_ptr

            def itr_fnc(et):
                # Creates a generator that will iteratively go through h5 file and return (source_gid, target_gid,
                # nsyn) values for connections with matching edge_type.edge_type_id
                edge_type_id = et.id
                for ep_indx in xrange(n_edge_ptr - 1):
                    trg_gid = ep_indx
                    for syn_indx in xrange(edge_ptr_ds[ep_indx], edge_ptr_ds[ep_indx + 1]):
                        if edge_types_ds[syn_indx] == edge_type_id:
                            src_gid = src_gids_ds[syn_indx]
                            n_syn = num_syns_ds[syn_indx]
                            yield (src_gid, trg_gid, n_syn)

            for edge in self._network.edge_types():
                # create iterator and directly add edges
                itr = itr_fnc(edge)
                self._network._add_edges(edge, itr)

            self.edges_built = True

    def load_dir(self, directory, metadata):
        def get_path(f):
            if f not in metadata:
                return None
            file_name = metadata[f]
            if directory is None or os.path.isabs(file_name):
                return file
            return os.path.join(directory, file_name)

        nodes_file = get_path('nodes_file')
        node_types_file = get_path('node_types_file')
        self.__load_nodes(nodes_file, node_types_file)

        edge_types_file = get_path('edge_types_file')
        edges_file = get_path('edges_file')
        self.__load_edge_types(edges_file, edge_types_file)

    def load(self, nodes_file=None, node_types_file=None, edges_file=None, edge_types_file=None):
        self.__load_nodes(nodes_file, node_types_file)
