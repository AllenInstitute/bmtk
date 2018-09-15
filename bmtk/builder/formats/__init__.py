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
""" network2.format

The XFormat classes are implemented within Network class to allow network objects to handle different data types.
Each class should be able to control both input and output file format (json, csv, h5, etc) and the expected parameters,
including their corresponding order.

Example:
    net = Network(format=ISeeFormat)
    ...
    net.save(cells="cells.csv", models="cell_models.csv", connections="connections.h5")

Todo:
    * change network.load(cls) to be format specific.
"""
import csv
import h5py
import numpy as np
import json
import pandas as pd

from ..node import Node

from iformats import IFormat


class DefaultFormat(IFormat):
    def save_nodes(self, file_name):
        raise NotImplementedError()

    def save_edges(self, file_name):
        raise NotImplementedError()

    def save(self, file_name):
        raise NotImplementedError()


class ISeeFormat(IFormat):
    """Controls the output of networks that will be used in the isee_engine simulator.

    The nodes are saved in a cells and cell_model csv files with predefined format. the edges/connections are
        saved in a connections h5 format.
    """
    def save_cells(self, filename, columns, position_labels=None):
        """Saves nodes/cell information and their model type metadata.

        :param cells_csv: name of csv file where cell information will be saved.
        :param models_csv: name of csv file where cell model information will be saved.
        """
        # TODO: add checks and warnings if parameters are missing.
        with open(filename, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=' ')
            header = []
            for col in columns:
                if col == 'position':
                    for label in position_labels:
                        if label:
                            header.append(label)
                else:
                    header.append(col)
            csvw.writerow(header)
            for nid, params in self._network.nodes():
                row_array = []
                for col in columns:
                    if col == 'position':
                        for i, label in enumerate(position_labels):
                            if label:
                                row_array.append(params['position'][i])
                    else:
                        row_array.append(params[col])

                csvw.writerow(row_array)

    def save_types(self, filename, columns, key=None):
        seen_types = set()

        with open(filename, 'w') as csvfile:
            csvw = csv.writer(csvfile, delimiter=' ')
            csvw.writerow(columns)
            #csvw.writerow(['model_id', 'electrophysiology' 'level_of_detail', 'morphology', 'rotation_angle_zaxis'])
            for node_set in self._network._node_sets:
                props = node_set.properties#['properties']

                if key is not None:
                    key_val = props.get(key, None)
                    if key_val is not None and key_val in seen_types:
                        continue
                    else:
                        seen_types.add(key_val)

                row_array = []
                for col in columns:
                    row_array.append(props.get(col, 'NA'))
                csvw.writerow(row_array)

    def save_edges(self, filename, include_nsyns=True):
        """Saves connection information into h5 format

        :param filename: Name of h5 file where connection information will be stored.
        :param include_nsyns: setting to false will omit the nsyns table in the h5 file, default
            true (nsyn table included).
        """
        print("save_edges")

        n_nodes = self._network.nnodes
        n_edges = self._network.nedges

        # TODO: check the order of the node list

        print("> building tables with %d nodes and %d edges" % (self._network.nnodes, self._network.nedges))
        indptr_table = [0]
        nsyns_table = []
        src_gids_table = []
        edge_types_table = []
        for trg in self._network.nodes():
            tid = trg[1]['id']
            for edges in self._network.edges([tid], rank=1):
                src_gids_table.append(edges[0])
                nsyns_table.append(edges[2])
                edge_types_table.append(edges[3])

            #if len(src_gids_table) == indptr_table[-1]:
            #    print "node %d doesn't have any edges" % (tid)
            indptr_table.append(len(src_gids_table))


        print("> saving tables to %s" % (filename))

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('edge_ptr', data=indptr_table)
            if include_nsyns:
                hf.create_dataset('num_syns', data=nsyns_table)
            hf.create_dataset('src_gids', data=src_gids_table)
            hf.create_dataset('edge_types', data=edge_types_table)
            hf.attrs["shape"] = (n_nodes, n_nodes)


        """
        temp = np.empty([n_edges, 3])
        for i, edge in enumerate(self._network.edges()):
            temp[i, 0] = edge[0]
            temp[i, 1] = edge[1]
            temp[i, 2] = edge[2]

        src_gids_new = np.array([])
        nsyns_new = np.array([])
        indptr_new = []
        counter = 0
        indptr_new.append(counter)
        print "Building database"
        for i in range(n_nodes):
            indicies = np.where(temp[:, 1] == i)

            src_gids_new = np.concatenate([src_gids_new, np.array(temp[indicies[0], 0])])
            nsyns_new = np.concatenate([nsyns_new, np.array(temp[indicies[0], 2])])

            counter += np.size(indicies[0])
            indptr_new.append(counter)

        print "Writing to h5"

        indptr_new = np.array(indptr_new)

        src_gids_new = src_gids_new.astype(int)
        print src_gids_new
        exit()

        nsyns_new = nsyns_new.astype(int)
        indptr_new = indptr_new.astype(int)

        with h5py.File(filename, 'w') as hf:
            hf.create_dataset('indptr', data=indptr_new)
            if include_nsyns:
                hf.create_dataset('nsyns', data=nsyns_new)
            hf.create_dataset('src_gids', data=src_gids_new)
            hf.attrs["shape"] = (n_nodes, n_nodes)
        """

    def save(self, cells_fname, cell_models_fname, connections_fname, include_nsyns=True):
        """Saves node (cells) and connection information to files.

        :param cells_fname: name of csv file where cell information will be saved.
        :param cell_models_fname: name of csv file where cell model information will be saved.
        :param connections_fname: Name of h5 file where connection information will be stored.
        :param include_nsyns: set to False to build h5 without nsyn table.
        """
        #self.save_nodes(cells_fname, cell_models_fname)
        self.save_edges(connections_fname, include_nsyns)

    def load(self, nodes, edge_types=None, node_types=None, edges=None, positions=None):
        # TODO: check imported ids

        df = pd.read_csv(nodes, sep=' ')
        if node_types is not None:
            types_df = pd.read_csv(node_types, sep=' ', index_col='node_type_id')
            df = pd.merge(left=df, right=types_df, how='left', left_on='node_type_id', right_index=True)

        gids_df = df['node_id'] if 'node_id' in df.columns else df['id']
        #df = df.drop(['id'], axis=1)

        positions_df = None
        if positions:
            positions_df = df[positions]
            df = df.drop(positions, axis=1)

        node_params = df.to_dict(orient='records')
        node_tuples = [Node(gids_df[i], gids_df[i], None, array_params=node_params[i])
                       for i in xrange(df.shape[0])]


        if positions:
            self._network.positions = position_set.PositionSet()
            posr = positioner.create('points', location=positions_df.as_matrix())
            #self._network.positions.add(posr(df.shape[0]), gids_df.tolist())
            self._network.positions.add(positions_df.values, gids_df.tolist())

            for i in xrange(df.shape[0]):
                node_tuples[i]['position'] = np.array(positions_df.loc[i])

            self._network.positions.finalize()

        self._network._initialize()
        self._network._add_nodes(node_tuples)
        self._network.nodes_built = True

