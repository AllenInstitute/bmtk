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
import sys

import h5py
import pandas as pd
import numpy as np

from . import utils
from .population import NodePopulation, EdgePopulation
from .types_table import NodeTypesTable, EdgeTypesTable


class FileRoot(object):
    """Base class for both /nodes and /edges root group in h5 file"""
    def __init__(self, root_name, h5_files, h5_mode, csv_files):
        """
        :param root_name: should either be 'nodes' or 'edges'
        :param h5_files: file (or list of files) containing nodes/edges
        :param h5_mode: currently only supporting 'r' mode in h5py
        :param csv_files: file (or list of files) containing node/edge types
        """
        self._root_name = root_name
        self._h5_handles = [utils.load_h5(f, h5_mode) for f in utils.listify(h5_files)]
        self._csv_handles = [(f, utils.load_csv(f)) for f in utils.listify(csv_files)]

        # merge and create a table of the types table(s)
        self._types_table = None
        self._build_types_table()

        # population_name->h5py.Group table (won't instantiate the population)
        self._populations_groups = {}
        self._store_groups()

        # A map between population_name -> Population object. Population objects aren't created until called, in the
        # case user wants to split populations among MPI nodes (instantiation will create node/edge indicies and other
        # overhead).
        self._populations_cache = {}

        self.check_format()

    @property
    def root_name(self):
        return self._root_name

    @property
    def population_names(self):
        return list(self._populations_groups.keys())

    @property
    def populations(self):
        return [self[name] for name in self.population_names]

    @property
    def types_table(self):
        return self._types_table

    @types_table.setter
    def types_table(self, types_table):
        self._types_table = types_table

    def _build_types_table(self):
        raise NotImplementedError

    def _store_groups(self):
        """Create a map between group population to their h5py.Group handle"""
        for h5handle in self._h5_handles:
            assert(self.root_name in h5handle.keys())
            for pop_name, pop_group in h5handle[self._root_name].items():
                if pop_name in self._populations_groups:
                    raise Exception('Multiple {} populations with name {}.'.format(self._root_name, pop_name))
                self._populations_groups[pop_name] = pop_group

    def _build_population(self, pop_name, pop_group):
        raise NotImplementedError

    def get_population(self, population_name, default=None):
        """Return a population group object based on population's name"""
        if population_name in self:
            return self[population_name]
        else:
            # need this for EdgeRoot.get_populations
            return default

    def check_format(self):
        if len(self._h5_handles) == 0:
            raise Exception('No {} hdf5 files specified.'.format(self.root_name))

        if len(self._csv_handles) == 0:
            raise Exception('No {} types csv files specified.'.format(self.root_name))

    def __contains__(self, population_name):
        # TODO: Add condition if user passes in io.Population object
        return population_name in self.population_names

    def __getitem__(self, population_name):
        if population_name not in self:
            raise Exception('{} does not contain a population with name {}.'.format(self.root_name, population_name))

        if population_name in self._populations_cache:
            return self._populations_cache[population_name]
        else:
            h5_grp = self._populations_groups[population_name]
            pop_obj = self._build_population(population_name, h5_grp)
            self._populations_cache[population_name] = pop_obj
            return pop_obj


class NodesRoot(FileRoot):
    def __init__(self, nodes, node_types, mode='r', gid_table=None):
        super(NodesRoot, self).__init__('nodes', h5_files=nodes, h5_mode=mode, csv_files=node_types)

        # load the gid <--> (node_id, population) map if specified.
        self._gid_table = gid_table
        self._gid_table_groupby = {}
        self._has_gids = False
        # TODO: Should we allow gid-table to be built into '/nodes' h5 groups, or must it always be a separat file?
        if gid_table is not None:
            self.set_gid_table(gid_table)

    @property
    def has_gids(self):
        return self._has_gids

    @property
    def node_types_table(self):
        return self.types_table

    def set_gid_table(self, gid_table, force=False):
        """Adds a map from a gids <--> (node_id, population) based on specification.

        :param gid_table: An h5 file/group containing map specifications
        :param force: Set to true to have it overwrite any exsiting gid table (default False)
        """
        assert(gid_table is not None)
        if self.has_gids and not force:
            raise Exception('gid table already exists (use force=True to overwrite)')

        self._gid_table = utils.load_h5(gid_table, 'r')
        # TODO: validate that the correct columns/dtypes exists.
        gid_df = pd.DataFrame()
        gid_df['gid'] = pd.Series(data=self._gid_table['gid'], dtype=self._gid_table['gid'].dtype)
        gid_df['node_id'] = pd.Series(data=self._gid_table['node_id'], dtype=self._gid_table['node_id'].dtype)
        gid_df['population'] = pd.Series(data=self._gid_table['population'])
        population_names_ds = self._gid_table['population_names']
        for pop_id, subset in gid_df.groupby(by='population'):
            pop_name = population_names_ds[pop_id]
            self._gid_table_groupby[pop_name] = subset
        self._has_gids = True

    def generate_gids(self, file_name, gids=None, force=False):
        """Creates a gid <--> (node_id, population) table based on sonnet specifications.

         Generating gids will take some time and so not recommend to call this during the simulation. Instead save
         the file to the disk and pass in h5 file during the simulation (using gid_table parameter). In fact if you're
         worried about efficeny don't use this method.

        :param file_name: Name of h5 file to save gid map to.
        :param gids: rule/list of gids to use
        :param force: set to true to overwrite existing gid map (default False).
        """

        # TODO: This is very inefficent, fix (although not a priority as this function should be called sparingly)
        # TODO: Allow users to pass in a list/function to determine gids
        # TODO: We should use an enumerated lookup table for population ds instead of storing strings
        # TODO: Move this to a utils function rather than a File
        if self.has_gids and not force:
            raise Exception('Nodes already have a gid table. Use force=True to overwrite existing gids.')

        dir_name = os.path.dirname(os.path.abspath(file_name))
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        with h5py.File(file_name, 'w') as h5:
            # TODO: should we use mode 'x', or give an option to overwrite existing files
            n_nodes = 0
            ascii_len = 0  # store max population name for h5 fixed length strings
            # Find population names and the total size of every population
            for node_pop in self.populations:
                n_nodes += len(node_pop)
                name_nchars = len(node_pop.name)
                ascii_len = ascii_len if ascii_len >= name_nchars else name_nchars

            # node_id and gid datasets should just be unsigned integers
            h5.create_dataset(name='gid', shape=(n_nodes,), dtype=np.uint64)
            h5.create_dataset(name='node_id', shape=(n_nodes,), dtype=np.uint64)
            # TODO: determine population precisions from num of populations
            h5.create_dataset(name='population', shape=(n_nodes,), dtype=np.uint16)

            # Create a lookup table for pop-name
            pop_name_list = [pname for pname in self.population_names]
            if utils.using_py3:
                dt = h5py.special_dtype(vlen=str)  # python 3
            else:
                dt = h5py.special_dtype(vlen=unicode)  # python 2
            h5.create_dataset(name='population_names', shape=(len(pop_name_list),), dtype=dt)
            # No clue why but just passing in the data during create_dataset doesn't work h5py
            for i, n in enumerate(pop_name_list):
                h5['population_names'][i] = n

            # write each (gid, node_id, population)
            indx = 0
            for node_pop in self.populations:
                # TODO: Block write if special gid generator isn't being used
                # TODO: Block write populations at least
                pop_name = node_pop.name # encode('ascii', 'ignore')
                pop_id = pop_name_list.index(pop_name)
                for node in node_pop:
                    h5['node_id'][indx] = node.node_id
                    h5['population'][indx] = pop_id
                    h5['gid'][indx] = indx
                    indx += 1

            # pass gid table to current nodes
            self.set_gid_table(h5)

    def _build_types_table(self):
        self.types_table = NodeTypesTable()
        for _, csvhandle in self._csv_handles:
            self.types_table.add_table(csvhandle)

    def _build_population(self, pop_name, pop_group):
        return NodePopulation(pop_name, pop_group, self.node_types_table)

    def __getitem__(self, population_name):
        # If their is a gids map then we must pass it into the population
        pop_obj = super(NodesRoot, self).__getitem__(population_name)
        if self.has_gids and (not pop_obj.has_gids) and (population_name in self._gid_table_groupby):
            pop_obj.add_gids(self._gid_table_groupby[population_name])

        return pop_obj


class EdgesRoot(FileRoot):
    def __init__(self, edges, edge_types, mode='r'):
        super(EdgesRoot, self).__init__(root_name='edges', h5_files=edges, h5_mode=mode, csv_files=edge_types)


    @property
    def edge_types_table(self):
        return self.types_table

    def get_populations(self, name=None, source=None, target=None):
        """Find all populations with matching criteria, either using the population name (which will return a list
        of size 0 or 1) or based on the source/target population.

        To return a list of all populations just use populations() method

        :param name: (str) name of population
        :param source: (str or NodePopulation) returns edges with nodes coming from matching source-population
        :param target: (str or NodePopulation) returns edges with nodes coming from matching target-population
        :return: A (potential empty) list of EdgePopulation objects filter by criteria.
        """
        assert((name is not None) ^ (source is not None or target is not None))
        if name is not None:
            return [self[name]]

        else:
            # TODO: make sure groups aren't built unless they are a part of the results
            selected_pops = self.population_names
            if source is not None:
                # filter out only edges with given source population
                source = source.name if isinstance(source, NodePopulation) else source
                selected_pops = [name for name in selected_pops
                                 if EdgePopulation.get_source_population(self._populations_groups[name]) == source]
            if target is not None:
                # filter out by target population
                target = target.name if isinstance(target, NodePopulation) else target
                selected_pops = [name for name in selected_pops
                                 if EdgePopulation.get_target_population(self._populations_groups[name]) == target]

            return [self[name] for name in selected_pops]

    def _build_types_table(self):
        self.types_table = EdgeTypesTable()
        for _, csvhandle in self._csv_handles:
            self.edge_types_table.add_table(csvhandle)

    def _build_population(self, pop_name, pop_group):
        return EdgePopulation(pop_name, pop_group, self.edge_types_table)
