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
import json
import functools
import nest
import numpy as np

from bmtk.simulator.core.simulator_network import SimNetwork
from bmtk.simulator.pointnet.sonata_adaptors import PointNodeAdaptor, PointEdgeAdaptor
from bmtk.simulator.pointnet import pyfunction_cache
from bmtk.simulator.pointnet.io_tools import io


class PointNetwork(SimNetwork):
    def __init__(self, **properties):
        super(PointNetwork, self).__init__(**properties)
        self._io = io

        self.__weight_functions = {}
        self._params_cache = {}

        self._virtual_ids_map = {}

        self._batch_nodes = True

        self._nest_id_map = {}
        self._nestid2nodeid_map = {}

        self._nestid2gid = {}

        self._nodes_table = {}
        self._gid2nestid = {}

    @property
    def py_function_caches(self):
        return pyfunction_cache

    def __get_params(self, node_params):
        if node_params.with_dynamics_params:
            # TODO: use property, not name
            return node_params['dynamics_params']

        params_file = node_params[self._params_column]
        # params_file = self._MT.params_column(node_params) #node_params['dynamics_params']
        if params_file in self._params_cache:
            return self._params_cache[params_file]
        else:
            params_dir = self.get_component('models_dir')
            params_path = os.path.join(params_dir, params_file)
            params_dict = json.load(open(params_path, 'r'))
            self._params_cache[params_file] = params_dict
            return params_dict

    def _register_adaptors(self):
        super(PointNetwork, self)._register_adaptors()
        self._node_adaptors['sonata'] = PointNodeAdaptor
        self._edge_adaptors['sonata'] = PointEdgeAdaptor

    # TODO: reimplement with py_modules like in bionet
    def add_weight_function(self, function, name=None):
        fnc_name = name if name is not None else function.__name__
        self.__weight_functions[fnc_name] = functools.partial(function)

    def set_default_weight_function(self, function):
        self.add_weight_function(function, 'default_weight_fnc', overwrite=True)

    def get_weight_function(self, name):
        return self.__weight_functions[name]

    def build_nodes(self):
        for node_pop in self.node_populations:
            nid2nest_map = {}
            nest2nid_map = {}
            if node_pop.internal_nodes_only:
                for node in node_pop.get_nodes():
                    node.build()
                    for nid, gid, nest_id in zip(node.node_ids, node.gids, node.nest_ids):
                        self._nestid2gid[nest_id] = gid
                        self._gid2nestid[gid] = nest_id
                        nid2nest_map[nid] = nest_id
                        nest2nid_map[nest_id] = nid

            elif node_pop.mixed_nodes:
                for node in node_pop.get_nodes():
                    if node.model_type != 'virtual':
                        node.build()
                        for nid, gid, nest_id in zip(node.node_ids, node.gids, node.nest_ids):
                            self._nestid2gid[nest_id] = gid
                            self._gid2nestid[gid] = nest_id
                            nid2nest_map[nid] = nest_id
                            nest2nid_map[nest_id] = nid

            self._nest_id_map[node_pop.name] = nid2nest_map
            self._nestid2nodeid_map[node_pop.name] = nest2nid_map

    def build_recurrent_edges(self):
        recurrent_edge_pops = [ep for ep in self._edge_populations if not ep.virtual_connections]
        if not recurrent_edge_pops:
            return

        for edge_pop in recurrent_edge_pops:
            src_nest_ids = self._nest_id_map[edge_pop.source_nodes]
            trg_nest_ids = self._nest_id_map[edge_pop.target_nodes]
            for edge in edge_pop.get_edges():
                nest_srcs = [src_nest_ids[nid] for nid in edge.source_node_ids]
                nest_trgs = [trg_nest_ids[nid] for nid in edge.target_node_ids]
                nest.Connect(nest_srcs, nest_trgs, conn_spec='one_to_one', syn_spec=edge.nest_params)

    def find_edges(self, source_nodes=None, target_nodes=None):
        # TODO: Move to parent
        selected_edges = self._edge_populations[:]

        if source_nodes is not None:
            selected_edges = [edge_pop for edge_pop in selected_edges if edge_pop.source_nodes == source_nodes]

        if target_nodes is not None:
            selected_edges = [edge_pop for edge_pop in selected_edges if edge_pop.target_nodes == target_nodes]

        return selected_edges

    def add_spike_trains(self, spike_trains, node_set):
        # Build the virtual nodes
        src_nodes = [node_pop for node_pop in self.node_populations if node_pop.name in node_set.population_names()]
        for node_pop in src_nodes:
            if node_pop.name in self._virtual_ids_map:
                 continue

            virt_node_map = {}
            if node_pop.virtual_nodes_only:
                for node in node_pop.get_nodes():
                    nest_ids = nest.Create('spike_generator', node.n_nodes, {})
                    for node_id, nest_id in zip(node.node_ids, nest_ids):
                        virt_node_map[node_id] = nest_id
                        nest.SetStatus([nest_id], {'spike_times': np.array(spike_trains.get_spikes(node_id))})

            elif node_pop.mixed_nodes:
                for node in node_pop.get_nodes():
                    if node.model_type != 'virtual':
                        continue

                    nest_ids = nest.Create('spike_generator', node.n_nodes, {})
                    for node_id, nest_id in zip(node.node_ids, nest_ids):
                        virt_node_map[node_id] = nest_id
                        nest.SetStatus([nest_id], {'spike_times': np.array(spike_trains.get_spikes(node_id))})

            self._virtual_ids_map[node_pop.name] = virt_node_map

        # Create virtual synaptic connections
        for source_reader in src_nodes:
            for edge_pop in self.find_edges(source_nodes=source_reader.name):
                src_nest_ids = self._virtual_ids_map[edge_pop.source_nodes]
                trg_nest_ids = self._nest_id_map[edge_pop.target_nodes]
                for edge in edge_pop.get_edges():
                    nest_srcs = [src_nest_ids[nid] for nid in edge.source_node_ids]
                    nest_trgs = [trg_nest_ids[nid] for nid in edge.target_node_ids]
                    nest.Connect(nest_srcs, nest_trgs, conn_spec='one_to_one', syn_spec=edge.nest_params)
