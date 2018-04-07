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


# import nrn


import numpy as np
from neuron import h

from bmtk.simulator.utils.graph import SimGraph, SimEdge, SimNode
import bmtk.simulator.bionet.config as cfg
# from bmtk.simulator.bionet.property_schemas import DefaultPropertySchema, CellTypes
from . import io
from property_map import NodePropertyMap, EdgePropertyMap
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.lifcell import LIFCell
from bmtk.simulator.bionet.stim import Stim
from bmtk.simulator.bionet.morphology import Morphology

# TODO: leave this import, it will initialize some of the default functions for building neurons/synapses/weights.
import bmtk.simulator.bionet.default_setters

pc = h.ParallelContext()  # object to access MPI methods
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())


class BioEdge(object):
    def __init__(self, sonata_edge, graph, prop_map):
        self._sonata_edge = sonata_edge
        self._graph = graph
        self._prop_map = prop_map

    @property
    def preselected_targets(self):
        return self._prop_map.preselected_targets

    def weight(self, source, target):
        return self._prop_map.syn_weight(self._sonata_edge, source, target)

    @property
    def target_sections(self):
        # TODO: Use property-map
        return self._sonata_edge['target_sections']

    @property
    def target_distance(self):
        # TODO: Use property map
        return self._sonata_edge['distance_range']

    @property
    def nsyns(self):
        return self._prop_map.nsyns(self._sonata_edge)

    def load_synapses(self, section_x, section_id):
        return self._prop_map.load_synapse_obj(self._sonata_edge, section_x, section_id)

    def __getitem__(self, item):
        return self._sonata_edge[item]


class BioNode(object):
    def __init__(self, node, property_map, graph):
        self._node = node
        self._prop_map = property_map
        self._graph = graph

    @property
    def model_type(self):
        return self._node['model_type']

    @property
    def node_id(self):
        return self._node.node_id

    @property
    def gid(self):
        return self._prop_map.gid(self._node)

    @property
    def positions(self):
        return self._prop_map.positions(self._node)

    def load_cell(self):
        return self._prop_map.load_cell(self._node)

    @property
    def rotation_angle_xaxis(self):
        return self._prop_map.rotation_angle_xaxis(self._node)

    @property
    def rotation_angle_yaxis(self):
        # TODO: Combine rotation alnges into a single property
        return self._prop_map.rotation_angle_yaxis(self._node)

    @property
    def rotation_angle_zaxis(self):
        return self._prop_map.rotation_angle_zaxis(self._node)

    def __getitem__(self, property_name):
        return self._node[property_name]


class BioGraph(SimGraph):
    model_type_col = 'model_type'

    def __init__(self):
        # property_schema = property_schema if property_schema is not None else DefaultPropertySchema
        super(BioGraph, self).__init__()
        self._io = io

        self._model_type_map = {
            'biophysical': BioCell,
            'point_process': LIFCell,
            'point_soma': lambda *x: None,
            'virtual': lambda *x: None
        }

        self._node_cache = {}
        self._local_cells_nid = {}
        self._local_cells_gid = {}
        # TODO: Find a way to add new types
        self._local_cells_type = {mtype: {} for mtype in self._model_type_map.keys()}

        self._connections_initialized = False

        self._virtual_cells_nid = {}

        self.__morphologies_cache = {}
        self._morphology_lookup = {}

        self._cells_built = False

    def _from_json(self, file_name):
        return cfg.Config.from_dict(file_name, validate=True)

    '''
    def __get_morphology(self, node):
        morphology_file = node['morphology_file']
        if morphology_file in self.__morphology_cache:
            return self.__morphology_cache[morphology_file]
        else:
            full_path = os.path.join(self.get_component('morphologies_dir'), morphology_file)
            self.__morphology_cache[morphology_file] = full_path
            return full_path
    '''

    def _create_edge(self, edge, dynamics_params):
        return BioEdge(edge, dynamics_params, self)

    def __avail_model_types(self, population):
        model_types = set()
        for grp in population.groups:
            if self.model_type_col not in grp.all_columns:
                self.io.log_exception('model_type is missing from nodes.')

            model_types.update(set(np.unique(grp.get_values(self.model_type_col))))
        return model_types

    def _create_nodes_prop_map(self, grp):
        return NodePropertyMap.build_map(grp, self)

    def _create_edges_prop_map(self, grp):
        return EdgePropertyMap.build_map(grp, self)

    @property
    def local_cells(self):
        return self._local_cells_gid

    def get_cells(self, model_type):
        return self._local_cells_type[model_type].values()

    @property
    def biopyhys_gids(self):
        return list(self._local_cells_type['biophysical'].keys())

    def get_local_cell(self, gid):
        return self._local_cells_gid[gid]

    def _build_cell(self, bionode):
        # TODO: use property to find model_type
        return self._model_type_map[bionode['model_type']](bionode, self)

    def build_nodes(self):
        # TODO: Raise a warning if more than one internal population and no gids (node_id collision)
        # TODO: Verify there actually is at least one internal population
        for node_pop in self._internal_populations_map.values():
            pop_name = node_pop.name
            prop_map = self._node_property_maps[pop_name]
            node_cache = {}  # TODO: See if we can preallocate
            local_cells = {}
            for node in node_pop[MPI_rank::MPI_size]:
                # Convert sonata node into a bionet node
                # TODO: It might be faster to build and cache all nodes, especially connection_function is used.
                bnode = BioNode(node, prop_map[node.group_id], self)
                node_cache[node.node_id] = bnode

                # build a Cell which contains NEURON objects
                cell = self._build_cell(bnode)
                if cell is not None:
                    self._local_cells_gid[cell.gid] = cell
                    self._local_cells_type[bnode.model_type][cell.gid] = cell
                    local_cells[bnode.node_id] = cell

            self._node_cache[pop_name] = node_cache
            self._local_cells_nid[pop_name] = local_cells

        self.make_morphologies()
        self.set_seg_props()  # set segment properties by creating Morphologies
        # self.set_tar_segs()  # set target segments needed for computing the synaptic innervations
        self.calc_seg_coords()  # use for computing the ECP
        self._cells_built = True

    def set_seg_props(self):
        """Set morphological properties for biophysically (morphologically) detailed cells"""
        for _, morphology in self.__morphologies_cache.items():
            morphology.set_seg_props()

        io.log_info("Set segment properties")

    def calc_seg_coords(self):
        """Needed for the ECP calculations"""
        # TODO: Is there any reason this function can't be moved to make_morphologies()
        for morphology_file, morphology in self.__morphologies_cache.items():
            morph_seg_coords = morphology.calc_seg_coords()   # needed for ECP calculations

            for gid in self._morphology_lookup[morphology_file]:
                self.get_local_cell(gid).calc_seg_coords(morph_seg_coords)

        io.log_info("Set segment coordinates")

    def make_morphologies(self):
        """Creating a Morphology object for each biophysical model"""
        # TODO: Let Morphology take care of the cache
        # TODO: Let other types have morphologies
        # TODO: Get all available morphologies from TypesTable or group
        for cell in self.get_cells('biophysical'):
            morphology_file = cell.morphology_file
            if morphology_file in self.__morphologies_cache:
                # create a single morphology object for each model_group which share that morphology
                morph = self.__morphologies_cache[morphology_file]

                # associate morphology with a cell
                cell.set_morphology(morph)
                self._morphology_lookup[morphology_file].append(cell.gid)

            else:
                hobj = cell.hobj  # get hoc object (hobj) from the first cell with a new morphologys
                morph = Morphology(hobj)

                # associate morphology with a cell
                cell.set_morphology(morph)

                # create a single morphology object for each model_group which share that morphology
                self.__morphologies_cache[morphology_file] = morph
                self._morphology_lookup[morphology_file] = [cell.gid]

        pc.barrier()

    def get_virt_node(self, population, node_id):
        pop_cache = self._node_cache[population]
        if node_id in pop_cache:
            return pop_cache[node_id]
        else:
            # Load node into cache.
            node_pop = self._virtual_populations_map[population]
            sonata_node = node_pop.get_node_id(node_id)
            prop_map = self._node_property_maps[population][sonata_node.group_id]
            bnode = BioNode(sonata_node, prop_map, self)
            pop_cache[node_id] = bnode
            return bnode

    def get_node(self, population, node_id):
        pop_cache = self._node_cache[population]
        if node_id in pop_cache:
            return pop_cache[node_id]
        else:
            # Load node into cache.
            node_pop = self._internal_populations_map[population]
            sonata_node = node_pop.get_node_id(node_id)
            prop_map = self._node_property_maps[population][sonata_node.group_id]
            bnode = BioNode(sonata_node, prop_map, self)
            pop_cache[node_id] = bnode
            return bnode

    def _init_connections(self):
        if not self._connections_initialized:
            io.log_info('Initializing connections.')
            for gid, cell in self._local_cells_gid.items():
                cell.init_connections()
            self._connections_initialized = True

    def build_recurrent_edges(self):
        if not self._recurrent_edges:
            return 0

        self._init_connections()
        syn_count = 0

        # TODO: Check the order, I believe this can be built faster
        for trg_pop_name, nid_table in self._local_cells_nid.items():
            for edge_pop in self._recurrent_edges[trg_pop_name]:
                src_pop_name = edge_pop.source_population
                prop_maps = self._edge_property_maps[edge_pop.name]
                for trg_nid, trg_cell in nid_table.items():
                    for edge in edge_pop.get_target(trg_nid):
                        # Create edge object
                        bioedge = BioEdge(edge, self, prop_maps[edge.group_id])
                        src_node = self.get_node(src_pop_name, edge.source_node_id)
                        syn_count += trg_cell.set_syn_connection(bioedge, src_node)

    def virtual_populations(self):
        return self._virtual_populations

    '''
    def external_edge_populations(self, src_pop, trg_pop):
        return self._external_edges.get((src_pop, trg_pop), [])
    '''

    def get_virt_cell(self, population, node_id, spike_train):
        pop_lkup = self._virtual_cells_nid[population]
        if node_id in pop_lkup:
            return pop_lkup[node_id]
        else:
            node_pop = self._virtual_populations_map[population]
            sonata_node = node_pop.get_node_id(node_id)
            prop_map = self._node_property_maps[population][sonata_node.group_id]
            bnode = BioNode(sonata_node, prop_map, self)
            stim = Stim(bnode, spike_train)
            self._virtual_cells_nid[population][node_id] = stim
            return stim

    def add_spike_trains(self, spike_trains):
        for pop_name in self._virtual_populations_map.keys():
            if pop_name not in spike_trains.populations:
                continue

            for trg_pop_name in self._local_cells_nid.keys():
                for edge_pop in self.external_edge_populations(src_pop=pop_name, trg_pop=trg_pop_name):
                    prop_maps = self._edge_property_maps[edge_pop.name]
                    for trg_nid, trg_cell in self._local_cells_nid[trg_pop_name].items():
                        for edge in edge_pop.get_target(trg_nid):
                            virt_edge = BioEdge(edge, self, prop_maps[edge.group_id])
                            src_cell = self.get_virt_cell(pop_name, edge.source_node_id, spike_trains)
                            trg_cell.set_syn_connection(virt_edge, src_cell, src_cell)
