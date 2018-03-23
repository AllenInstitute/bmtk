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
import ast

import nrn


import numpy as np
from neuron import h

from bmtk.simulator.utils.graph import SimGraph, SimEdge, SimNode
import bmtk.simulator.bionet.config as cfg
from bmtk.simulator.bionet.property_schemas import DefaultPropertySchema, CellTypes
from . import io
from property_map import PropertyMap, EdgePropertyMap
from bmtk.simulator.bionet.biocell import BioCell
from bmtk.simulator.bionet.lifcell import LIFCell
from bmtk.simulator.bionet.stim import Stim
from bmtk.simulator.bionet.morphology import Morphology

pc = h.ParallelContext()  # object to access MPI methods
MPI_size = int(pc.nhost())
MPI_rank = int(pc.id())


'''
class BioEdge(SimEdge):
    def __init__(self, original_params, dynamics_params, graph):
        super(BioEdge, self).__init__(original_params, dynamics_params)
        self._graph = graph
        self._preselected_targets = graph.property_schema.preselected_targets()
        self._target_sections = graph.property_schema.target_sections(original_params)
        self._target_distance = graph.property_schema.target_distance(original_params)
        self._nsyns = graph.property_schema.nsyns(original_params)

    @property
    def preselected_targets(self):
        return self._preselected_targets

    def weight(self, source, target):
        return self._graph.property_schema.get_edge_weight(source, target, self)

    @property
    def target_sections(self):
        return self._target_sections

    @property
    def target_distance(self):
        return self._target_distance

    @property
    def nsyns(self):
        return self._nsyns

    def load_synapses(self, section_x, section_id):
        return self._graph.property_schema.load_synapse_obj(self, section_x, section_id)
'''
class BioEdge(object):
    def __init__(self, sonata_edge, graph, prop_map):
        #super(BioEdge, self).__init__(original_params, dynamics_params)
        self._sonata_edge = sonata_edge
        self._graph = graph
        self._prop_map = prop_map
        #self._preselected_targets = graph.property_schema.preselected_targets()
        #self._target_sections = graph.property_schema.target_sections(original_params)
        #self._target_distance = graph.property_schema.target_distance(original_params)
        #self._nsyns = graph.property_schema.nsyns(original_params)

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
        #return self._graph.property_schema.load_synapse_obj(self, section_x, section_id)

    def __getitem__(self, item):
        return self._sonata_edge[item]



'''
class BioNode(SimNode):
    def __init__(self, node_id, graph, network, node_params):
        super(BioNode, self).__init__(node_id, graph, network, node_params)

        self._cell_type = graph.property_schema.get_cell_type(node_params)
        self._morphology_file = None
        self._positions = graph.property_schema.get_positions(node_params)  # TODO: implement with lazy evaluation

    @property
    def cell_type(self):
        return self._cell_type

    @property
    def positions(self):
        return self._positions

    @property
    def morphology_file(self):
        return self._morphology_file

    @property
    def rotation_angle_yaxis(self):
        if 'rotation_angle_yaxis' in self:
            return self['rotation_angle_yaxis']
        else:
            return 0.0

    @property
    def rotation_angle_zaxis(self):
        if 'rotation_angle_zaxis' in self:
            return self['rotation_angle_zaxis']
        else:
            return 0.0

    @morphology_file.setter
    def morphology_file(self, value):
        self._morphology_file = value
        self._updated_params['morphology_file'] = value

    def load_hobj(self):
        return self._graph.property_schema.load_cell_hobj(self)
'''
'''
class BioNode(object):
    pass
'''



class PropertyMapS(object):
    def __init__(self, biograph):
        self._template_cache = {}
        self._biograph = biograph

    def _parse_model_template(self, model_template):
        if model_template in self._template_cache:
            return self._template_cache[model_template]
        else:
            template_parts = model_template.split(':')
            assert(len(template_parts) == 2)
            directive, template = template_parts[0], template_parts[1]
            self._template_cache[model_template] = (directive, template)
            return directive, template

    def load_cell(self, node):
        model_template = node['model_template']
        model_type = node['model_type']
        if nrn.py_modules.has_cell_model(model_template, model_type):
            cell_fnc = nrn.py_modules.cell_model(model_template, node['model_type'])
            template_name = None
        else:
            directive, template_name = self._parse_model_template(node['model_template'])
            cell_fnc = nrn.py_modules.cell_model(directive, node['model_type'])

        dynamics_params = self.dynamics_params(node)
        return cell_fnc(node, template_name, dynamics_params)

    @classmethod
    def parse_group(cls, node_group, biograph):
        prop_map = cls(biograph)

        if 'positions' in node_group.all_columns:
            prop_map.positions = lambda self, node: node['positions']
        elif 'position' in node_group.all_columns:
            prop_map.positions = lambda self, node: node['position']
        else:
            prop_map.positions = lambda self, node: [0, 0, 0]

        # Use gids or node_ids
        if node_group.has_gids:
            prop_map.gid = lambda self, node: node.gid
        else:
            prop_map.gid = lambda self, node: node.node_id

        # dynamics_params
        if node_group.has_dynamics_params:
            prop_map.dynamics_params = lambda self, node: node.dynamics_params
        else:
            prop_map.dynamics_params = lambda self, node: node['dynamics_params']

        # Rotation angles
        if 'rotation_angle_xaxis' in node_group.all_columns:
            prop_map.rotation_angle_xaxis = lambda self, node: node['rotation_angle_xaxis']
        else:
            prop_map.rotation_angle_xaxis = lambda self, _: 0.0

        if 'rotation_angle_yaxis' in node_group.all_columns:
            cls.rotation_angle_yaxis = lambda self, node: node['rotation_angle_yaxis']
        else:
            cls.rotation_angle_yaxis = lambda self, _: 0.0

        if 'rotation_angle_zaxis' in node_group.all_columns:
            cls.rotation_angle_zaxis = lambda self, node: node['rotation_angle_zaxis']
        else:
            cls.rotation_angle_zaxis = lambda self, _: 0.0


        return prop_map


class BioGraph(SimGraph):
    model_type_col = 'model_type'

    def __init__(self, property_schema=None):
        property_schema = property_schema if property_schema is not None else DefaultPropertySchema
        super(BioGraph, self).__init__(property_schema)

        self.__local_nodes_table = {}
        self.__virtual_nodes_table = {}
        self.__morphology_cache = {}

        #self._params_cache = {}
        self._params_column = self.property_schema.get_params_column()
        self._io = io

        #self._virtual_node_groups = []
        #self._local_node_groups = []
        #self._nodes = {'biophysical': [], 'point_soma': [], 'point_process': [], 'virtual': []}

        self._internal_populations = []
        self._internal_pop_names = set()
        self._virtual_populations = []
        self._virtual_pop_names = set()

        self._recurrent_edges = {}  # organize edges by their target_population
        self._external_edges = {}

        self._node_property_maps = {}
        self._edge_property_maps = {}

        self._edge_params_cache = {}
        self._node_params_cache = {}
        self._internal_populations_map = {}
        self._virtual_populations_map = {}


    @staticmethod
    def get_default_property_schema():
        raise NotImplementedError()

    def _from_json(self, file_name):
        return cfg.Config.from_dict(file_name, validate=True)

    def __get_morphology(self, node):
        morphology_file = node['morphology_file']
        if morphology_file in self.__morphology_cache:
            return self.__morphology_cache[morphology_file]
        else:
            full_path = os.path.join(self.get_component('morphologies_dir'), morphology_file)
            self.__morphology_cache[morphology_file] = full_path
            return full_path

    def get_node_params(self, node):
        model_type = node['model_type']
        params_file = node['dynamics_params']
        key = (model_type, params_file)
        if key in self._node_params_cache:
            return self._node_params_cache[key]
        else:
            if model_type == 'biophysical':
                params_dir = self.get_component('biophysical_neuron_models_dir')
            elif model_type == 'point_process':
                params_dir = self.get_component('point_neuron_models_dir')
            elif model_type == 'point_soma':
                params_dir = self.get_component('point_neuron_models_dir')
            else:
                # Not sure what to do in this case, throw Exception?
                params_dir = self.get_component('custom_neuron_models')

            params_path = os.path.join(params_dir, params_file)

            # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
            # cell_model loader function handle the extension.
            try:
                params_val = json.load(open(params_path, 'r'))
            except Exception:
                # TODO: Check dynamics_params before
                self.io.log_exception('Could not find node dynamics_params file {}.'.format(params_path))

            # cache and return the value
            self._node_params_cache[key] = params_val
            return params_val


    '''
    def __get_params(self, node, cell_type):
        if node.with_dynamics_params:
            return node[self._params_column]

        params_file = node[self._params_column]

        if params_file in self._params_cache:
            return self._params_cache[params_file]

        else:
            # find the full path of the parameters file from the config file
            if cell_type == CellTypes.Biophysical:
                params_dir = self.get_component('biophysical_neuron_models_dir')
            elif cell_type == CellTypes.Point:
                params_dir = self.get_component('point_neuron_models_dir')
            else:
                # Not sure what to do in this case, throw Exception?
                params_dir = self.get_component('custom_neuron_models')

            params_path = os.path.join(params_dir, params_file)

            # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
            # cell_model loader function handle the extension.
            try:
                params_val = json.load(open(params_path, 'r'))
            except Exception:
                params_val = params_path

            # cache and return the value
            self._params_cache[params_file] = params_val
            return params_val
    '''

    def _create_edge(self, edge, dynamics_params):
        return BioEdge(edge, dynamics_params, self)

    def _add_node(self, node_params, network):
        node = BioNode(node_params.gid, self, network, node_params)
        if node.cell_type == CellTypes.Biophysical:
            node.morphology_file = self.__get_morphology(node_params)
            node.model_params = self.__get_params(node_params, node.cell_type)
            self._add_internal_node(node, network)

        elif node.cell_type == CellTypes.Point:
            node.model_params = self.__get_params(node_params, CellTypes.Point)
            self._add_internal_node(node, network)

        elif node.cell_type == CellTypes.Virtual:
            self.__virtual_nodes_table[node.node_id] = node
            self._add_external_node(node, network)

        else:
            raise Exception('Unknown model type {}'.format(node_params['model_type']))

    def __avail_model_types(self, population):
        model_types = set()
        for grp in population.groups:
            if self.model_type_col not in grp.all_columns:
                self.io.log_exception('model_type is missing from nodes.')

            model_types.update(set(np.unique(grp.get_values(self.model_type_col))))
        return model_types

    def __preprocess_node_types(self, node_population):
        # TODO: The following figures out the actually used node-type-ids. For mem and speed may be better to just process them all
        node_type_ids = node_population.type_ids
        # TODO: Verify all the node_type_ids are in the table
        node_types_table = node_population.types_table

        # TODO: Convert model_type to a enum
        morph_dir = self.get_component('morphologies_dir')
        if morph_dir is not None and 'morphology_file' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                if node_type['morphology_file'] is None:
                    continue
                # TODO: Check the file exits
                # TODO: See if absolute path is stored in csv
                node_type['morphology_file'] = os.path.join(morph_dir, node_type['morphology_file'])

        if 'dynamics_params' in node_types_table.columns and 'model_type' in node_types_table.columns:
            for nt_id in node_type_ids:
                node_type = node_types_table[nt_id]
                dynamics_params = node_type['dynamics_params']
                model_type = node_type['model_type']
                if model_type == 'biophysical':
                    params_dir = self.get_component('biophysical_neuron_models_dir')
                elif model_type == 'point_process':
                    params_dir = self.get_component('point_neuron_models_dir')
                elif model_type == 'point_soma':
                    params_dir = self.get_component('point_neuron_models_dir')
                else:
                    # Not sure what to do in this case, throw Exception?
                    params_dir = self.get_component('custom_neuron_models')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    node_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    self.io.log_exception('Could not find node dynamics_params file {}.'.format(params_path))


    def add_nodes(self, sonata_file, populations=None):
        """Add nodes from a network to the graph.

        :param nodes: A NodesFormat type object containing list of nodes.
        :param network: name/identifier of network. If none will attempt to retrieve from nodes object
        """
        nodes = sonata_file.nodes

        selected_populations = nodes.population_names if populations is None else populations
        for pop_name in selected_populations:
            if pop_name not in nodes:
                continue

            pop_metadata = NodePopulationMetaData(self)
            node_pop = nodes[pop_name]

            self.__preprocess_node_types(node_pop)

            # TODO: Check for population name collisions

            # Segregate into virtual populations and non-virtual populations
            model_types = self.__avail_model_types(node_pop)
            if 'virtual' in model_types:
                self._virtual_populations.append(node_pop)
                self._virtual_populations_map[pop_name] = node_pop
                self._virtual_pop_names.add(pop_name)
                model_types -= set(['virtual'])
                if model_types:
                    # We'll allow a population to have virtual and non-virtual nodes but it is not ideal
                    self.io.log_warning('Node population {} contains both virtual and non-virtual nodes which can ' +
                                        'cause memory and build-time inefficency. Consider separating virtual nodes ' +
                                        'into their own population'.format(pop_name))
                    pop_metadata.mixed_types = True

            if model_types:
                self._internal_populations.append(node_pop)
                self._internal_populations_map[pop_name] = node_pop
                self._internal_pop_names.add(pop_name)

            self._node_property_maps[pop_name] = {}
            for grp in node_pop.groups:
                # TODO: Use list since group_id's are ordered
                prop_map = PropertyMap.build_map(grp, self)
                self._node_property_maps[pop_name][grp.group_id] = prop_map


    #_local_gid_map = {}
    #_local_nid_map = {}
    #_global_gid_map = {}



    _model_type_map = {
        'biophysical': BioCell,
        'point_process': LIFCell,
        'point_soma': lambda *x: None,
        'virtual': lambda *x: None
    }

    #_local_nodes = {}
    _node_cache = {}
    _local_cells_nid = {}
    _local_cells_gid = {}
    _local_cells_type = {mtype: {} for mtype in _model_type_map.keys()}  # TODO: Find a way to add new types

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
        def rotation_angle_yaxis(self):
            # TODO: Combine rotation alnges into a single property
            return self._prop_map.rotation_angle_yaxis(self._node)

        @property
        def rotation_angle_zaxis(self):
            return self._prop_map.rotation_angle_zaxis(self._node)

        def __getitem__(self, property_name):
            return self._node[property_name]


    def build_nodes(self):
        # TODO: Raise a warning if more than one internal population and no gids (node_id collision)
        # TODO: Verify there actually is at least one internal population
        io.log_info('building cells.')

        for node_pop in self._internal_populations:
            pop_name = node_pop.name
            prop_map = self._node_property_maps[pop_name]
            node_cache = {}  # TODO: See if we can preallocate
            local_cells = {}
            for node in node_pop[MPI_rank::MPI_size]:
                # Convert sonata node into a bionet node
                # TODO: It might be faster to build and cache all nodes, especially connection_function is used.
                bnode = self.BioNode(node, prop_map[node.group_id], self)
                node_cache[node.node_id] = bnode

                # build a Cell which contains NEURON objects
                cell = self._build_cell(bnode)
                if cell is not None:
                    self._local_cells_gid[cell.gid] = cell
                    self._local_cells_type[bnode.model_type][cell.gid] = cell
                    local_cells[bnode.node_id] = cell

            self._node_cache[pop_name] = node_cache
            self._local_cells_nid[pop_name] = local_cells

        '''
        def internal_nodes_itr(self, population_name, start=0, step=1):
            prop_maps = self._node_property_maps[population_name]
            node_pop = self._internal_populations_map[population_name]
            # print prop_maps
            for node in node_pop[start::step]:
                # print prop_maps[node.group_id]
                yield node, prop_maps[node.group_id]
        '''

        self.make_morphologies()
        self.set_seg_props()  # set segment properties by creating Morphologies
        # self.set_tar_segs()  # set target segments needed for computing the synaptic innervations
        self.calc_seg_coords()  # use for computing the ECP
        self._cells_built = True

    __morphologies_cache = {}
    _morphology_lookup = {}


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
        #
        #for gid in self._cell_model_gids['biophysical']:
        #    cell = self._cells[gid]
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

    def get_node(self, population, node_id):
        pop_cache = self._node_cache[population]
        if node_id in pop_cache:
            return pop_cache[node_id]
        else:
            # Load node into cache.
            print node_id
            print population
            raise NotImplementedError

    _connections_initialized = False

    def _init_connections(self):
        if not self._connections_initialized:
            io.log_info('Initializing connections.')
            for gid, cell in self._local_cells_gid.items():
                cell.init_connections()
            self._connections_initialized = True


    def build_recurrent_edges(self):
        self._init_connections()
        io.log_info('building recurrent connections')
        syn_count = 0
        # TODO: Check the order, I believe this can be built faster
        for trg_pop_name, nid_table in self._local_cells_nid.items():
            for edge_pop in self._recurrent_edges[trg_pop_name]:
                src_pop_name = edge_pop.source_population
                prop_maps = self._edge_property_maps[edge_pop.name]
                for trg_nid, trg_cell in nid_table.items():
                    for edge in edge_pop.get_target(trg_nid):
                        # Create edge object
                        # TODO: Checking edge property for every group is not ideal. Force all groups to be uniform
                        bioedge = BioEdge(edge, self, prop_maps[edge.group_id])
                        src_node = self.get_node(src_pop_name, edge.source_node_id)
                        syn_count += trg_cell.set_syn_connection(bioedge, src_node)

        io.log_info('  Created {} synapses'.format(syn_count))


    def __preprocess_edge_types(self, edge_pop):
        edge_types_table = edge_pop.types_table
        edge_type_ids = edge_pop.type_ids
        if 'dynamics_params' in edge_types_table.columns:
            for et_id in edge_type_ids:
                edge_type = edge_types_table[et_id]
                dynamics_params = edge_type['dynamics_params']
                params_dir = self.get_component('synaptic_models_dir')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    edge_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    self.io.log_exception('Could not find edge dynamics_params file {}.'.format(params_path))

                # Split target_sections
                if 'target_sections' in edge_type:
                    trg_sec = edge_type['target_sections']
                    if trg_sec is not None:
                        try:
                            edge_type['target_sections'] = ast.literal_eval(trg_sec)
                        except Exception as exc:
                            io.log_warning('Unable to split target_sections list {}'.format(trg_sec))
                            edge_type['target_sections'] = None

                # Split target distances
                if 'distance_range' in edge_type:
                    dist_range = edge_type['distance_range']
                    if dist_range is not None:
                        try:
                            # TODO: Make the distance range has at most two values
                            edge_type['distance_range'] = json.loads(dist_range)
                        except Exception as e:
                            try:
                                edge_type['distance_range'] = [0.0, float(dist_range)]
                            except Exception as e:
                                io.log_warning('Unable to parse distance_range {}'.format(dist_range))
                                edge_type['distance_range'] = None



    def add_edges(self, sonata_file, populations=None, source_pop=None, target_pop=None):
        edges = sonata_file.edges

        selected_populations = edges.population_names if populations is None else populations
        for pop_name in selected_populations:
            if pop_name not in edges:
                continue

            edge_pop = edges[pop_name]

            # TODO: Preprocess edge_types_table
            # TODO: Need to move this out, do it for each file
            self.__preprocess_edge_types(edge_pop)

            # Check the source nodes exists
            src_pop = source_pop if source_pop is not None else edge_pop.source_population
            internal_src = src_pop in self._internal_pop_names
            external_src = src_pop in self._virtual_pop_names

            trg_pop = target_pop if target_pop is not None else edge_pop.target_population
            internal_trg = trg_pop in self._internal_pop_names

            if not internal_trg:
                self.io.log_exception(('Node population {} does not exists (or consists of only virtual nodes). ' +
                                      '{} edges cannot create connections.').format(trg_pop, pop_name))

            if not (internal_src or external_src):
                self.io.log_exception('Source node population {} not found. Please update {} edges'.format(src_pop,
                                                                                                           pop_name))

            if internal_src:
                if trg_pop not in self._recurrent_edges:
                    self._recurrent_edges[trg_pop] = []
                self._recurrent_edges[trg_pop].append(edge_pop)

            if external_src:
                if trg_pop not in self._external_edges:
                    self._external_edges[(src_pop, trg_pop)] = []
                self._external_edges[(src_pop, trg_pop)].append(edge_pop)

            # TODO: Just make all groups have the same connection properties
            self._edge_property_maps[pop_name] = {}
            for grp in edge_pop.groups:
                prop_map = EdgePropertyMap.build_map(grp, self)
                self._edge_property_maps[pop_name][grp.group_id] = prop_map


    '''
    def internal_nodes_itr(self, start=0, step=1):
        for node_pop in self._internal_populations:
            prop_maps = self._node_property_maps[node_pop.name]
            #print prop_maps
            for node in node_pop[start::step]:
                #print prop_maps[node.group_id]
                yield node, prop_maps[node.group_id]
    '''
    def internal_nodes_itr(self, population_name, start=0, step=1):
        prop_maps = self._node_property_maps[population_name]
        node_pop = self._internal_populations_map[population_name]
        #print prop_maps
        for node in node_pop[start::step]:
            #print prop_maps[node.group_id]
            yield node, prop_maps[node.group_id]

    def external_nodes_itr(self, population_name, start=0, step=1):
        node_pop = self._virtual_populations_map[population_name]
        for node in node_pop[start::step]:
            yield node

    def virtual_populations(self):
        return self._virtual_populations

    def virtual_pop_names(self):
        return list(self._virtual_pop_names)

    def external_edge_populations(self, src_pop, trg_pop):
        return self._external_edges.get((src_pop, trg_pop), [])

    '''
    def external_edges_itr(self, src_pop, trg_pop, node_ids):
        for edge_pop in self._graph.external_edge_populations(src_pop=pop_name, trg_pop=trg_pop):
            for trg_node in self._population_map[trg_pop]:
                for edge in edge_pop.get_target(trg_node.node_id):
                    stim = source_stims[edge.source_node_id]
                    syn_counter += trg_node.set_syn_connection(edge, stim, stim)
                    print
    '''

    def _get_spike_trains(self, src_gid, network):
        if network in self._spike_trains_ds:
            h5ds = self._spike_trains_ds[network]
            src_gid_str = str(src_gid)
            if src_gid_str in h5ds.keys():
                return h5ds[src_gid_str]['data']

        elif network in self._spike_trains_df:
            spikes_list = [float(t) for t in self._spike_trains_df[network].loc[src_gid]['spike-times'].split(',')]
            return spikes_list

        return []


    _stims = {}


    saved_gids = range(0,10)


    def make_stims(self):
        """Create the stims/virtual/external nodes.

        Make sure spike trains have been set before calling, otherwise it will creating spiking cells with no spikes.
        """
        #print self._stim_networks
        #print self._local_cells_nid
        #exit()

        syn_counter = 0
        for pop_name in self.virtual_pop_names():
            if pop_name not in self._stim_networks:
                continue

            # TODO: Do we need to save stims as an object property?
            self._stims[pop_name] = {}
            for node in self.external_nodes_itr(pop_name):
                spike_train = self._get_spike_trains(node.node_id, pop_name)
                self._stims[pop_name][node.node_id] = Stim(node, spike_train)

            self._init_connections()
            io.log_info('    Setting connections from {}'.format(pop_name))
            # TODO: skip if source_network is not in stims
            source_stims = self._stims[pop_name]
            # TODO: Iterate and do node_id lookup at the same time, cache results (temp)
            for trg_pop_name in self._local_cells_nid.keys():
                for edge_pop in self.external_edge_populations(src_pop=pop_name, trg_pop=trg_pop_name):
                    prop_maps = self._edge_property_maps[edge_pop.name]
                    for trg_nid, trg_cell in self._local_cells_nid[trg_pop_name].items():
                        for edge in edge_pop.get_target(trg_nid):
                            virt_edge = BioEdge(edge, self, prop_maps[edge.group_id])
                            stim = source_stims[edge.source_node_id]
                            syn_counter += trg_cell.set_syn_connection(virt_edge, stim, stim)

                    '''
                    for trg_node_ in self._local_cells_nid[trg_pop]:
                        print trg_node, trg_pop
                        exit()
                        trg_cell self.get_node()
                        #prop_map = prop_maps[trg_node.group_id]
                        for edge in edge_pop.get_target(trg_node.node_id):
                            bioedge = BioEdge(edge, self._graph, prop_map)
                            stim = source_stims[edge.source_node_id]
                            syn_counter += trg_node.set_syn_connection(bioedge, stim, stim)
                    '''

'''
    def build_recurrent_edges(self):
        self._init_connections()
        io.log_info('building recurrent connections')
        syn_count = 0
        # TODO: Check the order, I believe this can be built faster
        for trg_pop_name, nid_table in self._local_cells_nid.items():
            for edge_pop in self._recurrent_edges[trg_pop_name]:
                src_pop_name = edge_pop.source_population
                prop_maps = self._edge_property_maps[edge_pop.name]
                for trg_nid, trg_cell in nid_table.items():
                    for edge in edge_pop.get_target(trg_nid):
                        # Create edge object
                        # TODO: Checking edge property for every group is not ideal. Force all groups to be uniform
                        bioedge = BioEdge(edge, self, prop_maps[edge.group_id])
                        src_node = self.get_node(src_pop_name, edge.source_node_id)
                        syn_count += trg_cell.set_syn_connection(bioedge, src_node)

        io.log_info('  Created {} synapses'.format(syn_count))


    def build_nodes(self):
        # TODO: Raise a warning if more than one internal population and no gids (node_id collision)
        # TODO: Verify there actually is at least one internal population
        io.log_info('building cells.')

        for node_pop in self._internal_populations:
            pop_name = node_pop.name
            prop_map = self._node_property_maps[pop_name]
            node_cache = {}  # TODO: See if we can preallocate
            local_cells = {}
            for node in node_pop[MPI_rank::MPI_size]:
                # Convert sonata node into a bionet node
                # TODO: It might be faster to build and cache all nodes, especially connection_function is used.
                bnode = self.BioNode(node, prop_map[node.group_id], self)
                node_cache[node.node_id] = bnode

                # build a Cell which contains NEURON objects
                cell = self._build_cell(bnode)
                if cell is not None:
                    self._local_cells_gid[cell.gid] = cell
                    self._local_cells_type[bnode.model_type][cell.gid] = cell
                    local_cells[bnode.node_id] = cell

            self._node_cache[pop_name] = node_cache
            self._local_cells_nid[pop_name] = local_cells


'''



class NodePopulationMetaData(object):
    def __init__(self, graph):
        self.mixed_types = False
        self.property_maps = {}
