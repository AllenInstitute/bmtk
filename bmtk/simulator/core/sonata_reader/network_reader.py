import os
import numpy as np
import json
import ast

from bmtk.simulator.core.network_reader import NodesReader, EdgesReader
from bmtk.simulator.core.sonata_reader.node_adaptor import NodeAdaptor
from bmtk.simulator.core.sonata_reader.edge_adaptor import EdgeAdaptor
from bmtk.utils import sonata


def load_nodes(nodes_h5, node_types_csv, gid_table=None, selected_nodes=None, adaptor=NodeAdaptor):
    return SonataNodes.load(nodes_h5, node_types_csv, gid_table, selected_nodes, adaptor)


def load_edges(edges_h5, edge_types_csv, selected_populations=None, adaptor=EdgeAdaptor):
    return SonataEdges.load(edges_h5, edge_types_csv, selected_populations, adaptor)


class SonataNodes(NodesReader):
    def __init__(self, sonata_node_population, prop_adaptor):
        super(SonataNodes, self).__init__()
        self._node_pop = sonata_node_population
        self._pop_name = self._node_pop.name
        self._prop_adaptors = {}
        self._adaptor = prop_adaptor

    @property
    def name(self):
        return self._pop_name

    @property
    def adaptor(self):
        return self._adaptor

    def n_nodes(self):
        return len(self._node_pop)

    def nodes_df(self, **params):
        return self._node_pop.to_dataframe(**params)

    def initialize(self, network):
        # Determine the various mode-types available in the Node Population, whether or not a population of nodes
        # contains virtual/external nodes, internal nodes, or a mix of both affects how to nodes are built
        model_types = set()
        for grp in self._node_pop.groups:
            if self._adaptor.COL_MODEL_TYPE not in grp.all_columns:
                network.io.log_exception('property {} is missing from nodes.'.format(self._adaptor.COL_MODEL_TYPE))

            model_types.update(set(np.unique(grp.get_values(self._adaptor.COL_MODEL_TYPE))))

        if 'virtual' in model_types:
            self._has_virtual_nodes = True
            model_types -= set(['virtual'])
        else:
            self._has_virtual_nodes = False

        if model_types:
            self._has_internal_nodes = True

        self._adaptor.preprocess_node_types(network, self._node_pop)
        #self._preprocess_node_types(network)
        self._prop_adaptors = {grp.group_id: self._create_adaptor(grp, network) for grp in self._node_pop.groups}

    def _create_adaptor(self, grp, network):
        return self._adaptor.create_adaptor(grp, network)

    '''
    def _preprocess_node_types(self, network):
        # TODO: The following figures out the actually used node-type-ids. For mem and speed may be better to just
        # process them all
        node_type_ids = self._node_pop.type_ids
        # TODO: Verify all the node_type_ids are in the table
        node_types_table = self._node_pop.types_table

        # TODO: Convert model_type to a enum
        if network.has_component('morphologies_dir'):
            morph_dir = network.get_component('morphologies_dir')
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
                if isinstance(dynamics_params, dict):
                    continue

                model_type = node_type['model_type']
                if model_type == 'biophysical':
                    params_dir = network.get_component('biophysical_neuron_models_dir')
                elif model_type == 'point_process':
                    params_dir = network.get_component('point_neuron_models_dir')
                elif model_type == 'point_soma':
                    params_dir = network.get_component('point_neuron_models_dir')
                else:
                    # Not sure what to do in this case, throw Exception?
                    params_dir = network.get_component('custom_neuron_models')

                params_path = os.path.join(params_dir, dynamics_params)

                # see if we can load the dynamics_params as a dictionary. Otherwise just save the file path and let the
                # cell_model loader function handle the extension.
                try:
                    params_val = json.load(open(params_path, 'r'))
                    node_type['dynamics_params'] = params_val
                except Exception:
                    # TODO: Check dynamics_params before
                    network.io.log_exception('Could not find node dynamics_params file {}.'.format(params_path))

        # TODO: Use adaptor to validate model_type and model_template values
    '''

    @classmethod
    def load(cls, nodes_h5, node_types_csv, gid_table=None, selected_nodes=None, adaptor=NodeAdaptor):
        sonata_file = sonata.File(data_files=nodes_h5, data_type_files=node_types_csv, gid_table=gid_table)
        node_populations = []
        for node_pop in sonata_file.nodes.populations:
            node_populations.append(cls(node_pop, adaptor))

        return node_populations

    def get_node(self, node_id):
        return self._node_pop.get_node_id(node_id)

    def __getitem__(self, item):
        for base_node in self._node_pop[item]:
            snode = self._prop_adaptors[base_node.group_id].get_node(base_node)
            yield snode

    def __iter__(self):
        return self

    def filter(self, filter_conditons):
        for node in self._node_pop.filter(**filter_conditons):
            yield node

    def get_nodes(self):
        for node_group in self._node_pop.groups:
            node_adaptor = self._prop_adaptors[node_group.group_id]
            if node_adaptor.batch_process:
                for batch in node_adaptor.get_batches(node_group):
                    yield batch
            else:
                for node in node_group:
                    yield node_adaptor.get_node(node)


class SonataEdges(EdgesReader):
    def __init__(self, edge_population, adaptor):
        self._edge_pop = edge_population
        self._adaptor_cls = adaptor
        self._edge_adaptors = {}

    @property
    def name(self):
        return self._edge_pop.name

    @property
    def source_nodes(self):
        return self._edge_pop.source_population

    @property
    def target_nodes(self):
        return self._edge_pop.target_population

    def initialize(self, network):
        self._adaptor_cls.preprocess_edge_types(network, self._edge_pop)
        # self._preprocess_edge_types(network)
        self._edge_adaptors = {grp.group_id: self._adaptor_cls.create_adaptor(grp, network)
                               for grp in self._edge_pop.groups}

    def get_target(self, node_id):
        for edge in self._edge_pop.get_target(node_id):
            yield self._edge_adaptors[edge.group_id].get_edge(edge)

    def get_source(self, node_id):
        for edge in self._edge_pop.get_source(node_id):
            yield self._edge_adaptors[edge.group_id].get_edge(edge)

    def get_edges(self):
        for edge_group in self._edge_pop.groups:
            edge_adaptor = self._edge_adaptors[edge_group.group_id]
            if edge_adaptor.batch_process:
                for edge in edge_adaptor.get_batches(edge_group):
                    yield edge
            else:
                for edge in self._edge_pop:
                    yield edge_adaptor.get_edge(edge)

    '''
    def _preprocess_edge_types(self, network):
        edge_types_table = self._edge_pop.types_table
        edge_type_ids = np.unique(self._edge_pop.type_ids)

        for et_id in edge_type_ids:
            edge_type = edge_types_table[et_id]
            if 'dynamics_params' in edge_types_table.columns:
                dynamics_params = edge_type['dynamics_params']
                params_dir = network.get_component('synaptic_models_dir')

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
                        self.io.log_warning('Unable to split target_sections list {}'.format(trg_sec))
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
                            self.io.log_warning('Unable to parse distance_range {}'.format(dist_range))
                            edge_type['distance_range'] = None
    '''

    @classmethod
    def load(cls, edges_h5, edge_types_csv, selected_populations=None, adaptor=EdgeAdaptor):
        sonata_file = sonata.File(data_files=edges_h5, data_type_files=edge_types_csv)
        edge_populations = []
        for edge_pop in sonata_file.edges.populations:
            edge_populations.append(cls(edge_pop, adaptor))

        return edge_populations
