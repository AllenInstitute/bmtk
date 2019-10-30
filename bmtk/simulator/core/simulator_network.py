from six import string_types
import numpy as np

from bmtk.simulator.core.io_tools import io
#from bmtk.simulator.core.config import ConfigDict
from bmtk.simulator.utils.config import ConfigDict
from bmtk.simulator.core.node_sets import NodeSet, NodeSetAll
from bmtk.simulator.core import sonata_reader


class SimNetwork(object):
    def __init__(self):
        self._components = {}
        self._io = io

        self._node_adaptors = {}
        self._edge_adaptors = {}
        self._register_adaptors()

        self._node_populations = {}
        self._node_sets = {}

        self._edge_populations = []

    @property
    def io(self):
        return self._io

    @property
    def node_populations(self):
        return self._node_populations.values()

    @property
    def recurrent_edges(self):
        return [ep for ep in self._edge_populations if ep.recurrent_connections]

    @property
    def py_function_caches(self):
        return None

    def _register_adaptors(self):
        self._node_adaptors['sonata'] = sonata_reader.NodeAdaptor
        self._edge_adaptors['sonata'] = sonata_reader.EdgeAdaptor

    def get_node_adaptor(self, name):
        return self._node_adaptors[name]

    def get_edge_adaptor(self, name):
        return self._edge_adaptors[name]

    def add_component(self, name, path):
        self._components[name] = path

    def get_component(self, name):
        if name not in self._components:
            self.io.log_exception('No network component set with name {}'.format(name))
        else:
            return self._components[name]

    def has_component(self, name):
        return name in self._components

    def get_node_population(self, name):
        return self._node_populations[name]

    def get_node_populations(self):
        return self._node_populations.values()

    def add_node_set(self, name, node_set):
        self._node_sets[name] = node_set

    def get_node_set(self, node_set):
        if isinstance(node_set, string_types) and node_set in self._node_sets:
            return self._node_sets[node_set]

        elif isinstance(node_set, (dict, list)):
            return NodeSet(node_set, self)

        else:
            self.io.log_exception('Unable to load or find node_set "{}"'.format(node_set))

    def add_nodes(self, node_population):
        pop_name = node_population.name
        if pop_name in self._node_populations:
            # Make sure their aren't any collisions
            self.io.log_exception('There are multiple node populations with name {}.'.format(pop_name))

        node_population.initialize(self)
        self._node_populations[pop_name] = node_population
        if node_population.mixed_nodes:
            # We'll allow a population to have virtual and non-virtual nodes but it is not ideal
            self.io.log_warning(('Node population {} contains both virtual and non-virtual nodes which can cause ' +
                                 'memory and build-time inefficency. Consider separating virtual nodes into their ' +
                                 'own population').format(pop_name))

        # Used in inputs/reports when needed to get all gids belonging to a node population
        self._node_sets[pop_name] = NodeSet({'population': pop_name}, self)

    def node_properties(self, populations=None):
        if populations is None:
            selected_pops = self.node_populations

        elif isinstance(populations, string_types):
            selected_pops = [pop for pop in self.node_populations if pop.name == populations]

        else:
            selected_pops = [pop for pop in self.node_populations if pop.name in populations]

        all_nodes_df = None
        for node_pop in selected_pops:
            node_pop_df = node_pop.nodes_df()
            if 'population' not in node_pop_df:
                node_pop_df['population'] = node_pop.name

            node_pop_df = node_pop_df.set_index(['population', node_pop_df.index.astype(dtype=np.uint64)])
            if all_nodes_df is None:
                all_nodes_df = node_pop_df
            else:
                all_nodes_df = all_nodes_df.append(node_pop_df)

        return all_nodes_df

    def get_node_groups(self, populations=None):
        if populations is None:
            selected_pops = self.node_populations

        elif isinstance(populations, string_types):
            selected_pops = [pop for pop in self.node_populations if pop.name == populations]

        else:
            selected_pops = [pop for pop in self.node_populations if pop.name in populations]

        all_nodes_df = None
        for node_pop in selected_pops:
            node_pop_df = node_pop.nodes_df(index_by_id=False)
            print(node_pop_df)
            print('-----')
            if 'population' not in node_pop_df:
                node_pop_df['population'] = node_pop.name

            #node_pop_df = node_pop_df.set_index([node_pop_df.index.astype(dtype=np.uint64)])

            if all_nodes_df is None:
                all_nodes_df = node_pop_df
            else:
                all_nodes_df = all_nodes_df.append(node_pop_df, sort=False)

        return all_nodes_df

    def get_node_sets(self, populations=None, groupby=None, **filterby):
        selected_nodes_df = self.node_properties(populations)
        for k, v in filterby:
            if isinstance(v, (np.ndarray, list, tuple)):
                selected_nodes_df = selected_nodes_df[selected_nodes_df[k].isin(v)]
            else:
                selected_nodes_df = selected_nodes_df[selected_nodes_df[k].isin(v)]

        if groupby is not None:
            return {k: v.tolist() for k, v in selected_nodes_df.groupby(groupby).groups.items()}
        else:
            return selected_nodes_df.index.tolist()

    def add_edges(self, edge_population):
        edge_population.initialize(self)
        pop_name = edge_population.name

        # Check that source_population exists
        src_pop_name = edge_population.source_nodes
        if src_pop_name not in self._node_populations:
            self.io.log_exception('Source node population {} not found. Please update {} edges'.format(src_pop_name,
                                                                                                       pop_name))

        # Check that the target population exists and contains non-virtual nodes (we cannot synapse onto virt nodes)
        trg_pop_name = edge_population.target_nodes
        if trg_pop_name not in self._node_populations or self._node_populations[trg_pop_name].virtual_nodes_only:
            self.io.log_exception(('Node population {} does not exists (or consists of only virtual nodes). ' +
                                   '{} edges cannot create connections.').format(trg_pop_name, pop_name))

        edge_population.set_connection_type(src_pop=self._node_populations[src_pop_name],
                                            trg_pop = self._node_populations[trg_pop_name])
        self._edge_populations.append(edge_population)

    def build(self):
        self.build_nodes()
        self.build_recurrent_edges()

    def build_nodes(self):
        raise NotImplementedError()

    def build_recurrent_edges(self):
        raise NotImplementedError()

    def build_virtual_connections(self):
        raise NotImplementedError()

    @classmethod
    def from_config(cls, conf, **properties):
        """Generates a graph structure from a json config file or dictionary.

        :param conf: name of json config file, or a dictionary with config parameters
        :param properties: optional properties.
        :return: A graph object of type cls
        """
        network = cls(**properties)

        # The simulation run script should create a config-dict since it's likely to vary based on the simulator engine,
        # however in the case the user doesn't we will try a generic conversion from dict/json to ConfigDict
        if isinstance(conf, ConfigDict):
            config = conf
        else:
            try:
                config = ConfigDict.load(conf)
            except Exception as e:
                network.io.log_exception('Could not convert {} (type "{}") to json.'.format(conf, type(conf)))

        if not config.with_networks:
            network.io.log_exception('Could not find any network files. Unable to build network.')

        # TODO: These are simulator specific
        network.spike_threshold = config.spike_threshold
        network.dL = config.dL

        # load components
        for name, value in config.components.items():
            network.add_component(name, value)

        # load nodes
        gid_map = config.gid_mappings
        node_adaptor = network.get_node_adaptor('sonata')
        for node_dict in config.nodes:
            nodes = sonata_reader.load_nodes(node_dict['nodes_file'], node_dict['node_types_file'], gid_map,
                                             adaptor=node_adaptor)
            for node_pop in nodes:
                network.add_nodes(node_pop)

        # TODO: Raise a warning if more than one internal population and no gids (node_id collision)

        # load edges
        edge_adaptor = network.get_edge_adaptor('sonata')
        for edge_dict in config.edges:
            if not edge_dict.get('enabled', True):
                continue

            edges = sonata_reader.load_edges(edge_dict['edges_file'], edge_dict['edge_types_file'],
                                             adaptor=edge_adaptor)
            for edge_pop in edges:
                network.add_edges(edge_pop)

        # Add nodeset section
        network.add_node_set('all', NodeSetAll(network))
        for ns_name, ns_filter in config.node_sets.items():
            network.add_node_set(ns_name, NodeSet(ns_filter, network))

        return network

    @classmethod
    def from_manifest(cls, manifest_json):
        # TODO: Add adaptors to build a simulation network from model files downloaded celltypes.brain-map.org
        raise NotImplementedError()

    @classmethod
    def from_builder(cls, network):
        # TODO: Add adaptors to build a simulation network from a bmtk.builder Network object
        raise NotImplementedError()

