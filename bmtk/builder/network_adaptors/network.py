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
import numpy as np
import types
import csv
import six
import logging

from ..node_pool import NodePool
from ..connection_map import ConnectionMap
from ..node_set import NodeSet
from ..id_generator import IDGenerator
from ..builder_utils import mpi_rank, mpi_size, barrier, check_properties_across_ranks


logger = logging.getLogger(__name__)


class Network(object):
    """The Network class is used for building and saving a brain network/circuit. By default it will save to SONATA
    format for running network simulations using BioNet, PointNet, PopNet or FilterNet bmtk modules, however it can
    be generalized to build any time of network for any time of simulation.

    Building the network:
        For the general use-case building a network consists of 4 steps, each with a corresponding method.

        1. Initialize the network::

            net = Network("network_name")

        2. Create nodes (ie cells) using the **add_nodes()** method::

            net.add_nodes(N=80, model_type='Biophysical', ei='exc')
            net.add_nodes(N=20, model_type='IntFire', ei='inh')
            ...

        3. Create Connection rules between different subsets of nodes using **add_edges()** method::

            net.add_edges(source={'ei': 'exc'}, target={'ei': 'inh'},
                          connection_rule=my_conn_func, synaptic_model='e2i')
            ...

        4. Finally **build** the network and **save** the files::

            net.build()
            net.save(output_dir='network_path')

        See the bmtk documentation, or the method doc-strings for more advanced functionality


    Network Accessor methods:
        **nodes()**

        Will return a iterable of Node objects for each node created. The Node objects can be used like dictionaries to
        fetch their properties. By default returns all nodes in the network, but you can filter out a given subset by
        passing in property/values pairs::

            for node in net.nodes(model_type='Biophysical', ei='exc'):
                assert(node['ei'] == 'exc')
                ...

        **edges()**

        Like the nodes() methods, but insteads returns a list of Edge type objects. It too can be filtered by an edge
        property::

            for edge in net.edges(synaptic_model='exp2'):
                ...

        One can also pass in a list of source (or target) to filter out only those edges which belong to a specific
        subset of cells::

            for edge in net.edges(target_nodes=net.nodes(ei='exc')):
            ...


    Network Properties:
        * **name** - name of the network
        * **nnodes** - number of nodes (cells) in the network.
        * **nedges** - number of edges. Will be zero if build() method hasn't been called
    """

    def __init__(self, name, **network_props):
        if name is None or len(name) == 0:
            raise ValueError('Network name missing.')

        self._network_name = name
        self._nnodes = 0
        self._nodes_built = False
        self._nedges = 0
        self._edges_built = False
        
        self._node_sets = []
        self.__external_node_sets = []
        # self.__node_id_counter = 0

        self._node_types_properties = {}
        self._node_types_columns = {'node_type_id'}
        self._connection_maps = []

        self._node_id_gen = IDGenerator()
        self._node_type_id_gen = IDGenerator(100)
        self._edge_type_id_gen = IDGenerator(100)

        self._gj_id_gen = IDGenerator(network_props.get('gj_id_start', 0))

        self._network_conns = set()
        self._connected_networks = {}

    @property
    def name(self):
        """Get the name (string) of this network."""
        return self._network_name

    @property
    def nodes_built(self):
        """Returns True if nodes has been instantiated for this network."""
        return self._nodes_built

    @property
    def edges_built(self):
        """Returns True if the connectivity matrix has been instantiated for this network."""
        return self._edges_built

    @property
    def nnodes(self):
        """Returns the number of nodes for this network."""
        raise NotImplementedError

    @property
    def nedges(self):
        """Returns the total number of edges for this network."""
        raise NotImplementedError

    def get_connections(self):
        """Returns a list of all bmtk.builder.connection_map.ConnectionMap objects representing all edge-types for this
         network."""
        return self._connection_maps

    def _add_node_type(self, props):
        node_type_id = props.get('node_type_id', None)
        if node_type_id is None:
            node_type_id = self._node_type_id_gen.next()
        else:
            if node_type_id in self._node_types_properties:
                raise Exception('node_type_id {} already exists.'.format(node_type_id))
            self._node_type_id_gen.remove_id(node_type_id)

        props['node_type_id'] = node_type_id
        self._node_types_properties[node_type_id] = props

    def add_nodes(self, N=1, **properties):
        """Used to add nodes (eg cells) to a network. User should specify the number of Nodes (N) and can use any
        properties/attributes they require to define the nodes. By default all individual cells will be assigned a
        unique 'node_id' to identify each node in the network and a 'node_type_id' to identify each group of nodes.

        If a property is a singular value then said property will be shared by all the nodes in the group. If a value
        is a list of length N then each property will be uniquely assigned to each node. In the below example a group
        of 100 nodes is created, all share the same 'model_type' parameter but the pos_x values will be different for
        each node::

            net.add_nodes(N=100, pos_x=np.random.rand(100), model_type='intfire1', ...)

        You can use a tuple to store property values (in which the SONATA hdf5 will save it as a dataset with multiple
        columns). For example to have one property 'positions' which keeps track of the x/y/z coordinates of each cell::

            net.add_nodes(N=100, positions=[(rand(), rand(), rand()) for _ in range(100)], ...)

        :param N: number of nodes in this group
        :param properties: Individual and group properties of given nodes
        """
        self._clear()
        check_properties_across_ranks(properties)

        # categorize properties as either a node-params (for nodes file) or node-type-property (for node_types files)
        node_params = {}
        node_properties = {}
        for prop_name, prop_value in properties.items():
            if isinstance(prop_value, (list, np.ndarray)):
                n_props = len(prop_value)
                if n_props != N:
                    raise Exception('Trying to pass in array of length {} into N={} nodes'.format(n_props, N))
                node_params[prop_name] = prop_value

            elif isinstance(prop_value, (types.GeneratorType, six.moves.range)):
                vals = list(prop_value)
                assert(len(vals) == N)
                node_params[prop_name] = vals

            else:
                node_properties[prop_name] = prop_value
                self._node_types_columns.add(prop_name)

        # If node-type-id exists, make sure there is no clash, otherwise generate a new id.
        if 'node_type_id' in node_params:
            raise Exception('There can be only one "node_type_id" per set of nodes.')

        if 'node_id' in node_params:
            node_id_list = node_params['node_id']
            node_id_list = node_id_list if isinstance(node_id_list, (list, np.ndarray)) else [node_id_list]
            for nid in node_id_list:
                if nid in self._node_id_gen:
                    raise ValueError('Duplicate add node_id value {}.'.format(nid))
                self._node_id_gen.remove_id(nid)

        self._add_node_type(node_properties)
        self._node_sets.append(NodeSet(N, node_params, node_properties))

    def add_edges(self, source=None, target=None, connection_rule=1, connection_params=None, iterator='one_to_one',
                  **edge_type_properties):
        """Used to create the connectivity matrix between subsets of nodes. The actually connections will not be
        created until the build() method is called, using the 'connection_rule.

        Node Selection:
            To specify what subset of nodes will be used for the pre- and post-synaptic one can use a dictionary to
            filter the nodes. In the following all inh nodes will be used for the pre-synaptic neurons, but only exc
            fast-spiking neurons will be used in the post-synaptic neurons (If target or source is not specified all
            neurons will be used)::

                net.add_edges(source={'ei': 'inh'}, target={'ei': 'exc', 'etype': 'fast-spiking'},
                              dynamic_params='i2e.json', synaptic_model='alpha', ...)

            In the above code there is one connection between each source/target pair of nodes, but to create a
            multi-graph with N connections between each pair use 'connection_rule' parameter with an integer value::

                net.add_edges(
                    source={'ei': 'inh'},
                    target={'ei': 'exc', 'etype': 'fast-spiking'},
                    connection_rule=M,
                    ...
                )

        Connection rules:
            Usually the 'connection_rule' parameter will be the name of a function that takes in source-node and
            target-node object (which can be treated like dictionaries, and returns the number of connections (ie
            synapses, 0 or None if no synapses should exists) between the source and target cell::

                def my_conn_fnc(source_node, target_node):
                    src_pos = source_node['position']
                    trg_pos = target_node['position']
                    ...
                    return N_syns

                net.add_edges(source={'ei': 'exc'}, target={'ei': 'inh'}, connection_rule=my_conn_fnc, **opt_edge_attrs)

            If the connection_rule function requires addition arguments use the 'connection_params' option::

                def my_conn_fnc(source_node, target_node, min_edges, max_edges)
                    ...

                net.add_edges(connection_rule=my_conn_fnc, connection_params={'min_edges': 0, 'max_edges': 20}, ...)

            Sometimes it may be more efficient or even a requirement that multiple connections are created at the same
            time. For example a post-synaptic neuron may only be targeted by a limited number of sources which couldn't
            be done by the previous connection_rule function. But by setting property 'iterator' to value 'all_to_one'
            the connection_rule function now takes in as a value a list of N source neurons, a single target, and should
            return a list of size N::

                def bulk_conn_fnc(sources, target):
                    syn_list = np.zeros(len(sources))
                    for source in sources:
                        ....
                    return syn_list

                net.add_edges(connection_rule=bulk_conn_fnc, iterator='all_to_one', ...)

            There is also a 'all_to_one' iterator option that will pair each source node with a list of all available
            target nodes.

        Edge Properties:
            Normally the properties used when creating a given type of edge will be shared by all the indvidual
            connections. To create unique values for each edge, the add_edges() method returns a ConnectionMap object::

                def set_syn_weight_by_dist(source, target):
                    src_pos, trg_pos = source['position'], target['position']
                    ....
                    return syn_weight


                cm = net.add_edges(connection_rule=my_conn_fnc, model_template='Exp2Syn', ...)
                                delay=2.0)
                cm.add_properties('syn_weight', rule=set_syn_weight_by_dist)
                cm.add_properties('delay', rule=lambda *_: np.random.rand(0.01, 0.50))

            In this case the 'model_template' property has a value for all connections of this given type of edge. The
            'syn_weight' and 'delay' properties will (most likely) be unique values. See ConnectionMap documentation for
            more info.

        :param source: A dictionary or list of Node objects (see nodes() method). Used to filter out pre-synaptic
            subset of nodes.
        :param target: A dictionary or list of Node objects). Used to filter out post-synaptic subset of nodes
        :param connection_rule: Integer or a function that returns integer(s). Rule to determine number of connections
            between each source and target node
        :param connection_params: A dictionary, used when the 'connection_rule' is a function that requires additional
            argments
        :param iterator: 'one_to_one', 'all_to_one', 'one_to_all'. When 'connection_rule' is a function this sets
            how the subsets of source/target nodes are passed in. By default (one-to-one) the connection_rule is
            called for every source/target pair. 'all-to-one' will pass in a list of all possible source nodes for
            each target, and 'all-to-one' will pass in a list of all possible targets for each source.
        :param edge_type_properties: properties/attributes of the given edge type
        :return: A ConnectionMap object
        """
        check_properties_across_ranks(edge_type_properties)

        if not isinstance(source, NodePool):
            source = NodePool(self, **source or {})

        if not isinstance(target, NodePool):
            target = NodePool(self, **target or {})

        if edge_type_properties.get('is_gap_junction', False) and source.network_name != target.network_name:
            raise Exception("Gap junctions must consist of two cells on the same network.")

        self._network_conns.add((source.network_name, target.network_name))
        self._connected_networks[source.network_name] = source.network
        self._connected_networks[target.network_name] = target.network

        # TODO: make sure that they don't add a dictionary or some other weird property type.
        edge_type_id = edge_type_properties.get('edge_type_id', None)
        if edge_type_id is None:
            edge_type_id = self._edge_type_id_gen.next()
            edge_type_properties['edge_type_id'] = edge_type_id
        elif edge_type_id in self._edge_type_id_gen:
            raise Exception('edge_type_id {} already exists.'.format(edge_type_id))
        else:
            self._edge_type_id_gen.remove_id(edge_type_id)

        edge_type_properties['source_query'] = source.filter_str
        edge_type_properties['target_query'] = target.filter_str

        if 'nsyns' in edge_type_properties:
            connection_rule = edge_type_properties['nsyns']
            del edge_type_properties['nsyns']

        connection = ConnectionMap(source, target, connection_rule, connection_params, iterator, edge_type_properties)
        self._connection_maps.append(connection)
        return connection

    def add_gap_junctions(self, source=None, target=None, iterator='one_to_one', resistance=1,
                          target_sections=['somatic'], connection_rule=1, connection_params=None):
        """A special function for marking a edge group as gap junctions. Just a wrapper for add_edges"""
        if target_sections is not None:
            logger.warning(
                'For gap junctions, the target sections variable is used for both the source and target sections.'
            )

        return self.add_edges(
            source=source, target=target, iterator=iterator, distance_range=[0.0, 300.0], syn_weight=resistance,
            is_gap_junction=True, target_sections=target_sections, connection_rule=connection_rule,
            connection_params=connection_params
        )

    def nodes(self, **properties):
        """Returns an iterator of Node (glorified dictionary) objects, filtered by parameters.

        To get all nodes on a network::

            for node in net.nodes():
                ...

        To only get those nodes with properties that match a given list of parameter values::

            for nod in net.nodes(param1=value1, param2=value2, ...):
                ...

        :param properties: key-value pair of node attributes to filter returned nodes
        :return: An iterator of Node objects
        """
        if not self.nodes_built:
            self._build_nodes()

        return NodePool(self, **properties)

    def nodes_iter(self, nids=None):
        raise NotImplementedError

    def edges(self, target_nodes=None, source_nodes=None, target_network=None, source_network=None, **properties):
        """Returns a list of dictionary-like Edge objects, given filter parameters.

        To get all edges from a network::

            edges = net.edges()

        To specify the target and/or source node-set::

            edges = net.edges(target_nodes=net.nodes(type='biophysical'), source_nodes=net.nodes(ei='i'))

        To only get edges with a given edge_property::

          edges = net.edges(weight=100, syn_type='AMPA_Exc2Exc')

        :param target_nodes: gid, list of gid, dict or node-pool. Set of target nodes for a given edge.
        :param source_nodes: gid, list of gid, dict or node-pool. Set of source nodes for a given edge.
        :param target_network: name of network containing target nodes.
        :param source_network: name of network containing source nodes.
        :param properties: edge-properties used to filter out only certain edges.
        :return: list of bmtk.builder.edge.Edge properties.
        """
        def nodes2gids(nodes, network):
            """helper function for converting target and source nodes into list of gids"""
            if nodes is None or isinstance(nodes, list):
                return nodes, network
            if isinstance(nodes, int):
                return [nodes], network
            if isinstance(nodes, dict):
                network = network or self._network_name
                nodes = self._connected_networks[network].nodes(**nodes)
            if isinstance(nodes, NodePool):
                if network is not None and nodes.network_name != network:
                    logger.warning('Nodes and network do not match')
                return [n.node_id for n in nodes], nodes.network_name
            else:
                raise Exception('Couldnt convert nodes')

        def filter_edges(e):
            """Returns true only if all the properities match for a given edge"""
            for k, v in properties.items():
                if k not in e:
                    return False
                if e[k] != v:
                    return False
            return True

        if not self.edges_built:
            self.build()

        # trg_gids can't be none for edges_itr. if target-nodes is not explicity states get all target_gids that
        # synapse onto or from current network.
        if target_nodes is None:
            trg_gid_set = set(n.node_id for cm in self._connection_maps for n in cm.target_nodes)
            target_nodes = sorted(trg_gid_set)

        # convert target/source nodes into a list of their gids
        trg_gids, trg_net = nodes2gids(target_nodes, target_network)
        src_gids, src_net = nodes2gids(source_nodes, source_network)

        # use the iterator to get edges and return as a list
        if properties is None:
            edges = list(self.edges_iter(trg_gids=trg_gids, trg_network=trg_net, src_network=src_net))
        else:
            # filter out certain edges using the properties parameters
            edges = [e for e in self.edges_iter(trg_gids=trg_gids, trg_network=trg_net, src_network=src_net)
                     if filter_edges(e)]

        if src_gids is not None:
            # if src_gids are set filter out edges some more
            edges = [e for e in edges if e.source_gid in src_gids]

        return edges

    def edges_iter(self, trg_gids, src_network=None, trg_network=None):
        """Given a list of target gids, returns a generator for iteratoring over all possible edges.

        It is preferable to use edges() method instead, it allows more flexibility in the input and can better
        indicate if their is a problem.

        The order of the edges returned will be in the same order as the trg_gids list, but does not guarentee any
        secondary ordering by source-nodes and/or edge-type. If their isn't a edge with a matching target-id then
        it will skip that gid in the list, the size of the generator can 0 to arbitrarly large.

        :param trg_gids: list of gids to match with an edge's target.
        :param src_network: str, only returns edges coming from the specified source network.
        :param trg_network: str, only returns edges coming from the specified target network.
        :return: iteration of bmtk.build.edge.Edge objects representing given edge.
        """
        raise NotImplementedError

    def clear(self):
        """Resets the network removing the nodes and edges created."""
        self._nodes_built = False
        self._edges_built = False
        self._clear()

    def _build_nodes(self):
        """Builds or rebuilds all the nodes, clear out both node and edge sets."""
        logger.debug('Building nodes for population {}.'.format(self.name))
        self._clear()
        self._initialize()

        n_node_types = 0
        for ns in self._node_sets:
            nodes = ns.build(nid_generator=self._node_id_gen)
            self._add_nodes(nodes)
            n_node_types += 1

        self._nodes_built = True
        logger.debug('Nodes {} built with {} nodes, {} node-types'.format(self.name, self.nnodes, n_node_types))

    def __build_edges(self):
        """Builds network edges"""
        if not self.nodes_built:
            # only rebuild nodes if necessary.
            self._build_nodes()

        logger.debug('Building edges.')
        # for i, conn_map in enumerate(self._connection_maps):
        for i, conn_map in enumerate(self._connection_maps[mpi_rank::mpi_size]):
            self._add_edges(conn_map, i)

        # exit()
        self._edges_built = True

    def build(self, force=False):
        """Builds nodes and edges.

        :param force: set true to force complete rebuilding of nodes and edges, if nodes() or save_nodes() has been
            called before then forcing a rebuild may change gids of each node.
        """
        # if nodes() or save_nodes() is called by user prior to calling build() - make sure the nodes
        # are completely rebuilt (unless a node set has been added).
        if force:
            self._clear()
            self._initialize()
            self._build_nodes()

        # always build the edges.
        self.__build_edges()

    def __get_path(self, filename, path_dir, ftype):
        if filename is None:
            fname = '{}_{}'.format(self.name, ftype)
            return os.path.join(path_dir, fname)
        elif os.path.isabs(filename):
            return filename
        else:
            return os.path.join(path_dir, filename)

    def save(self, output_dir='.', force_overwrite=True):
        """Used to save the network files in the appropriate (eg SONATA) format into the output_dir directory. The file
        names will be automatically generated based on the network names.

        To have more control over the output and file names use the **save_nodes()** and **save_edges()** methods.

        :param output_dir: string, directory where network files will be generated. Default, current working directory.
        :param force_overwrite: Overwrites existing network files.
        """
        self.save_nodes(output_dir=output_dir, force_overwrite=force_overwrite)
        self.save_edges(output_dir=output_dir, force_overwrite=force_overwrite)

    def save_nodes(self, nodes_file_name=None, node_types_file_name=None, output_dir='.', force_overwrite=True):
        """Save the instantiated nodes in SONATA format files.

        :param nodes_file_name: file-name of hdf5 nodes file. By default will use <network.name>_nodes.h5.
        :param node_types_file_name: file-name of the csv node-types file. By default will use
            <network.name>_node_types.csv
        :param output_dir: Directory where network files will be generated. Default, current working directory.
        :param force_overwrite: Overwrites existing network files.
        """
        nodes_file = self.__get_path(nodes_file_name, output_dir, 'nodes.h5')
        if not force_overwrite and os.path.exists(nodes_file):
            raise Exception('File {} already exists. Please delete existing file, use a different name, or use force_overwrite.'.format(nodes_file))
        nf_dir = os.path.dirname(nodes_file)
        if not os.path.exists(nf_dir) and mpi_rank == 0:
            os.makedirs(nf_dir)
        barrier()

        node_types_file = self.__get_path(node_types_file_name, output_dir, 'node_types.csv')
        if not force_overwrite and os.path.exists(node_types_file):
            raise Exception('File {} exists. Please use different name or use force_overwrite'.format(node_types_file))
        ntf_dir = os.path.dirname(node_types_file)
        if not os.path.exists(ntf_dir) and mpi_rank == 0:
            os.makedirs(ntf_dir)
        barrier()

        self._save_nodes(nodes_file)
        self._save_node_types(node_types_file)

    def _save_nodes(self, nodes_file_name):
        raise NotImplementedError

    def _save_node_types(self, node_types_file_name):
        if mpi_rank == 0:
            logger.debug('Saving {} node-types to {}.'.format(self.name, node_types_file_name))

            node_types_cols = ['node_type_id'] + [col for col in self._node_types_columns if col != 'node_type_id']
            with open(node_types_file_name, 'w') as csvfile:
                csvw = csv.writer(csvfile, delimiter=' ')
                csvw.writerow(node_types_cols)
                for node_type in self._node_types_properties.values():
                    csvw.writerow([node_type.get(cname, 'NULL') for cname in node_types_cols])
        barrier()

    def import_nodes(self, nodes_file_name, node_types_file_name):
        raise NotImplementedError

    def save_edges(self, edges_file_name=None, edge_types_file_name=None, output_dir='.', src_network=None,
                   trg_network=None, name=None, force_build=True, force_overwrite=False):
        """Save the instantiated edges in SONATA format files.

        :param edges_file_name: file-name of hdf5 edges file. By default will use <src_network>_<trg_network>_edges.h5.
        :param edge_types_file_name: file-name of csv edge-types file. By default will use
            <src_network>_<trg_network>_edges.h5.
        :param output_dir: Directory where network files will be generated. Default, current working directory.
        :param src_network: Name of the source-node populations.
        :param trg_network: Name of the target-node populations.
        :param name: Name of file.
        :param force_build: Force to (re)build the connection matrix if it hasn't already been built.
        :param force_overwrite: Overwrites existing network files.
        """
        # Make sure edges exists and are built
        if len(self._connection_maps) == 0:
            logging.warning('No edges have been made for this network, skipping saving of edges file.')
            return

        if self._edges_built is False:
            if force_build:
                self.__build_edges()
            else:
                logger.warning("Edges are not built. Either call build() or use force_build parameter. Skip saving of edges file.")
                return

        network_params = [(s, t, s+'_'+t+'_edges.h5', s+'_'+t+'_edge_types.csv') for s, t in list(self._network_conns)]
        if src_network is not None:
            network_params = [p for p in network_params if p[0] == src_network]

        if trg_network is not None:
            network_params = [p for p in network_params if p[1] == trg_network]

        if len(network_params) == 0:
            logger.warning("Warning: couldn't find connections. Skip saving.")
            return

        if (edges_file_name or edge_types_file_name) is not None:
            network_params = [(network_params[0][0], network_params[0][1], edges_file_name, edge_types_file_name)]

        if not os.path.exists(output_dir) and mpi_rank == 0:
            os.mkdir(output_dir)
        barrier()

        self._save_gap_junctions(os.path.join(output_dir, self._network_name + '_gap_juncs.h5'))

        for p in network_params:
            if p[3] is not None:
                self._save_edge_types(os.path.join(output_dir, p[3]), p[0], p[1])

            if p[2] is not None:
                self._save_edges(os.path.join(output_dir, p[2]), p[0], p[1], name)

    def _save_edge_types(self, edge_types_file_name, src_network, trg_network):
        if mpi_rank == 0:
            # Get edge-type properties for connections with matching source/target networks
            matching_et = [c.edge_type_properties for c in self._connection_maps
                           if c.source_network_name == src_network and c.target_network_name == trg_network]

            # Get edge-type properties that are only relevant for this source-target network pair
            cols = ['edge_type_id', 'target_query', 'source_query']  # manditory and should come first
            merged_keys = [k for et in matching_et for k in et.keys() if k not in cols]
            cols += list(set(merged_keys))

            # Write to csv
            with open(edge_types_file_name, 'w') as csvfile:
                csvw = csv.writer(csvfile, delimiter=' ')
                csvw.writerow(cols)
                for edge_type in matching_et:
                    csvw.writerow([edge_type.get(cname, 'NULL') if edge_type.get(cname, 'NULL') is not None else 'NULL'
                                   for cname in cols])

        barrier()

    def _save_edges(self, edges_file_name, src_network, trg_network, pop_name=None, **opts):
        raise NotImplementedError

    def _save_gap_junctions(self, gj_file_name):
        raise NotImplementedError

    def _initialize(self):
        raise NotImplementedError

    def _add_nodes(self, node_tuples):
        raise NotImplementedError

    def _add_edges(self, edge_tuples, i):
        raise NotImplementedError

    def _clear(self):
        raise NotImplementedError
