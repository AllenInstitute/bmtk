from .network_adaptors.dm_network import DenseNetwork


"""
NetworkBuilder is just a slim container for the a NetworkAdaptor class object which does all the work of storing/saving
the network. Should make it easier to 
 1) Create/update the backend network-builder strategies on the fly by swapping out different network-adaptors
 2) Change the public interface/documentation without having to mess with the adaptor, see BioNetBuilder, 
    PointNetBuilder, etc.
"""


class NetworkBuilder(object):
    """The NetworkBuilder class is used for building and saving a brain network/circuit. By default it will save to
    SONATA format for running network simulations using BioNet, PointNet, PopNet or FilterNet bmtk modules, however it
    can be generalized to build any time of network for any time of simulation.

    Building the network:
        For the general use-case building a network consists of 4 steps, each with a corresponding method.

        1. Initialize the network::

            net = NetworkBuilder("network_name")

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

    def __init__(self, name, adaptor_cls=DenseNetwork, **network_props):
        self.adaptor = adaptor_cls(name, **network_props)

    @property
    def name(self):
        """Get the name (string) of this network."""
        return self.adaptor.name

    @property
    def nodes_built(self):
        """Returns True if nodes has been instantiated for this network."""
        return self.adaptor.nodes_built

    @property
    def edges_built(self):
        """Returns True if the connectivity matrix has been instantiated for this network."""
        return self.adaptor.edges_built

    @property
    def nnodes(self):
        """Returns the number of nodes for this network."""
        return self.adaptor.nnodes

    @property
    def nedges(self):
        """Returns the total number of edges for this network."""
        return self.adaptor.nedges

    def add_nodes(self, N=1, **properties):
        """Used to add nodes (eg cells) to a network. User should specify the number of Nodes (N) and can use any
        properties/attributes they require to define the nodes. By default all individual cells will be assigned a
        unique 'node_id' to identify each node in the network and a 'node_type_id' to identify each group of nodes.

        If a property is a singluar value then said property will be shared by all the nodes in the group. If a value
        is a list of length N then each property will be uniquly assigned the each node. In the below example a group
        of 100 nodes is created, all share the same 'model_type' parameter but the pos_x values will be different for
        each node::

            net.add_nodes(N=100, pos_x=np.random.rand(100), model_type='intfire1', ...)

        You can use a tuple to store property values (in which the SONATA hdf5 will save it as a dataset with multiple
        columns). For example to have one property 'positions' which keeps track of the x/y/z coordinates of each cell::

            net.add_nodes(N=100, positions=[(rand(), rand(), rand()) for _ in range(100)], ...)

        :param N: number of nodes in this group
        :param properties: Individual and group properties of given nodes
        """
        self.adaptor.add_nodes(N=N, **properties)

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
        return self.adaptor.nodes(**properties)

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
                    connection_rule=N,
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

            There is also a 'one_to_all' iterator option that will pair each source node with a list of all available
            target nodes.

        Edge Properties:
            Normally the properties used when creating a given type of edge will be shared by all the individual
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
            each target, and 'one-to-all' will pass in a list of all possible targets for each source.
        :param edge_type_properties: properties/attributes of the given edge type
        :return: A ConnectionMap object
        """
        return self.adaptor.add_edges(
            source=source,
            target=target,
            connection_rule=connection_rule,
            connection_params=connection_params,
            iterator=iterator,
            **edge_type_properties
        )

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
        return self.adaptor.edges(
            target_nodes=target_nodes,
            source_nodes=source_nodes,
            target_network=target_network,
            source_network=source_network,
            **properties
        )

    def build(self, force=False):
        """Builds nodes and edges.

        :param force: set true to force complete rebuilding of nodes and edges, if nodes() or save_nodes() has been
            called before then forcing a rebuild may change gids of each node.
        """
        self.adaptor.build(force=force)

    def save(self, output_dir='.', force_overwrite=True):
        """Used to save the network files in the appropriate (eg SONATA) format into the output_dir directory. The file
        names will be automatically generated based on the network names.

        To have more control over the output and file names use the **save_nodes()** and **save_edges()** methods.

        :param output_dir: string, directory where network files will be generated. Default, current working directory.
        :param force_overwrite: Overwrites existing network files.
        """
        self.adaptor.save(output_dir=output_dir, force_overwrite=force_overwrite)

    def save_nodes(self, nodes_file_name=None, node_types_file_name=None, output_dir='.', force_overwrite=True):
        """Save the instantiated nodes in SONATA format files.

        :param nodes_file_name: file-name of hdf5 nodes file. By default will use <network.name>_nodes.h5.
        :param node_types_file_name: file-name of the csv node-types file. By default will use
            <network.name>_node_types.csv
        :param output_dir: Directory where network files will be generated. Default, current working directory.
        :param force_overwrite: Overwrites existing network files.
        """
        self.adaptor.save_nodes(
            nodes_file_name=nodes_file_name,
            node_types_file_name=node_types_file_name,
            output_dir=output_dir,
            force_overwrite=force_overwrite
        )

    def save_edges(self, edges_file_name=None, edge_types_file_name=None, output_dir='.', src_network=None,
                   trg_network=None, name=None, force_build=True, force_overwrite=False):
        """Save the instantiated edges in SONATA format files.

        :param edges_file_name: file-name of hdf5 edges file. By default will use <src_network>_<trg_network>_edges.h5.
        :param edge_types_file_name: file-name of csv edge-types file. By default will use
            <src_network>_<trg_network>_edges.h5.
        :param output_dir: Directory where network files will be generated. Default, current working directory.
        :param src_network: Name of the source-node populations.
        :param trg_network: Name of the target-node populations.
        :param name: Name of edge populations, eg /edges/<name> in edges.h5 file.
        :param force_build: Force to (re)build the connection matrix if it hasn't already been built.
        :param force_overwrite: Overwrites existing network files.
        """
        self.adaptor.save_edges(
            edges_file_name=edges_file_name,
            edge_types_file_name=edge_types_file_name,
            output_dir=output_dir,
            src_network=src_network,
            trg_network=trg_network,
            name=name,
            force_build=force_build,
            force_overwrite=force_overwrite
        )

    def import_nodes(self, nodes_file_name, node_types_file_name):
        """Import nodes from an existing sonata file

        :param nodes_file_name:
        :param node_types_file_name:
        """
        self.adaptor.import_nodes(
            nodes_file_name=nodes_file_name,
            node_types_file_name=node_types_file_name
        )

    def clear(self):
        """Resets the network removing the nodes and edges created."""
        self.adaptor.clear()

    def __getattr__(self, item):
        """Catch-all for adaptor attributes"""
        return getattr(self.adaptor, item)


class BioNetBuilder(NetworkBuilder):
    """An instance of NetworkBuilder specifically designed for building BioNet (NEURON) complainant networks

    """
    def __init__(self, name, adaptor_cls=DenseNetwork, **network_props):
        super(BioNetBuilder, self).__init__(name=name, adaptor_cls=adaptor_cls, **network_props)

    def add_nodes(self, N=1, model_type=None, model_template=None, model_processing=None, dynamics_params=None,
                  morphology=None, x=None, y=None, z=None, rotation_angle_xaxis=None, rotation_angle_yaxis=None,
                  rotation_angle_zaxis=None, **properties):
        """Add a set of N NEURON type nodes/cells to the network

        :param N: number of nodes (cells) in this group
        :param model_type: the type of node, 'biophysical', 'virtual', 'single_compartment', or 'point_neuron'
        :param model_template: Template or class used to instantiate all instances of each cell, eg. a NeuroML or
            NEURON Hoc Template file. Should contain a prefix of the format type ('nml:<Cell.nml>' or
            'ctdb:Biophys1.hoc'
        :param model_processing: string, pre-defined or user function applied to instantiated cell property/morphology,
            if using Allen Cell-Type Database biophysical models use 'aibs_perisomatic' or 'aibs_allactive'.
        :param dynamics_params: A file-name or a dictionary of cell-dynamics parameter values used for generating the
            cell. If using models from Allen Cell-Type Database then passing in the name of <model>_fit.json file will
            apply the same model params to all N cells. If you want unique params for each cell pass in a dictionary
            where each key has an associated list of size N::

                dynamics_params={'g_bar': [0.5, 0.2, ...], 'na_bar': [1.653e-5, 9.235e-6, ...]}

        :param morphology: string, name of morphology swc file used for creating model_type=biophysical cells
        :param x: list/array of size N floats for each cell's x-axis soma position
        :param y: list/array of size N floats for each cell's y-axis soma position
        :param z: list/array of size N floats for each cell's z-axis soma position
        :param rotation_angle_xaxis: x-axis euler rotation angle around the soma in radians
        :param rotation_angle_yaxis: y-axis euler rotation angle around the soma in radians
        :param rotation_angle_zaxis: z-axis euler rotation angle around the soma in radians
        :param properties: Individual and group properties of given nodes
        """
        for arg_var, arg_val in locals().items():
            if arg_var in ['properties', 'self', 'N']:
                continue
            elif arg_val is not None:
                properties[arg_var] = arg_val

        super(BioNetBuilder, self).add_nodes(N=N, **properties)

    def add_edges(self, source=None, target=None, connection_rule=1, connection_params=None, iterator='one_to_one',
                  model_template=None, dynamics_params=None, syn_weight=None, delay=None, target_sections=None,
                  distance_range=None, **properties):
        """Add rules for creating edges between a subset of source and target cells.

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
            each target, and 'one-to-all' will pass in a list of all possible targets for each source.
        :param model_template: A predefined or user generated NEURON function name for generating connection. (default:
            exp2syn).
        :param dynamics_params: A json file path or a dictionary of edge/synapse dynamics parameter values used for
            generating the connection, according to the model_template function. If passing in the name or path of a
            json file the parameter values will be applied to a edges.
        :param syn_weight: float or list of N floats, weights applied to each edge/synpase, value dependent on model
        :param delay: float or list of N floats, delay in milliseconds for cell-cell events
        :param target_sections: list of strings, non-SONATA compliant, locations of where to place synpases when the
            target cell is model_type=biophysical. One or more of the following values: somatic, basal, apical, axon.
            If specified post-synaptic location will be randomly chosen from given sections (+ distance_range value).
            To assign specific locations use afferent_section_id/afferent_section_pos along with the ConnectionMap
            instance returned.
        :param distance_range: A numeric range of floats [beg, end]. non SONATA compliant. Start and end arc-length
            distance from SOMA where the post-synaptic location will be placed on a model_type=biophysical cell. Along
            with target_sections parameter can be used to randomly but strategically connection cells. To assign
            specific locations use afferent_section_id/afferent_section_pos along with the returned ConnectionMap.
        :param properties: properties/attributes of the given edge type
        :return: A ConnectionMap object
        """
        for arg_var, arg_val in locals().items():
            if arg_var in ['properties', 'source', 'target', 'connection_rule', 'connection_params', 'iterator']:
                continue
            elif arg_val is not None:
                properties[arg_var] = arg_val

        return super(BioNetBuilder, self).add_edges(
            source=source,
            target=target,
            connection_rule=connection_rule,
            connection_params=connection_params,
            iterator=iterator,
            **properties
        )


class PointNetBuilder(NetworkBuilder):
    """An instance of NetworkBuilder specifically designed for building PointNet (NEST) complainant networks

    """
    def __init__(self, name, adaptor_cls=DenseNetwork, **network_props):
        super(PointNetBuilder, self).__init__(name=name, adaptor_cls=adaptor_cls, **network_props)

    def add_nodes(self, N=1, model_type='point_neuron', model_template=None, dynamics_params=None, x=None, y=None,
                  z=None, **properties):
        """Add a set of N NEST model nodes/cells to the network.

        :param N:
        :param model_type: The type of node, should be value 'point_neuron'
        :param model_template: Name of NEST neuron model to used, should have a 'nest:' prefix; nest:iaf_psc_alpha,
            nest:glif_asc
        :param dynamics_params: A file-name or a dictionary of cell-dynamics parameter values used for generating the
            cell, with parameters matching that of the parameters used to initialize NEST model_template cell. If you
            want unique params for each cell pass in a dictionary where each key has an associated list of size N::
                
                dynamics_params={'v_init': [-90, -85, ...], 'v_reset': [-20, -20, ...]}
                
        :param x: list/array of size N floats for each cell's x-axis position
        :param y: list/array of size N floats for each cell's y-axis position
        :param z: list/array of size N floats for each cell's z-axis position
        :param properties: Individual and group properties of given nodes
        """
        for arg_var, arg_val in locals().items():
            if arg_var in ['properties', 'self', 'N']:
                continue
            elif arg_val is not None:
                properties[arg_var] = arg_val

        super(PointNetBuilder, self).add_nodes(N=N, **properties)

    def add_edges(self, source=None, target=None, connection_rule=1, connection_params=None, iterator='one_to_one',
                  model_template=None, dynamics_params=None, syn_weight=None, delay=None, **properties):
        """Add rules for creating edges between a subset of source and target cells.

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
            each target, and 'one-to-all' will pass in a list of all possible targets for each source.
        :param model_template: A predefined or user generated NEURON function name for generating connection. (default:
            exp2syn).
        :param dynamics_params: A json file path or a dictionary of edge/synapse dynamics parameter values used for
            generating the connection, according to the model_template function. If passing in the name or path of a
            json file the parameter values will be applied to a edges.
        :param syn_weight: float or list of N floats, weights applied to each edge/synpase, value dependent on model
        :param delay: float or list of N floats, delay in milliseconds for cell-cell events
        :param properties: properties/attributes of the given edge type
        :return: A ConnectionMap object
        """
        for arg_var, arg_val in locals().items():
            if arg_var in ['properties', 'source', 'target', 'connection_rule', 'connection_params', 'iterator']:
                continue
            elif arg_val is not None:
                properties[arg_var] = arg_val

        return super(PointNetBuilder, self).add_edges(
            source=source,
            target=target,
            connection_rule=connection_rule,
            connection_params=connection_params,
            iterator=iterator,
            **properties
        )


class PopNetBuilder(NetworkBuilder):
    """An instance of NetworkBuilder specifically designed for building PopNet (DiPDE) complainant networks

    """
    def __init__(self, name, adaptor_cls=DenseNetwork, **network_props):
        super(PopNetBuilder, self).__init__(name=name, adaptor_cls=adaptor_cls, **network_props)

    def add_nodes(self, model_type='population', model_template=None, dynamics_params=None, **properties):
        """Add a DiPDE cell population

        :param model_type: The type of node, should be value 'population'
        :param model_template: Should be either 'dipde:Internal' or 'dipde:External' (or 'virtual')
        :param dynamics_params: path to json file, or dictionary, of parameters for dipde population
        :param properties: Individual and group properties of given nodes
        """
        for arg_var, arg_val in locals().items():
            if arg_var in ['properties', 'self']:
                continue
            elif arg_val is not None:
                properties[arg_var] = arg_val

        super(PopNetBuilder, self).add_nodes(N=1, **properties)

    def add_edges(self, source=None, target=None, nsyns=1, syn_weight=None, delay=None, dynamics_params=None,
                  **properties):
        """

        :param source: A dictionary or list of efferent dipde Populations
        :param target: A dictionary or list of afferent dipde Populations
        :param nsyns: Number of connections between the populations
        :param syn_weight: connection weight between the populations
        :param delay: connection delay between the populations
        :param dynamics_params: Additional parameters used for initializing connection
        :param properties: properties/attributes of the given edge type
        :return: A ConnectionMap object
        """
        for arg_var, arg_val in locals().items():
            if arg_var in ['properties', 'source', 'target']:
                continue
            elif arg_val is not None:
                properties[arg_var] = arg_val

        return super(PopNetBuilder, self).add_edges(
            source=source,
            target=target,
            **properties
        )
