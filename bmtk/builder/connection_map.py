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
from . import connector
from . import iterator


class ConnectionMap(object):
    """Class for keeping track of connection rules. Used for saving unique edge properties in a network

    For every connection from source --> target this keeps track of rules (functions, literals, lists) for
      1. the number of synapses between source and target
      2. Used defined parameters (syn-weight, synaptic-location) for every synapse.

    The number of synapses rule (1) is stored as a connector. Individual synaptic parameters, if they exists, are stored
    as ParamsRules.

    Usage:
        Do not create a ConnectionMap object directly. Rather one is created and return every time NetworkBuilder's
        add_edges() method is called.

        For example, The following will create a set of edges where there is between 1 to 16 synapatic connections
        between each source and target node. Properties 'dynamics_params' and 'model_template' will be global
        properties::

            cm = net.add_edges(source={'ei': 'exc'}, target={'ei': 'inh'},
                               connection_rule=lambda *_: np.random.randn(1, 16),
                               dynamics_params='ExcToInh.json',
                               model_template='Exp2Syn',
                               ...)

        If we want all edges to have a unique 'syn_weight' value calculated from source/target locations, we create a
        function that takes in source and target (Node object) and returns the syn weight. This function will be called
        for every individual connection::

            def syn_weight_by_loc(source, target):
                src_loc, trg_loc = source['loc'], target['loc']
                ...
                return syn_weight

            cm.add_properties('syn_weight', rule=syn_weight_by_loc, dtypes=np.float)

        We can also add multiple properties at the same time::

            def get_target_loc_and_dist(source, target):
                ...
                return trg_loc, trg_dist

            cm.add_properties(['syn_location', 'syn_distance'],
                          rule=get_target_loc_and_dist,
                          dtypes=[string, np.float])

        Once NetworkBuilder.build() is called the 'connection_rule' will be called.
    """

    class ParamsRules(object):
        """A subclass to store indvidiual synpatic parameter rules"""
        def __init__(self, names, rule, rule_params, dtypes):
            self._names = names
            self._rule = rule
            self._rule_params = rule_params
            self._dtypes = self.__create_dtype_dict(names, dtypes)

        def __create_dtype_dict(self, names, dtypes):
            if isinstance(names, list):
                # TODO: compare size of names and dtypes
                return {n: dt for n, dt in zip(names, dtypes)}
            else:
                return {names: dtypes}

        @property
        def names(self):
            return self._names

        @property
        def rule(self):
            return connector.create(self._rule, **(self._rule_params or {}))

        @property
        def dtypes(self):
            return self._dtypes

        def get_prop_dtype(self, prop_name):
            return self._dtypes[prop_name]

    def __init__(self, sources=None, targets=None, connector=None, connector_params=None, iterator='one_to_one',
                 edge_type_properties=None):
        self._source_nodes = sources  # source nodes
        self._target_nodes = targets  # target nodes
        self._connector = connector  # function, list or value that determines connection between sources and targets
        self._connector_params = connector_params  # parameters passed into connector
        self._iterator = iterator  # rule for iterating between sources and targets
        self._edge_type_properties = edge_type_properties

        self._params = []
        self._param_keys = []

    @property
    def params(self):
        return self._params

    @property
    def source_nodes(self):
        return self._source_nodes

    @property
    def source_network_name(self):
        return self._source_nodes.network_name

    @property
    def target_nodes(self):
        return self._target_nodes

    @property
    def target_network_name(self):
        return self._target_nodes.network_name

    @property
    def connector(self):
        return self._connector

    @property
    def connector_params(self):
        return self._connector_params

    @property
    def iterator(self):
        return self._iterator

    @property
    def edge_type_properties(self):
        return self._edge_type_properties or {}

    @property
    def edge_type_id(self):
        # TODO: properly implement edge_type
        return self._edge_type_properties['edge_type_id']

    @property
    def property_names(self):
        if len(self._param_keys) == 0:
            return ['nsyns']
        else:
            return self._param_keys

    def properties_keys(self):
        ordered_keys = sorted(self.property_names)
        return str(ordered_keys)

    def max_connections(self):
        return len(self._source_nodes) * len(self._target_nodes)

    def add_properties(self, names, rule, rule_params=None, dtypes=None):
        """Add a synaptic property for an individual edge.

        Typically this requires a custom rule that will be used for every source/target pair of nodes and returns
        a value for the property. Such rule is a user defined function, and will be called by NetworkBuilder.build()::

            def syn_weight_by_loc(source, target):
                ...
                return syn_weight

            cm.add_properties('syn_weight', rule=syn_weight_by_loc, dtypes=np.float)

        If the connection_rule function requires additional parameters use the rule_params option::

            def syn_weight_by_loc(source, target, min_weight, max_weight):
                ...
                return syn_weight

            cm.add_properties('syn_weight',
                              rule=syn_weight_by_loc,
                              rule_params={'min_weigth':1.0e-7, 'max_weight': 5.0e-6},
                              dtypes=np.float)

        You may also have the rule define multiple connection parameters at one time::

            def conn_rule(source, target, min_weight, max_weight):
                ...
                return syn_weight, syn_location, delay

            cm.add_properties(['syn_weight', 'syn_loc', 'delay'],
                              rule=syn_weight_by_loc,
                              rule_params={'min_weigth':1.0e-7, 'max_weight': 5.0e-6},
                              dtypes=[np.float, str, np.float])

        :param names: list, or single string, of the property
        :param rule: function, list or value of property
        :param rule_params: when rule is a function, rule_params will be passed into function when called.
        :param dtypes: expected property type
        """
        self._params.append(self.ParamsRules(names, rule, rule_params, dtypes))
        self._param_keys += names

    def connection_itr(self):
        """Returns a generator that will iterate through the source/target pairs (as specified by the iterator function,
        and create a connection rule based on the connector.
        """
        conr = connector.create(self.connector, **(self.connector_params or {}))
        itr = iterator.create(self.iterator, conr, **({}))
        return itr(self.source_nodes, self.target_nodes, conr)
