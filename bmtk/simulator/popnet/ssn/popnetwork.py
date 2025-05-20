import numpy as np

from bmtk.simulator.core.simulator_network import SimNetwork
from .popnode import SSNNode

class PopNetwork(SimNetwork):
    def __init__(self, grouping_key='node_id', **opts):
        super(PopNetwork, self).__init__()
        # self.n_neu_total = 6
        self.grouping_key = grouping_key

        # self._pop_index = {}

        self._node_id_map = {}

        self._nnodes_recurrent = 0
        self._nnodes_external = 0

        
        self._scales = None
        self._exponents = None
        self._decay_constants = None
        self._initial_states = None

        self._nodeid2grp = {}
        self._nodes_idx = {}
        self._ssn_recurrent_nodes = set()
        self._ssn_external_nodes = set()
        self.gids = 0

        # Not the actual connection matrix used during sim, used for storing connections while building network
        self._conn_mat = [] 

        # The actual N x (N+M) connection matrix used during simulation, created before simulation begins
        self._connectivity_mat = None

        # Keeps track if connection matrix needs to be rebuilt (mainly if new nodes/edges are added after a simulation)
        self._conn_finalized = False

    @property 
    def network_finalized(self):
        return self._conn_finalized

    @property
    def target_simulator(self):
        return 'SSN'

    @property
    def n_neu_recurrent(self):
        return self._nnodes_recurrent

    @property
    def n_neu_total(self):
        return self._nnodes_recurrent + self._nnodes_external

    @property
    def connectivity_mat(self):
        if self._connectivity_mat is None or not self._conn_finalized:
            self._conn_finalized = True
            self._connectivity_mat = np.zeros((self._nnodes_recurrent, self.n_neu_total), dtype=float)
            for r, c, syn_w in self._conn_mat:
                self._connectivity_mat[r, c] = syn_w

        return self._connectivity_mat
    
    @connectivity_mat.setter
    def connectivity_mat(self, value):
        self._connectivity_mat = value
        self._conn_finalized = True

    @property
    def scales(self):
        if self._scales is None or not self._conn_finalized:
            self._scales = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._scales[n.gid] = np.mean(n.scaling_coef)
        
        return self._scales
    
    @scales.setter
    def scales(self, value):
        self._scales = value

    @property
    def initial_states(self):
        if self._initial_states is None or not self._conn_finalized:
            self._initial_states = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._initial_states[n.gid] = np.mean(n.initial_value)
        
        return self._initial_states

    @property
    def exponents(self):
        if self._exponents is None or not self._conn_finalized:
            self._exponents = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._exponents[n.gid] = np.mean(n.exponent)
        
        return self._exponents
    
    @exponents.setter
    def exponents(self, value):
        self._exponents = value

    @property
    def decay_constants(self):
        if self._decay_constants is None or self._conn_finalized:
            self._decay_constants = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._decay_constants[n.gid] = np.mean(n.decay_const)
        
        return self._decay_constants
    
    @decay_constants.setter
    def decay_constants(self, value):
        self._decay_constants = value

    def build_nodes(self):
        for node_pop in self.node_populations:
            for node in node_pop.get_nodes():
                model_type = node['model_type'].lower()
                
                if model_type in ['population', 'rate_population', 'recurrent']:
                    params = node.dynamics_params if node.dynamics_params is not None else {}
                    ssn_attrs = ['scaling_coef', 'initial_value', 'exponent', 'decay_const']
                    for attr_name in ssn_attrs:
                        if attr_name in node:
                            params[attr_name] = node[attr_name]

                    params['node'] = node
                    self.add_recurrent_node(
                        population_id=node_pop.name, 
                        node_id=node['node_id'], 
                        **params
                    )
                
                elif model_type in ['external', 'virtual']:
                    self.add_external_node(
                        population_id=node_pop.name, 
                        node_id=node[self.grouping_key],
                        node=node
                    )

    def build_edges(self):
        self._conn_finalized = False
        for edge_pop in self._edge_populations:
            for edge in edge_pop.get_edges():
                src_node = self._node_id_map[edge.source_population][int(edge.source_node_id)]
                trg_node = self._node_id_map[edge.target_population][int(edge.target_node_id)]
                
                # TODO: Move to add_edge function
                self._conn_mat.append([trg_node.gid, src_node.gid, edge['syn_weight']])


    def get_node(self, population_id, node_id):
        return self._node_id_map[population_id][node_id]

    def get_ssn_node(self, population_id, node_id, **node_properties):
        if population_id not in self._node_id_map:
            ssn_node = SSNNode(population_id, node_id, gid=self.gids, **node_properties)
            self._node_id_map[population_id] = {int(node_id): ssn_node}
            self.gids += 1
        
        elif int(node_id) not in self._node_id_map:
            ssn_node = SSNNode(population_id, node_id, gid=self.gids, **node_properties)
            self._node_id_map[population_id][int(node_id)] = ssn_node
            self.gids += 1

        else:
            ssn_node = self._node_id_map[population_id][node_id]

        return ssn_node

    def add_recurrent_node(self, population_id, node_id, scaling_coef, exponent, decay_const, initial_value=0.0, **node_properties):
        self._conn_finalized = False
        self._nnodes_recurrent += 1       
        
        ssn_obj = self.get_ssn_node(population_id=population_id, node_id=node_id, **node_properties)
        ssn_obj.type='internal'
        # ssn_obj.input_offset.append(input_offset)
        ssn_obj.scaling_coef.append(scaling_coef)
        ssn_obj.exponent.append(exponent)
        ssn_obj.decay_const.append(decay_const)
        self._ssn_recurrent_nodes.add(ssn_obj)
        
    def add_external_node(self, population_id, node_id, **node_properties):
        self._conn_finalized = False
        self._nnodes_external += 1
        
        ssn_obj = self.get_ssn_node(population_id=population_id, node_id=node_id, **node_properties)
        ssn_obj.type = 'external'
        self._ssn_external_nodes.add(ssn_obj)
