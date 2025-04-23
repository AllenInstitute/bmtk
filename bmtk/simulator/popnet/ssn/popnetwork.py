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

        self._connectivity_mat = None
        self._scales = None
        self._exponents = None
        self._decay_constants = None
        self._initial_states = None

        self._nodeid2grp = {}

        # self._recurrent_nodes = {}
        # self._external_nodes = {}


        self._nodes_idx = {}
        self._ssn_recurrent_nodes = set()
        self._ssn_external_nodes = set()
        self.gids = 0

        self._conn_mat = []
        # sefl._connectivity

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
        if self._connectivity_mat is None:
            self._connectivity_mat = np.zeros((self._nnodes_recurrent, self.n_neu_total), dtype=float)
            for r, c, syn_w in self._conn_mat:
                # print(r, c, syn_w)
                self._connectivity_mat[r, c] = syn_w

            # print(self._connectivity_mat)

        return self._connectivity_mat


    @property
    def scales(self):
        if self._scales is None:
            self._scales = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._scales[n.gid] = np.mean(n.scaling_coef)
        
        return self._scales

    @property
    def initial_states(self):
        if self._initial_states is None:
            self._initial_states = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._initial_states[n.gid] = np.mean(n.initial_value)
        
        return self._initial_states

    @property
    def exponents(self):
        if self._exponents is None:
            self._exponents = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._exponents[n.gid] = np.mean(n.exponent)
        
        return self._exponents

    @property
    def decay_constants(self):
        if self._decay_constants is None:
            self._decay_constants = np.zeros(self._nnodes_recurrent)
            for n in self._ssn_recurrent_nodes:
                self._decay_constants[n.gid] = np.mean(n.decay_const)

            # print(self._decay_constants)
        
        return self._decay_constants

    def build_nodes(self):
        for node_pop in self.node_populations:
            for node in node_pop.get_nodes():
                model_type = node['model_type'].lower()
                
                if model_type in ['population', 'rate_population', 'recurrent']:
                    self.add_recurrent_node(
                        population_id=node_pop.name, 
                        node_id=node['node_id'], 
                        input_offset=node['input_offset'], 
                        scaling_coef=node['scaling_coef'], 
                        exponent=node['exponent'], 
                        decay_const=node['decay_const'],
                        initial_value=node.get['initial_value'] if 'initial_value' in node else 0.0,
                        node=node
                    )
                
                elif model_type in ['external', 'virtual']:
                    self.add_external_node(
                        population_id=node_pop.name, 
                        node_id=node[self.grouping_key],
                        node=node
                    )


    def build_edges(self):
        for edge_pop in self._edge_populations:
            for edge in edge_pop.get_edges():
                # print(edge.source_population, edge.source_node_id, edge.target_population, edge.target_node_id, edge['syn_weight'])
                # print(edge.source_population in self._node_id_map)
                # print(list(self._node_id_map[edge.source_population].keys()))
                # print(edge.source_node_id in self._node_id_map[edge.source_population])
                # print(self.get_ssn_node(edge.source_population, edge.source_node_id).gid)
                src_node = self._node_id_map[edge.source_population][int(edge.source_node_id)]
                trg_node = self._node_id_map[edge.target_population][int(edge.target_node_id)]
                
                self._conn_mat.append([trg_node.gid, src_node.gid, edge['syn_weight']])


    def get_node(self, population_id, node_id):
        return self._node_id_map[population_id][node_id]


    def get_ssn_node(self, population_id, node_id, **node_properties):
        if population_id not in self._node_id_map:
            # print('new population')
            ssn_node = SSNNode(population_id, node_id, gid=self.gids, **node_properties)
            self._node_id_map[population_id] = {int(node_id): ssn_node}
            self.gids += 1
        
        elif int(node_id) not in self._node_id_map:
            # print(f'new node {population_id}, {node_id}')
            ssn_node = SSNNode(population_id, node_id, gid=self.gids, **node_properties)
            self._node_id_map[population_id][int(node_id)] = ssn_node
            self.gids += 1

        else:
            # print('found')
            ssn_node = self._node_id_map[population_id][node_id]

        return ssn_node


    def add_recurrent_node(self, population_id, node_id, input_offset, scaling_coef, exponent, decay_const, initial_value=0.0, **node_properties):
        self._nnodes_recurrent += 1       
        
        ssn_obj = self.get_ssn_node(population_id=population_id, node_id=node_id, **node_properties)
        ssn_obj.type='internal'
        ssn_obj.input_offset.append(input_offset)
        ssn_obj.scaling_coef.append(scaling_coef)
        ssn_obj.exponent.append(exponent)
        ssn_obj.decay_const.append(decay_const)
        self._ssn_recurrent_nodes.add(ssn_obj)
        

    def add_external_node(self, population_id, node_id, **node_properties):
        self._nnodes_external += 1
        ssn_obj = self.get_ssn_node(population_id=population_id, node_id=node_id, **node_properties)
        ssn_obj.type = 'external'
        self._ssn_external_nodes.add(ssn_obj)
