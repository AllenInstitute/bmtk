from bmtk.simulator import ssn
import numpy as np
from numba import njit
from six import string_types
from pprint import pprint
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

from bmtk.simulator.core.simulation_config import SimulationConfig
from bmtk.simulator.core.simulator_network import SimNetwork
from bmtk.simulator.core import sonata_reader
import bmtk.simulator.utils.simulation_inputs as inputs


class RatesRecorderMod:
    def __init__(self, name, output_type, module, file_name, **kwargs):
        self._name = name
        self._output_type = output_type
        self._module = module
        self._params = kwargs
        self.file_name = file_name
        self._output_dir = kwargs.get('output_dir', '.')
        self._node_set = kwargs.get('cells', None)

        self.file_path = Path(self.file_name)
        if not self.file_path.is_absolute():
            self.file_path = (Path(self._output_dir) / self.file_path).absolute()

        if self._output_type not in ['csv', 'h5']:
            raise ValueError(f'{self.__name__}: Invalid output_type "{self._output_type}". [Valid options: csv, h5]')

    def initialize(self, sim):
        pass

    def finalize(self, sim):
        if self._output_type == 'csv':
            self._to_csv(sim)
        elif self._output_type == 'h5':
            self._to_hdf5(sim)

    def _get_nodes(self, sim):
        if self._node_set is not None:
            node_set = sim.network.get_node_set(self._node_set)
            return [sim.network.get_node(n.population_name, n.node_id) for n in node_set.fetch_nodes()]
        else:
            return list(sim.network._ssn_recurrent_nodes)

    def _get_timestamps(self, sim):
        return np.linspace(0.0, sim.tstop, num=sim.nsteps, endpoint=False)

    def _to_csv(self, sim):
        output_df = None
        times = self._get_timestamps(sim)
        ssn_nodes = self._get_nodes(sim)
        for node in ssn_nodes:
            tmp_df = pd.DataFrame({
                'population': node.population,
                'node_id': node.node_id,
                'timestamps': times,
                'firing_rates': sim.results[:, node.gid]
            })
            
            output_df = tmp_df if output_df is None else pd.concat([output_df, tmp_df], ignore_index=True)
            # print(node.population, node.node_id, node.gid)
            
            # exit()
        if output_df is not None:
            output_df.to_csv(self.file_path, sep=' ', index=False)
        

    def _to_hdf5(self, sim):
        mode = self._params.get('modd', 'a')
        times = self._get_timestamps(sim)
        nsteps = len(times)
        ssn_nodes = self._get_nodes(sim)
        mappings = {}
        for n in ssn_nodes:
            subpop = mappings.get(n.population, {'node_id': [], 'gid': []})
            subpop['node_id'].append(n.node_id)
            subpop['gid'].append(n.gid)
            mappings[n.population] = subpop

        with h5py.File(self.file_path, mode) as h5:
            ratesgrp = h5['rates'] if 'rates' in h5 else h5.create_group('rates')
            for pop_name, pop_data in mappings.items():
                subgrp = ratesgrp.create_group(pop_name)
                subgrp.create_dataset('mapping/time', data=times)
                subgrp.create_dataset('mapping/node_ids', data=pop_data['node_id'])

                data = subgrp.create_dataset('data', shape=(nsteps, len(pop_data['gid'])), dtype=float)
                for col, gid in enumerate(pop_data['gid']):
                    data[:, col] = sim.results[:, gid]







class ExternalRatesMod:
    def __init__(self, name, input_type, module, **kwargs):
        self._name = name
        self._input_type = input_type
        self._module = module
        self._params = kwargs

    def initialize(self, sim):
        if self._module == 'npy':
            npy_path = self._params['file']
            inputs_arr = np.load(npy_path)

            
            if sim.tstop is None:
                # WARNING THAT TSTOP IS BEING SET BY INPUT

                sim.tstop = len(inputs_arr)*sim.dt

            if sim.nsteps < len(inputs_arr):
                # WARN THAT input is being cut
                inputs_arr = inputs_arr[:sim.nsteps]

            elif sim.nsteps > len(inputs_arr):
                # GIVE WARNING
                inputs_arr = np.append(inputs_arr, np.zeros(sim.nsteps - len(inputs_arr)))
            
            node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
        
            for node in node_set.fetch_nodes():
                ssn_node = sim.network.get_node(node.population_name, node.node_id)
                ssn_node.external_inputs = inputs_arr.flatten()

    def finalize(self, sim):
        pass


class InitStatesMod:
    def __init__(self, name, input_type, module, **kwargs):
        self._name = name
        self._input_type = input_type
        self._module = module
        self._params = kwargs

    def initialize(self, sim):
        # node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))

        if self._module == 'csv':
            self.from_csv(sim)
        elif self._module == 'constant':
            self.from_pregenerated(sim, itype='const')
        elif self._module == 'random':
            self.from_pregenerated(sim, itype='random')
        elif self._module == 'list':
            self.from_pregenerated(sim, itype='list')
        elif self._module == 'function':
            raise NotImplementedError()
        else:
            raise ValueError(f'{self.__name__}: Error in {self._name} input module, no valid module [options: csv, constant, random, function]')


    def from_csv(self, sim):
        csv_path = self._params['file']
        sep = self._params.get('sep', ' ')
        index_col = self._params.get('index_col', 'node_id')
        value_col = self._params.get('value_col', 'initial_state')
        strict_mapping = self._params.get('strict_mapping', False)

        init_df = pd.read_csv(csv_path, sep=sep).set_index(index_col)
        # init_df = init_df.set_index(init_df.columns['node_id'])
        
        node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
        
        for node in node_set.fetch_nodes():
            ssn_node = sim.network.get_node(node.population_name, node.node_id)
            if ssn_node.node_id not in init_df.index:
                if strict_mapping:
                    raise Exception('COULD NOT FIND APPROPIATE ID IN CSV')
                else:
                    # TODO: warning message
                    pass
            else:
                 ssn_node.initial_value = init_df.loc[ssn_node.node_id][value_col]
            
    def from_pregenerated(self, sim, itype):
        node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
        nsize = len(node_set)

        if itype == 'const':
            init_states = [self._params['initial_state']]*nsize

        if itype == 'list':
            init_states = self._params['initial_states']
            if len(init_states) != nsize:
                raise Exception('SIZE OF LIST DOES NOT MATCH NUMBER OF NODES')
            
            if self._params.get('shuffle', False):
                np.random.shuffle(init_states)

        elif itype == 'random':
            dist = self._params['distribution']
            if dist == 'uniform':
                init_states = np.random.normal(
                    low=self._params.get('low', 0.0),
                    high=self._params.get('high', 1.0),
                    size=nsize
                )
            elif dist == 'normal':
                init_states = np.random.normal(
                    loc=self._params.get('mean', 0.0),
                    scale=self._params.get('std', 1.0),
                    size=nsize
                )
            elif dist == 'poisson':
                init_states = np.random.poisson(
                    lam=self._params.get('lambda', 1.0), 
                    size=nsize
                )
            elif dist == 'lognormal':
                init_states = np.random.lognormal(
                    mean=self._params.get('mean', 0.0), 
                    sigma=self._params.get('sigma', 1.0),
                    size=nsize
                )
            else:
                raise Exception("AAAA")

        for idx, node in enumerate(node_set.fetch_nodes()):
            ssn_node = sim.network.get_node(node.population_name, node.node_id)
            ssn_node.initial_value = init_states[idx]

    def finalize(self, sim):
        pass



@njit
def relu2(array):
    # if the element is negative, set it to zero using loop
    # destructive method (alters the original array), but faster than the above one.
    for i in range(len(array)):
        if array[i] < 0:
            array[i] = 0
    return array


class Config(SimulationConfig):
    pass


class SSNNode:
    def __init__(self, population, node_id, gid):
        self.population = population
        self.node_id = node_id 
        self.gid = gid

        self.type = None
        self.input_offset = []
        self.scaling_coef = [] 
        self.exponent = []
        self.decay_const = []
        self.init_value = 0.0
        self.external_inputs = None


    def __repr__(self) -> str:
        return f'{self.gid} > ({self.population}.{self.node_id})'


class SSNNetwork(SimNetwork):
    def __init__(self, grouping_key='node_id', **opts):
        super(SSNNetwork, self).__init__()
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

        self._recurrent_nodes = {}
        self._external_nodes = {}


        self._nodes_idx = {}
        self._ssn_recurrent_nodes = set()
        self._ssn_external_nodes = set()
        self.gids = 0

        self._conn_mat = []
        # sefl._connectivity



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
                        initial_value=node.get['initial_value'] if 'initial_value' in node else 0.0
                    )
                
                elif model_type in ['external', 'virtual']:
                    self.add_external_node(
                        population_id=node_pop.name, 
                        node_id=node[self.grouping_key]
                    )

        # pprint(self._node_id_map)
        # exit()
        
        #         print(node.population_name)
        #         print(node['model_type'])

        # exit()

    def build_edges(self):
        for edge_pop in self._edge_populations:
            for edge in edge_pop.get_edges():
                # print(edge.source_population, edge.source_node_id, edge.target_population, edge.target_node_id, edge['syn_weight'])
                # print(edge.source_population in self._node_id_map)
                # print(list(self._node_id_map[edge.source_population].keys()))
                # print(edge.source_node_id in self._node_id_map[edge.source_population])
                # exit()
                # print(self.get_ssn_node(edge.source_population, edge.source_node_id).gid)
                src_node = self._node_id_map[edge.source_population][int(edge.source_node_id)]
                trg_node = self._node_id_map[edge.target_population][int(edge.target_node_id)]
                
                self._conn_mat.append([trg_node.gid, src_node.gid, edge['syn_weight']])
                # print(src_node, trg_node)

        # exit()

    def get_node(self, population_id, node_id):
        return self._node_id_map[population_id][node_id]


    def get_ssn_node(self, population_id, node_id):
        if population_id not in self._node_id_map:
            # print('new population')
            ssn_node = SSNNode(population_id, node_id, gid=self.gids)
            self._node_id_map[population_id] = {int(node_id): ssn_node}
            self.gids += 1
        
        elif int(node_id) not in self._node_id_map:
            # print(f'new node {population_id}, {node_id}')
            ssn_node = SSNNode(population_id, node_id, gid=self.gids)
            self._node_id_map[population_id][int(node_id)] = ssn_node
            self.gids += 1

        else:
            # print('found')
            ssn_node = self._node_id_map[population_id][node_id]

        return ssn_node


    def add_recurrent_node(self, population_id, node_id, input_offset, scaling_coef, exponent, decay_const, initial_value=0.0):
        self._nnodes_recurrent += 1
        
        
        ssn_obj = self.get_ssn_node(population_id=population_id, node_id=node_id)
        ssn_obj.type='internal'
        ssn_obj.input_offset.append(input_offset)
        ssn_obj.scaling_coef.append(scaling_coef)
        ssn_obj.exponent.append(exponent)
        ssn_obj.decay_const.append(decay_const)
        self._ssn_recurrent_nodes.add(ssn_obj)
        

    def add_external_node(self, population_id, node_id):
        self._nnodes_external += 1
        ssn_obj = self.get_ssn_node(population_id=population_id, node_id=node_id)
        ssn_obj.type = 'external'
        self._ssn_external_nodes.add(ssn_obj)


class SSNSimulator:
    def __init__(self, network, dt=1.0, tstart=0.0, tstop=None, **opts):
        self.dt = dt
        self.tstart = tstart
        self.tstop = tstop
        self.network = network

        self._fr_results = None

        self._mods = []


    @property
    def nsteps(self):
        return int((self.tstop - self.tstart)/self.dt)

    @property
    def results(self):
        if self._fr_results is None:
            self._fr_results = np.zeros((self.nsteps, self.network.n_neu_total), dtype=float)
            # print(self.network.initial_states)
            self._fr_results[0, :self.network.n_neu_recurrent] = self.network.initial_states
            
            for ext_node in self.network._ssn_external_nodes:
                if ext_node.external_inputs is not None:
                    self._fr_results[:, ext_node.gid] = ext_node.external_inputs
                # print(ext_node.gid, ext_node.external_inputs)
            # exit()
            
            # exit()

        return self._fr_results


    def step(self, state, mat, scales, exponents, decay_constants, dt):
        input = relu2(np.dot(mat, state) * scales) ** exponents
        n_neu_recurrent = mat.shape[0]
        recurrent_state = state[:n_neu_recurrent]
        dr = (-recurrent_state + input) / decay_constants * dt
        return relu2(recurrent_state + dr)

    def add_mod(self, mod):
        mod.initialize(self)
        self._mods.append(mod)

    def run(self):
        # print(self.network.connectivity_mat)
        # print(self.results)
        # exit()        
        for t in range(self.nsteps-1):
            self.results[t+1, :self.network.n_neu_recurrent] = self.step(
                self.results[t, :],
                self.network.connectivity_mat,
                self.network.scales,
                self.network.exponents,
                self.network.decay_constants,
                self.dt,
            )

        for mod in self._mods:
            mod.finalize(self)


        # print(self.results)
        # print(self.results.shape)

        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(self.results[:, :])
        # # ax[0].legend(nodes_recurrent["cell_types"].values)
        # ax[0].title.set_text("Entire simulation")
        # plt.show()

    @classmethod
    def from_config(cls, configure, network, **opts):
        # load the json file or object
        if isinstance(configure, string_types):
            config = Config.from_json(configure, validate=True)
        elif isinstance(configure, dict):
            config = configure
        else:
            raise Exception('Could not convert {} (type "{}") to json.'.format(configure, type(configure)))

        sim = cls(network, dt=config.dt, tstart=config.tstart, tstop=config.tstop, **opts)
        
        # if 'output_dir' in config['output']:
        #     network.output_dir = config['output']['output_dir']

        network.io.log_info('Building nodes.')
        network.build_nodes()

        network.io.log_info('Building recurrent connections')
        network.build_edges()

        for sim_input in inputs.from_config(config):
            if sim_input.input_type == 'init_states':
                mod = InitStatesMod(
                    name=sim_input.name,
                    input_type=sim_input.input_type,
                    module=sim_input.module,
                    **sim_input.params
                )
                sim.add_mod(mod)
    
            if sim_input.input_type == 'external_rates':
                mod = ExternalRatesMod(
                    name=sim_input.name,
                    input_type=sim_input.input_type,
                    module=sim_input.module,
                    **sim_input.params
                )
                sim.add_mod(mod)

        
        if 'rates_file' in config.output:
            mod = RatesRecorderMod(
                name='RecordRatesH5', 
                module='rates',
                output_type='csv',
                file_name=config.output['rates_file'],
                **config.output
            )
            sim.add_mod(mod)

        if 'rates_file_csv' in config.output:
            mod = RatesRecorderMod(
                name='RecordRatesH5', 
                module='rates',
                output_type='csv',
                file_name=config.output['rates_file_csv'],
                **config.output
            )
            sim.add_mod(mod)

        if 'rates_file_h5' in config.output:
            mod = RatesRecorderMod(
                name='RecordRatesH5', 
                module='rates',
                output_type='h5',
                file_name=config.output['rates_file_h5'],
                **config.output
            )
            sim.add_mod(mod)

        return sim


configure = Config.from_json('config.simulation.json')
configure.build_env()

# print(configure)

network = SSNNetwork.from_config(configure)

sim = SSNSimulator.from_config(configure, network)
sim.run()
