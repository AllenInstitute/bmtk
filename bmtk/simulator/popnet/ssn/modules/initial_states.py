import pandas as pd
import numpy as np

from ..pyfunction_cache import py_modules
from .sim_module import SimulatorMod


class InitStatesMod(SimulatorMod):
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
            self.from_user_function(sim)
        else:
            raise ValueError(f'{self.__name__}: Error in {self._name} input module, no valid module [options: csv, constant, random, function]')

    def from_user_function(self, sim):
        fnc_name = self._params['init_function']
        generator_fnc = py_modules.user_function(fnc_name)

        node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
        for node in node_set.fetch_nodes():
            ssn_node = sim.network.get_node(node.population_name, node.node_id)
            init_val = generator_fnc(ssn_node, sim)
            ssn_node.initial_value = init_val

    def from_csv(self, sim):
        csv_path = self._params['file']
        sep = self._params.get('sep', ' ')
        index_col = self._params.get('index_col', 'node_id')
        value_col = self._params.get('value_col', 'initial_state')
        strict_mapping = self._params.get('strict_mapping', False)

        init_df = pd.read_csv(csv_path, sep=sep).set_index(index_col)
        
        node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))       
        for node in node_set.fetch_nodes():
            ssn_node = sim.network.get_node(node.population_name, node.node_id)
            if ssn_node.node_id not in init_df.index:
                if strict_mapping:
                    raise Exception('COULD NOT FIND APPROPRIATE ID IN CSV')
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
                init_states = np.random.uniform(
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

    # def finalize(self, sim):
    #     pass

