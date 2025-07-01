import numpy as np
import pandas as pd
import h5py

from ..pyfunction_cache import py_modules
from .sim_module import SimulatorMod


class ExternalRatesMod(SimulatorMod):
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
                # TODO: WARNING THAT TSTOP IS BEING SET BY INPUT
                sim.tstop = len(inputs_arr)*sim.dt

            if sim.nsteps < len(inputs_arr):
                # TODO: WARN THAT input is being cut
                inputs_arr = inputs_arr[:sim.nsteps]

            elif sim.nsteps > len(inputs_arr):
                # GIVE WARNING
                inputs_arr = np.append(inputs_arr, np.zeros(sim.nsteps - len(inputs_arr)))
            
            node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
        
            for node in node_set.fetch_nodes():
                ssn_node = sim.network.get_node(node.population_name, node.node_id)
                ssn_node.external_inputs = inputs_arr.flatten()


        elif self._module == 'function':
            fnc_name = self._params['inputs_generator']
            generator_fnc = py_modules.user_function(fnc_name)

            node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
            for node in node_set.fetch_nodes():
                ssn_node = sim.network.get_node(node.population_name, node.node_id)
                external_inputs = generator_fnc(ssn_node, sim)
                ssn_node.external_inputs = external_inputs

                if sim.tstop is None:
                    # TODO: WARNING THAT TSTOP IS BEING SET BY INPUT
                    sim.tstop = len(external_inputs)*sim.dt

        elif self._module == 'csv':
            # TODO: Check Timestamps match, line up if needed
            rates_df = pd.read_csv(self._params['file'], sep=self._params.get('sep', ' '))
            node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
            for node in node_set.fetch_nodes():
                ssn_node = sim.network.get_node(node.population_name, node.node_id)
                
                # TODO: Check that node exists in file
                inputs = rates_df[(rates_df['node_id'] == ssn_node.node_id) & (rates_df['population'] == ssn_node.population)]['firing_rates']
                if len(inputs) > 0:
                    ssn_node.external_inputs = inputs

                    if sim.tstop is None:
                        # TODO: WARNING THAT TSTOP IS BEING SET BY INPUT
                        sim.tstop = len(inputs)*sim.dt
        
        elif self._module in ['h5', 'sonata']:
            with h5py.File(self._params['file'], 'r') as h5:
                ratesgrp = h5['/rates']
                node_set = sim.network.get_node_set(self._params.get('node_set', 'all'))
                for node in node_set.fetch_nodes():
                    ssn_node = sim.network.get_node(node.population_name, node.node_id)

                    node_id_map = ratesgrp[f'{ssn_node.population}/mapping/node_ids'][()]
                    idx = np.argwhere(node_id_map == ssn_node.node_id)[0][0]
                    inputs = ratesgrp[f'{ssn_node.population}/data'][:, idx]

                    ssn_node.external_inputs = inputs

                    if sim.tstop is None:
                        # TODO: WARNING THAT TSTOP IS BEING SET BY INPUT
                        sim.tstop = len(inputs)*sim.dt
        else:
            raise ValueError('Unknown module')
