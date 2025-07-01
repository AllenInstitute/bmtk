from six import string_types
import numpy as np

import bmtk.simulator.utils.simulation_inputs as inputs
from .modules import InitStatesMod, ExternalRatesMod, RatesRecorderMod
from .pyfunction_cache import py_modules
from bmtk.simulator.core.io_tools import io
from bmtk.simulator.core.simulation_config import SimulationConfig as Config

# TODO: leave this import, it will initialize some of the default functions for building neurons/synapses/weights.
import bmtk.simulator.popnet.ssn.default_setters

try:
    from numba import njit
except ImportError as ie:
    from bmtk.simulator.popnet.ssn.utils import empty_decorator
    njit = empty_decorator

@njit
def step(activation_function, state, mat, scales, exponents, decay_constants, dt):
    """ A function to compute one step of the network.
    This function needs to be here so that run_njit can be compiled.
    """

    input = activation_function(np.dot(mat, state) * scales) ** exponents
    n_neu_recurrent = mat.shape[0]
    recurrent_state = state[:n_neu_recurrent]
    dr = (-recurrent_state + input) / decay_constants * dt
    return activation_function(recurrent_state + dr)

@njit
def run_ssn(activation_function, results, connectivity_mat, scales, exponents, decay_constants, dt, nsteps, n_neu_recurrent):
    """
    A function to run the whole simulation.
    The results will be modified in place.
    """
    for t in range(nsteps-1):
        results[t+1, :n_neu_recurrent] = step(
            activation_function,
            results[t, :],
            connectivity_mat,
            scales,
            exponents,
            decay_constants,
            dt,
        )
    return


class PopSimulator:
    def __init__(self, network, dt=1.0, tstart=0.0, tstop=None, **opts):
        self.dt = dt
        self.tstart = tstart
        self.tstop = tstop
        self.network = network
        self._fr_results = None
        self._mods = []
        self.activation_function = py_modules.activation_function('default')

    @property
    def nsteps(self):
        return int((self.tstop - self.tstart)/self.dt)

    @property
    def results(self):
        if self._fr_results is None:
            self._fr_results = np.zeros((self.nsteps, self.network.n_neu_total), dtype=float)
            self._fr_results[0, :self.network.n_neu_recurrent] = self.network.initial_states
            
            for ext_node in self.network._ssn_external_nodes:
                if ext_node.external_inputs is not None:
                    self._fr_results[:, ext_node.gid] = ext_node.external_inputs

        return self._fr_results


    def add_mod(self, mod):
        mod.initialize(self)
        self._mods.append(mod)
        

    def run(self, return_output=False):
        
        activation_function = self.activation_function
        if not return_output:
            io.log_info('Running Simulation.')

        # self.results will be modified in place
        run_ssn(
            activation_function,
            self.results,
            self.network.connectivity_mat,
            self.network.scales,
            self.network.exponents,
            self.network.decay_constants,
            self.dt,
            self.nsteps,
            self.network.n_neu_recurrent,
        )

        if return_output:
            return self.results
        io.log_info('Simulation Finished.')

        for mod in self._mods:
            mod.finalize(self)

    def set_activation_function(self, fnc):
        self.activation_function = fnc

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
        
        act_fnc_sig = config.run.get('activation_function', None)
        if act_fnc_sig:
            if act_fnc_sig not in py_modules.activation_functions:
                io.log_error(f'Could not find activation fucntion with signature {act_fnc_sig} registered to PopNet')
            else:
                io.log_debug(f'Setting simulation activation function to {act_fnc_sig}')
                sim.set_activation_function(py_modules.activation_function(act_fnc_sig))


        network.io.log_info('Building nodes.')
        network.build_nodes()

        network.io.log_info('Building connections.')
        network.build_edges()

        for sim_input in inputs.from_config(config):
            if sim_input.input_type == 'init_states':
                mod = InitStatesMod(
                    name=sim_input.name,
                    input_type=sim_input.input_type,
                    module=sim_input.module,
                    **sim_input.params
                )
                network.io.log_info(f'Add input module "{sim_input.name}"')
                sim.add_mod(mod)
    
            elif sim_input.input_type == 'external_rates':
                mod = ExternalRatesMod(
                    name=sim_input.name,
                    input_type=sim_input.input_type,
                    module=sim_input.module,
                    **sim_input.params
                )
                network.io.log_info(f'Add input module "{sim_input.name}"')
                sim.add_mod(mod)
            
            else:
                network.io.log_info(f'Unknown input module "{sim_input.name}" (module={sim_input.module})')
        
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
