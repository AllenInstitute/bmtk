"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""


import sys, os
import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork
from bmtk.simulator.bionet import Config

from bmtk.utils.io import TabularNetwork_AI
from bmtk.simulator.bionet.property_schemas import AIPropertySchema


#import set_weights
#import set_cell_params
#import set_syn_params


def run(config_file):
    conf = Config.from_json(config_file, validate=True)
    conf.create_output_dir()
    conf.copy_to_output()
    conf.load_nrn_modules()
    # TODO: add option build_env() to call the above
    # conf.build_env()

    '''
    conf = config.from_json(config_file)        # build configuration
    io.setup_output_dir(conf)                   # set up output directories
    nrn.load_neuron_modules(conf)               # load NEURON modules and mechanisms
    nrn.load_py_modules(cell_models=set_cell_params,  # load custom Python modules
                        syn_models=set_syn_params,
                        syn_weights=set_weights)
    '''

    graph = BioGraph.from_config(conf)

    #net = BioNetwork.from_config(conf, graph)   # create network of in NEURON

    sim = Simulation.from_config(conf, network=graph)
    # sim = Simulation(conf, network=net)         # initialize a simulation
    # sim.set_recordings()                        # set recordings of relevant variables to be saved as an ouput
    sim.run()                                   # run simulation

    #assert (spike_files_equal(conf['output']['spikes_file_csv'], 'expected/spikes.txt'))

    nrn.quit_execution()                        # exit



if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
