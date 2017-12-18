"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""


import sys, os
import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork

from bmtk.utils.io import TabularNetwork_AI
from bmtk.simulator.bionet.property_schemas import AIPropertySchema


import set_weights
import set_cell_params
import set_syn_params


def run(config_file):
    conf = config.from_json(config_file)        # build configuration
    io.setup_output_dir(conf)                   # set up output directories
    nrn.load_neuron_modules(conf)               # load NEURON modules and mechanisms
    nrn.load_py_modules(cell_models=set_cell_params,  # load custom Python modules
                        syn_models=set_syn_params,
                        syn_weights=set_weights)

    graph = BioGraph.from_config(conf,                # create network graph containing parameters of the model
                                 network_format=TabularNetwork_AI,
                                 property_schema=AIPropertySchema)

    net = BioNetwork.from_config(conf, graph)   # create network of in NEURON
    sim = Simulation.from_config(conf, network=net)
    # sim = Simulation(conf, network=net)         # initialize a simulation
    # sim.set_recordings()                        # set recordings of relevant variables to be saved as an ouput
    sim.run()                                   # run simulation

    assert (spike_files_equal(conf['output']['spikes_ascii_file'], 'expected/spikes.txt'))

    nrn.quit_execution()                        # exit


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
