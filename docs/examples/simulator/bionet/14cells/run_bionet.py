"""Simulates an example network of 14 cell receiving two kinds of exernal input as defined in configuration file"""

import os, sys

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork


import set_weights
import set_cell_params
import set_syn_params


def run():

    config_file = str(sys.argv[-1])             # Get configuration file name from the command line argument

    conf = config.from_json(config_file)        # build configuration
    io.setup_output_dir(conf)                   # set up output directories
    nrn.load_neuron_modules(conf)               # load NEURON modules and mechanisms
    nrn.load_py_modules(cell_models=set_cell_params, # load custom Python modules
                        syn_models=set_syn_params,
                        syn_weights=set_weights)

    graph = BioGraph.from_config(conf)          # create network graph containing parameters of the model

    net = BioNetwork.from_config(conf, graph)   # create network of in NEURON
    sim = Simulation(conf, network=net)         # initialize a simulation
    sim.set_recordings()                        # set recordings of relevant variables to be saved as an ouput
    sim.run()                                   # run simulation

    nrn.quit_execution()                        # exit


if __name__ == '__main__':
    run()
