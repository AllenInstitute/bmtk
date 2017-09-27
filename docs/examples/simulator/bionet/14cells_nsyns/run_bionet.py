# -*- coding: utf-8 -*-

"""Builds and simulates Bionet from old file format with 14 internal cells based on V1 sample."""

import sys
import os

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork

from bmtk.utils.io import TabularNetwork_AI
from bmtk.simulator.bionet.property_schemas import AIPropertySchema

# Contains specialized user functions that will be called during the simulation
import weight_funcs
import set_model_params
import set_syn_params


def run():
    # Set up the graph structure
    conf = config.from_json('config.json')
    io.setup_output_dir(conf)
    nrn.load_neuron_modules(conf)
    graph = BioGraph.from_config(conf, network_format=TabularNetwork_AI, property_schema=AIPropertySchema)

    # Set up the network and simulate
    net = BioNetwork.from_config('simulator_config.json', graph)
    sim = Simulation(conf, network=net)
    sim.set_recordings()
    sim.run()

    # validate output
    assert (spike_files_equal(conf['output']['spikes_ascii'], 'expected/spikes.txt'))


if __name__ == '__main__':
    run()
