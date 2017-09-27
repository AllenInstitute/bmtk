# -*- coding: utf-8 -*-

"""Builds and simulates a 14 cell V1 example using new network format."""

import os, sys

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork


import weight_funcs
import set_model_params
import set_syn_params


def run():
    conf = config.from_json('config.json')
    io.setup_output_dir(conf)
    nrn.load_neuron_modules(conf)
    graph = BioGraph.from_config(conf)

    net = BioNetwork.from_config('simulator_config.json', graph)
    sim = Simulation(conf, network=net)
    sim.set_recordings()
    sim.run()

    assert (spike_files_equal(conf['output']['spikes_ascii'], 'expected/spikes.txt'))


if __name__ == '__main__':
    run()
