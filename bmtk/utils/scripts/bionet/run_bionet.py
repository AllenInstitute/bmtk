# -*- coding: utf-8 -*-

"""Simulates an example network of 14 cell receiving two kinds of exernal input as defined in configuration file"""

import os
import sys

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork
from bmtk.simulator.bionet.property_schemas import AIPropertySchema


def run(config_file):
    conf = config.from_json(config_file)
    io.setup_output_dir(conf)
    nrn.load_neuron_modules(conf)
    graph = BioGraph.from_config(conf, property_schema=AIPropertySchema)
    net = BioNetwork.from_config(conf, graph)
    sim = Simulation.from_config(conf, network=net)
    sim.run()
    nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
