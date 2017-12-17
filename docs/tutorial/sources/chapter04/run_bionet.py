# -*- coding: utf-8 -*-

"""Simulates an example network of 14 cell receiving two kinds of exernal input as defined in configuration file"""

import os
import sys
import math

import bmtk.simulator.bionet.config as config
from bmtk.simulator.bionet import io, nrn
from bmtk.simulator.bionet.pyfunction_cache import add_weight_function
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.simulator.bionet.biograph import BioGraph
from bmtk.simulator.bionet.bionetwork import BioNetwork
from bmtk.simulator.bionet.property_schemas import AIPropertySchema


def gaussianLL(tar_prop, src_prop, con_prop):
    src_tuning = src_prop['tuning_angle']
    tar_tuning = tar_prop['tuning_angle']

    w0 = con_prop["weight_max"]
    sigma = con_prop["weight_sigma"]

    delta_tuning = abs(abs(abs(180.0 - abs(float(tar_tuning) - float(src_tuning)) % 360.0) - 90.0) - 90.0)
    weight = w0 * math.exp(-(delta_tuning / sigma) ** 2)

    return weight



def run(config_file):
    conf = config.from_json(config_file)
    io.setup_output_dir(conf)
    add_weight_function(gaussianLL)
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
