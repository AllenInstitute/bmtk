import os
import sys


import bmtk.simulator.pointnet.config as config
from bmtk.simulator.pointnet.pointgraph import PointGraph
from bmtk.simulator.pointnet.simulation import Simulation
#from bmtk.simulator.pointnet.pointnetwork import PointNetwork
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.pointnet.pyfunction_cache import py_modules

import nest


def process_model(model_type, node, dynamics_params):
    return nest.Create(model_type, 1, dynamics_params)


def gaussianLL(edge_props, source_node, target_node):
    return edge_props['syn_weight']


def wmax(edge_props, source_node, target_node):
    return edge_props['syn_weight']*1000


def main(config_file):
    configure = config.from_json(config_file)
    configure.build_env()

    graph = PointGraph.from_config(configure)

    py_modules.add_cell_processor('process_model', process_model)
    py_modules.add_synaptic_weight('gaussianLL', gaussianLL)
    py_modules.add_synaptic_weight('wmax', wmax)
    #graph.add_weight_function(fn.wmax)
    #graph.add_weight_function(fn.gaussianLL)

    sim = Simulation.from_config(configure, graph)
    sim.run()
    #net = PointNetwork.from_config(configure, graph)
    # net.run()
    # print nest.GetConnections()
    # assert (spike_files_equal(configure['output']['spikes_file_csv'], 'expected/spikes.csv'))


if __name__ == '__main__':
    main('config.json')
