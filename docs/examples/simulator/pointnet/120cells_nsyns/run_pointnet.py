import os
import sys
import nest
import bmtk.simulator.pointnet.config as config
from bmtk.simulator.pointnet.pointgraph import PointGraph
from bmtk.simulator.pointnet.pointnetwork import PointNetwork
from bmtk.simulator.pointnet import io
from bmtk.utils.io import TabularNetwork_AI

from bmtk.simulator.pointnet.property_schemas import AIPropertySchema
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from mpi4py import MPI
comm = MPI.COMM_WORLD

import weight_funcs as fn

def main(config_file):
    nest.SetKernelStatus({"total_num_virtual_procs": comm.Get_size()})
    print nest.NumProcesses()
    print comm.Get_size()
    configure = config.from_json(config_file)
    io.setup_output_dir(configure)
    graph = PointGraph.from_config(configure, network_format=TabularNetwork_AI, property_schema=AIPropertySchema)

    graph.add_weight_function(fn.wmax)
    graph.add_weight_function(fn.gaussianLL)

    net = PointNetwork.from_config(configure, graph)
    net.run(configure['run']['duration'])


    assert(spike_files_equal(configure['output']['spikes_ascii'], 'expected/spikes.txt'))


if __name__ == '__main__':
    main('config.json')
