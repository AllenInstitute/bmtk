import os
import sys
import bmtk.simulator.utils.config as config
from bmtk.simulator.pointnet.graph import Graph
from bmtk.simulator.pointnet import nest_construction
from bmtk.analyzer.spikes_analyzer import spike_files_equal

import nest

nest.Install('glifmodule.so')

import weight_funcs as fn

def main(config_file):
    configure = config.from_json(config_file)
    graph = Graph(configure)
    graph.add_weight_function(fn.wmax)
    graph.add_weight_function(fn.gaussianLL)

    net = nest_construction.Network(configure, graph)
    net.run(configure['run']['duration'])

    #assert(spike_files_equal(configure['output']['spikes_ascii'], 'expected/spikes.txt'))

    #plot_spikes(configure, 'pop_name')

    # Check all the output was created

    """
    out_dir = configure['output']['output_dir']
    spikes_gdf = os.path.join(out_dir, 'spikes.txt')
    assert(os.path.exists(spikes_gdf))
    n_spikes = sum(1 for _ in open(spikes_gdf))
    assert(n_spikes > 328)
    """

if __name__ == '__main__':
    main('config.json')
    #if __file__ != sys.argv[-1]:
    #    main(sys.argv[-1])
    #else:
    #    print "HERE"
    #    main('config.json')
