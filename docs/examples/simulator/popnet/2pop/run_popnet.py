import sys
import os

import bmtk.simulator.utils.config as config
from bmtk.simulator.popnet.popgraph import PopGraph
from bmtk.simulator.popnet.popnetwork import PopNetwork
from bmtk.analyzer.firing_rates import firing_rates_equal


def main(config_file):
    configure = config.from_json(config_file)
    graph = PopGraph.from_config(config_file, group_by='node_type_id')
    #v1_pops = graph.get_populations('V1')
    #    print pop
    print graph.get_populations('LGN')
    pop0 = graph.get_population('LGN', 0)
    #pop0.firing_rate = 12.2

    net = PopNetwork.from_json(configure, graph)
    net.run()

    assert (firing_rates_equal('expected/spike_rates.txt', configure['output']['rates_file']))



if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.json')
