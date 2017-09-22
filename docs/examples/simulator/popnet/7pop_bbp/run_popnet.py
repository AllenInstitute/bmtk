import sys
import os

import bmtk.simulator.utils.config as config
from bmtk.simulator.popnet.popgraph import PopGraph
from bmtk.simulator.popnet.popnetwork import PopNetwork


def main(config_file):
    configure = config.from_json(config_file)
    graph = PopGraph.from_json(config_file, group_by='node_type_id')
    net = PopNetwork.from_json(configure, graph)
    net.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.json')