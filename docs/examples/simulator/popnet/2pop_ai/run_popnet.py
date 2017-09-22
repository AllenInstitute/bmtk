import sys
import os

import bmtk.simulator.utils.config as config
from bmtk.simulator.popnet.popgraph import PopGraph
from bmtk.simulator.popnet.popnetwork import PopNetwork

# from modelingsdk.utils.networkformat import AIFormat
from bmtk.utils.io import TabularNetwork_AI
from bmtk.simulator.popnet.property_schemas import AIPropertySchema


def main(config_file):
    configure = config.from_json(config_file)
    graph = PopGraph.from_json(config_file, network_format=TabularNetwork_AI, property_schema=AIPropertySchema,
                               group_by='ei')
    net = PopNetwork.from_json(configure, graph)
    net.run()

if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.json')
