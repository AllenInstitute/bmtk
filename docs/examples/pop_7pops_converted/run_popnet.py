import sys
import os

import bmtk.simulator.popnet.config as config
from bmtk.simulator.popnet.popgraph import PopGraph
from bmtk.simulator.popnet.simulation import Simulation


def main(config_file):
    configure = config.from_json(config_file)
    graph = PopGraph.from_config(configure, group_by='node_type_id')
    sim = Simulation.from_config(configure, graph)
    sim.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.json')