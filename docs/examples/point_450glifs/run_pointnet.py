import os, sys
from bmtk.simulator import pointnet

import nest

try:
    nest.Install('glifmodule')
except Exception as e:
    pass


def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()


if __name__ == '__main__':
    main('config.json')
