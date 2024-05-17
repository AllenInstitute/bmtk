# -*- coding: utf-8 -*-

"""Simulates an example network of 14 cell receiving two kinds of exernal input as defined in configuration file"""

import os, sys
from bmtk.simulator import filternet


def run(config_file):
    config = filternet.Config.from_json(config_file)
    config.build_env()

    net = filternet.FilterNetwork.from_config(config)
    sim = filternet.FilterSimulator.from_config(config, net)
    sim.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('${CONFIG}')
