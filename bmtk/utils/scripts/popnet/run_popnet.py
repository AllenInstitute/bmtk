import sys
import os

from bmtk.simulator import popnet


def main(config_file):
    configure = popnet.config.from_json(config_file)
    configure.build_env()
    network = popnet.PopNetwork.from_config(configure)
    sim = popnet.PopSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.json')