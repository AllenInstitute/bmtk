"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""


import sys, os

from bmtk.simulator import bionet
from bmtk.analyzer.cell_vars import plot_report


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    net = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=net)
    sim.run()

    plot_report(config_file='config.json', node_ids=[0, 10, 20, 30])

    # bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
