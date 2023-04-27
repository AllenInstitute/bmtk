"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import os
import sys
from bmtk.simulator import bionet


def run(config_path):
    conf = bionet.Config.from_json(config_path, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
        run(config_path)
    else:
        run('config.simulation_syns.json')
        # run('config.simulation_iclamp.json')

