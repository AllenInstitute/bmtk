"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import sys, os
from bmtk.simulator import bionet


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    net = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=net)
    sim.run()

    bionet.nrn.quit_execution()


if __name__ == '__main__':
    # run('config.simulation.json')
    # run('config.simulation_feedforward.json')
    # run('config.simulation_recurrent.json')
    # run('config.simulation_recreated.json')
    run('config.simulation_exc2scnn1a.json')
