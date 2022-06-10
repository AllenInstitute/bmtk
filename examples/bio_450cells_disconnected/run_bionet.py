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
    # run('config.recurrent.json')
    # run('config.feedforward.json')
    run('config.all_cells.json')
    # run('config.w_extern.json')
    # run('config.biophys_cells.json')
    # run('config.pv_cells.json')
