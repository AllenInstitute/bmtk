"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""


import sys, os

from bmtk.simulator.bionet import nrn, Config
#from bmtk.simulator.bionet import io, nrn, Config
from bmtk.simulator.bionet.simulation import Simulation
from bmtk.simulator.bionet.biograph import BioGraph

from bmtk.analyzer.spikes_analyzer import spike_files_equal


def run(config_file):
    conf = Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = BioGraph.from_config(conf)
    sim = Simulation.from_config(conf, network=graph)
    sim.run()

    # assert (spike_files_equal(conf['output']['spikes_file_csv'], 'expected/spikes.csv'))
    nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
