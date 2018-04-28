"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""


import sys, os

from bmtk.simulator import bionet


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)

    #net = BioNetwork.from_config(conf, graph)   # create network of in NEURON

    sim = bionet.BioSimulator.from_config(conf, network=graph)
    # sim = Simulation(conf, network=net)         # initialize a simulation
    # sim.set_recordings()                        # set recordings of relevant variables to be saved as an ouput
    sim.run()                                   # run simulation

    #assert (spike_files_equal(conf['output']['spikes_file_csv'], 'expected/spikes.txt'))

    bionet.nrn.quit_execution()                        # exit



if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.json')
