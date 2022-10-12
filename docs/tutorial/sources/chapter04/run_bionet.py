# Example call:
# $ python run_bionet.py simulation_config.json
#
# If the configuration file is not specified, it will look for config.json, simulation_config.json, or config.simulation.json
#
# -*- coding: utf-8 -*-

"""Simulates an example network of 14 cell receiving two kinds of exernal input as defined in configuration file"""

import os, sys
from bmtk.simulator import bionet


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    bionet.nrn.quit_execution()


# Compile NEURON mod files
# Typically these would be located within the components folder, but here they have been placed in a single folder used by all tutorial chapters
os.system('cd ../bionet_files/components/mechanisms/; nrnivmodl modfiles')


# Run simulation
if __name__ == '__main__':
    # Find the appropriate config.json file
    config_path = None
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
        if not os.path.exists(config_path):
            raise AttributeError('configuration file {} does not exist.'.format(config_path))
    else:
        for cfg_path in ['config.json', 'config.simulation.json', 'simulation_config.json']:
            if os.path.exists(cfg_path):
                config_path = cfg_path
                break
        else:
            raise AttributeError('Could not find configuration json file.')

    run(config_path)
