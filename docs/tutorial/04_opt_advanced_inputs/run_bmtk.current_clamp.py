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
        run('config.current_clamp.json')

