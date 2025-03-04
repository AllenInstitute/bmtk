import os
import sys

from bmtk.simulator import bionet
from bmtk.analyzer.compartment import plot_traces



def run(config_path):
    conf = bionet.Config.from_json(config_path, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    plot_traces(config_file=config_path, report_name='membrane_potential', population='net', node_ids=[0, 1])
    # bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
        run(config_path)
    else:
        run('config.current_clamp.json')

