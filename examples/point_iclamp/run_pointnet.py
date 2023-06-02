import sys

from bmtk.simulator import pointnet
from bmtk.analyzer.compartment import plot_traces


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()

    plot_traces(config_file=config_file, report_name='membrane_potential', population='cortex')

if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        # run('config.simulation_iclamp.json')
        # run('config.simulation_iclamp.aslist.json')
        # run('config.simulation_iclamp.csv.json')
        run('config.simulation_iclamp.nwb.json')
