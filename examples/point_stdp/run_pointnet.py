import os, sys

from bmtk.simulator import pointnet


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    if os.path.basename(__file__) != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run('config.simulation_iclamp.json')