import os, sys
# import nest

from bmtk.simulator import pointnet


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    if sys.argv[-1] != __file__:
        run(sys.argv[-1])
    else:
        run('config.nestml.json')
