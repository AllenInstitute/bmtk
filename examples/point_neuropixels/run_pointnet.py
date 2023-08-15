import os, sys

from bmtk.simulator import pointnet


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    # Find the appropriate config.json file
    # run('config.simulation.sample.json')
    # run('config.simulation.units_map.json')
    run('config.simulation.multi_sessions.json')
