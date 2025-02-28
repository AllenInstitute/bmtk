from bmtk.simulator import pointnet
import argparse


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run point network simulation.')
    parser.add_argument('config_file',
        type=str,
        nargs='?',
        default='config.simulation.json',
        help='Path to the configuration file'
    )
    args = parser.parse_args()

    run(args.config_file)
