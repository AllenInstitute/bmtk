import argparse
from bmtk.simulator import pointnet


def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    default_config = 'config.simulation.json'
    # default_config = 'config.simulation_perturbations.json'

    parser = argparse.ArgumentParser(description='Run PointNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default=default_config, 
                        help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)

