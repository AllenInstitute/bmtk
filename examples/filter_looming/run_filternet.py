import argparse
from bmtk.simulator import filternet


def run(config_file):
    config = filternet.Config.from_json(config_file)
    config.build_env()

    net = filternet.FilterNetwork.from_config(config)
    sim = filternet.FilterSimulator.from_config(config, net)
    sim.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FilterNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default='config.simulation.json', 
                        help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)
