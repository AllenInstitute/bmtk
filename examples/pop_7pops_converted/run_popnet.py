import argparse
from bmtk.simulator import popnet


def run(config_file):
    configure = popnet.config.from_json(config_file)
    configure.build_env()

    network = popnet.PopNetwork.from_config(configure, group_by='node_type_id')
    sim = popnet.PopSimulator.from_config(configure, network)
    sim.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PopNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default='config.simulation.json', 
                        help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)

