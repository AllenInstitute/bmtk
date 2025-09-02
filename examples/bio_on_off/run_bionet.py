from bmtk.simulator import bionet


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    parser = bionet.ArgumentParser(description='Run BioNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default='config.simulation.json', help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)
