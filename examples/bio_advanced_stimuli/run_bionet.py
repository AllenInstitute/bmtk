from bmtk.simulator import bionet
from bmtk.analyzer.compartment import plot_traces


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    plot_traces(config_file=config_file, report_name='membrane_potential', population='bio')
    bionet.nrn.quit_execution()


if __name__ == '__main__':
    # default_config = 'config.simulation_iclamp.json'
    # default_config = 'config.simulation_iclamp.aslist.json'
    # default_config = 'config.simulation_iclamp.csv.json'
    # default_config = 'config.simulation_iclamp.nwb.json'
    default_config = 'config.simulation_xstim.json'
    # default_config = 'config.simulation_spikes.json'
    # default_config = 'config.simulation_spont_activity.json'

    parser = bionet.ArgumentParser(description='Run BioNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default=default_config, help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)
