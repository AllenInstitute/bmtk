import argparse
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
    default_config = 'config.simulation_iclamp.json'
    # default_config = 'config.simulation_iclamp.aslist.json'
    # default_config = 'config.simulation_iclamp.csv.json'
    # default_config = 'config.simulation_iclamp.nwb.json'

    parser = argparse.ArgumentParser(description='Run PointNet network simulation.')
    parser.add_argument('config_path', type=str, nargs='?', default='config.simulation.json', 
                        help='Path to the SONATA configuration file')

    args, _ = parser.parse_known_args()
    run(args.config_path)
