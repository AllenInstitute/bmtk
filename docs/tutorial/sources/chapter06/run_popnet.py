import sys
import os

from bmtk.simulator import popnet

from bmtk.analyzer.firing_rates import plot_rates_popnet


def run(config_file):
    configure = popnet.config.from_json(config_file)
    configure.build_env()
    network = popnet.PopNetwork.from_config(configure)
    sim = popnet.PopSimulator.from_config(configure, network)
    sim.run()

    cells_file = 'network/V1_node_types.csv'
    rates_file = 'output/firing_rates.csv'
    plot_rates_popnet(cells_file, rates_file, model_keys='pop_name')


if __name__ == '__main__':
    # Find the appropriate config.json file
    config_path = None
    if __file__ != sys.argv[-1]:
        config_path = sys.argv[-1]
        if not os.path.exists(config_path):
            raise AttributeError('configuration file {} does not exist.'.format(config_path))
    else:
        for cfg_path in ['config.json', 'config.simulation.json', 'simulation_config.json']:
            if os.path.exists(cfg_path):
                config_path = cfg_path
                break
        else:
            raise AttributeError('Could not find configuration json file.')

    run(config_path)
