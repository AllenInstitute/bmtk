import sys
import os

from bmtk.simulator import popnet
from bmtk.analyzer.firing_rates import plot_rates_popnet


def plot_rates(cells_path='network/internal_node_types.csv', rates_path='output/spike_rates.csv'):
    plot_rates_popnet(cells_path, rates_path, model_keys='pop_name')


def main(config_file):
    # initialize and run the simulation
    configure = popnet.config.from_json(config_file)
    configure.build_env()

    network = popnet.PopNetwork.from_config(configure)
    sim = popnet.PopSimulator.from_config(configure, network)
    sim.run()

    # plot the results
    plot_rates()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.simulation.json')
