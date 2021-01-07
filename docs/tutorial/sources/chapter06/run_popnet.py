import sys
import os

from bmtk.simulator import popnet

from bmtk.analyzer.firing_rates import plot_rates_popnet

def main(config_file):
    configure = popnet.config.from_json(config_file)
    configure.build_env()
    network = popnet.PopNetwork.from_config(configure)
    sim = popnet.PopSimulator.from_config(configure, network)
    sim.run()

    cells_file = 'network/brunel_node_types.csv'
    rates_file = 'output/spike_rates.csv'
    plot_rates_popnet(cells_file, rates_file, model_keys='pop_name')


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        main(sys.argv[-1])
    else:
        main('config.json')