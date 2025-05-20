import numpy as np
import argparse

from bmtk.simulator import popnet
from bmtk.analyzer.firing_rates import plot_rates


@popnet.inputs_generator
def load_bkg_inputs(node, sim, **opts):
    # print(node.node_id, node.population)
    # print(sim.dt, sim.tstart, sim.tstop, sim.nsteps)
    return np.ones(sim.nsteps)


@popnet.init_function
def set_init_states(node, sim, **opts):
    if node['pop_name'] == 'Exc':
        return 1.38853652
    elif node['pop_name'] == 'PV':
        return 3.50304166
    elif node['pop_name'] == 'SST':
        return 1.61686557
    else:
        return 5.02447877


@popnet.activation_function
def tanh(state_arr):
    return np.tanh(state_arr)


def run(configuration_path):
    configure = popnet.Config.from_json(configuration_path)
    configure.build_env()

    network = popnet.PopNetwork.from_config(configure)
    sim = popnet.PopSimulator.from_config(configure, network)
    # sim.set_activation_function(tanh)
    sim.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', nargs='?', default='config.simulation.json', help='SONATA configuration json file.')
    args = parser.parse_args()
    run(args.config)

    plot_rates(args.config, label_column='pop_name')
