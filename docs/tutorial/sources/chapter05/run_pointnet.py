import nest

from bmtk.simulator import pointnet
from bmtk.analyzer.visualization.spikes import plot_spikes, plot_rates


def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()

    plot_spikes('network/V1_nodes.h5', 'network/V1_node_types.csv', 'output/spikes.h5')
    # assert (spike_files_equal('output/spikes.csv', 'expected/spikes.csv'))


if __name__ == '__main__':
    main('config.json')