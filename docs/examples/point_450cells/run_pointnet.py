import os, sys
from bmtk.simulator import pointnet
from bmtk.analyzer.visualization.spikes import plot_spikes


def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    graph = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, graph)
    sim.run()

    # plot_spikes('network/internal_nodes.h5', 'network/internal_node_types.csv', 'output/spikes.h5')


if __name__ == '__main__':
    main('config.json')
