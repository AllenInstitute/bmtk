from bmtk.simulator import pointnet
from bmtk.analyzer.spike_trains import plot_raster


def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()

    plot_raster(config_file='simulation_config.json', group_by='pop_name')
    # assert (spike_files_equal('output/spikes.csv', 'expected/spikes.csv'))


if __name__ == '__main__':
    main('config.json')