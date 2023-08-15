import os, sys
from bmtk.simulator import pointnet
from bmtk.analyzer.spike_trains import plot_raster
import matplotlib.pyplot as plt

def run(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()

    plot_raster(config_file='simulation_config.json', group_by='pop_name')
    # assert (spike_files_equal('output/spikes.csv', 'expected/spikes.csv'))


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
