import os
import sys
import nest

from bmtk.simulator import pointnet
from bmtk.analyzer.spikes_analyzer import spike_files_equal
from bmtk.simulator.pointnet.pyfunction_cache import py_modules


def process_model(model_type, node, dynamics_params):
    return nest.Create(model_type, 1, dynamics_params)


def gaussianLL(edge_props, source_node, target_node):
    return edge_props['syn_weight']


def main(config_file):
    configure = pointnet.Config.from_json(config_file)
    configure.build_env()

    network = pointnet.PointNetwork.from_config(configure)
    py_modules.add_cell_processor('process_model', process_model)
    py_modules.add_synaptic_weight('gaussianLL', gaussianLL)

    sim = pointnet.PointSimulator.from_config(configure, network)
    sim.run()
    assert (spike_files_equal('output/spikes.csv', 'expected/spikes.csv'))


if __name__ == '__main__':
    main('config.json')
