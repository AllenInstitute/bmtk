"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import sys
import matplotlib.pyplot as plt
import h5py
import numpy as np

from bmtk.simulator import bionet
from bmtk.analyzer.compartment import plot_traces


def show_cell_var(conf, var_name):
    plot_traces(config_file=conf, report_name=var_name)


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    show_cell_var(config_file, 'membrane_potential')


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        # Make sure to run only one at a time
        run('config_iclamp.json')  # Current clamp stimulation
        # run('config_xstim.json')  # Extracellular electrode stimulation
        # run('config_spikes_input.json')  # Synaptic stimulation with external virtual cells
