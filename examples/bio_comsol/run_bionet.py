"""Simulates an example network of 100 cells receiving two kinds of exernal input as defined in the configuration file"""
import sys
import matplotlib.pyplot as plt
import h5py
import numpy as np
import json
import os 

sys.path.append('../..')
sys.path.append('..')

from bmtk.simulator import bionet
from bmtk.analyzer.compartment import plot_traces
from bio_components.voltage_waveform import CreateVoltageWaveform
    
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# MPI_RANK = comm.Get_rank()

def show_cell_var(conf, var_name):
    plot_traces(config_file=conf, report_name=var_name)


def run(config_file):

    # with open (config_file, "r") as f:
    #     parameters = json.load(f)

    # time_step = parameters["run"]["dt"]
    # currentamplitude = parameters["run"]["current_amplitude"]

    # print("time_step =", time_step)
    # print("current_amplitude =", currentamplitude)

    # #creates the file xstim
    # CreateVoltageWaveform(current_amplitude=currentamplitude, timestep=time_step, plotting=False) 

    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    # show_cell_var(config_file, 'membrane_potential')


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        # Make sure to run only one at a time
        run('config_iclamp.json')  # Current clamp stimulation
        # run('config_xstim.json')  # Extracellular electrode stimulation
        # run('config_spikes_input.json')  # Synaptic stimulation with external virtual cells
