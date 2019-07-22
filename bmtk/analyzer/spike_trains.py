import numpy as np
import pandas as pd
import h5py


# from bmtk.analyzer.visualization.spikes import plot_spikes as raster_plot
# from bmtk.analyzer.visualization.spikes import plot_rates as rates_plot
# from .io_tools import load_config
import matplotlib.pyplot as plt


from bmtk.simulator.utils.config import ConfigDict
from bmtk.utils.reports import SpikeTrains
from bmtk.utils.reports.spike_trains import plotting


def load_spikes_file(config_file=None, spikes_file=None):
    if spikes_file is not None:
        return SpikeTrains.load(spikes_file)

    elif config_file is not None:
        config = ConfigDict.from_json(config_file)
        return SpikeTrains.load(config.spikes_file)


def to_dataframe(config_file, spikes_file=None):
    spike_trains = load_spikes_file(config_file=config_file, spikes_file=spikes_file)
    return spike_trains.to_dataframe()


def plot_raster(config_file, spikes_file=None):
    spike_trains = load_spikes_file(config_file=config_file, spikes_file=spikes_file)
    plotting.plot_raster(spike_trains)
    plt.show()


def plot_rates(config_file):
    spike_trains = load_spikes_file(config_file)
    plotting.plot_rates(spike_trains)

