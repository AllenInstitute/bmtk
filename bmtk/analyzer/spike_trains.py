import numpy as np
import pandas as pd
import h5py


from bmtk.analyzer.visualization.spikes import plot_spikes as raster_plot
from bmtk.analyzer.visualization.spikes import plot_rates as rates_plot
from .io_tools import load_config
from bmtk.utils.spike_trains import SpikesFile


def to_dataframe(config_file, spikes_file=None):
    config = load_config(config_file)
    spikes_file = SpikesFile(config.spikes_file)
    return spikes_file.to_dataframe()

