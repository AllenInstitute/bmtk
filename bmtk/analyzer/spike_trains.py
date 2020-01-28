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
from bmtk.utils.sonata import File


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


def spike_statistics(spikes_file, simulation=None, simulation_time=None, groupby=None, network=None, **filterparams):
    spike_trains = SpikeTrains.load(spikes_file)

    def calc_stats(r):
        d = {}
        vals = np.sort(r['timestamps'])
        diffs = np.diff(vals)
        if diffs.size > 0:
            d['isi'] = np.mean(np.diff(vals))
        else:
            d['isi'] = 0.0

        d['count'] = len(vals)

        return pd.Series(d, index=['count', 'isi'])

    spike_counts_df = spike_trains.to_dataframe().groupby(['population', 'node_ids']).apply(calc_stats)
    spike_counts_df = spike_counts_df.rename({'timestamps': 'counts'}, axis=1)
    spike_counts_df.index.names = ['population', 'node_id']

    if simulation is not None:
        nodes_df = simulation.net.node_properties(**filterparams)
        sim_time_s = simulation.simulation_time(units='s')
        spike_counts_df['firing_rate'] = spike_counts_df['count'] / sim_time_s

        vals_df = pd.merge(nodes_df, spike_counts_df, left_index=True, right_index=True, how='left')
        vals_df = vals_df.fillna({'count': 0.0, 'firing_rate': 0.0, 'isi': 0.0})

        vals_df = vals_df.groupby(groupby)[['firing_rate', 'count', 'isi']].agg([np.mean, np.std])
        return vals_df
    else:
        return spike_counts_df
