import numpy as np
import pandas as pd
import six
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import gridspec

from .spike_trains import SpikeTrains
from .core import STReader

from bmtk.utils.sonata.file import File
"""
plot_raster('output/spikes.h5',
            # time_window=(500.0, 1000.0),
            # node_ids=range(50,150),
            # with_histogram=True,
            # overlapping=True,
            # with_labels=True,
            nodes_file='network/internal_nodes.h5',
            node_types_file='network/internal_node_types.csv',
            group_by='model_name',
            )
'''

plot_raster(('output/spikes.h5', 'output.old/spikes.h5'),
            # time_window=(500.0, 1000.0),
            # node_ids=range(50,150),
            with_histogram=True,
            overlapping=False,
            with_labels=['bmtk', 'sonata'],
            # nodes_file='network/internal_nodes.h5',
            # node_types_file='network/internal_node_types.csv',
            # group_by='model_name',
            # with_labels=['bmtk', 'sonata']
            )

'''
plot_raster('output/spikes.h5',
            time_window=(500.0, 1000.0), node_ids=range(50,150),
            with_histogram=True,
            overlapping=True,
            with_labels=True, #['Hello']
            )
'''

plt.figure()
plot_raster('output/spikes.h5',
            # time_window=(500.0, 1000.0),
            # node_ids=range(50,150),
            # with_histogram=True,
            # overlapping=True,
            # with_labels=True,
            nodes_file='network/internal_nodes.h5',
            node_types_file='network/internal_node_types.csv',
            group_by='model_name',
            show_plot=False
            )

plt.figure()
plot_rates('output/spikes.h5',
           # node_ids=range(50, 150),
           nodes_file='network/internal_nodes.h5',
           node_types_file='network/internal_node_types.csv',
           group_by='model_name',
           show_plot=False)
plt.show()

"""





color_map = defaultdict(lambda: np.random.rand(3,1))
color_map.update({0: 'b', 1: 'r', 2: 'g', 3: 'y', 4: 'k'})

def _get_spike_trains(spike_trains):
    """Returns a list of reports.spike_trains.SpikeTrains objects"""
    if isinstance(spike_trains, six.string_types):
        return [SpikeTrains.load(spike_trains)]

    elif isinstance(spike_trains, (SpikeTrains, STReader)):
        return [spike_trains]

    elif isinstance(spike_trains, (list, tuple, np.ndarray, pd.Series)):
        return [_get_spike_trains(st)[0] for st in spike_trains]

    raise Exception('Could not parse spiketrains. Pass in file-path, SpikeTrains object, or list of the previous')


def _get_labels(labels):
    if isinstance(labels, six.string_types):
        return [labels]

    elif isinstance(labels, (list, tuple, np.ndarray, pd.Series)):
        labels = []
        for st in labels:
            if isinstance(st, six.string_types):
                labels.append(st)
            else:
                return None

        return labels

    else:
        return None


def _find_time_window(spike_trains, populations):
    """Use the spike-train(s) to find the time-window to plot"""

    if isinstance(spike_trains, (SpikeTrains, STReader)):
        return spike_trains.time_range(populations=populations)

    else:
        t_mins = [st.time_range(populations=populations)[0] for st in spike_trains]
        t_maxs = [st.time_range(populations=populations)[1] for st in spike_trains]
        return min(t_mins), max(t_maxs)


class Filter(object):
    def __init__(self, node_ids, label):
        self.node_ids = node_ids
        self.label = label

    @classmethod
    def build(cls, nodes_file, node_types_file, populations, group_by, group_exclude, node_ids_filter):
        if group_by is not None:
            sonata_file = File(data_files=nodes_file, data_type_files=node_types_file)
            pop = populations or sonata_file.nodes.populations[0].name
            node_types_df = sonata_file.nodes[pop].node_types_table.to_dataframe()
            pop_nodes = sonata_file.nodes[pop]
            nodes_df = pd.DataFrame({'node_ids': pop_nodes.node_ids, 'node_type_ids': pop_nodes.type_ids})
            nodes_df = pd.merge(left=nodes_df, right=node_types_df, how='left', left_on='node_type_ids',
                                right_index=True)

            for grp_name, grp in nodes_df.groupby(group_by):
                print (grp_name, grp['node_ids'][:10])


def _create_grouping(nodes_file, node_types_file, populations, group_by, node_ids_filter):
    if isinstance(group_by, dict):
        def intersection(node_ids):
            if node_ids_filter is None:
                return node_ids
            else:
                intersection = set(node_ids_filter) & set(node_ids)
                return sorted(list(intersection))

        return [(label, intersection(node_ids)) for label, node_ids in group_by.items()]

    elif group_by is not None:
        sonata_file = File(data_files=nodes_file, data_type_files=node_types_file)
        pop = populations or sonata_file.nodes.populations[0].name
        node_types_df = sonata_file.nodes[pop].node_types_table.to_dataframe()
        pop_nodes = sonata_file.nodes[pop]
        nodes_df = pd.DataFrame({'node_ids': pop_nodes.node_ids, 'node_type_ids': pop_nodes.type_ids})
        if node_ids_filter is not None:
            nodes_df = nodes_df[nodes_df['node_ids'].isin(node_ids_filter)]

        nodes_df = pd.merge(left=nodes_df, right=node_types_df, how='left', left_on='node_type_ids',
                            right_index=True)

        return [(grp_name, np.array(grp['node_ids'])) for grp_name, grp in nodes_df.groupby(group_by)]

    else:
        return [(None, node_ids_filter)]


def _build_labels_lu(labels, spike_trains_file):
    if not labels:
        labels = defaultdict(lambda: '')
    elif isinstance(labels, (list, tuple, np.ndarray, pd.Series)):
        labels = labels
    else:
        labels = [labels] if isinstance(spike_trains_file, six.string_types) else ['']

    return labels


def plot_raster(spike_trains, population=None, time_window=None, node_ids=None, ts_units=None, overlapping=False,
                with_histogram=True,
                show_plot=True, save_as=None, with_labels=True,
                nodes_file=None, node_types_file=None, group_by=None):
    # TODO: This is assuming all the units are the same (ms). Will cause problem if not.
    spike_trains_l = _get_spike_trains(spike_trains)
    if overlapping and len(spike_trains_l) > 1:
        return plot_raster_cmp(spike_trains, population=population, time_window=time_window, node_ids=node_ids,
                               ts_units=ts_units, show_plot=show_plot, save_as=save_as, with_labels=with_labels)


    # if user doesn't specifiy time-window then get it from the spike trains
    time_window = time_window or _find_time_window(spike_trains_l, population)

    labels = _build_labels_lu(with_labels, spike_trains)

    '''
    # Get the labels
    if not with_labels:
        labels = defaultdict(lambda: '')
    elif isinstance(with_labels, (list, tuple, np.ndarray, pd.Series)):
        labels = with_labels
    else:
        labels = [spike_trains] if isinstance(spike_trains, six.string_types) else spike_trains
    '''

    groups = _create_grouping(nodes_file, node_types_file, population, group_by, node_ids)

    n_rasters = len(spike_trains_l)
    '''
    if overlapping and n_rasters > 1:
        gs = gridspec.GridSpec(1, 1)
        raster_gs = [gs[0]]*n_rasters
        hists_gs = []
        last_gs = raster_gs[-1]
    '''
    if with_histogram:
        gs = gridspec.GridSpec(n_rasters*2, 1, height_ratios=[7 if i%2 == 0 else 1 for i in range(n_rasters*2)])
        raster_gs = [gs[i*2] for i in range(n_rasters)]
        hists_gs = [gs[i*2 + 1] for i in range(n_rasters)]
        last_gs = hists_gs[-1]
    else:
        gs = gridspec.GridSpec(n_rasters, height_ratios=[1]*len(n_rasters))
        raster_gs = [gs[i] for i in range(n_rasters)]
        hists_gs = []
        last_gs = raster_gs[-1]

    for i, spikes in enumerate(spike_trains_l):
        gs_i = raster_gs[i]
        for grp_label, grp_nodes in groups:
            spikes_df = spikes.to_dataframe(populations=population, time_window=time_window, node_ids=grp_nodes)
            ax1 = plt.subplot(gs_i)
            ax1.scatter(spikes_df['timestamps'], spikes_df['node_ids'], lw=0, s=5, label=grp_label or labels[i])
            ax1.legend(loc=1, prop={'size': 10})
            ax1.set_xlim([time_window[0], time_window[1]])
            ax1.set_ylabel('node_id')
        if gs_i != last_gs:
            ax1.set_xticks([])
        else:
            ax1.set_xlabel('time ({})'.format(ts_units))

    for spikes, gs_i in zip(spike_trains_l, hists_gs):
        spikes_df = spikes.to_dataframe(populations=population, time_window=time_window, node_ids=node_ids)
        ax_hist = plt.subplot(gs_i)
        ax_hist.hist(spikes_df['timestamps'], 100)
        ax_hist.axes.get_yaxis().set_visible(False)
        ax_hist.set_xlim([time_window[0], time_window[1]])
        if gs_i != last_gs:
            ax_hist.set_xticks([])
        else:
            ax_hist.set_xlabel('time ({})'.format(ts_units))

    if save_as is not None:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def plot_raster_cmp(spike_trains, population=None, time_window=None, node_ids=None, ts_units=None, show_plot=True,
                    save_as=None, with_labels=False):
    spike_trains_l = _get_spike_trains(spike_trains)
    time_window = time_window or _find_time_window(spike_trains_l, population)
    labels = _build_labels_lu(with_labels, spike_trains)

    gs = gridspec.GridSpec(1, 1)
    for i, spikes in enumerate(spike_trains_l):
        spikes_df = spikes.to_dataframe(populations=population, time_window=time_window, node_ids=node_ids)
        ax1 = plt.subplot(gs[0])
        ax1.scatter(spikes_df['timestamps'], spikes_df['node_ids'], lw=0, s=5, label=labels[i])
        ax1.legend(loc=1, prop={'size': 10})
        ax1.set_xlim([time_window[0], time_window[1]])

    if save_as is not None:
        plt.savefig(save_as)

    if show_plot:
        plt.show()


def plot_rates(spike_trains, population=None, time_window=None, node_ids=None, show_plot=True,
               smoothed=True, window_size=20,
               nodes_file=None, node_types_file=None, group_by=None
               ):
    def smooth(data):
        h = int(window_size/2)
        x_max = len(data)
        return [np.mean(data[max(0, x-h):min(x_max, x+h)]) for x in xrange(0, x_max)]


    spike_trains_l = _get_spike_trains(spike_trains)
    time_window = time_window or _find_time_window(spike_trains_l, population)
    sim_time = time_window[1] - time_window[0]

    groups = _create_grouping(nodes_file, node_types_file, population, group_by, node_ids)

    for i, spikes in enumerate(spike_trains_l):
        for grp_label, grp_nodes in groups:
            spikes_df = spikes.to_dataframe(populations=population, time_window=time_window, node_ids=grp_nodes)
            spike_rates = spikes_df.groupby('node_ids').size()/(sim_time/1000.0)
            # Fill in missing gids with 0.0
            spike_rates = spike_rates.reindex(range(spike_rates.index.min(), int(spike_rates.index.max()+1))).fillna(0.0)
            if smoothed:
                plt.plot(spike_rates.index, smooth(spike_rates), label=grp_label)
            else:
                plt.plot(spike_rates.index, spike_rates)
            plt.legend(loc=1, prop={'size': 10})

    plt.ylabel('Firing Rate (Hz)')
    plt.xlabel('node_id')
    if show_plot:
        plt.show()
