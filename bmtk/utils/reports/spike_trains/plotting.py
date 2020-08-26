# Copyright 2020. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import six
import matplotlib.pyplot as plt
import types
import copy
from functools import partial

from .spike_trains import SpikeTrains
from .spike_trains_api import SpikeTrainsAPI


def __get_spike_trains(spike_trains):
    """Make sure SpikeTrainsAPI object is always returned"""
    if isinstance(spike_trains, six.string_types):
        # Load spikes from file
        return SpikeTrains.load(spike_trains)

    elif isinstance(spike_trains, (SpikeTrains, SpikeTrainsAPI)):
        return spike_trains

    raise AttributeError('Could not parse spiketrains. Pass in file-path, SpikeTrains object, or list of the previous')


def __get_population(spike_trains, population):
    """Helper function to figure out which population of nodes to use."""
    pops = spike_trains.populations
    if population is None:
        # If only one population exists in spikes object/file select that one
        if len(pops) > 1:
            raise Exception('SpikeTrains contains more than one population of nodes. Use "population" parameter '
                            'to specify population to display.')

        else:
            return pops[0]

    elif population not in pops:
        raise Exception('Could not find node population "{}" in SpikeTrains, only found {}'.format(population, pops))

    else:
        return population


def __get_node_groups(spike_trains, node_groups, population):
    """Helper function for parsing the 'node_groups' params"""
    if node_groups is None:
        # If none are specified by user make a 'node_group' consisting of all nodes
        selected_nodes = spike_trains.node_ids(population=population)
        return [{'node_ids': selected_nodes, 'c': 'b'}], selected_nodes
    else:
        # Fetch all node_ids which can be used to filter the data.
        node_groups = copy.deepcopy(node_groups)  # Make a copy since later we may be altering the dictionary
        selected_nodes = np.array(node_groups[0]['node_ids'])
        for grp in node_groups[1:]:
            if 'node_ids' not in grp:
                raise AttributeError('Could not find "node_ids" key in node_groups parameter.')
            selected_nodes = np.concatenate((selected_nodes, np.array(grp['node_ids'])))

        return node_groups, selected_nodes


def plot_raster(spike_trains, with_histogram=True, population=None, node_groups=None, times=None, title=None,
                show=True):
    """will create a raster plot (plus optional histogram) from a SpikeTrains object or SONATA Spike-Trains file. Will
    return the figure

    By default will display all nodes, if you want to only display a subset of nodes and/or group together different
    nodes (by node_id) by dot colors and labels then you can use the node_groups, which should be a list of dicts::

        plot_raster('/path/to/my/spike.h5',
        node_groups=[{'node_ids': range(0, 70), 'c': 'b', 'label': 'pyr'},      # first 70 nodes are blue pyr cells
                     {'node_ids': range(70, 100), 'c': 'r', 'label': 'inh'}])   # last 30 nodes are red inh cells

    The histogram will not be grouped.

    :param spike_trains: SpikeTrains object or path to a (SONATA) spikes file.
    :param with_histogram: If True the a histogram will be shown as a small subplot below the scatter plot. Default
        True.
    :param population: string. If a spikes-file contains more than one population of nodes, use this to determine which
        nodes to actually plot. If only one population exists and population=None then the function will find it by
        default.
    :param node_groups: None or list of dicts. Used to group sets of nodes by labels and color. Each grouping should
        be a dictionary with a 'node_ids' key with a list of the ids. You can also add 'label' and 'c' keys for
        label and color. If None all nodes will be labeled and colored the same.
    :param times: (float, float). Used to set start and stop time. If not specified will try to find values from spiking
        data.
    :param title: str, Use to add a title. Default no tile
    :param show: bool to display or not display plot. default True.
    :return: matplotlib figure.Figure object
    """

    spike_trains = __get_spike_trains(spike_trains=spike_trains)
    pop = __get_population(spike_trains=spike_trains, population=population)
    node_groups, selected_ids = __get_node_groups(spike_trains=spike_trains, node_groups=node_groups, population=pop)

    # Only show a legend if one of the node_groups have an explicit label, otherwise matplotlib will show an empty
    # legend box which looks bad
    show_legend = False

    # Situation where if the last (or first) M nodes don't spike matplotlib will cut off the y range, but it should
    # show these as empty rows. To do this need to keep track of range of all node_ids
    min_id, max_id = np.inf, -1

    spikes_df = spike_trains.to_dataframe(population=pop, with_population_col=False)
    spikes_df = spikes_df[spikes_df['node_ids'].isin(selected_ids)]
    if times is not None:
        min_ts, max_ts = times[0], times[1]
        spikes_df = spikes_df[(spikes_df['timestamps'] >= times[0]) & (spikes_df['timestamps'] <= times[1])]
    else:
        min_ts = np.min(spikes_df['timestamps'])
        max_ts = np.max(spikes_df['timestamps'])

    # Used to determine
    if with_histogram:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [7, 1]}, squeeze=True)
        raster_axes = axes[0]
        bottom_axes = hist_axes = axes[1]
    else:
        fig, axes = plt.subplots(1, 1)
        bottom_axes = raster_axes = axes
        hist_axes = None

    for node_grp in node_groups:
        grp_ids = node_grp.pop('node_ids')
        grp_spikes = spikes_df[spikes_df['node_ids'].isin(grp_ids)]

        # If label exists for at-least one group we want to show
        show_legend = show_legend or 'label' in node_grp

        # Finds min/max node_id for all node groups
        min_id = np.min([np.min(grp_ids), min_id])
        max_id = np.max([np.max(grp_ids), max_id])

        raster_axes.scatter(grp_spikes['timestamps'], grp_spikes['node_ids'], lw=0, s=8, **node_grp)

    if show_legend:
        raster_axes.legend(loc='upper right')

    if title:
        raster_axes.set_title(title)

    raster_axes.set_ylabel('node_ids')
    raster_axes.set_ylim(min_id - 0.5, max_id + 1)  # add buffering to range else the rows at the ends look cut-off.
    raster_axes.set_xlim(min_ts, max_ts + 1)
    bottom_axes.set_xlabel('timestamps ({})'.format(spike_trains.units(population=pop)))

    if with_histogram:
        # Add a histogram if necessarry
        hist_axes.hist(spikes_df['timestamps'], 100)
        hist_axes.set_xlim(min_ts - 0.5, max_ts + 1)
        hist_axes.axes.get_yaxis().set_visible(False)
        raster_axes.set_xticks([])

    if show:
        plt.show()

    return fig


def moving_average(data, window_size=10):
    h = int(window_size / 2)
    x_max = len(data)
    return [np.mean(data[max(0, x - h):min(x_max, x + h)]) for x in range(0, x_max)]


def plot_rates(spike_trains, population=None, node_groups=None, times=None, smoothing=False,
               smoothing_params=None, title=None, show=True):
    """Calculate and plot the rates of each node in a SpikeTrains object or SONATA Spike-Trains file. If start and stop
    times are not specified from the "times" parameter, will try to parse values from the timestamps data.

    If you want to only display a subset of nodes and/or group together different nodes (by node_id) by dot colors and
    labels then you can use the node_groups, which should be a list of dicts::

        plot_rates('/path/to/my/spike.h5',
                    node_groups=[{'node_ids': range(0, 70), 'c': 'b', 'label': 'pyr'},
                                 {'node_ids': range(70, 100), 'c': 'r', 'label': 'inh'}])

    :param spike_trains: SpikeTrains object or path to a (SONATA) spikes file.
    :param population: string. If a spikes-file contains more than one population of nodes, use this to determine which
        nodes to actually plot. If only one population exists and population=None then the function will find it by
        default.
    :param node_groups: None or list of dicts. Used to group sets of nodes by labels and color. Each grouping should
        be a dictionary with a 'node_ids' key with a list of the ids. You can also add 'label' and 'c' keys for
        label and color. If None all nodes will be labeled and colored the same.
    :param times: (float, float). Used to set start and stop time. If not specified will try to find values from spiking
        data.
    :param smoothing: Bool or function. Used to smooth the data. By default (False) no smoothing will be done. If True
        will using a moving average smoothing function. Or use a function pointer.
    :param smoothing_params: dict, parameters when using a function pointer smoothing value.
    :param title: str, Use to add a title. Default no tile
    :param show:  bool to display or not display plot. default True.
    :return: matplotlib figure.Figure object
    """

    spike_trains = __get_spike_trains(spike_trains=spike_trains)
    pop = __get_population(spike_trains=spike_trains, population=population)
    node_groups, selected_ids = __get_node_groups(spike_trains=spike_trains, node_groups=node_groups, population=pop)

    # Determine if smoothing will be applied to the data
    smoothing_params = smoothing_params or {}  # pass in empty parameters
    if isinstance(smoothing, types.FunctionType):
        smoothing_fnc = partial(smoothing, **smoothing_params)
    elif smoothing:
        smoothing_fnc = partial(moving_average, **smoothing_params)
    else:
        smoothing_fnc = lambda d: d  # Use a filler function that won't do anything

    # get data
    spikes_df = spike_trains.to_dataframe(population=pop, with_population_col=False)
    spikes_df = spikes_df[spikes_df['node_ids'].isin(selected_ids)]
    if times is not None:
        recording_interval = times[1] - times[0]
        spikes_df = spikes_df[(spikes_df['timestamps'] >= times[0]) & (spikes_df['timestamps'] <= times[1])]
    else:
        recording_interval = np.max(spikes_df['timestamps']) - np.min(spikes_df['timestamps'])

    # Iterate through each group of nodes and add to the same plot
    fig, axes = plt.subplots()
    show_legend = False  # Only show labels if one of the node group has label value
    for node_grp in node_groups:
        show_legend = show_legend or 'label' in node_grp # If label exists for at-least one group we want to show

        grp_ids = node_grp.pop('node_ids')
        grp_spikes = spikes_df[spikes_df['node_ids'].isin(grp_ids)]
        spike_rates = grp_spikes.groupby('node_ids').size() / (recording_interval / 1000.0)
        axes.plot(np.array(spike_rates.index), smoothing_fnc(spike_rates), '.', **node_grp)

    axes.set_ylabel('Firing Rates (Hz)')
    axes.set_xlabel('node_ids')
    if show_legend:
        axes.legend()  # loc='upper right')

    if title:
        axes.set_title(title)

    if show:
        plt.show()

    return fig


def plot_rates_boxplot(spike_trains, population=None, node_groups=None, times=None, title=None, show=True):
    """Creates a box plot of the firing rates taken from a SpikeTrains object or SONATA Spike-Trains file. If start
    and stop times are not specified from the "times" parameter, will try to parse values from the timestamps data.

    By default will plot all nodes together. To only display a subset of the nodes and/or create groups of nodes use
    the node_groups options::

        plot_rates_boxplot(
            '/path/to/my/spike.h5',
            node_groups=[{'node_ids': range(0, 70), 'label': 'pyr'},
                         {'node_ids': range(70, 100), 'label': 'inh'}]
        )

    :param spike_trains: SpikeTrains object or path to a (SONATA) spikes file.
    :param population: string. If a spikes-file contains more than one population of nodes, use this to determine which
        nodes to actually plot. If only one population exists and population=None then the function will find it by
        default.
    :param node_groups: None or list of dicts. Used to group sets of nodes by labels and color. Each grouping should
        be a dictionary with a 'node_ids' key with a list of the ids. You can also add 'label' and 'c' keys for
        label and color. If None all nodes will be labeled and colored the same.
    :param title: str, Use to add a title. Default no tile
    :param show:  bool to display or not display plot. default True.
    :return: matplotlib figure.Figure object
    """
    spike_trains = __get_spike_trains(spike_trains=spike_trains)
    pop = __get_population(spike_trains=spike_trains, population=population)
    node_groups, selected_ids = __get_node_groups(spike_trains=spike_trains, node_groups=node_groups, population=pop)

    spikes_df = spike_trains.to_dataframe(population=pop, with_population_col=False)
    spikes_df = spikes_df[spikes_df['node_ids'].isin(selected_ids)]
    if times is not None:
        recording_interval = times[1] - times[0]
        spikes_df = spikes_df[(spikes_df['timestamps'] >= times[0]) & (spikes_df['timestamps'] <= times[1])]
    else:
        recording_interval = np.max(spikes_df['timestamps']) - np.min(spikes_df['timestamps'])

    fig, axes = plt.subplots()
    rates_data = []
    rates_labels = []

    if len(node_groups) == 1 and 'label' not in node_groups[0]:
        node_groups[0]['label'] = 'All Nodes'

    for i, node_grp in enumerate(node_groups):
        rates_labels.append(node_grp.get('label', 'Node Group {}'.format(i)))

        grp_ids = node_grp.pop('node_ids')
        grp_spikes = spikes_df[spikes_df['node_ids'].isin(grp_ids)]
        spike_rates = grp_spikes.groupby('node_ids').size() / (recording_interval / 1000.0)

        rates_data.append(spike_rates)

    axes.boxplot(rates_data)
    axes.set_ylabel('Firing Rates (Hz)')
    axes.set_xticklabels(rates_labels)

    if title:
        axes.set_title(title)

    if show:
        plt.show()

    return fig