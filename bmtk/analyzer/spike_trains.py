import os
import numpy as np
import pandas as pd
from functools import partial
from six import string_types

from bmtk.utils import sonata
from bmtk.utils.sonata.config import SonataConfig
from bmtk.utils.reports import SpikeTrains
from bmtk.utils.reports.spike_trains import plotting
from bmtk.simulator.utils import simulation_reports


def _find_spikes(spikes_file=None, config_file=None, population=None):
    candidate_spikes = []

    # Get spikes file(s)
    if spikes_file:
        # User has explicity set the location of the spike files
        candidate_spikes.append(spikes_file)

    elif config_file is not None:
        # Otherwise search the config.json for all possible output spikes_files. We can use the simulation_reports
        # module to find any spikes output file specified in config's "output" or "reports" section.
        config = SonataConfig.from_json(config_file)
        sim_reports = simulation_reports.from_config(config)
        for report in sim_reports:
            if report.module == 'spikes_report':
                # BMTK can end up output the same spikes file in SONATA, CSV, and NWB format. Try fetching the SONATA
                # version first, then CSV, and finally NWB if it exists.
                spikes_sonata = report.params.get('spikes_file', None)
                spikes_csv = report.params.get('spikes_file_csv', None)
                spikes_nwb = report.params.get('spikes_file_nwb', None)

                if spikes_sonata is not None:
                    candidate_spikes.append(spikes_sonata)
                elif spikes_csv is not None:
                    candidate_spikes.append(spikes_csv)
                elif spikes_csv is not None:
                    candidate_spikes.append(spikes_nwb)

        # TODO: Should we also look in the "inputs" for displaying input spike statistics?

    if not candidate_spikes:
        raise ValueError('Could not find an output spikes-file. Use "spikes_file" parameter option.')

    # Find file that contains spikes for the specified "population" of nodes. If "population" parameter is not
    # specified try to guess that spikes that the user wants to visualize.
    if population is not None:
        spikes_obj = None
        for spikes_f in candidate_spikes:
            st = SpikeTrains.load(spikes_f)
            if population in st.populations:
                if spikes_obj is None:
                    spikes_obj = st
                else:
                    spikes_obj.merge(st)

        if spikes_obj is None:
            raise ValueError('Could not fine spikes file with node population "{}".'.format(population))
        else:
            return population, spikes_obj

    else:
        if len(candidate_spikes) > 1:
            raise ValueError('Found more than one spike-trains file')

        spikes_f = candidate_spikes[0]
        if not os.path.exists(spikes_f):
            raise ValueError('Did not find spike-trains file {}. Make sure the simulation has completed.'.format(
                spikes_f))

        spikes_obj = SpikeTrains.load(spikes_f)

        if len(spikes_obj.populations) > 1:
            raise ValueError('Spikes file {} contains more than one node population.'.format(spikes_f))
        else:
            return spikes_obj.populations[0], spikes_obj


def _find_nodes(population, config=None, nodes_file=None, node_types_file=None):
    if nodes_file is not None:
        network = sonata.File(data_files=nodes_file, data_type_files=node_types_file)
        if population not in network.nodes.population_names:
            raise ValueError('node population "{}" not found in {}'.format(population, nodes_file))
        return network.nodes[population]

    elif config is not None:
        for nodes_grp in config.nodes:
            network = sonata.File(data_files=nodes_grp['nodes_file'], data_type_files=nodes_grp['node_types_file'])
            if population in network.nodes.population_names:
                return network.nodes[population]

    raise ValueError('Could not find nodes file with node population "{}".'.format(population))


def _plot_helper(plot_fnc, config_file=None, population=None, times=None, title=None, show=True, save_as=None,
                 group_by=None, group_excludes=None,
                 spikes_file=None, nodes_file=None, node_types_file=None):
    sonata_config = SonataConfig.from_json(config_file) if config_file else None
    pop, spike_trains = _find_spikes(config_file=config_file, spikes_file=spikes_file, population=population)

    # Create the title
    title = title if title is not None else "Nodes in network '{}'".format(pop)

    # Get start and stop times from config if needed
    if sonata_config and times is None:
        times = (sonata_config.tstart, sonata_config.tstop)

    # Create node-groups
    if group_by is not None:
        node_groups = []
        nodes = _find_nodes(population=pop, config=sonata_config, nodes_file=nodes_file,
                            node_types_file=node_types_file)
        grouped_df = None
        for grp in nodes.groups:
            if group_by in grp.all_columns:
                grp_df = grp.to_dataframe()
                grp_df = grp_df[['node_id', group_by]]
                grouped_df = grp_df if grouped_df is None else grouped_df.append(grp_df, ignore_index=True)

        if grouped_df is None:
            raise ValueError('Could not find any nodes with group_by attribute "{}"'.format(group_by))

        # Convert from string to list so we can always use the isin() method for filtering
        if isinstance(group_excludes, string_types):
            group_excludes = [group_excludes]
        elif group_excludes is None:
            group_excludes = []

        for grp_key, grp in grouped_df.groupby(group_by):
            if grp_key in group_excludes:
                continue
            node_groups.append({'node_ids': np.array(grp['node_id']), 'label': grp_key})

    else:
        node_groups = None

    return plot_fnc(
        spike_trains=spike_trains, node_groups=node_groups, population=pop, times=times, title=title, show=show,
        save_as=save_as
    )


def plot_raster(config_file=None, population=None, with_histogram=True, times=None, title=None, show=True,
                save_as=None, group_by=None, group_excludes=None,
                spikes_file=None, nodes_file=None, node_types_file=None, plt_style=None):
    """Create a raster plot (plus optional histogram) from the results of the simulation.

    Will using the SONATA simulation configs "output" section to locate where the spike-trains file was created and
    display them::

        plot_raster(config_file='config.json')

    If the path the the report is different (or missing) than what's in the SONATA config then use the "spikes_file"
    option instead::

        plot_raster(spikes_file='/my/path/to/membrane_potential.h5')

    You may also group together different subsets of nodes using specific attributes of the network using the "group_by"
    option, and the "group_excludes" option to exclude specific subsets. For example to color and label different
    subsets of nodes based on their cortical "layer", but exlcude plotting the L1 nodes::

        plot_raster(config_file='config.json', groupy_by='layer', group_excludes='L1')

    :param config_file: path to SONATA simulation configuration.
    :param population: name of the membrane_report "report" which will be plotted. If only one compartment report
        in the simulation config then function will find it automatically.
    :param with_histogram: If True the a histogram will be shown as a small subplot below the scatter plot. Default
        True.
    :param times: (float, float), start and stop times of simulation. By default will get values from simulation
        configs "run" section.
    :param title: str, adds a title to the plot. If None (default) then name will be automatically generated using the
        report_name.
    :param show: bool to display or not display plot. default True.
    :param save_as: None or str: file-name/path to save the plot as a png/jpeg/etc. If None or empty string will not
        save plot.
    :param group_by: Attribute of the "nodes" file used to group and average subsets of nodes.
    :param group_excludes: list of strings or None. When using the "group_by", allows users to exclude certain groupings
        based on the attribute value.
    :param spikes_file: Path to SONATA spikes file. Do not use with "config_file" options.
    :param nodes_file: path to nodes hdf5 file containing "population". By default this will be resolved using the
        config.
    :param node_types_file: path to node-types csv file containing "population". By default this will be resolved using
        the config.
    :return: matplotlib figure.Figure object
    """
    plot_fnc = partial(plotting.plot_raster, with_histogram=with_histogram, plt_style=plt_style)
    return _plot_helper(
        plot_fnc,
        config_file=config_file, population=population, times=times, title=title, show=show, save_as=save_as,
        group_by=group_by, group_excludes=group_excludes,
        spikes_file=spikes_file, nodes_file=nodes_file, node_types_file=node_types_file
    )


def plot_rates(config_file=None, population=None, smoothing=False, smoothing_params=None, times=None, title=None,
               show=True, save_as=None, group_by=None, group_excludes=None, spikes_file=None, nodes_file=None,
               node_types_file=None, plt_style=None):
    """Calculate and plot the rates of each node recorded during the simulation - averaged across the entirety of the
    simulation.

    Will using the SONATA simulation configs "output" section to locate where the spike-trains file was created and
    display them::

        plot_rates(config_file='config.json')

    If the path the the report is different (or missing) than what's in the SONATA config then use the "spikes_file"
    option instead::

        plot_rates(spikes_file='/my/path/to/membrane_potential.h5')

    You may also group together different subsets of nodes using specific attributes of the network using the "group_by"
    option, and the "group_excludes" option to exclude specific subsets. For example to color and label different
    subsets of nodes based on their cortical "layer", but exlcude plotting the L1 nodes::

        plot_rates(config_file='config.json', groupy_by='layer', group_excludes='L1')


    :param config_file: path to SONATA simulation configuration.
    :param population: name of the membrane_report "report" which will be plotted. If only one compartment report
        in the simulation config then function will find it automatically.
    :param smoothing: Bool or function. Used to smooth the data. By default (False) no smoothing will be done. If True
        will using a moving average smoothing function. Or use a function pointer.
    :param smoothing_params: dict, parameters when using a function pointer smoothing value.
    :param times: (float, float), start and stop times of simulation. By default will get values from simulation
        configs "run" section.
    :param title: str, adds a title to the plot. If None (default) then name will be automatically generated using the
        report_name.
    :param show: bool to display or not display plot. default True.
    :param save_as: None or str: file-name/path to save the plot as a png/jpeg/etc. If None or empty string will not
        save plot.
    :param group_by: Attribute of the "nodes" file used to group and average subsets of nodes.
    :param group_excludes: list of strings or None. When using the "group_by", allows users to exclude certain groupings
        based on the attribute value.
    :param spikes_file: Path to SONATA spikes file. Do not use with "config_file" options.
    :param nodes_file: Path to nodes hdf5 file containing "population". By default this will be resolved using the
        config.
    :param node_types_file: Path to node-types csv file containing "population". By default this will be resolved using
        the config.
    :return: matplotlib figure.Figure object
    """
    plot_fnc = partial(plotting.plot_rates, smoothing=smoothing, smoothing_params=smoothing_params, plt_style=plt_style)
    return _plot_helper(
        plot_fnc,
        config_file=config_file, population=population, times=times, title=title, show=show, save_as=save_as,
        group_by=group_by, group_excludes=group_excludes,
        spikes_file=spikes_file, nodes_file=nodes_file, node_types_file=node_types_file
    )


def plot_rates_boxplot(config_file=None, population=None, times=None, title=None, show=True, save_as=None,
                       group_by=None, group_excludes=None,
                       spikes_file=None, nodes_file=None, node_types_file=None, plt_style=None):
    """Creates a box plot of the firing rates taken from nodes recorded during the simulation.

    Will using the SONATA simulation configs "output" section to locate where the spike-trains file was created and
    display them::

        plot_rates_boxplot(config_file='config.json')

    If the path the the report is different (or missing) than what's in the SONATA config then use the "spikes_file"
    option instead::

        plot_rates_boxplot(spikes_file='/my/path/to/membrane_potential.h5')

    You may also group together different subsets of nodes using specific attributes of the network using the "group_by"
    option, and the "group_excludes" option to exclude specific subsets. For example to color and label different
    subsets of nodes based on their cortical "layer", but exlcude plotting the L1 nodes::

        plot_rates_boxplot(config_file='config.json', groupy_by='layer', group_excludes='L1')

    :param config_file: path to SONATA simulation configuration.
    :param population: name of the membrane_report "report" which will be plotted. If only one compartment report
        in the simulation config then function will find it automatically.
    :param times: (float, float), start and stop times of simulation. By default will get values from simulation
        configs "run" section.
    :param title: str, adds a title to the plot. If None (default) then name will be automatically generated using the
        report_name.
    :param show: bool to display or not display plot. default True.
    :param save_as: None or str: file-name/path to save the plot as a png/jpeg/etc. If None or empty string will not
        save plot.
    :param group_by: Attribute of the "nodes" file used to group and average subsets of nodes.
    :param group_excludes: list of strings or None. When using the "group_by", allows users to exclude certain groupings
        based on the attribute value.
    :param spikes_file: Path to SONATA spikes file. Do not use with "config_file" options.
    :param nodes_file: Path to nodes hdf5 file containing "population". By default this will be resolved using the
        config.
    :param node_types_file: Path to node-types csv file containing "population". By default this will be resolved using
        the config.
    :return: matplotlib figure.Figure object
    """
    plot_fnc = partial(plotting.plot_rates_boxplot, plt_style=plt_style)
    return _plot_helper(
        plot_fnc,
        config_file=config_file, population=population, times=times, title=title, show=show, save_as=save_as,
        group_by=group_by, group_excludes=group_excludes,
        spikes_file=spikes_file, nodes_file=nodes_file, node_types_file=node_types_file
    )


def spike_statistics(spikes_file, simulation=None, population=None, simulation_time=None, group_by=None, network=None,
                     config_file=None, **filterparams):
    """Get spike statistics (firing_rate, counts, inter-spike interval) of the nodes.

    :param spikes_file: Path to SONATA spikes file. Do not use with "config_file" options.
    :param simulation:
    :param population:
    :param simulation_time:
    :param groupby:
    :param network:
    :param config_file:
    :param filterparams:
    :return: pandas dataframe
    """

    # TODO: Should be implemented in bmtk.utils.spike_trains.stats.py
    pop, spike_trains = _find_spikes(config_file=config_file, spikes_file=spikes_file, population=population)
    # spike_trains = SpikeTrains.load(spikes_file)

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

        vals_df = vals_df.groupby(group_by)[['firing_rate', 'count', 'isi']].agg([np.mean, np.std])
        return vals_df
    else:
        return spike_counts_df


def to_dataframe(config_file, spikes_file=None, population=None):
    """

    :param config_file:
    :param spikes_file:
    :param population:
    :return:
    """
    # _, spike_trains = _find_spikes(config_file=config_file, spikes_file=spikes_file, population=population)
    pop, spike_trains = _find_spikes(config_file=config_file, spikes_file=spikes_file, population=population)

    return spike_trains.to_dataframe()
