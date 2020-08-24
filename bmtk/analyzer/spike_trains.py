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


def _plot_helper(plot_fnc, config_file=None, population=None, times=None, title=None, show=True,
                 group_by=None, group_excludes=None,
                 spikes_file=None, nodes_file=None, node_types_file=None):
    sonata_config = SonataConfig.from_json(config_file) if config_file else None
    pop, spike_trains = _find_spikes(config_file=config_file, spikes_file=spikes_file, population=population)

    # Create the title
    title = title if title is not None else '{} Nodes'.format(pop)

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

    plot_fnc(spike_trains=spike_trains, node_groups=node_groups, population=pop, times=times, title=title, show=show)


def plot_raster(config_file=None, population=None, with_histogram=True, times=None, title=None, show=True,
                group_by=None, group_excludes=None,
                spikes_file=None, nodes_file=None, node_types_file=None):

    plot_fnc = partial(plotting.plot_raster, with_histogram=with_histogram)
    return _plot_helper(plot_fnc,
                        config_file=config_file, population=population, times=times, title=title, show=show,
                        group_by=group_by, group_excludes=group_excludes,
                        spikes_file=spikes_file, nodes_file=nodes_file, node_types_file=node_types_file
    )


def plot_rates(config_file=None, population=None, smoothing=False, smoothing_params=None, times=None, title=None,
               show=True, group_by=None, group_excludes=None, spikes_file=None, nodes_file=None, node_types_file=None):

    plot_fnc = partial(plotting.plot_rates, smoothing=smoothing, smoothing_params=smoothing_params)
    return _plot_helper(plot_fnc,
                        config_file=config_file, population=population, times=times, title=title, show=show,
                        group_by=group_by, group_excludes=group_excludes,
                        spikes_file=spikes_file, nodes_file=nodes_file, node_types_file=node_types_file
    )


def plot_rates_boxplot(config_file=None, population=None, times=None, title=None, show=True,
                       group_by=None, group_excludes=None,
                       spikes_file=None, nodes_file=None, node_types_file=None):

    plot_fnc = partial(plotting.plot_rates_boxplot)
    return _plot_helper(plot_fnc,
        config_file=config_file, population=population, times=times, title=title, show=show,
        group_by=group_by, group_excludes=group_excludes,
        spikes_file=spikes_file, nodes_file=nodes_file, node_types_file=node_types_file
    )


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


def to_dataframe(config_file, spikes_file=None, population=None):
    _, spike_trains = _find_spikes(config_file=config_file, spikes_file=spikes_file, population=population)
    return spike_trains.to_dataframe()
