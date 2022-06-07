import matplotlib.pyplot as plt
import numpy as np
import copy
import warnings

from .compartment_report import CompartmentReport


def __get_population(report, population):
    """Helper function to figure out which population of nodes to use."""
    pops = report.populations
    if population is None:
        # If only one population exists in spikes object/file select that one
        if len(pops) > 1:
            raise Exception('Compartment Report contains more than one population of nodes. Use "population" parameter '
                            'to specify population to display.')

        else:
            return pops[0]

    elif population not in pops:
        raise Exception('Could not find node population "{}" in , only found {}'.format(population, pops))

    else:
        return population

def __get_node_groups(report, node_groups, population):
    """Helper function for parsing the 'node_groups' params"""
    if node_groups is None:
        # If none are specified by user make a 'node_group' consisting of all nodes
        selected_nodes = report.node_ids(population=population)
        return [{'node_ids': selected_nodes, 'c': 'b', 'label': 'averaged'}], selected_nodes
    else:
        # Fetch all node_ids which can be used to filter the data.
        node_groups = copy.deepcopy(node_groups)  # Make a copy since later we may be altering the dictionary
        selected_nodes = np.array(node_groups[0]['node_ids'])
        for grp in node_groups[1:]:
            if 'node_ids' not in grp:
                raise AttributeError('Could not find "node_ids" key in node_groups parameter.')
            selected_nodes = np.concatenate((selected_nodes, np.array(grp['node_ids'])))

        return node_groups, selected_nodes


def plot_traces(report, population=None, node_ids=None, sections='origin', average=False, node_groups=None, times=None,
                title=None, show_legend=None, show=True, save_as=None, plt_style=None):
    """Displays the time trace of one or more nodes from a SONATA CompartmentReport file.

    To plot a group of individual variable traces (based on their soma)::

       plot_traces('/path/to/report.h5', node_ids=[0, 10, 20, ...])

    If node_ids=None (default) then all nodes in the report will be displayed. For large networks then can become
    difficult to visualize so it's recommended you use average=True::

        plot_traces('/path/to/report.h5', average=True)

    Users also have the option of taking the averages of multiple subsets of nodes using the "node_groups" options.
    "node_groups" should be a list of dictionary, each dict with a 'node_ids': [list of ids], and optionally
    a 'label' and 'c' (color). For example support nodes [0, 70) are excitatory cells, nodes [70, 100) are inhibitory,
    and we want display excitatory and inhibitory averages and blue and red, respectivly::

        plot_traces('/path/to/report.h5',
                    node_groups=[{'node_ids': range(0, 70), 'label': 'exc', 'c': 'b'},
                                 {'node_ids': range(70, 100), 'label': 'inh', 'c': 'r'}])

    :param report: Path to SONATA report file or CompartmentReport object
    :param population: string. If the report more than one population of nodes, use this to determine which nodes to
           plot. If only one population exists and population=None then the function will find it by default.
    :param node_ids: int or list of ints. Individual node to display the variable
    :param sections: 'origin', 'all', or list of ids, Compartments/elements to display, By default will only show values
           at the soma.
    :param average: If True will take the averages of all/selected nodes. Default False
    :param node_groups: None or list of dicts. Used to group sets of nodes by labels and color. Each grouping should
        be a dictionary with a 'node_ids' key with a list of the ids. You can also add 'label' and 'c' keys for
        label and color. If None all nodes will be labeled and colored the same.
    :param times: (float, float), start and stop times of simulation
    :param title: str, adds a title to the plot
    :param show_legend: Set True or False to determine if legend should be displayed on the plot. The default (None)
           function itself will guess if legend should be shown.
    :param show: bool to display or not display plot. default True.
    :param save_as: None or str: file-name/path to save the plot as a png/jpeg/etc. If None or empty string will not
        save plot.
    :return: matplotlib figure.Figure object
    """
    if plt_style is not None:
        plt.style.use(plt_style)

    if node_groups is not None and node_ids is not None:
        warnings.warn('plot_traces is called with both "node_ids" and "node_groups" parameters.', UserWarning)

    elif node_ids is not None:
        if average and (not np.isscalar(node_ids)) and len(node_ids) > 1:
            node_groups = {'node_ids': node_ids, 'label': average}
            return plot_traces_averaged(report=report, population=population, sections=sections,
                                        node_groups=node_groups, times=times, title=title, show_legend=show_legend,
                                        show=show, save_as=save_as)

        return plot_traces_individual(report=report, population=population, node_ids=node_ids, sections=sections,
                                      times=times, title=title, show_legend=show_legend, show=show, save_as=save_as)

    elif node_groups is not None:
        return plot_traces_averaged(report=report, population=population, sections=sections, node_groups=node_groups,
                                    times=times, title=title, show_legend=show_legend, show=show, save_as=save_as)

    elif average is not None:
        return plot_traces_averaged(report=report, population=population, sections=sections, times=times, title=title,
                                    show_legend=show_legend, show=show, save_as=save_as)

    else:
        return plot_traces_individual(report=report, population=population, sections=sections, times=times, title=title,
                                      show_legend=show_legend, show=show, save_as=save_as)


def plot_traces_individual(report, population=None, node_ids=None, sections='origin', times=None, title=None,
                           show_legend=None, show=True, save_as=None):
    """Used the plot time traces of individual nodes from a SONATA compartment report file.

    Recommended use plot_traces instead, which will call this function if necessarcy.

    :param report: Path to SONATA report file or CompartmentReport object.
    :param population: string. If the report more than one population of nodes, use this to determine which nodes to
           plot. If only one population exists and population=None then the function will find it by default.
    :param node_ids: int or list of ints. Individual node to display the variable.
    :param sections: 'origin', 'all', or list of ids, Compartments/elements to display, By default will only show values
           at the soma.
    :param times: (float, float), start and stop times of simulation.
    :param title: str, adds a title to the plot.
    :param show_legend: Set True or False to determine if legend should be displayed on the plot. The default (None)
           function itself will guess if legend should be shown.
    :param show: bool to display or not display plot. default True.
    :param save_as: None or str: file-name/path to save the plot as a png/jpeg/etc. If None or empty string will not
        save plot.
    :return: matplotlib figure.Figure object
    """

    cr = CompartmentReport.load(report)
    pop = __get_population(report=cr, population=population)  # get node populations name

    if node_ids is None:
        node_ids = cr.node_ids(population=pop)
    elif np.isscalar(node_ids):
        node_ids = [node_ids]

    trace_times = cr.time_trace(population=pop)
    if times is not None:
        times_indx = np.argwhere((trace_times >= times[0]) & (trace_times <= times[1]))
        trace_times = trace_times[times_indx]
    else:
        # Get start and stop time from the traces data
        times = (trace_times[0], trace_times[-1])

    fig, axes = plt.subplots()
    for node_id in node_ids:
        trace_data = cr.data(node_id=node_id, sections=sections, population=pop, time_window=times)
        axes.plot(trace_times, trace_data, label=node_id)

    axes.set_xlim(times[0], times[1])
    axes.set_xlabel('time ({})'.format('ms'))

    if (show_legend is None and len(node_ids) <= 10) or show_legend:
        axes.legend()

    if title:
        axes.set_title(title)

    if save_as:
        plt.savefig(save_as)

    if show:
        plt.show()

    return fig


def plot_traces_averaged(report, population=None, sections='origin', node_groups=None, times=None, title=None,
                         show_background=True, show_legend=None, show=True, save_as=None):
    """Used to plot averages across multiple nodes in a SONATA Compartment Report file.

    Recommended that you use "plot_traces" function.

    To plot multiple averages use the "node_groups" options. "node_groups" should be a list of dictionary, each dict
    with a 'node_ids': [list of ids], and optionally a 'label' and 'c' (color). For example support nodes [0, 70) are
    excitatory cells, nodes [70, 100) are inhibitory, and we want display excitatory and inhibitory averages and blue
    and red, respectively::

        plot_traces('/path/to/report.h5',
                    node_groups=[{'node_ids': range(0, 70), 'label': 'exc', 'c': 'b'},
                                 {'node_ids': range(70, 100), 'label': 'inh', 'c': 'r'}])

    :param report: Path to SONATA report file or CompartmentReport object.
    :param population: string. If the report more than one population of nodes, use this to determine which nodes to
           plot. If only one population exists and population=None then the function will find it by default.
    :param sections: 'origin', 'all', or list of ids, Compartments/elements to display, By default will only show values
           at the soma.
    :param node_groups: None or list of dicts. Used to group sets of nodes by labels and color. Each grouping should
        be a dictionary with a 'node_ids' key with a list of the ids. You can also add 'label' and 'c' keys for
        label and color. If None all nodes will be labeled and colored the same.
    :param times: (float, float), start and stop times of simulation.
    :param title: str, adds a title to the plot.
    :param show_background: shows all the individual traces greyed in the background.
    :param show_legend: Set True or False to determine if legend should be displayed on the plot. The default (None)
           function itself will guess if legend should be shown.
    :param show: bool to display or not display plot. default True.
    :param save_as: None or str: file-name/path to save the plot as a png/jpeg/etc. If None or empty string will not
        save plot.
    :return: matplotlib figure.Figure object
    """
    cr = CompartmentReport.load(report)
    pop = __get_population(report=cr, population=population)  # get node populations name
    node_groups, selected_ids = __get_node_groups(report=cr, node_groups=node_groups, population=pop)

    nodes_array = cr.node_ids(population=pop)
    trace_times = cr.time_trace(population=pop)
    traces_data = cr.data(sections=sections, population=pop)

    if times is not None:
        min_ts, max_ts = times[0], times[1]
        times_indx = np.argwhere((trace_times >= min_ts) & (trace_times <= max_ts))
        times_indx = times_indx.flatten()
        traces_data = traces_data[times_indx, :]
        trace_times = trace_times[times_indx]
    else:
        min_ts, max_ts = trace_times[0], trace_times[-1]

    fig, axes = plt.subplots()
    has_labels = False

    if show_background:
        nodes_indx = np.argwhere(np.isin(nodes_array, selected_ids))
        nodes_indx = nodes_indx.flatten()
        background_data = traces_data[:, nodes_indx]
        axes.plot(trace_times, background_data, c='lightgray')

    for node_grp in node_groups:
        grp_ids = node_grp.pop('node_ids')
        has_labels = has_labels or 'label' in node_grp

        nodes_indx = np.argwhere(np.isin(nodes_array, grp_ids))
        nodes_indx = nodes_indx.flatten()
        if len(nodes_indx) == 0:
            continue

        grp_data = traces_data[:, nodes_indx]
        grp_mean = grp_data.mean(axis=1)
        axes.plot(trace_times, grp_mean, **node_grp)

    axes.set_xlim(min_ts, max_ts)
    axes.set_xlabel('time ({})'.format('ms'))

    y_label = cr.variable(population=pop)
    if cr.units(population=pop) is not None:
        y_label += '({})'.format(cr.units(population=pop))
    axes.set_ylabel(cr.variable(population=pop))

    if title:
        axes.set_title(title)

    if (show_legend is None or show_legend) and has_labels:
        axes.legend()

    if save_as:
        plt.savefig(save_as)

    if show:
        plt.show()

    return fig
