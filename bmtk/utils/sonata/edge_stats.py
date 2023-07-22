import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

from bmtk.utils.sonata.config import SonataConfig
from bmtk.utils.sonata.utils import get_attribute_h5
from bmtk.utils.sonata.config import SonataConfig


def to_edges_dataframe(edges_pop_h5, edge_types_path=None, with_properties=True):
    edges_df = pd.DataFrame({
        'source_node_id': edges_pop_h5['source_node_id'][()],
        'target_node_id': edges_pop_h5['target_node_id'][()],
        'edge_type_id': edges_pop_h5['edge_type_id'][()],
        'edge_group_id': edges_pop_h5['edge_group_id'][()],
        'edge_group_index': edges_pop_h5['edge_group_index'][()],
    })

    if with_properties:
        if isinstance(with_properties, (list, tuple)):
            include_prop = lambda s: s in with_properties
        elif isinstance(with_properties, str):
            include_prop = lambda s: s == with_properties
        else:
            include_prop = lambda s: True
            
        if edge_types_path:
            edge_types_df = pd.read_csv(edge_types_path, sep=' ')
            edge_types_df = edge_types_df[[c for c in edge_types_df.columns if include_prop(c) or c == 'edge_type_id']]
            edges_df = pd.merge(edges_df, edge_types_df, how='left', on='edge_type_id')
            
        grp_ids = np.unique(edges_pop_h5['edge_group_id'][()])
        edge_props_cols = set()
        for grp_id in grp_ids:
            edge_group = edges_pop_h5[str(grp_id)]
            for n, g in edge_group.items():
                if isinstance(g, h5py.Dataset) and include_prop(n):
                    edge_props_cols.add(n)

        for col in edge_props_cols:
            edges_df[col] = None
    
        ind_beg = 0
        for egid, egid_df in edges_df.groupby('edge_group_id'):
            ind_end = ind_beg + len(egid_df)
            for n, d in edges_pop_h5[str(egid)].items():
                if include_prop(n):
                    prop_data = d[()]
                    edges_df.iloc[ind_beg:ind_end, edges_df.columns.get_loc(n)] = prop_data
            
            ind_beg = ind_end
   
    edges_df = edges_df.drop(columns=['edge_group_id', 'edge_group_index'])
    return edges_df


def to_nodes_dataframe(nodes_pop_h5, node_types_path=None, with_properties=True):
    nodes_df = pd.DataFrame({
        'node_id': nodes_pop_h5['node_id'][()],
        'node_type_id': nodes_pop_h5['node_type_id'][()],
        'node_group_id': nodes_pop_h5['node_group_id'][()],
        'node_group_index': nodes_pop_h5['node_group_index'][()],
    })

    if with_properties:
        if isinstance(with_properties, (list, tuple)):
            include_prop = lambda s: s in with_properties
        elif isinstance(with_properties, str):
            include_prop = lambda s: s == with_properties
        else:
            include_prop = lambda s: True
            
        if node_types_path:
            node_types_df = pd.read_csv(node_types_path, sep=' ')
            node_types_df = node_types_df[[c for c in node_types_df.columns if include_prop(c) or c == 'node_type_id']]
            nodes_df = pd.merge(nodes_df, node_types_df, how='left', on='node_type_id')
            
        grp_ids = np.unique(nodes_pop_h5['node_group_id'][()])
        node_props_cols = set()
        for grp_id in grp_ids:
            edge_group = nodes_pop_h5[str(grp_id)]
            for n, g in edge_group.items():
                if isinstance(g, h5py.Dataset) and include_prop(n):
                    node_props_cols.add(n)

        for col in node_props_cols:
            nodes_df[col] = None
    
        ind_beg = 0
        for egid, egid_df in nodes_df.groupby('node_group_id'):
            ind_end = ind_beg + len(egid_df)
            for n, d in nodes_pop_h5[str(egid)].items():
                if include_prop(n):
                    prop_data = d[()]
                    nodes_df.iloc[ind_beg:ind_end, nodes_df.columns.get_loc(n)] = prop_data
            
            ind_beg = ind_end
   
    nodes_df = nodes_df.drop(columns=['node_group_id', 'node_group_index'])
    return nodes_df


class _SonataFP(object):
    def __init__(self, name, h5_grp, csv):
        self.name = name
        self.h5_grp = h5_grp
        self.csv = csv


def __read_sonata(h5_file, csv_file=None):
    """Open an individual sonata file and returns path to each nodes/edges groups inside hdf5"""
    edge_populations = {}
    node_populations = {}
    h5 = h5py.File(h5_file, 'r')
    if 'edges' in h5:
        for ename, epop in h5['/edges'].items():
            if isinstance(epop, h5py.Group):
                edge_populations[ename] = _SonataFP(ename, epop, csv_file)
    
    if 'nodes' in h5:
        for ename, epop in h5['/nodes'].items():
            if isinstance(epop, h5py.Group):
                node_populations[ename] = _SonataFP(ename, epop, csv_file)

    return node_populations, edge_populations        


def __read_sonata_files(edge_objs):
    edge_objs = __to_list(edge_objs)

    # The list of edge objects can include sonata configs or raw edges files, in which case we want
    # to convert them to a dict {'edges_files': ..., 'edge_types_file': ...} where it can be easily 
    # processed below.
    edge_objs_update = []
    for eo in edge_objs:
        if isinstance(eo, str):
            try:
                cfg = SonataConfig.from_json(eo)
                for net_dict in cfg.nodes + cfg.edges:
                    edge_objs_update.append(net_dict)
            except Exception:
                edge_objs_update.append({'edges_file': eo})

        else:
            edge_objs_update.append(eo)

    edges = {}
    nodes = {}
    for eo in edge_objs_update:
        if isinstance(eo, dict):
            edges_file = eo.get('edges_file', None)
            edge_types_file = eo.get('edge_types_file', None)
            if edges_file:
                sonata_nodes, sonata_edges = __read_sonata(edges_file, edge_types_file)
                nodes.update(sonata_nodes)
                edges.update(sonata_edges)

            nodes_file = eo.get('nodes_file', None)
            node_types_file = eo.get('node_types_file', None)
            if nodes_file:
                sonata_nodes, sonata_edges = __read_sonata(nodes_file, node_types_file)
                nodes.update(sonata_nodes)
                edges.update(sonata_edges)
        
    return nodes, edges


def __to_list(vals):
    """Helper function for turning parameters into lists"""
    if vals is None:
        return []
    elif isinstance(vals, (list, tuple, np.ndarray, pd.Series)):
        return vals
    else:
        return [vals]
      

def __load_dist_as_df(edges_obj, **kwopts):
    # If edges_obj is a csv file, or path to a csv, open it up and return
    # it as a pandas data frame
    if isinstance(edges_obj, pd.DataFrame):
        return edges_obj
    
    if isinstance(edges_obj, str):
        # Check to see if file can be open as a csv-file and return it.
        try:
            return pd.read_csv(edges_obj, sep=' ')
        except Exception:
            pass

    # Otherwise assume edges_obj is a config file or sonata file in which case we 
    # need to read the raw edges and return it as a pandas DataFrame.
    if kwopts.get('edge_props', '') in ['nconns', 'nconnections', 'n_connections']:
        # The normal edge_props_distribtion doesn't accurately handle returning just
        # the number of cell-to-cell connections, need to call different function
        return nconnections_distributions(edges_obj, **kwopts)
    else:
        return edge_props_distribution(edges_obj, **kwopts)


def __combine_data(edges_obj1, edges_obj2, edge_prop, fill_val=None, **kwargs):
    """A helper function for doing a join of two edges DataFrames, for comparing the "edge_prop" values between the two edges."""
    # convert two edges objects (config files, csv, or paths to sonata files, etc.) to dataframe
    df1 = __load_dist_as_df(edges_obj1, edge_prop=edge_prop, **kwargs)
    df2 = __load_dist_as_df(edges_obj2, edge_prop=edge_prop, **kwargs)

    # find columns to join on, don't include property being study and any other population columns
    orig_cols = set([c for c in df1.columns if c not in [edge_prop] + ['population', 'source_population', 'target_population']])
    new_cols = set([c for c in df2.columns if c not in [edge_prop] + ['population', 'source_population', 'target_population']])
    join_cols = list(orig_cols and new_cols)

    # join the two DataFrames and return
    comb_df = df1.merge(df2, how='inner', on=join_cols, suffixes=('_orig', '_new'))
    if fill_val:
        comb_df = comb_df.fillna(fill_val)
    return comb_df


def edge_props_distribution(edge_files, edge_prop, populations=None, 
                            edge_props_grouping=None, source_props_grouping=None, target_props_grouping=None, 
                            fill_val=1, operation='sum', population_columns=False):
    """Reads in one or more SONATA edges files and return a DataFrame consisting of the distribution of a given edge property
    across an arbitary grouping of cells. For example return the total number of synapses between each source/target node-type,
    or the mean syn_weights for edge edge-type, or the variance of connecting in-degrees across morphologies.

    By default will return a DataFrame of each "source_node_id", "target_node_id", and "<edge_prop>" row found in the edge file(s).
    But you can group the rows by any property/column found in the edges, source or target populations; and apply an arbitary
    <operation> on the results.

    The <edge_prop> can be any property/column found in the edges, target-nodes, or source-nodes populations. But if the property
    is not a numeric the <operation> applied to it may fail.


    :param edge_files: str, dict, list of str or dict. Is the edges (and optional nodes) files paths. It can be a SONATA h5 file,
        a dictionary of h5/csv files, or the location(s) of SONATA config files.
    :param edge_prop: string, property name (column if h5 or csv) file that is being investigated.
    :param populations: string or list of strings. If SONATA file(s) contains multiple edge populations you can specify .
    :param edge_props_grouping: str or list[str]. List of columns in edges file(s) to group results by.
    :param source_props_grouping: str or list[str]. List of columns in source-node file(s) to group results by.
    :param target_props_grouping: str or list[str]. List of columns in target-node file(s) to group results by.
    :param fill_val: If <edge_prop> has missing/None/NaN values will fill in with given value. set to None to Turn off.
    :param operation: Str or function: pandas or numpy function to apply to when doing the grouping, eg. 'sum', 'mean', np.std.
    :param population_columns: If set to true will return extra column describing the edges/nodes populations for each row.
    """
    edge_props_grouping = __to_list(edge_props_grouping)
    source_props_grouping = __to_list(source_props_grouping)
    target_props_grouping = __to_list(target_props_grouping)
    populations = __to_list(populations)
    
    nodes, edges = __read_sonata_files(edge_files)
    ret_edges_df = None
    for edge_pop_name, edges_fp in edges.items():
        if populations and edge_pop_name not in populations:
            continue

        # return edges population as a dataframe without only relevant columns 
        edges_df = to_edges_dataframe(edges_fp.h5_grp, edges_fp.csv, with_properties=edge_props_grouping + [edge_prop])

        # When grouping/filtering by properties in the source-nodes population
        if source_props_grouping:
            # Find the population-name of the source-nodes and get the nodes as a dataframe
            source_node_pop = get_attribute_h5(edges_fp.h5_grp['source_node_id'], 'node_population')
            nodes_fp = nodes[source_node_pop]
            src_nodes_df = to_nodes_dataframe(nodes_fp.h5_grp, nodes_fp.csv, with_properties=['node_id'] + source_props_grouping)
            # prepend "source_" to all source-node columns so it can be distinguished from the target nodes column sets
            src_nodes_df.columns = ['source_{}'.format(c) for c in src_nodes_df.columns]
            source_props_grouping = ['source_{}'.format(c) for c in source_props_grouping]
            # merge source-nodes dataframe onto the edges
            edges_df = pd.merge(edges_df, src_nodes_df, how='left', on='source_node_id')

        if target_props_grouping:
            # Same as with source-nodes but this time for the target/post-synaptic population of cells.
            target_node_pop = get_attribute_h5(edges_fp.h5_grp['target_node_id'], 'node_population')
            nodes_fp = nodes[target_node_pop]
            trg_nodes_df = to_nodes_dataframe(nodes_fp.h5_grp, nodes_fp.csv, with_properties=['node_id'] + target_props_grouping)
            trg_nodes_df.columns = ['target_{}'.format(c) for c in trg_nodes_df.columns]
            target_props_grouping = ['target_{}'.format(c) for c in target_props_grouping]
            edges_df = pd.merge(edges_df, trg_nodes_df, how='left', on='target_node_id')

        if fill_val is not False:
            edges_df[edge_prop] = fill_val if edge_prop not in edges_df.columns else edges_df[edge_prop].fillna(fill_val)
            
        grouping_cols = edge_props_grouping + source_props_grouping + target_props_grouping
        if not grouping_cols:
            dist_df = edges_df
        else:
            dist_df = edges_df[grouping_cols + [edge_prop]].groupby(grouping_cols)[edge_prop].agg(operation).reset_index()
        
        if population_columns and 'population' not in dist_df:
            dist_df['population'] = edge_pop_name

        # TODO: Instead of concatenating together multiple DataFrames, we should return them as a list, tuple, or Dict?
        ret_edges_df = dist_df if ret_edges_df is None else pd.concat([ret_edges_df, dist_df])

    return ret_edges_df


def nsyns_distribution(edge_files, populations=None, edge_props_grouping=None, source_props_grouping=None, target_props_grouping=None):
    """Reads in one or more SONATA edges files and return a DataFrame consisting of the total number of synapses given any arbitary 
    grouping of network properties. The property will be called "nsyns" in the returned table. Similar to edge_props_distribution().

    Note: Each cell-to-cell connection may have mutiliple synapses. Use nconnections_distributions() to get distribution of the 
    raw connectivity map.

    :param edge_files: str, dict, list of str or dict. Is the edges (and optional nodes) files paths. It can be a SONATA h5 file,
        a dictionary of h5/csv files, or the location(s) of SONATA config files.
    :param populations: string or list of strings. If SONATA file(s) contains multiple edge populations you can specify .
    :param source_props_grouping: str or list[str]. List of columns in source-node file(s) to group results by.
    :param target_props_grouping: str or list[str]. List of columns in target-node file(s) to group results by.
    """
    return edge_props_distribution(
        edge_files=edge_files, 
        edge_prop='nsyns', 
        populations=populations,
        edge_props_grouping=edge_props_grouping, 
        source_props_grouping=source_props_grouping, 
        target_props_grouping=target_props_grouping, 
        fill_val=1, 
        operation='sum'
    )


def nconnections_distributions(edge_files, populations=None, edge_props_grouping=None, source_props_grouping=None, target_props_grouping=None, **kwopts):
    """Reads in one or more SONATA edges files and return a DataFrame consisting of the total number of connection given any arbitary 
    grouping of network properties. The property will be called "nconns" in the returned table. Similar to edge_props_distribution().

    Note: A connection here just refers to whether-or-not two cells are connected (including autapses) and ignores the number of synapses between each cell. 
    Use nsyns_distribution() to get distribution of the number of synapses.

    :param edge_files: str, dict, list of str or dict. Is the edges (and optional nodes) files paths. It can be a SONATA h5 file,
        a dictionary of h5/csv files, or the location(s) of SONATA config files.
    :param populations: string or list of strings. If SONATA file(s) contains multiple edge populations you can specify .
    :param source_props_grouping: str or list[str]. List of columns in source-node file(s) to group results by.
    :param target_props_grouping: str or list[str]. List of columns in target-node file(s) to group results by.
    """    
    tmp_edge_props = __to_list(edge_props_grouping) + ['source_node_id', 'target_node_id'] 
    
    edges_df = edge_props_distribution(
        edge_files=edge_files, 
        edge_prop='_conns_', 
        populations=populations,
        edge_props_grouping=tmp_edge_props, 
        source_props_grouping=source_props_grouping, 
        target_props_grouping=target_props_grouping, 
        fill_val=1, 
        operation='count'
    )
    edges_df = edges_df.drop(columns=['_conns_'])
    edges_df = edges_df.drop_duplicates()
    if len(edges_df.columns) > 2:
        # Drop the source_node_id and target_node_id, but only if some type of edge/node grouping is specified, otherwise just return
        # list of source_node_id and target_node_id
        edges_df = edges_df.drop(columns=['source_node_id', 'target_node_id'])
    
    return edges_df.value_counts().reset_index(name='nconnections')


def plot_distribution(edges_data, edge_prop, names=None, log_scale=False, ax=None, show=True, **kwopts):
    """Plots the distribution of an edge property given an arbitary grouping of rows based on edge, target or source node
    columns. 


    :param edge_files: str, dict, list of str or dict. Is the edges (and optional nodes) files paths. It can be a SONATA h5 file,
        a dictionary of h5/csv files, or the location(s) of SONATA config files. Can also be a csv file containg results from
        edge_props_distribution() function
    :param edge_prop: string, property name (column if h5 or csv) file that is being investigated.
    :param names: A list of names/titles to use for the labeling of the distribution(s) curves. If None then will infer from data.
    :param log_scale: If set to True then use log-scale on the x-axis. Default: False.
    :param ax: If true will save distribution plot to pre-generated matplotlib axis, for merging into another figure. Default: None.
    :param show: If true will plot distribution. Default: True.
    :param kwargs: optional args that will be passed into edge_props_distribution() function.
    """
    edges_data = __to_list(edges_data)
    names = [str(ed) for ed in edges_data] if names is None else names

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for csv_path, name in zip(edges_data, names):
        dist_df = __load_dist_as_df(csv_path, edge_prop=edge_prop, **kwopts)
        prop_counts = dist_df[edge_prop].values
        min_val, max_val = np.min(prop_counts), np.max(prop_counts)
        bins = np.logspace(np.log10(min_val), np.log10(max_val)) if log_scale else np.linspace(min_val, max_val)
        hist, bin_edges = np.histogram(prop_counts, bins=bins)
        
        ax.plot(bin_edges[:-1], hist, label=name)
        if log_scale:
            ax.set_xscale('log')
        ax.set_xlabel(edge_prop)
        ax.set_title(', '.join([c for c in dist_df.columns if c not in [edge_prop] + ['population', 'source_population', 'target_population']]))
        ax.legend(fontsize='xx-small')

    if show:
        # plt.legend()
        plt.show()
    
    return ax


def plot_correlation(edges_orig, edges_new, edge_prop, log_scale=False, ax=None, show=True):
    combined_df = __combine_data(edges_orig, edges_new, edge_prop)
    data_org = combined_df[edge_prop + '_orig'].values
    data_new = combined_df[edge_prop + '_new'].values
    if log_scale:
        data_org = np.log(data_org)
        data_new = np.log(data_new)
    
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(data_org, data_new, '.')

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if show:
        plt.show()


def edge_stats_table(edges_data):
    edges_data = __to_list(edges_data)
    
    _, edges = __read_sonata_files(edges_data)
    pop_stats = {}
    for edge_pop_name, edges_fp in edges.items():
        edges_df = to_edges_dataframe(edges_fp.h5_grp, edges_fp.csv, with_properties=['nsyns'])
        
        n_src_nodes = edges_df['source_node_id'].nunique()
        n_trg_nodes = edges_df['target_node_id'].nunique()
        n_edge_types = edges_df.pop('edge_type_id').nunique()
        
        edges_df['nsyns'] = 1 if 'nsyns' not in edges_df.columns else edges_df['nsyns'].fillna(1)
        conns_se = edges_df.groupby(['source_node_id', 'target_node_id'])['nsyns'].agg('sum')
        n_conns = len(conns_se)
        n_syns = np.sum(conns_se.values)
        
        pop_stats[edge_pop_name] = [n_src_nodes, n_trg_nodes, n_edge_types, n_conns, n_syns]
        
    stats_df = pd.DataFrame.from_dict(pop_stats)
    stats_df.index = ['n_sources', 'n_targets', 'n_edge_types', 'n_connections', 'n_synapses']
    return stats_df


def pearson_r(edges_orig, edges_new, edge_prop, **kwargs):
    combined_df = __combine_data(edges_orig, edges_new, edge_prop, **kwargs)
    data_org = combined_df[edge_prop + '_orig'].values
    data_new = combined_df[edge_prop + '_new'].values
    return np.corrcoef(data_org, data_new)[0, 1]


def chisquare(edges_orig, edges_new, edge_prop, raw_counts=False, **kwargs):
    from scipy.stats import chisquare

    combined_df = __combine_data(edges_orig, edges_new, edge_prop, **kwargs)
    data_org = combined_df[edge_prop + '_orig'].values
    data_new = combined_df[edge_prop + '_new'].values
    if not raw_counts:
        data_org = data_org / np.sum(data_org)
        data_new = data_new / np.sum(data_new)
    
    results = chisquare(data_org, data_new)
    return results.statistic, results.pvalue


def kolmogorov_smirnov(edges_orig, edges_new, edge_prop, **kwargs):
    from scipy.stats import ks_2samp

    combined_df = __combine_data(edges_orig, edges_new, edge_prop, **kwargs)
    data_org = combined_df[edge_prop + '_orig'].values
    data_new = combined_df[edge_prop + '_new'].values

    results = ks_2samp(data_org, data_new)
    return results.statistic, results.pvalue


if __name__ == '__main__':
    # stat, pval = kolmogorov_smirnov('v1_edges.nsyns.orig.csv', 'v1_edges.nsyns.rebuilt.csv', 'nsyns')
    # stat, pval = chisquare('v1_edges.nsyns.orig.csv', 'v1_edges.nsyns.rebuilt.csv', 'nsyns', raw_counts=True)
    # r1 = pearson_r('v1_edges.nsyns.orig.csv', 'v1_edges.nsyns.rebuilt.csv', 'nsyns')
    # print(pval)
    # plot_correlation('v1_edges.nsyns.orig.csv', 'v1_edges.nsyns.rebuilt.csv', 'nsyns', log_scale=True)
    # plot_correlation('v1_edges.nconns.orig.csv', 'v1_edges.nconns.rebuilt.csv', 'nconnections')
    # plot_distribution(['v1_edges.nsyns.orig.csv', 'v1_edges.nsyns.rebuilt.csv'], edge_prop='nsyns', log_scale=True, show=False)
    # plot_distribution(['v1_edges.nconns.orig.csv', 'v1_edges.nconns.rebuilt.csv'], edge_prop='nconnections', log_scale=True, show=False)
    # plot_distribution('config.orig.json', 'nsyns')
    # nconn_edges_df = nconnections_distributions(
    #     'config.orig.json',                                        
    #     source_props_grouping='pop_name',
    #     target_props_grouping='pop_name'
    # )
    # plot_distribution(
    #     ['config.orig.json', 'config.rebuilt.json'],
    #     edge_prop='nsyns',
    #     populations='v1_to_v1',
    #     source_props_grouping='pop_name',
    #     target_props_grouping='pop_name',
    #     log_scale=True
    # )
    # plot_distribution(
    #     ['config.orig.json', 'config.rebuilt.json'],
    #     edge_prop='nconns',
    #     # operation='nconns',
    #     populations='v1_to_v1',
    #     source_props_grouping='pop_name',
    #     target_props_grouping='pop_name',
    #     log_scale=True
    # )

    # stat, pval = kolmogorov_smirnov(
    #     'config.orig.json', 
    #     'config.rebuilt.json', 
    #     edge_prop='nsyns',
    #     source_props_grouping='pop_name',
    #     target_props_grouping='pop_name',
    # )
    # print(stat, pval)
    # r2 = pearson_r(
    #     'config.orig.json', 
    #     'config.rebuilt.json', 
    #     edge_prop='nsyns',
    #     source_props_grouping='pop_name',
    #     target_props_grouping='pop_name',
    # )
    # print(r2)

    # stat, pval = chisquare(
    #     'config.orig.json', 
    #     'config.rebuilt.json', 
    #     edge_prop='nsyns',
    #     source_props_grouping='pop_name',
    #     target_props_grouping='pop_name',
    # )
    # print(stat, pval)

    # print(nconn_edges_df)
    # plt.show()
    edge_stats_table('config.orig.json')
    # edge_stats_table({"edges_file": "bio_450cells/internal_internal_edges.h5", "edge_types_file": "bio_450cells/internal_internal_edge_types.csv"})
