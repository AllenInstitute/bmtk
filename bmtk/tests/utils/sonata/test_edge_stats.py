import pytest
import os
import numpy as np
import tempfile
import json

import bmtk
from bmtk.builder.networks import NetworkBuilder
from bmtk.utils.sonata import edge_stats


@pytest.fixture
def net():
    net = NetworkBuilder('test')
    net.add_nodes(5, pop_name='e1', ei='e', morph='e1.swc', tau_i=np.full(5, 0), tau_m=np.ones(5))
    net.add_nodes(5, pop_name='e2', ei='e', tau_i=np.full(5, 1), ki=np.random.uniform(0, 10.0, 5))
    net.add_nodes(10, pop_name='i1', ei='i', morph='i1.swc', tau_i=np.full(10, 2), tau_m=np.ones(10)*2)
    cm = net.add_edges(source={'ei': 'i'}, target={'ei': 'e'}, connection_rule=5, syn_model='i2e', syn_weight=0.1)
    cm = net.add_edges(source={'ei': 'e'}, target={'ei': 'i'}, connection_rule=2, syn_model='e2i', syn_location='soma')
    cm.add_properties('syn_weight', rule=lambda *_: np.random.uniform(0.001, 0.1), dtypes=float)
    cm.add_properties('syn_dist', rule=lambda *_: np.random.uniform(0, 10), dtypes=float)

    net.build()
    net_dir = tempfile.mkdtemp()
    net.save_nodes('nodes.h5', 'node_types.csv', output_dir=net_dir)
    net.save_edges('edges.h5', 'edge_types.csv', output_dir=net_dir)
    
    config_path = os.path.join(net_dir, 'config.json') # tempfile.NamedTemporaryFile(suffix='.json')
    nodes_path = os.path.join(net_dir, 'nodes.h5')
    node_types_path = os.path.join(net_dir, 'node_types.csv')
    edges_path = os.path.join(net_dir, 'edges.h5')
    edge_types_path = os.path.join(net_dir, 'edge_types.csv')
    cfg_dict = {
        'networks': {
            'nodes': [{'nodes_file': nodes_path, 'node_types_file': node_types_path}],
            'edges': [{'edges_file': edges_path, 'edge_types_file': edge_types_path}],
        }
    }
    json.dump(cfg_dict, open(config_path, 'w'))

    return config_path, nodes_path, node_types_path, edges_path, edge_types_path


def test_to_edges_dataframe(net):
    config_path, edges_path, edge_types_path, _, _ = net
    nodes_pop, edges_pop = edge_stats.__read_sonata_files(config_path)
    edges_df = edge_stats.to_edges_dataframe(edges_pop['test_to_test'].h5_grp, edges_pop['test_to_test'].csv)
    assert(len(edges_df) == 300)
    assert(len(edges_df.columns) == 10)
    assert('syn_model' in edges_df.columns)
    assert('syn_weight' in edges_df.columns)
    
    edges_df = edge_stats.to_edges_dataframe(edges_pop['test_to_test'].h5_grp)
    assert(len(edges_df) == 300)
    assert(len(edges_df.columns) == 6)
    assert('syn_model' not in edges_df.columns)
    assert('syn_weight' in edges_df.columns)

    edges_df = edge_stats.to_edges_dataframe(edges_pop['test_to_test'].h5_grp, edges_pop['test_to_test'].csv, with_properties=False)
    assert(len(edges_df) == 300)
    assert(set(edges_df.columns) == {'source_node_id', 'target_node_id', 'edge_type_id'})


def test_to_nodes_dataframe(net):
    config_path, _, _, _, _ = net
    nodes_pop, edges_pop = edge_stats.__read_sonata_files(config_path)
    nodes_df = edge_stats.to_nodes_dataframe(nodes_pop['test'].h5_grp, nodes_pop['test'].csv)
    assert(len(nodes_df) == 20)
    assert(len(nodes_df.columns) == 8)

    nodes_df = edge_stats.to_nodes_dataframe(nodes_pop['test'].h5_grp, nodes_pop['test'].csv, with_properties=False)
    assert(len(nodes_df) == 20)
    assert(set(nodes_df.columns) == {'node_id', 'node_type_id'})


def test_edge_props_distribution(net):
    config_path = net[0]
    dist_df = edge_stats.edge_props_distribution(config_path, 'syn_weight')
    assert(len(dist_df) == 300)
    assert('syn_weight' in dist_df.columns)
    assert('source_node_id' in dist_df.columns)
    assert('target_node_id' in dist_df.columns)

    dist_df = edge_stats.edge_props_distribution(config_path, 'syn_weight', edge_props_grouping='syn_model')
    assert(len(dist_df) == 2)
    assert('syn_model' in dist_df.columns)
    assert('syn_weight' in dist_df.columns)

    dist_df = edge_stats.edge_props_distribution(config_path, 'syn_weight', source_props_grouping='ei', target_props_grouping='ei')
    assert(len(dist_df)) == 2
    assert('source_ei' in dist_df.columns)
    assert('target_ei' in dist_df.columns)
    assert('syn_weight' in dist_df.columns)

    dist_df = edge_stats.edge_props_distribution(config_path, 'syn_dist', edge_props_grouping='edge_type_id', fill_val=0)
    assert(len(dist_df) == 2)
    assert('edge_type_id' in dist_df.columns)
    assert('syn_dist' in dist_df.columns)


def test_nsyns_distribution(net):
    config_path = net[0]
    dist_df = edge_stats.nsyns_distribution(config_path)
    assert(len(dist_df) == 300)
    assert('nsyns' in dist_df.columns)
    assert('source_node_id' in dist_df.columns)
    assert('target_node_id' in dist_df.columns)

    dist_df = edge_stats.nsyns_distribution(config_path, edge_props_grouping='syn_model')
    assert(len(dist_df) == 2)
    assert(np.sum(dist_df['nsyns'].values) == 200 + 500)
    assert('syn_model' in dist_df.columns)
    assert('nsyns' in dist_df.columns)

    dist_df = edge_stats.nsyns_distribution(config_path, source_props_grouping='ei', target_props_grouping='ei')
    assert(len(dist_df) == 2)
    assert(np.sum(dist_df['nsyns'].values) == 200 + 500)
    assert('source_ei' in dist_df.columns)
    assert('target_ei' in dist_df.columns)
    assert('nsyns' in dist_df.columns)


def test_nconnections_distributions(net):
    config_path = net[0]
    dist_df = edge_stats.nconnections_distributions(config_path)
    assert(len(dist_df) == 200)
    assert(np.all(dist_df['nconnections'].unique() == [1]))
    assert('source_node_id' in dist_df.columns)
    assert('target_node_id' in dist_df.columns)

    dist_df = edge_stats.nconnections_distributions(config_path, edge_props_grouping='edge_type_id', source_props_grouping='ei', target_props_grouping='ei')
    assert(len(dist_df) == 2)
    assert(np.array_equal(dist_df['nconnections'].values, [100, 100]))
    assert('edge_type_id' in dist_df.columns)
    assert('source_ei' in dist_df.columns)
    assert('nconnections' in dist_df.columns)


def test_edge_stats_table(net):
    config_path = net[0]
    stats_df = edge_stats.edge_stats_table(config_path)
    assert(stats_df.loc['n_sources', 'test_to_test'] == 20)
    assert(stats_df.loc['n_targets', 'test_to_test'] == 20)
    assert(stats_df.loc['n_edge_types', 'test_to_test'] == 2)
    assert(stats_df.loc['n_connections', 'test_to_test'] == 200)
    assert(stats_df.loc['n_synapses', 'test_to_test'] == 700)


if __name__ == '__main__':
    # test_to_edges_dataframe(net())
    # test_to_nodes_dataframe(net())
    # test_edge_props_distribution(net())
    # test_nsyns_distribution(net())
    # test_nconnections_distributions(net())
    test_edge_stats_table(net())
