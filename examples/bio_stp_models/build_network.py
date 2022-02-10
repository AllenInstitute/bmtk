import os
from optparse import OptionParser
import numpy as np
import pandas as pd
import h5py

from bmtk.builder.networks import NetworkBuilder


# Create 10 identical cells where 5 of the receive inputs of synaptic type 'a' and 5 of synaptic type 'b'. 
# This setup will allow comparing results between two different synaptic types
firing_rate = [10, 20, 50, 100, 200]


def build_net():
    net = NetworkBuilder("slice")
    net.add_nodes(
        N=5,
        pop_name='Scnn1a',
        synapse_model='a',
        firing_rate=firing_rate,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        dynamics_params='472363762_fit.json',
        morphology='Scnn1a_473845048_m.swc',
        rotation_angle_zaxis=3.646878266,
        model_processing='aibs_perisomatic,extracellular'
    )

    net.add_nodes(
        N=5,
        pop_name='Scnn1a',
        synapse_model='b',
        firing_rate=firing_rate,
        model_type='biophysical',
        model_template='ctdb:Biophys1.hoc',
        model_processing='aibs_perisomatic,extracellular',
        dynamics_params='472363762_fit.json',
        morphology='Scnn1a_473845048_m.swc',
        rotation_angle_zaxis=3.646878266
    )

    net.build()
    net.save_nodes(nodes_file_name='network/slice_nodes.h5', node_types_file_name='network/slice_node_types.csv')

    return net


# Create 5 external cells each stimulating at different frequency 10,20,50,100,200. 
# Connect them to the nodes such that each cell connects to one cell of synaptic type 'a' and one cell of type 'b'.
def build_ext5_nodes():
    if not os.path.exists('network'):
        os.makedirs('network')

    ext = NetworkBuilder("EXT")
    # need 5 cells to stimulate at 5 different frequencies
    ext.add_nodes(N=5, pop_name='EXT', model_type='virtual', firing_rate=firing_rate)

    # Save cells.csv and cell_types.csv
    ext.save_nodes(nodes_file_name='network/ext_nodes.h5', node_types_file_name='network/ext_node_types.csv')

    net = NetworkBuilder('slice')
    net.import_nodes(nodes_file_name='network/slice_nodes.h5', node_types_file_name='network/slice_node_types.csv')

    net.add_edges(
        source=ext.nodes(firing_rate=10), target=net.nodes(firing_rate=10, synapse_model='a'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='expsyn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=20), target=net.nodes(firing_rate=20, synapse_model='a'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='expsyn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=50), target=net.nodes(firing_rate=50, synapse_model='a'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='expsyn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=100), target=net.nodes(firing_rate=100, synapse_model='a'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='expsyn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=200), target=net.nodes(firing_rate=200, synapse_model='a'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='AMPA_ExcToExc.json',
        model_template='expsyn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=10), target=net.nodes(firing_rate=10, synapse_model='b'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic','basal', 'apical'],
        delay=2.0,
        dynamics_params='pvalb_pvalb.json',
        model_template='stp2syn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=20), target=net.nodes(firing_rate=20, synapse_model='b'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='pvalb_pvalb.json',
        model_template='stp2syn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=50), target=net.nodes(firing_rate=50, synapse_model='b'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='pvalb_pvalb.json',
        model_template='stp2syn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=100), target=net.nodes(firing_rate=100, synapse_model='b'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='pvalb_pvalb.json',
        model_template='stp2syn'
    )

    net.add_edges(
        source=ext.nodes(firing_rate=200), target=net.nodes(firing_rate=200, synapse_model='b'),
        connection_rule=5,
        syn_weight=0.002,
        distance_range=[0.0, 50.0],
        target_sections=['somatic', 'basal', 'apical'],
        delay=2.0,
        dynamics_params='pvalb_pvalb.json',
        model_template='stp2syn'
    )

    net.build()
    net.save_edges(
        edges_file_name='network/ext_to_slice_edges.h5',
        edge_types_file_name='network/ext_to_slice_edge_types.csv'
    )


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--force-overwrite", dest="force_overwrite", action="store_true", default=False)
    parser.add_option("--out-dir", dest="out_dir", default='./output/')
    parser.add_option("--percentage", dest="percentage", type="float", default=1.0)
    parser.add_option("--with-stats", dest="with_stats", action="store_true", default=False)
    (options, args) = parser.parse_args()

    my_network = build_net()
    build_ext5_nodes()
