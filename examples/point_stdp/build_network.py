import os, sys

from bmtk.builder import NetworkBuilder
from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator


def build_network():
    # target cells recieve inputs with plastic synapses
    net_sources = NetworkBuilder('targets')
    net_sources.add_nodes(
        N=3,
        model_type='point_neuron',
        model_template='nest:iaf_psc_delta',
        dynamics_params='iaf_psc_delta_exc.json'
    )
    net_sources.build()
    net_sources.save(output_dir='network')
    
    # source cells that synapse onto targets
    net_feedforward = NetworkBuilder('sources')
    net_feedforward.add_nodes(
        N=2,
        model_type='point_neuron',
        model_template='nest:iaf_psc_delta',
        dynamics_params='iaf_psc_delta_exc.json'
    )

    # connection 1: cells 0 --> 0
    net_feedforward.add_edges(
        source=net_feedforward.nodes(node_id=0),
        target=net_sources.nodes(node_id=0),
        connection_rule=1,
        syn_weight=15.0,
        delay=1.0,
        dynamics_params='stdp_exc_1.json',
        model_template='stdp_synapse'
    )

    # connection: cells 1 --> 1
    net_feedforward.add_edges(
        source=net_feedforward.nodes(node_id=1),
        target=net_sources.nodes(node_id=1),
        connection_rule=1,
        syn_weight=15.0,
        delay=1.0,
        dynamics_params='stdp_exc_2.json',
        model_template='stdp_synapse'
    )

    # connection: cells [0, 1] --> 2
    net_feedforward.add_edges(
        source=net_feedforward.nodes(),
        target=net_sources.nodes(node_id=2),
        connection_rule=1,
        syn_weight=15.0,
        delay=1.0,
        dynamics_params='stdp_exc_3.json',
        model_template='stdp_synapse'
    )

    net_feedforward.build()
    net_feedforward.save(output_dir='network')

    # virtual inputs into feedforward cells
    net_virts = NetworkBuilder('virtual')
    net_virts.add_nodes(
        N=10,
        model_type='virtual'
    )
    net_virts.add_edges(
        target=net_feedforward.nodes(),
        connection_rule=1,
        syn_weight=4.2,
        delay=1.5,
        dynamics_params='static_exc.json',
        model_template='static_synapse'
    )
    net_virts.build()
    net_virts.save(output_dir='network')


def generate_virt_spikes():
    psg = PoissonSpikeGenerator()
    psg.add(node_ids='network/virtual_nodes.h5', firing_rate=20.0, times=(0.0, 5.0), population='virtual')
    psg.to_sonata('inputs/virtual_spikes.h5')


if __name__ == '__main__':
    build_network()
    generate_virt_spikes()