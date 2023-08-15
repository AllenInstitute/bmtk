import os
import numpy as np

from bmtk.builder import NetworkBuilder
from bmtk.utils.io.spike_trains import PoissonSpikesGenerator
from bmtk.builder.auxi.node_params import positions_columinar, xiter_random

# Brunel network parameters -------------------------------------------------------------------------
N_LIF_exc = 10000  # number of excitatory LIF neurons 80%
N_LIF_inh = 2500  # number of inhibitory LIF neurons 20%
n_cells = N_LIF_exc + N_LIF_inh  # total number of internal neurons
N_ext = 100  # number of neurons in the external network

JE = 0.2  # excitatory synaptic strength (mV)
g = 2.5  # ratio of inhibitory synpatic strength to excitatory synaptic strength (unitless)*************
JI = -g * JE  # inhibitory synpatic strength (mv)
eps = 0.1  # percentage of all possible target neurons that are connected to any given source
CE = int(eps * N_LIF_exc)  # number of internal connections with excitatory source neurons
C_ext = CE  # number of connections to each internal cell from the input network
n_syn = int(C_ext / N_ext)  # number of synapses per neuron in the input network
CI = int(eps * N_LIF_inh)  # number of internal connections with inhibitory source neurons
C = CE + CI  # number of internal connections
tau_e = 20.  # membrane time constant of excitatory neurons (ms)
theta = 20.  # neuron firing threshold (mV)
Vr = 10.  # neuron rest potential (mV)
tau_rp = 2.  # refractory period (ms)
D = 1.5  # transmission delay (ms)
v_ext_v_thr_ratio = 10.  # ratio of v_ext to v_thr (unitless)***************
eps_ext = C_ext / float(n_cells)  # external connection probability
v_thr = theta / (JE * CE * tau_e)  # frequency needed for a neuron to reach threshold in absence of feedback
v_ext = v_ext_v_thr_ratio * v_thr  # external firing frequency

# -----------------------------------------------------------------------------------------------
def generate_random_positions(N):
    x = np.random.random(N)  # x-axis location
    y = np.random.random(N)  # y-axis location
    z = np.random.random(N)  # z-axis location

    positions = np.column_stack((x, y, z))
    return positions


def random_connections(source, target, p=0.1):
    sid = source['node_id']  # Get source id
    tid = target['node_id']  # Get target id

    # Avoid self-connections.
    if sid == tid:
        if sid % 1000 == 0:
            print(sid)
        return None

    # print np.random.binomial(1, p)
    return np.random.binomial(1, p)  # nsyns


def build_lif_network():
    ### Define all the cell models in a dictionary.
    print('Building Internal Network')

    LIF_models = {
        'LIF_exc': {
            'N': N_LIF_exc,
            'ei': 'e',
            'pop_name': 'LIF_exc',
            'model_type': 'point_process',
            'model_template': 'nest:iaf_psc_exp',
            'dynamics_params': 'excitatory.json'
        },
        'LIF_inh': {
            'N': N_LIF_inh,
            'ei': 'i',
            'pop_name': 'LIF_inh',
            'model_type': 'point_process',
            'model_template': 'nest:iaf_psc_exp',
            'dynamics_params': 'inhibitory.json'
        }
    }
    net = NetworkBuilder('internal')

    for model in LIF_models:
        params = LIF_models[model].copy()
        positions = positions_columinar(N=LIF_models[model]['N'], center=[0, 10.0, 0], max_radius=50.0, height=200.0)
        net.add_nodes(x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                      **params)

    net.add_edges(source={'ei': 'e'},
                  target={'ei': 'e'},
                  connection_rule=random_connections,
                  connection_params={'p': eps},
                  syn_weight=200,
                  delay=D,
                  dynamics_params='ExcToExc.json',
                  model_template='static_synapse')

    net.add_edges(source={'ei': 'e'},
                  target={'ei': 'i'},
                  connection_rule=random_connections,
                  connection_params={'p': eps},
                  syn_weight=300,
                  delay=D,
                  dynamics_params='ExcToInh.json',
                  model_template='static_synapse')

    net.add_edges(source={'ei': 'i'},
                  target={'ei': 'e'},
                  connection_rule=random_connections,
                  connection_params={'p': eps},
                  syn_weight=-550,
                  delay=D,
                  dynamics_params='InhToExc.json',
                  model_template='static_synapse')

    net.add_edges(source={'ei': 'i'},
                  target={'ei': 'i'},
                  connection_rule=random_connections,
                  connection_params={'p': eps},
                  syn_weight=-300,
                  delay=D,
                  dynamics_params='InhToInh.json',
                  model_template='static_synapse')

    net.build()
    net.save(output_dir='network')
    # net.save_edges(edges_file_name='edges.h5', edge_types_file_name='edge_types.csv', output_dir='network')
    # net.save_nodes(nodes_file_name='nodes.h5', node_types_file_name='node_types.csv', output_dir='network')
    return net


def build_source_network(target_net):
    input_network_model = {
        'external': {
            'N': 1,
            'ei': 'e',
            'pop_name': 'input_network',
            'model_type': 'virtual'
        }
    }

    inputNetwork = NetworkBuilder("external")
    inputNetwork.add_nodes(**input_network_model['external'])

    inputNetwork.add_edges(target=target_net.nodes(pop_name='LIF_exc'),
                           connection_rule=random_connections,
                           connection_params={'p': 0.1},
                           syn_weight=400,
                           delay=D,
                           dynamics_params='ExcToExc.json',
                           model_template='static_synapse')

    inputNetwork.add_edges(target=target_net.nodes(pop_name='LIF_inh'),
                           connection_rule=random_connections,
                           connection_params={'p': 0.1},
                           syn_weight=400,
                           delay=D,
                           dynamics_params='ExcToExc.json',
                           model_template='static_synapse')

    inputNetwork.build()
    net.save(output_dir='network')
    #inputNetwork.save_nodes(nodes_file_name='one_input_node.h5', node_types_file_name='one_input_node_type.csv',
    #                        output_dir='lif_network')
    #inputNetwork.save_edges(edges_file_name='one_input_edges.h5', edge_types_file_name='one_input_edge_type.csv',
    #                        output_dir='lif_network')
    return inputNetwork

def injective_connections(sources, target):
    if target.node_id % 1000 == 0:
        print(target.node_id)

    # Since we used the 'all_to_one' iterator "sources" is a list of all input nodes and function returns a list.
    nsyns = np.zeros(len(sources))
    nsyns[target.node_id] = 1
    return nsyns


def build_injective_inputs(target_net):
    print('Building External Network')

    input_network_model = {
        'external': {
            'N': len(target_net.nodes()),  # Need one virtual node for every LIF_network node
            'ei': 'e',
            'pop_name': 'input_network',
            'model_type': 'virtual'
        }
    }

    inputNetwork = NetworkBuilder("external")
    inputNetwork.add_nodes(**input_network_model['external'])

    inputNetwork.add_edges(target=target_net.nodes(pop_name='LIF_exc'),
                           connection_rule=injective_connections,
                           iterator='all_to_one',  # will make building a little faster
                           syn_weight=200,
                           delay=D,
                           dynamics_params='ExcToExc.json',
                           model_template='static_synapse')

    inputNetwork.add_edges(target=target_net.nodes(pop_name='LIF_inh'),
                           connection_rule=injective_connections,
                           iterator='all_to_one',
                           syn_weight=100,
                           delay=D,
                           dynamics_params='ExcToExc.json',
                           model_template='static_synapse')

    inputNetwork.build()
    inputNetwork.save(output_dir='network')


def build_spike_trains(N=N_LIF_exc + N_LIF_inh):
    from bmtk.utils.reports.spike_trains import PoissonSpikeGenerator
    from bmtk.utils.reports.spike_trains import sort_order

    # Steadily increase the firing rate of each virtual cell from 1.0 Hz (node 0) to 50.0 Hz (node 12499)
    firing_rates = 500*np.ones(N)

    psg = PoissonSpikeGenerator(population='external')
    for i in range(N):
        # create a new spike-train for each node.
        psg.add(node_ids=i, firing_rate=firing_rates[i], times=(0.0, 3.0))

    psg.to_sonata('inputs/injective_500hz.h5', sort_order=sort_order.by_id)


if __name__ == '__main__':
    lif_net = build_lif_network()

    # Create a new source network that has 12500 virtual nodes
    build_injective_inputs(lif_net)

    # create a spikes file that has different firing rates for each virtual node
    build_spike_trains()

    

