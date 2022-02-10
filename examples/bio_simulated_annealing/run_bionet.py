"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import sys
import os
import numpy as np
import pandas as pd

from neuron import h

from bmtk.simulator import bionet
from bmtk.simulator.bionet.nrn import synaptic_weight
from bmtk.simulator.bionet.modules import SaveSynapses, SpikesMod
from bmtk.analyzer.spike_trains import spike_statistics


pc = h.ParallelContext()
MPI_Rank = int(pc.id())
MPI_Size = int(pc.nhost())

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


@synaptic_weight
def default_weight_fnc(edge_props, src_props, trg_props):
    return edge_props['syn_weight'] * np.random.uniform(0.6, 1.4)


@synaptic_weight
def wmax(edge_props, src_props, trg_props):
    return edge_props["syn_weight"]* np.random.uniform(0.6, 1.4)


@synaptic_weight
def gaussianLL(edge_props, src_props, trg_props):
    src_tuning = src_props['tuning_angle']
    tar_tuning = trg_props['tuning_angle']

    w0 = edge_props["syn_weight"]
    sigma = edge_props["weight_sigma"]

    delta_tuning = abs(abs(abs(180.0 - abs(float(tar_tuning) - float(src_tuning)) % 360.0) - 90.0) - 90.0)
    weight = w0 * np.exp(-(delta_tuning / sigma) ** 2)

    return weight * np.random.uniform(0.6, 1.4)


target_frs = {
    'LIF_exc': 15.0,
    'LIF_inh': 15.0,
    'Nr5a1': 15.0,
    'PV1': 15.0,
    'PV2': 15.0,
    'Rorb': 15.0,
    'Scnn1a': 15.0
}


def get_grads(spike_stats, targets_frs, update_rule=0.0005):
    """Calculate gradients for updating the synaptic weights"""
    mean_frs = spike_stats['firing_rate']['mean']
    fr_diffs = {pop_name: (trg_fr - mean_frs.loc[pop_name]) for pop_name, trg_fr in targets_frs.items()}
    mse = np.sum(np.power(list(fr_diffs.values()), 2))/float(len(fr_diffs))
    rel_grads = {pop_name: min(update_rule * d / 100.0, 1.0) for pop_name, d in fr_diffs.items()}
    return rel_grads, mse


def update_syn_weights(net, gradients):
    for gid, cell in net.get_local_cells().items():
        trg_pop = cell['pop_name']

        for con in cell.connections():
            if con.is_virtual:
                continue

            src_node = con.source_node
            src_type = src_node['ei']
            con.syn_weight += gradients[trg_pop]*(-1.0 if src_type == 'i' else 1.0)

            con.syn_weight = max(con.syn_weight, 0.0)


def update_rates_table(rates_table, sim_stats, mse):
    if rates_table is None:
        pd.DataFrame(columns=sim_stats.columns + ['MSE'])
    firing_rates_df = sim_stats['firing_rate']['mean']
    sim_num = len(rates_table)
    nxt_row = pd.DataFrame(data=firing_rates_df.to_dict(), index=[sim_num])
    nxt_row['MSE'] = mse
    return rates_table.append(nxt_row, sort=False)


def run_iteration(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    # load network using config.json params
    graph = bionet.BioNetwork.from_config(conf)

    # run initial simulation using config.json params
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    # Get the spike statistics of the output, using "groupby" will get averaged firing rates across each model
    spike_stats_df = spike_statistics('output/spikes.h5', simulation=sim, group_by='pop_name', populations='v1')

    # Calculate gradients
    gradients, mse = get_grads(spike_stats_df, target_frs)

    # Update the synaptic weights
    update_syn_weights(graph, gradients)

    # Keep track of firing rates to display results at the end of run
    rates_table = pd.DataFrame(columns=list(target_frs.keys()) + ['MSE'])
    rates_table = update_rates_table(rates_table, spike_stats_df, mse)

    for i in range(2, 12):
        # initialize a new simulation
        sim_step = bionet.BioSimulator(network=graph, dt=conf.dt, tstop=conf.tstop, v_init=conf.v_init,
                                       celsius=conf.celsius, nsteps_block=conf.block_step)

        # Attach mod to simulation that will be used to keep track of spikes.
        spikes_recorder = SpikesMod(spikes_file='spikes.h5', tmp_dir='output', spikes_sort_order='gid', mode='w')
        sim_step.add_mod(spikes_recorder)

        # run simulation
        sim_step.run()

        # Get latest spiking statistics, calcuate gradients and update the weights
        spike_stats_df = spike_statistics('output/spikes.h5', simulation=sim, group_by='pop_name', populations='v1')
        gradients, mse = get_grads(spike_stats_df, target_frs)
        update_syn_weights(graph, gradients)
        rates_table = update_rates_table(rates_table, spike_stats_df, mse)

    # Save the connections in update_weights/ folder
    connection_recorder = SaveSynapses('updated_weights')
    connection_recorder.initialize(sim_step)
    connection_recorder.finalize(sim_step)
    pc.barrier()

    if MPI_Rank == 0:
        print(rates_table)

    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run_iteration(sys.argv[-1])
    else:
        run_iteration('config.simulation.json')
