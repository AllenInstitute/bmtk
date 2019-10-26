"""Simulates an example network of 450 cell receiving two kinds of exernal input as defined in the configuration file"""
import sys
import os
import numpy as np
import pandas as pd

from neuron import h

from bmtk.simulator import bionet
#from bmtk.simulator.bionet.modules.record_spikes import SpikesMod
from bmtk.simulator.bionet.modules import SaveSynapses, SpikesMod
from bmtk.utils.reports.spike_trains import SpikeTrains


pc = h.ParallelContext()
MPI_Rank = int(pc.id())
MPI_Size = int(pc.nhost())


def run(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    #graph.build()
    #node_props = graph.node_properties()
    #node_ids = {k: v.tolist() for k, v in node_props['v1'].groupby('pop_name').groups.items()}
    #print(node_ids)

    #exit()


    #graph.get_node_ids

    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()

    node_props = graph.node_properties()
    node_groups = {k: v.tolist() for k, v in node_props['v1'].groupby('pop_name').groups.items()}

    st = SpikeTrains.load('output/spikes.h5')
    #print(conf.tstart, conf.tstop)
    #print()
    #exit()

    max_fr = 100.0
    min_fr = 0.0
    target_fr = 15.0
    learning_rate = 5.0
    gradients = {
        'PV1': 0.0,
        'PV2': 0.0,
        'LIF_inh': 0.0,
        'LIF_exc': 0.0,
        'Rorb': 0.0,
        'Scnn1a': 0.0,
        'Nr5a1': 0.0
    }

    sim_time_s = sim.simulation_time(units='s')
    for grp_key, node_ids in node_groups.items():
        total_spikes = np.sum([len(st.get_times(node_id)) for node_id in node_ids])
        n_nodes = len(node_ids)
        spiking_avg = 0.0 if n_nodes == 0 else total_spikes / (sim_time_s * n_nodes)  # caclucate pop avg (assuming n_nodes != 0)
        spiking_avg = min(spiking_avg, max_fr)  # set upper bound for firing rate
        gradients[grp_key] = learning_rate*(target_fr - spiking_avg)/max_fr  # get gradient

        #print(grp_key, spiking_avg)

        #time_coeff = 1.0/
        #print(np.sum([len()*time_coeff for node_id in ]))
        #print('#####')
        #print(grp_key)
        #print(len(node_ids))
        #print('-------')
        #for node_id in node_ids:
        #    print(st.get_times(node_id=node_id))
        #    print(len(st.get_times(node_id=node_id)))
        #print(grp)
        #print()

    #exit()


    # print(graph.get_local_cells())

    # Reset the netcons
    for gid, cell in graph.get_local_cells().items():
        trg_pop = cell['pop_name']
        #print(cell['pop_name'])
        for i, nc in enumerate(cell._netcons):
            #print(cell._edge_props[i]['dynamics_params'])
            if cell._src_gids[i] == -1:
                continue
            pop_id = graph.gid_pool.get_pool_id(cell._src_gids[i])
            src_type = graph.get_node_id(pop_id.population, pop_id.node_id)['ei']

            nc.weight[0] += gradients[trg_pop]*(-1.0 if src_type == 'i' else 1.0)
            nc.weight[0] = max(nc.weight[0], 0.0)

    sim2 = bionet.BioSimulator(network=graph, dt=conf.dt, tstop=conf.tstop, v_init=conf.v_init, celsius=conf.celsius,
                               nsteps_block=conf.block_step)

    spikes_recorder = SpikesMod(spikes_file='spikes2.h5', tmp_dir='output', spikes_sort_order='gid')
    sim2.add_mod(spikes_recorder)
    sim2.run()

    st = SpikeTrains.load('output/spikes2.h5')
    for grp_key, node_ids in node_groups.items():
        total_spikes = np.sum([len(st.get_times(node_id)) for node_id in node_ids])
        n_nodes = len(node_ids)
        spiking_avg = 0.0 if n_nodes == 0 else total_spikes / (sim_time_s * n_nodes)  # caclucate pop avg (assuming n_nodes != 0)
        spiking_avg = min(spiking_avg, max_fr)  # set upper bound for firing rate
        #print(grp_key, spiking_avg)

    bionet.nrn.quit_execution()


def calc_gradients(spikes_file, node_groups, target_fr, max_fr, learning_rate, sim_time_s):
    gradients = {}
    spiking_rates = []

    st = SpikeTrains.load(spikes_file)
    for grp_key, node_ids in node_groups.items():
        total_spikes = np.sum([len(st.get_times(node_id, population=pop)) for pop, node_id in node_ids])
        #print(total_spikes)
        #exit()
        n_nodes = len(node_ids)
        spiking_avg = 0.0 if n_nodes == 0 else total_spikes / (sim_time_s * n_nodes)  # caclucate pop avg (assuming n_nodes != 0)
        spiking_avg = min(spiking_avg, max_fr)  # set upper bound for firing rate
        spiking_rates.append((grp_key, spiking_avg))
        rel_fr = min((target_fr - spiking_avg) / max_fr, 1.0)

        #if grp_key == 'Scnn1a':
        #    print(rel_fr, spiking_avg, n_nodes, st.get_times(0))
        # exit()

        gradients[grp_key] = learning_rate * rel_fr  # get gradient

    return gradients, spiking_rates


def update_weights(graph, gradients):
    for gid, cell in graph.get_local_cells().items():
        trg_pop = cell['pop_name']
        # print(gid, trg_pop)
        for i, nc in enumerate(cell._netcons):
            if cell._src_gids[i] == -1:
                continue
            pop_id = graph.gid_pool.get_pool_id(cell._src_gids[i])
            src_type = graph.get_node_id(pop_id.population, pop_id.node_id)['ei']

            nc.weight[0] += gradients[trg_pop]*(-1.0 if src_type == 'i' else 1.0)
            nc.weight[0] = max(nc.weight[0], 0.0)
            #if gid == 0:
            #    print(nc.weight[0])

            #print(nc.weight[0])

    # exit()

def track_rates(rates_table, sim_rates):
    for k, v in sim_rates:
        if k not in rates_table:
            rates_table[k] = [v]
        else:
            rates_table[k].append(v)


def spike_statistics(spikes_file, network=None, groupby=None):
    spike_trains = SpikeTrains.load(spikes_file)
    spike_trains_df = spike_trains.to_dataframe()
    return spike_trains.groupby(['population', 'node_ids'])['timestamps'].agg(np.mean)





def run_iteration(config_file):
    conf = bionet.Config.from_json(config_file, validate=True)
    conf.build_env()

    graph = bionet.BioNetwork.from_config(conf)
    sim = bionet.BioSimulator.from_config(conf, network=graph)
    sim.run()
    pc.barrier()

    #print(spike_statistics(spikes_file='output/spikes.h5'))
    #exit()

    rates_table = {}
    sim_time_s = sim.simulation_time(units='s')
    node_props = graph.node_properties(populations='v1')
    node_groups = {k: v.tolist() for k, v in node_props.groupby('pop_name').groups.items()}
    #print(node_groups)
    #exit()
    #exit()

    #for pop, grp in node_props.groupby('pop_name'):
    #    print(pop)
    #    print(grp.index.values.tolist())
    #print(node_props.groupby('pop_name').groups.items())
    #exit()

    #node_groups = {k: v.tolist() for k, v in node_props['v1'].groupby('pop_name').groups.items()}
    gradients, sim_rates = calc_gradients(spikes_file='output/spikes.h5', node_groups=node_groups, target_fr=15.0, max_fr=100.0,
                               learning_rate=0.0003, sim_time_s=sim_time_s)

    #if MPI_Rank == 0:
    #    print(node_groups)

    # print(gradients)
    update_weights(graph, gradients)
    track_rates(rates_table, sim_rates)

    #exit()

    for i in range(2, 3):
        spikes_file = 'spikes.h5'.format(i)

        pc.barrier()
        sim_step = bionet.BioSimulator(network=graph, dt=conf.dt, tstop=conf.tstop, v_init=conf.v_init, celsius=conf.celsius, nsteps_block=conf.block_step)
        spikes_recorder = SpikesMod(spikes_file=spikes_file, tmp_dir='output', spikes_sort_order='gid', mode='w')
        sim_step.add_mod(spikes_recorder)
        sim_step.run()
        gradients, sim_rates = calc_gradients(spikes_file='output/{}'.format(spikes_file), node_groups=node_groups, target_fr=15.0,
                                   max_fr=100.0, learning_rate=0.0003, sim_time_s=sim_time_s)

        update_weights(graph, gradients)
        track_rates(rates_table, sim_rates)

    connection_recorder = SaveSynapses('updated_weights')
    #connection_recorder.syn_statistics(sim_step)
    connection_recorder.initialize(sim_step)
    connection_recorder.finalize(sim_step)

    summary_df = pd.DataFrame(rates_table)
    summary_df['MSE'] = summary_df.apply(lambda r: np.sum(np.power(15.0 - r, 2))/(len(r)), axis=1)
    pc.barrier()

    if MPI_Rank == 0:
        print(summary_df)

    bionet.nrn.quit_execution()


if __name__ == '__main__':
    if __file__ != sys.argv[-1]:
        run(sys.argv[-1])
    else:
        run_iteration('config.json')
        #run('config.json')
