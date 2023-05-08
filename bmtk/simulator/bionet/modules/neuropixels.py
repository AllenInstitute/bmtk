import numpy as np

from bmtk.simulator.core.modules.iclamp import IClampMod


class NeuropixelsNWBReader(IClampMod):
    def __init__(self, **kwargs):
        self._node_set = kwargs['node_set']
        self._nwb_file = kwargs['input_file']

    def get_spikes(self, node_id):
        # firing_rate = np.random.uniform(1.0, 20.0, size=1)
        firing_rate = np.random.randint(1.0, 15.0, size=1)
        # print()
        spikes = np.random.uniform(0.0, 3000.0, firing_rate[0]*2)
        return np.sort(spikes)
        # print(spikes)
        # exit()
        # return []        

    def initialize(self, sim):
        net = sim.net
        net._init_connections()
        node_set = net.get_node_set(self._node_set)

        src_nodes = [node_pop for node_pop in net.node_populations if node_pop.name in node_set.population_names()]
        for src_node_pop in src_nodes:
            source_population = src_node_pop.name
            
            for edge_pop in net.find_edges(source_nodes=source_population):
                # print(source_population, edge_pop.target_nodes)
                if edge_pop.virtual_connections:
                    for trg_nid, trg_cell in net._rank_node_ids[edge_pop.target_nodes].items():
                        for edge in edge_pop.get_target(trg_nid):
                            # print(edge.source_node_id, '-->', trg_nid)
                            source_node_id = edge.source_node_id
                            spike_trains = self.get_spikes(source_node_id)

                            src_cell = net.get_virtual_cells(source_population, source_node_id, spike_trains)
                            trg_cell.set_syn_connection(edge, src_cell, src_cell)

                elif edge_pop.mixed_connections:
                    raise NotImplementedError()
