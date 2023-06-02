from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.core.modules.ecephys_module import ECEphysUnitsModule


class BioECEphysUnitsModule(ECEphysUnitsModule):
    def initialize(self, sim):
        io.log_info('Building virtual cell stimulations for {}'.format(self._name))

        net = sim.net
        net._init_connections()
        node_set = net.get_node_set(self._node_set)

        self._mapping_strategy.build_map(node_set=node_set)

        src_nodes = [node_pop for node_pop in net.node_populations if node_pop.name in node_set.population_names()]
        for src_node_pop in src_nodes:
            source_population = src_node_pop.name
            
            for edge_pop in net.find_edges(source_nodes=source_population):
                if edge_pop.virtual_connections:
                    for trg_nid, trg_cell in net._rank_node_ids[edge_pop.target_nodes].items():
                        for edge in edge_pop.get_target(trg_nid):
                            source_node_id = edge.source_node_id
                            spike_trains = self._mapping_strategy.get_spike_trains(source_node_id, source_population)

                            src_cell = net.get_virtual_cells(source_population, source_node_id, spike_trains)
                            trg_cell.set_syn_connection(edge, src_cell, src_cell)

                elif edge_pop.mixed_connections:
                    raise NotImplementedError()
