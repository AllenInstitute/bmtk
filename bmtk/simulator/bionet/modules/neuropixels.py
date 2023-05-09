from pathlib import Path
import numpy as np
import pynwb

from bmtk.simulator.core.modules.iclamp import IClampMod


file_dir = Path(__file__).parent
namespace_path = (file_dir/"ndx-aibs-ecephys.namespace.yaml").resolve()
pynwb.load_namespaces(str(namespace_path))


class NeuropixelsNWBReader(IClampMod):
    def __init__(self, **kwargs):
        self._node_set = kwargs['node_set']
        self._nwb_path = kwargs['input_file']
        self._mapping = kwargs['mapping']

        io = pynwb.NWBHDF5IO(self._nwb_path, 'r')
        self._nwb_file = io.read()

        units = self._nwb_file.units.to_dataframe()
        units = units.rename(columns={'peak_channel_id': 'channel_id'})
        channels = self._nwb_file.electrodes.to_dataframe()
        channels = channels.reset_index().rename(columns={'id': 'channel_id'})
        self._units_table = units.merge(channels, how='left', on='channel_id')
        self._filters = kwargs['filter']
        # print(self._filters)
        mask = True
        for col_name, filter_val in self._filters.items():
            if isinstance(filter_val, str):
                mask &= self._units_table[col_name] == filter_val
            elif isinstance(filter_val, (list, np.ndarray, tuple)):
                mask &= self._units_table[col_name].isin(filter_val)
                # print(self._units_table[col_name].value_counts())
                # exit()

        self._units_table = self._units_table[mask]
        self._num_units = len(self._units_table)
        # print(self._units_table)
        # exit()

    def get_spikes(self, node_id):
        if self._mapping == 'sample_with_replace':
            row_num = np.random.randint(0, self._num_units)
            select_row = self._units_table.iloc[row_num]

            return select_row['spike_times']
            exit()

        # exit()
       
        # # firing_rate = np.random.uniform(1.0, 20.0, size=1)
        # firing_rate = np.random.randint(1.0, 15.0, size=1)
        # # print()
        # spikes = np.random.uniform(0.0, 3000.0, firing_rate[0]*2)
        # return np.sort(spikes)
        # # print(spikes)
        # # exit()
        # # return []        

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
