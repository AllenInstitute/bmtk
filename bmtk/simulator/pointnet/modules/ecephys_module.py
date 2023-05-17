import nest
import numpy as np
import pandas as pd

from bmtk.simulator.core.modules.ecephys_module import ECEphysUnitsModule
from bmtk.simulator.pointnet.nest_utils import nest_version
from bmtk.simulator.pointnet.io_tools import io
from bmtk.utils.reports.spike_trains.spike_trains import SpikeTrains


def set_spikes_nest2(node_id, nest_obj, spike_trains):
    if isinstance(spike_trains, SpikeTrains):
        st = spike_trains.get_times(node_id)
    elif isinstance(spike_trains, (list, np.ndarray, pd.Series)):
        st = spike_trains

    if st is None or len(st) == 0:
        return

    st = np.array(st)
    if np.any(st <= 0.0):
        # NRN will fail if VecStim contains negative spike-time, throw an exception and log info for user
        io.log_exception('spike train {} contains negative/zero time, unable to run virtual cell in NEST'.format(st))
    st.sort()
    nest.SetStatus([nest_obj], {'spike_times': st})


def set_spikes_nest3(node_id, nest_obj, spike_trains):
    if isinstance(spike_trains, SpikeTrains):
        st = spike_trains.get_times(node_id)
    elif isinstance(spike_trains, (list, np.ndarray, pd.Series)):
        st = spike_trains
    
    if st is None or len(st) == 0:
        return

    st = np.array(st)
    if np.any(st <= 0.0):
        io.log_exception('spike train {} contains negative/zero time, unable to run virtual cell in NEST'.format(st))
    st.sort()
    nest.SetStatus(nest_obj, {'spike_times': st})


if nest_version[0] >= 3:
    set_spikes = set_spikes_nest3
else:
    set_spikes = set_spikes_nest2


class PointECEphysUnitsModule(ECEphysUnitsModule):
    def initialize(self, sim):
        net = sim.net
        # print('HERE')
        sg_params={'precise_times': True}
        node_set = net.get_node_set(self._node_set)
        self._mapping_strategy.build_map(node_set=node_set)

        src_nodes = [node_pop for node_pop in net.node_populations if node_pop.name in node_set.population_names()]
        virt_gid_map = net._virtual_gids
        for node_pop in src_nodes:
            if node_pop.name in net._virtual_ids_map:
                continue

            virt_node_map = {}
            if node_pop.virtual_nodes_only:
                for node in node_pop.get_nodes():
                    nest_objs = nest.Create('spike_generator', node.n_nodes, sg_params)
                    nest_ids = nest_objs.tolist() if nest_version[0] >= 3 else nest_objs

                    virt_gid_map.add_nestids(name=node_pop.name, nest_ids=nest_ids, node_ids=node.node_ids)
                    for node_id, nest_obj, nest_id in zip(node.node_ids, nest_objs, nest_ids):
                        spike_trains = self._mapping_strategy.get_spike_trains(node_id, '')
                        print(spike_trains)

                        virt_node_map[node_id] = nest_id
                        set_spikes(node_id=node_id, nest_obj=nest_obj, spike_trains=spike_trains)

            elif node_pop.mixed_nodes:
                for node in node_pop.get_nodes():
                    if node.model_type != 'virtual':
                        continue

                    nest_ids = nest.Create('spike_generator', node.n_nodes, sg_params)
                    for node_id, nest_id in zip(node.node_ids, nest_ids):
                        virt_node_map[node_id] = nest_id
                        set_spikes(node_id=node_id, nest_id=nest_id, spike_trains=spike_trains)

            net._virtual_ids_map[node_pop.name] = virt_node_map
        
        # Create virtual synaptic connections
        for source_reader in src_nodes:
            for edge_pop in net.find_edges(source_nodes=source_reader.name):
                for edge in edge_pop.get_edges():
                    nest_trgs = net.gid_map.get_nestids(edge_pop.target_nodes, edge.target_node_ids)
                    nest_srcs = virt_gid_map.get_nestids(edge_pop.source_nodes, edge.source_node_ids)
                    if np.isscalar(edge.nest_params['weight']):
                        edge.nest_params['weight'] = np.full(shape=len(nest_srcs),
                                                             fill_value=edge.nest_params['weight'])
                    net._nest_connect(nest_srcs, nest_trgs, conn_spec='one_to_one', syn_spec=edge.nest_params)
