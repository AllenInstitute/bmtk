from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pynwb

from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.utils import lazy_property

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    bcast = comm.bcast
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()
    has_mpi = True
except:
    MPI_rank = 0
    MPI_size = 1
    bcast = lambda v, n: v
    has_mpi = False


file_dir = Path(__file__).parent
namespace_path = (file_dir/"ndx-aibs-ecephys.namespace.yaml").resolve()
pynwb.load_namespaces(str(namespace_path))


class NeuropixelsNWBReader(SimulatorMod):
    def __init__(self, name, **kwargs):
        self._name = name
        self._node_set = kwargs['node_set']
        
        # Load a Strategy for mapping SONATA node_ids to NWB unit_ids
        self._mapping_name = kwargs.get('mapping', 'invalid_strategy').lower()
        if self._mapping_name in ['units_map']:
            self._mapping_strategy = UnitIdMapStrategy(**kwargs)

        elif self._mapping_name in ['sample', 'sample_without_replacement']:
            self._mapping_strategy = SamplingStrategy(with_replacement=False, **kwargs)

        elif self._mapping_name in ['sample_with_replacement']:
            self._mapping_strategy = SamplingStrategy(with_replacement=True, **kwargs)

        else:
            io.log_exception('NeuropixelsNWBReader: Invalid "mapping" parameters, options: units_map, sample, sample_with_replacement')

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


class NWBFileWrapper(object):
    # TODO: Implement Singleton
    def __init__(self, nwb_path):
        self.path = nwb_path
        self._io = pynwb.NWBHDF5IO(nwb_path, 'r').read()
        
    @property
    def uuid(self):
        return self.path
    
    def __getattr__(self, name):
        return getattr(self.__dict__['_io'], name)


class TimeWindow(object):
    def __init__(self, defaults=None, nwb_files=None):
        self._units_lu = None
        self._default_windows = None
        self.conversion_factor = 1/1000.0

        # By requestion, the "time_window" option can 
        #  - None 
        #  - a single window; "time_window": [100, 200] 
        #  - A stim_table filter: {stim_name: gratings, ori: 90.0, tf: 2.0, ...}
        #  - one unique window for each nwb_ids/files; "time_window": [[0, 100], [300, 400], ...]
        # To handle this we will 1) convert each possible option into a list of 0, 1, or more windows. 2) Check
        # that the num of time windows makes sense for the number of nwb_ids/files. And 3) Create a map of
        # defaults for each nwb_ids/file
        if defaults is not None:
            time_windows = self._tolist(defaults)
            n_windows = len(time_windows)
            if n_windows > 1 and len(nwb_files) != n_windows:
                # There can be no default time window, or one default for all nwb_files, otherwise the number
                # of time_windows must correspond to the number of nwb_files
                io.log_error('NeuropixelsNWBReader: Cannot match each "time_window" with the "input_file"s.')
        
            if n_windows == 1:
                # convert [interal] -> [interval, interval, interval, ...]
                time_windows = time_windows*n_windows
            self._default_windows = {nwb.uuid: self._parse_tw(tw, nwb) for tw, nwb in zip(time_windows, nwb_files)}

    @property
    def units_lu(self):
        return self._units_lu
    
    @units_lu.setter
    def units_lu(self, units_table):
        units_table['start_times'] = units_table['start_times']*self.conversion_factor
        units_table['stop_times'] = units_table['stop_times']*self.conversion_factor
        self._units_lu = units_table
        # print(units_table)
       
        # exit()

    def _tolist(self, window):
        if isinstance(window, dict):
            # Is a stimulus_table filter, ex. {"stimulus_name": "gratings", "ori": 90.0, ...}
            return [window]
        elif isinstance(window, (tuple, list)) and len(window) == 0:
            # Is an empyt list
            return []
        elif isinstance(window, (tuple, list)) and isinstance(window[0], (tuple, list)):
            # is a list of intervals, ex. [[0.0, 100.0], [200.0, 300.0], {stim:gratings} ...]
            return window
        else:
            # assume is a interval, ex. [0.0, 100.0]
            return [window]

    def _parse_tw(self, interval, nwb_file):
        if isinstance(interval, dict):
            filter = interval.copy()
            stim_name = filter.pop('stimulus_name', None)
            stim_idx = filter.pop('stimulus_index', 'all')

            if stim_name is None:
                io.log_error('Stimulus table filter missing "stimulus_name"')

            # stim_name = interval['stimulus_name']
            if stim_name in nwb_file.intervals.keys():
                interval_df = nwb_file.intervals[stim_name].to_dataframe()
            elif stim_name + '_presentations' in nwb_file.intervals.keys():
                interval_df = nwb_file.intervals[stim_name + '_presentations'].to_dataframe()
            else:
                io.log_error('stimulus name "{}" not found in {}'.format(stim_name, nwb_file.uuid))

            interval_df = filter_table(interval_df, filter)
            if len(interval_df) == 0:
                return [0.0, np.inf]
            
            if stim_idx == 'all':
                start_time = interval_df['start_time'].min()
                stop_time = interval_df['stop_time'].max()
            else:
                start_time = interval_df.iloc[stim_idx]['start_time']
                stop_time = interval_df.iloc[stim_idx]['stop_time']

            return [start_time*self.conversion_factor, stop_time*self.conversion_factor]

        else:
            return [interval[0]*self.conversion_factor, interval[1]*self.conversion_factor]

    def __getitem__(self, unit_info):
        unit_id, nwb_uuid = unit_info[0], unit_info[1]
        if (self.units_lu is not None) and (unit_id in self.units_lu.index):
            return self.units_lu.loc[unit_id].values
        elif self._default_windows and nwb_uuid in self._default_windows.keys():
            return self._default_windows[nwb_uuid]
        else:
            return None


class MappingStrategy(object):
    def __init__(self, **kwargs):
        self._nwb_paths = kwargs['input_file']
        self._filters = kwargs.get('filter', {})       
        self._simulation_onset = kwargs.get('simulation_onset', 0.0)/1000.0
        # self._mapping_path = kwargs.get('mapping_file', None)
        self._missing_ids = kwargs.get('missing_ids', 'fail')
        self._cache_spike_times = kwargs.get('cache', False)
        self._spike_times_cache = {}
        
        default_window = kwargs.get('time_window', None)
        self._time_window = TimeWindow(defaults=default_window, nwb_files=self.nwb_files)
       
        self._units_table = None
        self._units2nodes_map = None

    @lazy_property
    def nwb_files(self):
        if not isinstance(self._nwb_paths, (list, tuple)):
            self._nwb_paths = [self._nwb_paths] 

        nwb_files = []
        for nwb_path in self._nwb_paths:
            # io = pynwb.NWBHDF5IO(nwb_path, 'r')
            # nwb_files.append(io.read())
            nwb_files.append(NWBFileWrapper(nwb_path))
        return nwb_files


    @property
    def units_table(self):
        if self._units_table is None:
            # Combine the units and channels table from the nwb file
            merged_table = None
            for nwb_file in self.nwb_files:
                units_table = self._load_units_table(nwb_file)
                units_table = self._filter_units_table(units_table)
                units_table = units_table[['spike_times']]
                units_table['nwb_uid'] = nwb_file.uuid
                merged_table = units_table if merged_table is None else pd.concat((merged_table, units_table))

            if merged_table is None or len(merged_table) == 0:
                io.log_error('NeuropixelsNWBReader: Could not parse units table from nwb_file(s).')

            self._units_table = merged_table

        return self._units_table
    
    def _load_units_table(self, nwb_file):
        units = nwb_file.units.to_dataframe().reset_index()
        units = units.rename(columns={'id': 'unit_id', 'peak_channel_id': 'channel_id'})
        channels = nwb_file.electrodes.to_dataframe()
        channels = channels.reset_index().rename(columns={'id': 'channel_id'})
        return units.merge(channels, how='left', on='channel_id').set_index('unit_id')


    def _filter_units_table(self, units_table):
        if self._filters:
            # Filter out only those specified units
            mask = True
            for col_name, filter_val in self._filters.items():
                if isinstance(filter_val, str):
                    mask &= units_table[col_name] == filter_val
                
                elif isinstance(filter_val, (list, np.ndarray, tuple)):
                    mask &= units_table[col_name].isin(filter_val)
                
                elif isinstance(filter_val, dict):
                    col = filter_val.get('column', col_name)
                    op = filter_val['operation']
                    val = filter_val['value']
                    val = '"{}"'.format(val) if isinstance(val, str) else val
                    expr = 'units_table["{}"] {} {}'.format(col, op, val)
                    mask &= eval(expr)

                else:
                    mask &= units_table[col_name] == filter_val

            units_table = units_table[mask]

        return units_table        

    def build_map(self, node_set):
        raise NotImplementedError()

    def get_spike_trains(self, node_id, source_population):
        # TODO: Is it worth caching spike-trains so we don't have to do a lookup + filtering 
        # every time?
        unit_id = self._units2nodes_map[node_id]
        spike_times = np.array(self.units_table.loc[unit_id]['spike_times'])
        nwb_uid = self.units_table.loc[unit_id]['nwb_uid']
        time_window = self._time_window[unit_id, nwb_uid]
        if time_window is not None:
            spike_times = spike_times[
                (time_window[0] <= spike_times) & (spike_times <= time_window[1])
            ]
            spike_times = spike_times - time_window[0] + self._simulation_onset

        spike_times = spike_times*1000.0  # Convert from seconds to miliseconds
        # print(node_id, unit_id, time_window, spike_times)
        return spike_times


class UnitIdMapStrategy(MappingStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._mapping_path = kwargs['mapping_file']
        except KeyError:
            io.log_exception('{}: Could not find "mapping_file" csv path for units-to-nodes mapping.'.format(NeuropixelsNWBReader.__name__))

    def _filter_units_table(self, units_table):
        return units_table

    def build_map(self, node_set):
        try:
            # TODO: Include population name
            mapping_file_df = pd.read_csv(self._mapping_path, sep=' ')
            self._units2nodes_map = mapping_file_df[['node_ids', 'unit_ids']].set_index('node_ids').to_dict()['unit_ids']
            if 'start_times' in mapping_file_df:
                self._time_window.units_lu = mapping_file_df[['unit_ids', 'start_times', 'stop_times']].set_index('unit_ids')

        except (FileNotFoundError, UnicodeDecodeError):
            io.log_exception('{}: {} should be a space separated file with columns "node_ids", "unit_ids"'.format(NeuropixelsNWBReader.__name__, self._mapping_path))


class SamplingStrategy(MappingStrategy):
    def __init__(self, with_replacement=False, **kwargs):
        super().__init__(**kwargs)
        self._with_replacement = with_replacement

    def build_map(self, node_set):
        node_ids = node_set.node_ids
        unit_ids = self.units_table.index.values

        # There is no way to randomly map a subset of possible unit_ids to all possible
        # node_ids. Ignore, warn user, or fail depending on config file option
        if (not self._with_replacement) and len(node_ids) > len(unit_ids):
            if self._missing_ids == 'fail':
                # Fail application
                io.log_error('NeuropixelsNWBReader: Not enough NWB unit_ids to map onto node_set.')

            # Not all node_ids will have spikes, TODO: Make do this at random?            
            node_ids = node_ids[:len(unit_ids)]

        # When running with MPI, need to make sure the sampling of the unit_id maps is
        # consistent across all cores. Shuffle on rank 0 and broadcast the new order to all
        # other ranks 
        if MPI_rank == 0:
            shuffled_unit_ids = np.random.choice(
                unit_ids, size=len(node_ids), replace=self._with_replacement
            )
        else:
            shuffled_unit_ids = None
        shuffled_unit_ids = bcast(shuffled_unit_ids, 0)

        # TODO: Include population name
        # Creates an mapping between SONTA node_ids and NWB unit_ids
        self._units2nodes_map = pd.DataFrame({
            'node_ids': node_ids,
            'unit_ids': shuffled_unit_ids
        }).set_index('node_ids').to_dict()['unit_ids']


def filter_table(table_df, filters_dict):
    if filters_dict:
        # Filter out only those specified units
        mask = True
        for col_name, filter_val in filters_dict.items():
            if isinstance(filter_val, str):
                mask &= table_df[col_name] == filter_val
            
            elif isinstance(filter_val, (list, np.ndarray, tuple)):
                mask &= table_df[col_name].isin(filter_val)
            
            elif isinstance(filter_val, dict):
                col = filter_val.get('column', col_name)
                op = filter_val['operation']
                val = filter_val['value']
                val = '"{}"'.format(val) if isinstance(val, str) else val
                expr = 'table_df["{}"] {} {}'.format(col, op, val)
                mask &= eval(expr)

            else:
                mask &= table_df[col_name] == filter_val

        table_df = table_df[mask]

    return table_df