from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .simulator_module import SimulatorMod
from bmtk.simulator.core.io_tools import io
from bmtk.utils import lazy_property


try:
    import pynwb
    has_pynwb = True
except ImportError as ie:
    has_pynwb = False


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


# The ndx-aibs extensions for nwb are required to load the neuropixel data modules, 
# warning: can take some time to load
file_dir = Path(__file__).parent
namespace_path = (file_dir/"ndx-aibs-ecephys.namespace.yaml").resolve()
pynwb.load_namespaces(str(namespace_path))


class ECEphysUnitsModule(SimulatorMod):
    """
    TODO:
    - Have option to specify the nwb file units and/or get them from the NWB
    - Have option to save units-node mapping to output folder
    """
    def __init__(self, name, **kwargs):
        self._name = name
        self._node_set = kwargs['node_set']

        if not has_pynwb:
            io.log_exception('ECEphysUnitsModule: pynwb is not installed (pip install pynwb), unable to use module.')
       
        # Load a Strategy for mapping SONATA node_ids to NWB unit_ids
        self._mapping_name = kwargs.get('mapping', 'invalid_strategy').lower()
        if self._mapping_name in ['units_map']:
            self._mapping_strategy = UnitIdMapStrategy(**kwargs)

        elif self._mapping_name in ['sample', 'sample_without_replacement']:
            self._mapping_strategy = SamplingStrategy(with_replacement=False, **kwargs)

        elif self._mapping_name in ['sample_with_replacement']:
            self._mapping_strategy = SamplingStrategy(with_replacement=True, **kwargs)

        else:
            io.log_exception('ECEphysUnitsModule: Invalid "mapping" parameters, options: units_map, sample, sample_with_replacement')

    def initialize(self, sim):
        raise NotImplementedError()


class NWBFileWrapper(object):
    # A Simple wrapper class for nwb files, mainly to keep track of file-path which I can't get from pynwb
    # TODO: Implement a Singleton so that the same nwb file isn't loaded multiple times
    def __init__(self, nwb_path):
        if isinstance(nwb_path, pynwb.file.NWBFile):
            self._id = nwb_path.identifier
            self._io = nwb_path
        else:
            self._id = nwb_path
            self._io = pynwb.NWBHDF5IO(nwb_path, 'r').read()
        
    @property
    def uuid(self):
        return self._id
    
    def __getattr__(self, name):
        return getattr(self.__dict__['_io'], name)


class TimeWindow(object):
    """
    A class for dealing with different strategies for storing and intrepreting time windows intervals [start, stop], 
    mainly for use in filtering spike times. Including converting between seconds/miliseconds, parsing an NWB stimulus 
    table, and look up for individual unit time, and dealing with defaults. To initialize::

        tw = TimeWindow(defaults=[interval1, interval2, ...], nwb_files=[session1.nwb, session2.nwb, ...])

    Where interval<i> is a time-window associated with all unit spikes in session<i>.nwb, and can include None (in which
    case it will not filter unit spike times).

    If individual units will have unique time intervals then you can pass in a pandas DataFrame with columns 
    [unit_ids, start_times, stop_times], with values in ms::

        tw.units_lu = units_times_table_df

    And to fetch the time window associated with unit, for example unit_id 9999 in session_0.nwb, then call::

        window = tw[9999, 'session_0.nwb']

    If will first check to see if unit 9999 has a special [start_time, stop_time] in units_lu table, and if not then fall
    back to the default for 'session_0.nwb', and in seconds. If unit_id/default_session is then it returns a None value.
    """
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
                io.log_exception('ECEphysUnitsModule: Cannot match each "time_window" with the "input_file"s.')
        
            if n_windows == 1:
                # convert [interal] -> [interval, interval, interval, ...]
                time_windows = time_windows*n_windows
            self._default_windows = {nwb.uuid: self._parse_tw(tw, nwb) for tw, nwb in zip(time_windows, nwb_files)}

    @property
    def units_lu(self):
        return self._units_lu
    
    @units_lu.setter
    def units_lu(self, units_table):
        if 'start_times' in units_table and 'stop_times' in units_table:
            units_table['start_times'] = units_table['start_times']*self.conversion_factor
            units_table['stop_times'] = units_table['stop_times']*self.conversion_factor

            if 'unit_ids' in units_table.columns:
                units_table = units_table.set_index('unit_ids')
            
            self._units_lu = units_table

    def _tolist(self, window):
        if isinstance(window, dict):
            # Is a stimulus_table filter, ex. {"interval_name": "gratings", "ori": 90.0, ...}
            return [window]
        elif isinstance(window, (tuple, list)) and len(window) == 0:
            # Is an empyt list
            return []
        elif isinstance(window, (tuple, list)) and isinstance(window[0], (tuple, list, dict, np.ndarray)):
            # is a list of intervals, ex. [[0.0, 100.0], [200.0, 300.0], {stim:gratings} ...]
            return window
        else:
            # assume is a interval, ex. [0.0, 100.0]
            return [window]

    def _parse_tw(self, interval, nwb_file):
        """Converts intervals, including windows and stim-table filters, into appropiate format [start (s), stop (s)]"""
        if isinstance(interval, dict):
            # If it is a dictionary try to find time interval by filtering on nwb.intervals, eg stimulus_table. The filter
            # is a dictionary {'interval_name': 'flashes', 'col1': val1, 'col2': val2, ...}
            filter = interval.copy()
            stim_name = filter.pop('interval_name', None)
            stim_idx = filter.pop('interval_index', 'all')

            # In the NWB there are separate tables for each stimulus, and sometimes they are stored in the
            # nwb as <flashes>_presentations.
            if stim_name is None:
                io.log_exception('Stimulus table filter missing "interval_name"')
            if stim_name in nwb_file.intervals.keys():
                interval_df = nwb_file.intervals[stim_name].to_dataframe()
            elif stim_name + '_presentations' in nwb_file.intervals.keys():
                interval_df = nwb_file.intervals[stim_name + '_presentations'].to_dataframe()
            else:
                io.log_exception('interval name "{}" not found in {}'.format(stim_name, nwb_file.uuid))

            # In most cases 
            interval_df = filter_table(interval_df, filter)
            if len(interval_df) == 0:
                return [0.0, np.inf]
            
            if stim_idx == 'all':
                start_time = interval_df['start_time'].min()
                stop_time = interval_df['stop_time'].max()
            else:
                start_time = interval_df.iloc[stim_idx]['start_time']
                stop_time = interval_df.iloc[stim_idx]['stop_time']

            # In the NWB stim_table and units_tables uses seconds, do not convert time-window
            return [start_time, stop_time]

        else:
            # Is an interval [stop, start] that is entered in manually in units of ms, convert 
            # to seconds so it matches nwb spike_times units (s)
            return [interval[0]/1000.0, interval[1]/1000.0]

    def __getitem__(self, unit_info):
        unit_id, nwb_uuid = unit_info[0], unit_info[1]
        if (self.units_lu is not None) and (unit_id in self.units_lu.index):
            unit = self.units_lu.loc[unit_id]
            return [unit['start_times'], unit['stop_times']]
        elif self._default_windows and nwb_uuid in self._default_windows.keys():
            return self._default_windows[nwb_uuid]
        else:
            return None


class MappingStrategy(object):
    def __init__(self, **kwargs):
        self._nwb_paths = kwargs['input_file']
        self._filters = kwargs.get('units', {})       
        self._simulation_onset = kwargs.get('interval_offset', 0.0)/1000.0
        self._missing_ids = kwargs.get('missing_ids', 'fail')
        self._cache_spike_times = kwargs.get('cache', False)
        self._spike_times_cache = {}
        
        default_window = kwargs.get('interval', None)
        self._time_window = TimeWindow(defaults=default_window, nwb_files=self.nwb_files)
       
        self._units_table = None
        self._units2nodes_map = None

    @lazy_property
    def nwb_files(self):
        if not isinstance(self._nwb_paths, (list, tuple)):
            self._nwb_paths = [self._nwb_paths] 

        nwb_files = []
        for nwb_path in self._nwb_paths:            
            nwb_files.append(NWBFileWrapper(nwb_path))

        return nwb_files
    
    @property
    def units2nodes_map(self):
        return self._units2nodes_map

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

            # if merged_table is None or len(merged_table) == 0:
            #     io.log_exception('ECEphysUnitsModule: Could not parse units table from nwb_file(s).')

            self._units_table = merged_table

        return self._units_table
    
    def _load_units_table(self, nwb_file):
        units = nwb_file.units.to_dataframe().reset_index()
        units = units.rename(columns={'id': 'unit_id', 'peak_channel_id': 'channel_id'})
        channels = nwb_file.electrodes.to_dataframe()
        channels = channels.reset_index().rename(columns={'id': 'channel_id'})
        return units.merge(channels, how='left', on='channel_id').set_index('unit_id')

    def _filter_units_table(self, units_table):
        units_table = filter_table(units_table, self._filters)
        return units_table        

    def build_map(self, node_set):
        raise NotImplementedError()

    def get_spike_trains(self, node_id, source_population):
        # TODO: Is it worth caching spike-trains so we don't have to do a lookup + filtering 
        # every time?
        if node_id not in self._units2nodes_map:
            msg = 'ECEphysUnitsModule: Could not find mapping for node_id {}.'.format(node_id)
            if self._missing_ids == 'fail':
                io.log_exception(msg) 
            elif self._missing_ids == 'warn':
               io.log_warning(msg)
               return np.array([])

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
        return spike_times


class UnitIdMapStrategy(MappingStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            self._mapping_path = kwargs['units']
        except KeyError:
            io.log_exception('ECEphysUnitsModule: Could not find "units" csv path for units-to-nodes mapping.')

    def _filter_units_table(self, units_table):
        return units_table

    def build_map(self, node_set):
        try:
            # TODO: Include population name
            mapping_file_df = self._mapping_path if isinstance(self._mapping_path, pd.DataFrame) else pd.read_csv(self._mapping_path, sep=' ')
            self._units2nodes_map = mapping_file_df[['node_ids', 'unit_ids']].set_index('node_ids').to_dict()['unit_ids']
            if 'start_times' in mapping_file_df:
                self._time_window.units_lu = mapping_file_df[['unit_ids', 'start_times', 'stop_times']].set_index('unit_ids')

        except (FileNotFoundError, UnicodeDecodeError):
            io.log_exception('ECEphysUnitsModule: {} should be a space separated file with columns "node_ids", "unit_ids"'.format(self._mapping_path))


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
                io.log_exception('ECEphysUnitsModule: Not enough NWB unit_ids to map onto node_set.')

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
            try:
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
            
            except KeyError as ke:
                col = filter_val.get('column', col_name) if isinstance(filter_val, dict) else col_name
                io.log_exception('ECEphysUnitsModule: Could not find "{}" column in units/electrodes table.'.format(col))

        table_df = table_df[mask]

    return table_df