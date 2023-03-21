import numpy as np
import pandas as pd
import h5py

from .simulator_module import SimulatorMod
from bmtk.simulator.core.io_tools import io


class AmpsReader(object):
    def __init__(self, **args):
        self.has_delays = True

        try:
            self.amps = self.__to_list(args['amp'])
            self.delays = self.__to_list(args['delay'])
            self.durations = self.__to_list(args['duration'])
        except KeyError as ke:
            io.log_exception('{}: missing current-clamp parameter {}.'.format(IClampMod.__name__, ke))

        if not len(self.amps) == len(self.delays) == len(self.durations):
            io.log_exception('{}: current clamp parameters amp, delay, duration must be same length when using list.'.format(
                IClampMod.__name__
            ))

    @staticmethod
    def __to_list(value):
        # Helper function so that amp, delay, duration can be passed as either a list of scalar
        return value if isinstance(value, (list, tuple, np.ndarray)) else [value]


class CSVAmpReader(object):
    def __init__(self, **args):
        self.has_delays = False

        try:
            self._csv_path = args['file']
            self._sep = args.get('separator', ' ')
            self._ts_col = args.get('timestamps_column', 'timestamps')
            self._amps_col = args.get('amplitudes_column', 'amps')
        except KeyError as ke:
            io.log_exception('{}: missing current-clamp parameter {}.'.format(IClampMod.__name__, ke))

        self._inputs_df = pd.read_csv(self._csv_path, sep=self._sep)

        # Make sure csv has atleast two values
        if len(self._inputs_df) < 2:
            io.log_exception('{}: csv file must be row oriented and contain two or more values.'.format(
                IClampMod.__name__)
            )

        # Check that csv has both a timestamp and amplitudes column, if not, or set to a different column name than
        # the default, indicate problem to user
        for pname, cname in zip(['timestamps_col', 'amplitudes_col'], [self._ts_col, self._amps_col]):
            if cname not in self._inputs_df.columns:
                io.log_exception('{}: csv file missing column "{}". Use "{}" option to specify.'.format(
                    IClampMod.__name__, cname, pname)
                )

        self.delays = self._inputs_df[self._ts_col].values
        self.amps = self._inputs_df[self._amps_col].values


class NWBReader(object):
    def __init__(self, **args):
        self.has_delays = False 

        try:
            self._nwb_path = args['file']
            self._sweep_id = args['sweep_id']
            self._downsample = args.get('downsample', None)
            self._sweep_window = args.get('sweep_window', None)
            self._offset = args.get('delay', 0.0)
        except KeyError as ke:
            io.log_exception('{}: missing current-clamp parameter {}.'.format(IClampMod.__name__, ke))

        try:
            # if the "sweep_id" is a integer type value: 6, "6", 6.0; then change it to Sweep_<id> which is
            # how it is saved in the Allen NWB files.
            sweep_id_num = int(self._sweep_id)
            self._sweep_id = 'Sweep_{}'.format(sweep_id_num)
        except ValueError as ve:
            self._sweep_id = self._sweep_id

        with h5py.File(self._nwb_path, 'r') as h5:
            if self._sweep_id not in h5['/epochs']:
                io.log_exception('{}: {} missing sweep group {}.'.format(
                    IClampMod.__name__, self._nwb_path, self._sweep_id)
                )

            # Get timestamps and amplitudes from Allen Cell-Types filefile
            sweep_grp = h5['epochs/{}/stimulus/timeseries'.format(self._sweep_id)]
            self._idt = 1.0/sweep_grp['starting_time'].attrs['rate']
            self.amps = sweep_grp['data'][()]*1.0e9
            self.delays = np.arange(len(self.amps))*self._idt*1000

            # If the "downsample" option is used take every n'th value in the nwb stimulus
            if self._downsample is not None and self._downsample > self._idt:
                stride = int(np.round(self._downsample/self._idt))
                self.amps = self.amps[::stride]
                self.delays = self.delays[::stride]

            # If "sweep_window" filter out stimuli that falls outside the time window
            if self._sweep_window is not None:
                if len(self._sweep_window) != 2 or self._sweep_window[0] >= self._sweep_window[1] or self._sweep_window[0] < 0:
                    io.log_exception('{}: "sweep_window" parameter must be set to [start_time, stop_time] where 0 <= start_time < stop_time.'.format(
                        IClampMod.__name__
                    ))
                idx_beg = np.argwhere(self.delays >= self._sweep_window[0])
                idx_end = np.argwhere(self.delays <= self._sweep_window[1]) + 1
                self.amps = self.amps[idx_beg:idx_end]
                self.delays = self.delays[idx_beg:idx_end]

            # Adds an offset for the simulation time, needs to be done after the sweep_window
            # is applied
            self.delays += self._offset


class IClampMod(SimulatorMod):
    """
    A Module to help with creating a current-clamp/generator that can be injected into a selected subset 
    of cells.

    This class if primarily focused on reading the inputs parameters and/or configuration files to determine
    the current injection onsets, amplitudes, and stop times depending on the input-type. The actual implementation
    of the current injections is dependent on the simulator (IClamp for BioNet, step_current_generator for PointNet)
    and implementation is done in their respective modules.iclamps.py files.

    Input Parameters:
    -----------------
    input_type : string
        Specifies if input parameters are passed in directory, or needs to be read from a csv or nwb file.
    node_set: string, list or dictionary
        A filter to determine which subset of nodes/cells that injection current will be applied too.
    """
    input2reader_map = {
        'current_clamp': AmpsReader,
        'csv': CSVAmpReader,
        'file': CSVAmpReader,
        'nwb': NWBReader,
        'allen': NWBReader
    }

    def __init__(self, input_type, **mod_args):
        if input_type not in IClampMod.input2reader_map:
            err_msg = '{}: invalid input_type value "{}",'.format(self.__class__.__name__, input_type)
            err_msg += ' unable to parse current clamp parameters.'
            err_msg += ' Valid options: {}'.format(', '.join(list(self.input2reader_map.keys())))
            io.log_exception(err_msg)

        self._node_set = mod_args.get('node_set', 'all')       
        self._amp_reader = IClampMod.input2reader_map[input_type](**mod_args)
