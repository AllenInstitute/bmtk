import numpy as np
import pandas as pd
import h5py
from neuron import h

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.io_tools import io


class AmpsReader(object):
    def __init__(self, **args):
        try:
            self._amps = self.__to_list(args['amp'])
            self._delays = self.__to_list(args['delay'])
            self._durations = self.__to_list(args['duration'])
        except KeyError as ke:
            io.log_exception('{}: missing current-clamp parameter {}.'.format(IClampMod.__name__, ke))

        if not len(self._amps) == len(self._delays) == len(self._durations):
            io.log_exception('{}: current clamp parameters amp, delay, duration must be same length when using list.')

    def create_clamps(self, hobj):
        clamps = []
        for amp, delay, dur in zip(self._amps, self._delays, self._durations):
            clamp = h.IClamp(hobj)
            clamp.amp = amp
            clamp.delay = delay
            clamp.dur = dur
            clamps.append(clamp)

        return clamps

    @staticmethod
    def __to_list(value):
        # Helper function so that amp, delay, duration can be passed as either a list of scalar
        return value if isinstance(value, (list, tuple, np.ndarray)) else [value]


class CSVAmpReader(object):
    def __init__(self, **args):
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

        # NRN Vector.play function requires consistent
        dts = np.unique(np.diff(self._inputs_df[self._ts_col].values))
        if len(dts) > 1:
            io.log_exception('{}: csv timestamps column ({}) must have a consistent intervals.'.format(
                IClampMod.__name__, self._ts_col)
            )
        self._idt = dts[0]
        self._istart = self._inputs_df[self._ts_col].values[0]
        self._amps_vals = self._inputs_df[self._amps_col].values
        self._ts_vals = self._inputs_df[self._ts_col].values

        # The way NEURON's Vector.play([amps], dt) works is that it will access the [amps] at every dt interval (0.0,
        #  dt, 2dt, 3dt, ...) in ms regardless of when the IClamp starts. Thus if the initial onset stimuli is > 0.0 ms,
        #  we need to update the amps array so that the following occurs:
        #    1. amps[0] is the injection current (usually 0.0mA) that occurs at the very start of the simulation.
        #    2. the above _istart, and all other timestamps, is a multiple of dt.
        #  this may require decreasing dt which will require increasing the the amp array. eg:
        #      times: [200, 800, 1200, 1600] ==> [  0, 200, 400, 800, 1000, 1200, 1400, 1600]
        #       amps: [0.1, 0.2,  0.3,  0.0] ==> [0.0, 0.1, 0.1, 0.2,  0.2,  0.3,  0.3,  0.0]
        if self._istart < 0.0:
            # Check in trying to start a current clamp before simulation starts
            io.log_exception('{}: initial onset of stimulus ({}) cannot be a negative number.'.format(
                IClampMod.__name__, self._ts_col)
            )
        elif self._istart > 0.0:
            if self._istart - self._idt == 0.0:
                # Simplist case initial onset time occurs at first dt after start of simulation, add a 0.0 current
                # clamp at the very beginning of the simulation.
                self._amps_vals = np.concatenate(([0.0], self._amps_vals))
                self._ts_vals = np.concatenate(([0.0], self._ts_vals))
                self._istart = 0.0

            else:
                warn_msg = '{}: initial onset of stimulus at {} does not occur at a {} timestep. Attempting to update' \
                          'timesteps and amp values'.format(IClampMod.__name__, self._istart, self._idt)
                io.log_warning(warn_msg)

                gcd = np.gcd(int(self._idt), int(self._istart))
                amps = np.repeat(self._amps_vals, int(self._idt)/gcd)
                zero_curr = np.zeros(int(self._istart/gcd))
                self._amps_vals = np.concatenate((zero_curr, amps))
                self._ts_vals = np.concatenate(([0.0], self._ts_vals))
                self._idt = gcd
                self._istart = 0.0

        # Try to determine IClamp stop time (eg max duration)
        if self._amps_vals[-1] != 0:
            self._istop = self._ts_vals[-1] + self._idt
            warn_msg = '{}: Stimulus of {} does not end with a 0.0, attempting to set turn off at time {}.'.format(
                IClampMod.__name__, self._csv_path, self._istop)
            io.log_warning(warn_msg)
        else:
            self._istop = self._ts_vals[-1]

    @property
    def amps(self):
        return self._amps_vals

    @property
    def dt(self):
        return self._idt

    def create_clamps(self, hobj):
        clamp = h.IClamp(hobj)
        clamp.delay = self._istart
        clamp.dur = self._istop

        vect_stim = h.Vector(self._amps_vals)
        vect_stim.play(clamp._ref_amp, self._idt)

        return [(vect_stim, clamp)]


class NWBReader(object):
    def __init__(self, **args):

        try:
            self._nwb_path = args['file']
            self._sweep_id = args['sweep_id']
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

            sweep_grp = h5['epochs/{}/stimulus/timeseries'.format(self._sweep_id)]
            self._idt = 1.0/sweep_grp['starting_time'].attrs['rate']
            self._amps_vals = sweep_grp['data'][()]*1.0e9
            self._istart = 0.0

            nsamples = len(self._amps_vals)
            self._istop = nsamples*self._idt*1000.0

    def create_clamps(self, hobj):
        clamp = h.IClamp(hobj)
        clamp.delay = self._istart
        clamp.dur = self._istop

        vect_stim = h.Vector(self._amps_vals)
        vect_stim.play(clamp._ref_amp, self._idt)

        return [(vect_stim, clamp)]


class IClampMod(SimulatorMod):
    input2reader_map = {
        'current_clamp': AmpsReader,
        'csv': CSVAmpReader,
        'file': CSVAmpReader,
        'nwb': NWBReader,
        'allen': NWBReader
    }

    def __init__(self, input_type, **mod_args):
        # Check input_type parameter
        if input_type not in IClampMod.input2reader_map:
            err_msg = '{}: invalid input_type value "{}",'.format(self.__class__.__name__, input_type)
            err_msg += ' unable to parse current clamp parameters.'
            err_msg += ' Valid options: {}'.format(', '.join(list(self.input2reader_map.keys())))
            io.log_exception(err_msg)

        # self.amps_reader = AmpsReader.load(input_type, **mod_args)
        self._node_set = mod_args.get('node_set', 'all')
        self._section_name = mod_args.get('section_name', 'soma')
        self._section_index = mod_args.get('section_index', 0)
        self._section_dist = mod_args.get('section_dist', 0.5)

        self._amp_reader = IClampMod.input2reader_map[input_type](**mod_args)
        self._iclamps = []

    def initialize(self, sim):
        # Get select node gids, but only for those nodes that are on the current rank (if running on multiple cores)
        select_gids = list(sim.net.get_node_set(self._node_set).gids())
        gids_on_rank = list(set(select_gids) & set(select_gids))

        for gid in gids_on_rank:
            cell = sim.net.get_cell_gid(gid)
            hobj_sec = getattr(cell.hobj, self._section_name)[self._section_index](self._section_dist)
            self._iclamps.extend(self._amp_reader.create_clamps(hobj_sec))
