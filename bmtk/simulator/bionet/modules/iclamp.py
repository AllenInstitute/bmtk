import numpy as np
import pandas as pd
import h5py
import six
from neuron import h

from bmtk.simulator.core.modules import iclamp
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

    @property
    def amps(self):
        return self._amps

    @property
    def start(self):
        return np.min(self._delays)

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
            self._ts_col = args.get('timestamps_col', 'timestamps')
            self._amps_col = args.get('amplitudes_col', 'amps')
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
                self._ts_vals = np.concatenate([0.0], self._ts_vals)
                self._istart = 0.0

            else:
                wrn_msg = '{}: initial onset of stimulus at {} does not occur at a {} timestep. Attempting to update' \
                          'timesteps and amp values'.format(IClampMod.__name__, self._istart, self._idt)
                io.log_warning(wrn_msg)

                gcd = np.gcd(np.int(self._idt), np.int(self._istart))
                amps = np.repeat(self._amps_vals, int(self._idt)/gcd)
                zero_curr = np.zeros(int(self._istart/gcd))
                self._amps_vals = np.concatenate((zero_curr, amps))
                self._ts_vals = np.concatenate(([0.0], self._ts_vals))
                self._idt = gcd
                self._istart = 0.0

    @property
    def amps(self):
        return self._amps_vals

    @property
    def timestamps(self):
        return self._delays

    @property
    def start(self):
        return np.min(self._delays)

    @property
    def stop(self):
        return np.max([delay + dur for delay, dur in zip(self._delays, self._durations)])

    def create_clamps(self, hobj):
        """
        clamp = h.IClamp(hobj)
        clamp.delay = 0.0 # self._istart
        clamp.dur = self._istop

        vect_stim = h.Vector(self._amps_vals)
        vect_stim.play(clamp._ref_amp, self._idt)
        """

        # ts = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0])
        # amps = np.array([0.0, -0.2, 0.0, 0.2, 0.0])

        # ts = np.array([500.0, 1000.0, 1500.0, 2000.0])
        # amps = np.array([-0.2, 0.0, 0.2, 0.0])

        # ts = np.array([700.0, 1200.0, 1700.0, 2200.0])
        # amps = np.array([-0.2, 0.0, 0.2, 0.0])

        # ts = np.array([300.0, 800.0, 1300.0, 1800.0])
        # amps = np.array([-0.2, 0.0, 0.2, 0.0])

        # ts = np.array([700.0, 1700.0, 2200.0, 2900.0])
        # amps = np.array([-0.2, 0.0, 0.2, 0.0])

        # ts = np.array([200.0, 1200.0, 1700.0, 2700.0])
        # amps = np.array([-0.2, 0.0, 0.2, 0.0])

        # ts = np.array([600.0, 800.0, 1000.0, 1200.0])
        # amps = np.array([.1, .2, 0.3, 0.0])

        # dt = np.diff(ts)[0]
        # istart = ts[0]
        #
        # if istart > 0.0:
        #     if istart - dt == 0.0:
        #         amps = np.concatenate(([0.0], amps))
        #         istart = 0.0
        #
        #     else:
        #         print('HERE')
        #         gcd = np.gcd(np.int(dt), np.int(istart))
        #         print(gcd)
        #         amps = np.repeat(amps, int(dt)/gcd)
        #         zero_curr = np.zeros(int(istart/gcd))
        #         amps = np.concatenate((zero_curr, amps))
        #         print(gcd, amps)
        #         dt = gcd
        #         istart = 0.0
        #         # exit()

        # istop =

        # onset = ts[0]
        # print(dt, onset)
        # print(np.int(dt), np.int(onset))
        #
        # gcd = np.gcd(np.int(dt), np.int(onset))
        # amps = np.repeat(amps, int(onset/gcd))
        # dt = gcd
        #
        # print(np.repeat(amps, int(onset/gcd)))
        # print(np.arange(onset, ts[-1], step=gcd))
        # # print(np.gcd((int(ts), int(onset))))
        # # exit()


        clamp = h.IClamp(hobj)
        clamp.delay = self._istart
        clamp.dur = 4000.0  # self._istop

        vect_stim = h.Vector(self._amps_vals)
        vect_stim.play(clamp._ref_amp, self._idt)

        return [(vect_stim, clamp)]

        # for amp, delay, dur in zip(self._amps, self._delays, self._durations):
        #     clamp = h.IClamp(hobj)
        #     clamp.amp = amp
        #     clamp.delay = delay
        #     clamp.dur = dur
        #     clamps.append(clamp)
        #
        # return clamps

#     'input_type': 'current_clamp',
#     'module': 'IClamp',
#     'node_set': 'biophys_cells',
#     'amp': [0.1, 0.2, 0.3],
#     'delay': [100.0, 200.0, 300.0],
#     'duration': [50.0, 50.0, 50.0]
# }
#
# iclamp_csv = {
#     'input_type': 'csv',
#     'module': 'IClamp',
#     'node_set': 'biophys_cells',
#     'file': 'clamp_amplitdues.csv',
#     'separator': ' ',
#     'timestamps_col': 'timestamps',
#     'amplitudes_col': 'amps'
# }
#
# iclamp_nwb = {
#     'input_type': 'nwb',
#     'module': 'IClamp',
#     'node_set': 'biophys_cells',
#     'file': '487667203_ephys.nwb',
#     'sweep_id': 8,
#     'delay': 0.0
# }


class NWBReader(object):
    def __init__(self, **args):
        import matplotlib.pyplot as plt

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
            sweep_grp = h5['epochs/{}/stimulus/timeseries'.format(self._sweep_id)]
            self._idt = 1.0/sweep_grp['starting_time'].attrs['rate']

            self._amps_vals = sweep_grp['data'][int(17/self._idt):int(20/self._idt)]*1.0e9
            self._istart = 0.0

            nsamples = len(self._amps_vals)
            stop_time = nsamples*self._idt
            ts = np.arange(0.0, stop_time, step=self._idt)
            plt.plot(ts, self._amps_vals)
            # plt.show()

    def create_clamps(self, hobj):
        clamp = h.IClamp(hobj)
        clamp.delay = self._istart
        clamp.dur = 4000.0  # self._istop

        vect_stim = h.Vector(self._amps_vals)
        vect_stim.play(clamp._ref_amp, self._idt)

        return [(vect_stim, clamp)]


class IClampMod(iclamp.IClampMod):
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
