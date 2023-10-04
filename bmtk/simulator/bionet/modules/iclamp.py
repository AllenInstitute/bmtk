import numpy as np
from neuron import h

# from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.core.modules import iclamp


class AmpsReaderNRN(iclamp.AmpsReader):
    def create_clamps(self, hobj):
        clamps = []
        for amp, delay, dur in zip(self.amps, self.delays, self.durations):
            clamp = h.IClamp(hobj)
            clamp.amp = amp
            clamp.delay = delay
            clamp.dur = dur
            clamps.append(clamp)

        return clamps


class CSVAmpReaderNRN(iclamp.CSVAmpReader):
    def __init__(self, **args):
        super().__init__(**args)

        # NRN Vector.play function requires consistent
        dts = np.unique(np.diff(self._inputs_df[self._ts_col].values))
        if len(dts) > 1:
            io.log_exception('{}: csv timestamps column ({}) must have a consistent intervals.'.format(
                IClampMod.__name__, self._ts_col)
            )
        self._idt = dts[0]
        self._istart = self.delays[0]  # self._inputs_df[self._ts_col].values[0]
        # self._amps_vals = self._inputs_df[self._amps_col].values
        # self._ts_vals = self._inputs_df[self._ts_col].values

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
                IClampMod.__name__, self.delays)
            )
        elif self._istart > 0.0:
            if self._istart - self._idt == 0.0:
                # Simplist case initial onset time occurs at first dt after start of simulation, add a 0.0 current
                # clamp at the very beginning of the simulation.
                self.amps = np.concatenate(([0.0], self.amps))
                self.delays = np.concatenate(([0.0], self.delays))
                self._istart = 0.0

            else:
                warn_msg = '{}: initial onset of stimulus at {} does not occur at a {} timestep. Attempting to update' \
                          'timesteps and amp values'.format(IClampMod.__name__, self._istart, self._idt)
                io.log_warning(warn_msg)

                gcd = np.gcd(int(self._idt), int(self._istart))
                amps = np.repeat(self.amps, int(self._idt)/gcd)
                zero_curr = np.zeros(int(self._istart/gcd))
                self.amps = np.concatenate((zero_curr, amps))
                self.delays = np.concatenate(([0.0], self.delays))
                self._idt = gcd
                self._istart = 0.0

        # Try to determine IClamp stop time (eg max duration)
        if self.amps[-1] != 0:
            self._istop = self.delays[-1] + self._idt
            warn_msg = '{}: Stimulus of {} does not end with a 0.0, attempting to set turn off at time {}.'.format(
                IClampMod.__name__, self._csv_path, self._istop)
            io.log_warning(warn_msg)
        else:
            self._istop = self.delays[-1]

    def create_clamps(self, hobj):
        clamp = h.IClamp(hobj)
        clamp.delay = self._istart
        clamp.dur = self._istop

        vect_stim = h.Vector(self.amps)
        vect_stim.play(clamp._ref_amp, self._idt)

        return [(vect_stim, clamp)]


class NWBReaderNRN(iclamp.NWBReader):
    def __init__(self, **args):
        super().__init__(**args)
        self._istop = len(self.delays)*self._idt*1000.0

    def create_clamps(self, hobj):
        clamp = h.IClamp(hobj)
        clamp.delay = self._offset
        clamp.dur = self._istop

        vect_stim = h.Vector(self.amps)
        vect_stim.play(clamp._ref_amp, self._idt)

        return [(vect_stim, clamp)]


class IClampMod(iclamp.IClampMod):
    @property
    def input2reader_map(self):
        return {
            'current_clamp': AmpsReaderNRN,
            'csv': CSVAmpReaderNRN,
            'file': CSVAmpReaderNRN,
            'nwb': NWBReaderNRN,
            'allen': NWBReaderNRN
        }

    def __init__(self, input_type, **mod_args):
        super().__init__(input_type, **mod_args)
       
        # Select location to place iclamp, if not specified use the center of the soma
        self._section_name = mod_args.get('section_name', 'soma')
        self._section_index = mod_args.get('section_index', 0)
        self._section_dist = mod_args.get('section_dist', 0.5)

        # IClamp objects need to be saved in memory otherwise NRN will try to garbage collect them
        # prematurly
        self._iclamps = []

    def initialize(self, sim):
        # Get select node gids, but only for those nodes that are on the current rank (if running on multiple cores)
        select_gids = list(sim.net.get_node_set(self._node_set).gids())
        local_gids = sim.net.get_local_cells()
        gids_on_rank = list(set(select_gids) & set(local_gids))

        for gid in gids_on_rank:
            cell = sim.net.get_cell_gid(gid)
            hobj_sec = getattr(cell.hobj, self._section_name)[self._section_index](self._section_dist)
            self._iclamps.extend(self._amp_reader.create_clamps(hobj_sec))
