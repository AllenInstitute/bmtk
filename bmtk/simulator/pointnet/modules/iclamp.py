import nest

from bmtk.simulator.core.modules import iclamp
from bmtk.simulator.pointnet.io_tools import io


class IClampMod(iclamp.IClampMod):
    def initialize(self, sim):
        """Creates a NEST current clamp (eg step-generator) and attaches it to the given cells."""
        
        # Get a list of amplitude times and values, making sure values are valid for NEST and
        # also turning off iclamp after "duration"
        # TODO: May be faster if put the has_delays check first 
        amp_times = []
        amp_vals = []
        n_steps = len(self._amp_reader.delays)
        for idx in range(n_steps):
            start_time = self._amp_reader.delays[idx]
            if start_time <= sim.dt:
                # NEST has issues with amplitude_values that occur before the dt.
                io.log_warning('IClamp Stimulus has onset ("delay") skipping values that occur before "dt"')
                continue

            # Add a time and amp value for turning on iclamp
            amp_times.append(start_time)
            amp_vals.append(self._amp_reader.amps[idx])
            
            # if there is a duration, make sure to turn off iclamp (set to 0.0) at timestamp
            # delay + duration
            if self._amp_reader.has_delays:
                duration = self._amp_reader.durations[idx]
                stop_time = start_time + duration
                last_element = idx + 1 == n_steps
                if last_element or stop_time <= self._amp_reader.delays[idx+1]:
                    amp_times.append(stop_time)
                    amp_vals.append(0.0)

        # Create iclamp/current generator
        scg = nest.Create(
            "step_current_generator",
            params={
                'amplitude_times': amp_times, 
                'amplitude_values': amp_vals
            }
        )

        # attach iclamp to all the specified cells.
        # TODO: Check it will work if multiple Node populations are selected!
        node_set = sim.net.get_node_set(self._node_set)
        nest_ids = node_set.gids()
        nest.Connect(scg, list(nest_ids), syn_spec={'delay': sim.dt})
