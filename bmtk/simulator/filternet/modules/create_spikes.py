import os
import numpy as np
import random
import six

from .base import SimModule
from bmtk.utils.reports.spike_trains import SpikeTrains, pop_na
from bmtk.simulator.filternet.lgnmodel import poissongeneration as pg


class SpikesGenerator(SimModule):
    def __init__(self, spikes_file_csv=None, spikes_file=None, spikes_file_nwb=None, tmp_dir='output'):
        def _get_file_path(file_name):
            if file_name is None or os.path.isabs(file_name):
                return file_name

            return os.path.join(tmp_dir, file_name)

        self._csv_fname = _get_file_path(spikes_file_csv)
        self._save_csv = spikes_file_csv is not None

        self._h5_fname = _get_file_path(spikes_file)
        self._save_h5 = spikes_file is not None

        self._nwb_fname = _get_file_path(spikes_file_nwb)
        self._save_nwb = spikes_file_nwb is not None

        self._tmpdir = tmp_dir

        # self._spike_writer = SpikeTrainWriter(tmp_dir=tmp_dir)
        self._spike_writer = SpikeTrains(cache_dir=tmp_dir)

    def save(self, sim, cell, times, rates):
        try:
            spike_trains = np.array(f_rate_to_spike_train(times*1000.0, rates, np.random.randint(10000),
                                                          1000.*min(times), 1000.*max(times), 0.1))
        except:
            # convert to milliseconds and hence the multiplication by 1000
            spike_trains = 1000.0*np.array(pg.generate_inhomogenous_poisson(times, rates,
                                                                            seed=np.random.randint(10000)))

        # self._spike_writer.add_spikes(times=spike_trains, gid=gid)
        self._spike_writer.add_spikes(node_ids=cell.gid, timestamps=spike_trains, population=cell.population)


    def finalize(self, sim):
        self._spike_writer.flush()

        if self._save_csv:
            self._spike_writer.to_csv(self._csv_fname)

        if self._save_h5:
            self._spike_writer.to_sonata(self._h5_fname)

        if self._save_nwb:
            self._spike_writer.to_nwb(self._nwb_fname)

        self._spike_writer.close()


def f_rate_to_spike_train(t, f_rate, random_seed, t_window_start, t_window_end, p_spike_max):
    # t and f_rate are lists containing time stamps and corresponding firing rate values;
    # they are assumed to be of the same length and ordered with the time strictly increasing;
    # p_spike_max is the maximal probability of spiking that we allow within the time bin; it is used to decide on the size of the time bin; should be less than 1!

    #if np.max(f_rate) * np.max(np.diff(t))/1000. > 0.1:   #Divide by 1000 to convert to seconds
    #    print('Firing rate to high for time interval and will not estimate spike correctly. Spikes will ' \
    #        'be calculated with the slower inhomogenous poisson generating fucntion')
    #    raise Exception()

    spike_times = []

    # Use seed(...) to instantiate the random number generator.  Otherwise, current system time is used.
    random.seed(random_seed)

    # Assume here for each pair (t[k], f_rate[k]) that the f_rate[k] value applies to the time interval [t[k], t[k+1]).
    for k in six.moves.range(0, len(f_rate)-1):
        t_k = t[k]
        t_k_1 = t[k+1]
        if ((t_k >= t_window_start) and (t_k_1 <= t_window_end)):
            delta_t = t_k_1 - t_k
            # Average number of spikes expected in this interval (note that firing rate is in Hz and time is in ms).
            av_N_spikes = f_rate[k] / 1000.0 * delta_t

            if (av_N_spikes > 0):
                if (av_N_spikes <= p_spike_max):
                    N_bins = 1
                else:
                    N_bins = int(np.ceil(av_N_spikes / p_spike_max))

            t_base = t[k]
            t_bin = 1.0 * delta_t / N_bins
            p_spike_bin = 1.0 * av_N_spikes / N_bins
            for i_bin in six.moves.range(0, N_bins):
                rand_tmp = random()
                if rand_tmp < p_spike_bin:
                    spike_t = t_base + random() * t_bin
                    spike_times.append(spike_t)

                t_base += t_bin

    return spike_times