# Copyright 2020. Allen Institute. All rights reserved
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from six import string_types

from .core import find_file_type, MPI_size
from .spike_train_readers import load_sonata_file, CSVSTReader, NWBSTReader
from .spike_train_buffer import STMemoryBuffer, STCSVBuffer, STMPIBuffer, STCSVMPIBufferV2
from bmtk.utils.sonata.utils import get_node_ids
from scipy.stats import gamma
import warnings

class SpikeTrains(object):
    """A class for creating and reading spike files.

    """
    def __init__(self, spikes_adaptor=None, **kwargs):
        # There are a number of strategies for reading and writing spike trains, depending on memory limitations, if
        #  MPI is being used, or if there if read-only from disk. I'm using a decorator/adaptor pattern and moving
        #  the actual functionality to the Buffered/ReadOnly classes that implement SpikeTrainsAPI.
        if spikes_adaptor is not None:
            self.adaptor = spikes_adaptor
        else:
            # TODO: Check that comm has gather, reduce, etc methods; if not can't use STMPIBuffer
            use_mpi = MPI_size > 1
            cache_to_disk = 'cache_dir' in kwargs and kwargs.get('cache_to_disk', True)

            if use_mpi and cache_to_disk:
                self.adaptor = STCSVMPIBufferV2(**kwargs)
            elif cache_to_disk:
                self.adaptor = STCSVBuffer(**kwargs)
            elif use_mpi:
                self.adaptor = STMPIBuffer(**kwargs)
            else:
                self.adaptor = STMemoryBuffer(**kwargs)

    @classmethod
    def from_csv(cls, path, **kwargs):
        return cls(spikes_adaptor=CSVSTReader(path, **kwargs))

    @classmethod
    def from_sonata(cls, path, **kwargs):
        sonata_adaptor = load_sonata_file(path, **kwargs)
        return cls(spikes_adaptor=sonata_adaptor)

    @classmethod
    def from_nwb(cls, path, **kwargs):
        return cls(spikes_adaptor=NWBSTReader(path, **kwargs))

    @classmethod
    def load(cls, path, file_type=None, **kwargs):
        file_type = file_type.lower() if file_type else find_file_type(path)
        if file_type == 'h5' or file_type == 'sonata':
            return cls.from_sonata(path, **kwargs)
        elif file_type == 'nwb':
            return cls.from_nwb(path, **kwargs)
        elif file_type == 'csv':
            return cls.from_csv(path, **kwargs)

    def __getattr__(self, item):
        return getattr(self.adaptor, item)

    def __setattr__(self, key, value):
        if key == 'adaptor':
            self.__dict__[key] = value
        else:
            self.adaptor.__dict__[key] = value

    def __len__(self):
        return self.adaptor.__len__()

    def __eq__(self, other):
        return self.adaptor.__eq__(other)

    def __lt__(self, other):
        return self.adaptor.__lt__(other)

    def __le__(self, other):
        return self.adaptor.__le__(other)

    def __gt__(self, other):
        return self.adaptor.__gt__(other)

    def __ge__(self, other):
        return self.adaptor.__ge__(other)

    def __ne__(self, other):
        return self.adaptor.__ne__(other)


class SpikeGenerator(SpikeTrains):
    def __init__(self, population=None, seed=None, output_units='ms', **kwargs):
        max_spikes_per_node = 10000000
        if population is not None and 'default_population' not in kwargs:
            kwargs['default_population']= population

        if seed:
            np.random.seed(seed)

        super(SpikeGenerator, self).__init__(units=output_units, **kwargs)
        # self.units = units
        if output_units.lower() in ['ms', 'millisecond', 'milliseconds']:
            self._units = 'ms'
            self.output_conversion = 1000.0
        elif output_units.lower() in ['s', 'second', 'seconds']:
            self._units = 's'
            self.output_conversion = 1.0
        else:
            raise AttributeError('Unknown output_units value {}'.format(output_units))

class PoissonSpikeGenerator(SpikeGenerator):
    """ A Class for generating spike-trains with a homogeneous and inhomogeneous Poisson distribution.

    Uses the methods describe in Dayan and Abbott, 2001.
    """
    def __init__(self, population=None, seed=None, output_units='ms', **kwargs):
        super(PoissonSpikeGenerator, self).__init__(population, seed, output_units, **kwargs)

    def add(self, node_ids, firing_rate, population=None, times=(0.0, 1.0), abs_ref=0, tau_ref=0):
        """
        :param firing_rate: Scalar stationary firing rate or array of values for inhomogeneous (Hz)
        :param times: Start and end time for spike train (s)
        :param abs_ref: Absolute refractory period (s)
        :param tau_ref: Relative refractory period time constant for exponential recovery (s)
        """
        if isinstance(node_ids, string_types):
            # if user passes in path to nodes.h5 file count number of nodes
            node_ids = get_node_ids(node_ids, population)
        if np.isscalar(node_ids):
            # In case user passes in single node_id
            node_ids = [node_ids]
        if np.isscalar(firing_rate):
            self._build_fixed_fr(node_ids, population, firing_rate, times, abs_ref, tau_ref)
        elif isinstance(firing_rate, (list, np.ndarray)):
            self._build_inhomogeneous_fr(node_ids, population, firing_rate, times, abs_ref, tau_ref)
        if tau_ref < 0:
            raise ValueError('Refractory period time constant (sec) cannot be negative.')
        if abs_ref < 0:
            raise ValueError('Absolute refractory period (sec) cannot be negative.')

    def time_range(self, population=None):
        df = self.to_dataframe(populations=population, with_population_col=False)
        timestamps = df['timestamps']
        return np.min(timestamps), np.max(timestamps)

    def _build_fixed_fr(self, node_ids, population, fr, times, abs_ref, tau_ref):
        if np.isscalar(times) and times > 0.0:
            tstart = 0.0
            tstop = times
        else:
            tstart = times[0]
            tstop = times[-1]
            if tstart >= tstop:
                raise ValueError('Invalid start and stop times.')
        if fr < 0:
            raise ValueError('Firing rates must not be negative.')

        #rs2 = np.random.RandomState(0)
        count = 0
        for node_id in node_ids:
            c_time = tstart
            while True:
                interval = -np.log(1.0 - np.random.uniform()) / fr
                preceding_time = c_time
                c_time += interval
                if c_time > tstop:
                    break
                if tau_ref != 0:
                    w = 1 - np.exp(-(c_time-preceding_time-abs_ref)/tau_ref)
                else:
                    w = 1  # To avoid divide by zero warning
                if abs_ref != 0:
                    w = w*(interval>abs_ref)
                #if (w == 1) or (rs2.uniform() < w):
                if (w == 1) or (np.random.uniform() < w):
                    self.add_spike(node_id=node_id, population=population, timestamp=c_time*self.output_conversion)
                    count = count+1
        if (abs_ref != 0) or (tau_ref != 0):
            fr_actual = count/(tstop-tstart)/len(node_ids)
            str = (f'When using refractory periods, the actual firing rate ({fr_actual} spk/s) may be less than '
                   f'the desired firing rate ({fr} spk/s), particularly for high rates, and saturates at 1/abs_ref. '
                   'See also GammaSpikeGenerator for more exact firing rates with refractory periods.')
            warnings.warn(str)

    def _build_inhomogeneous_fr(self, node_ids, population, fr, times, abs_ref, tau_ref):
        if np.min(fr) < 0:
            raise ValueError('Firing rates must not be negative')
        if len(fr) != len(times):
            raise ValueError('If using a time series for firing rate, times must be an array of equal length')

        max_fr = np.max(fr)

        times = times
        tstart = times[0]
        tstop = times[-1]

        for node_id in node_ids:
            c_time = tstart
            time_indx = 0
            while True:
                # Using the pruning method, see Dayan and Abbott Ch 2
                interval = -np.log(1.0 - np.random.uniform()) / max_fr
                preceding_time = c_time
                c_time += interval
                if c_time > tstop:
                    break
                if tau_ref != 0:
                    w = 1 - np.exp(-(c_time-preceding_time-abs_ref)/tau_ref)
                else:
                    w = 1  # To avoid divide by zero warning
                if abs_ref != 0:
                    w = w * (interval > abs_ref)
                # A spike occurs at t_i, find index j st times[j-1] < t_i < times[j], and interpolate the firing rates
                # using fr[j-1] and fr[j]
                while times[time_indx] <= c_time:
                    time_indx += 1

                fr_i = _interpolate_fr(c_time, times[time_indx-1], times[time_indx],
                                       fr[time_indx-1], fr[time_indx])

                if not fr_i/max_fr*w < np.random.uniform():
                    self.add_spike(node_id=node_id, population=population, timestamp=c_time*self.output_conversion)

        if (abs_ref != 0) or (tau_ref != 0):
            str = ('When using refractory periods, the actual firing rates may be less than '
                   'the desired firing rates, particularly for high rates, and saturates at 1/abs_ref.')
            warnings.warn(str)

class GammaSpikeGenerator(SpikeGenerator):
    """ A Class for generating spike-trains based on a gamma-distributed renewal process.
    """
    def __init__(self, population=None, seed=None, output_units='ms', **kwargs):
        super(GammaSpikeGenerator, self).__init__(population, seed, output_units, **kwargs)

    def add(self, node_ids, firing_rate, a, population=None, times=(0.0, 1.0)):
        """
        :param firing_rate: Stationary firing rate (Hz)
        :param a: Shape parameter (a>0). For a=1, this becomes a Poisson distribution.
        :param times: Start and end time for spike train (s)
        """
        if isinstance(node_ids, string_types):
            # if user passes in path to nodes.h5 file count number of nodes
            node_ids = get_node_ids(node_ids, population)
        if np.isscalar(node_ids):
            # In case user passes in single node_id
            node_ids = [node_ids]

        if np.isscalar(firing_rate):
            self._build_fixed_fr(node_ids, population, firing_rate, a, times)
        elif isinstance(firing_rate, (list, np.ndarray)):
            raise Exception('Firing rate must be stationary for GammaSpikeGenerator')

    def time_range(self, population=None):
        df = self.to_dataframe(populations=population, with_population_col=False)
        timestamps = df['timestamps']
        return np.min(timestamps), np.max(timestamps)

    def _build_fixed_fr(self, node_ids, population, fr, a, times):
        if np.isscalar(times) and times > 0.0:
            tstart = 0.0
            tstop = times
        else:
            tstart = times[0]
            tstop = times[-1]
            if tstart >= tstop:
                raise ValueError('Invalid start and stop times.')
        if fr < 0:
            raise Exception('Firing rates must not be negative.')
        if a < 0:
            raise ValueError('Shape parameter `a` cannot be negative.')

        for node_id in node_ids:
            c_time = tstart
            n_spikes_avg = (tstop-tstart)*fr
            intervals = gamma.rvs(a, loc=0, scale=1 / (a * fr), size=round(n_spikes_avg * 1.5))

            for i in range(len(intervals)):
                preceding_time = c_time
                c_time += intervals[i]
                if c_time > tstop:
                    break
                self.add_spike(node_id=node_id, population=population, timestamp=c_time*self.output_conversion)

def _interpolate_fr(t, t0, t1, fr0, fr1):
    # Used to interpolate the firing rate at time t from a discrete list of firing rates
    return fr0 + (fr1 - fr0)*(t - t0)/(t1 - t0)
