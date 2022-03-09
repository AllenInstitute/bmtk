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
import warnings

from .core import SortOrder
from .spikes_file_writers import write_csv, write_sonata


class SpikeTrainsAPI(object):
    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        """Add a single spike

        :param node_id: integer, id of node/cell that spike belongs too.
        :param timestamp: double, time that spike occurred.
        :param population: string, name of population belong to spike. If none will try to use the default population.
        :param kwargs: optional arguments.
        """
        raise NotImplementedError()

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        """Add a sequence of spikes.

        :param node_ids: list of ints or int. If a list is used, it should be the same lenght as the corresponding
            timestamps. If a singluar integer value is used then assumes all timestamps corresponds with said node_id.
        :param timestamps: A list of doubles
        :param population: The population to which the node(s) belong too
        :param kwargs: optional arguments.
        """
        raise NotImplementedError()

    def import_spikes(self, obj, **kwargs):
        """Import spikes from another spike-trains or other object. Highly dependent on stragey"""
        warnings.warn('method import_spikes has not been implemented for this type of strategy.')
        pass

    def flush(self):
        """Used by some underlying strategies to finish saving spikes."""
        pass

    def close(self):
        pass

    @property
    def populations(self):
        """Get all available spike population names

        :return: A list of strings
        """
        raise NotImplementedError()

    def units(self, population=None):
        """Returns the units used in the timestamps.

        :return: str
        """

        # for backwards comptability assume everything is in milliseconds unless overwritten
        # TODO: Use an enum/struct to pre-define the avilable units.
        return 'ms'

    def set_units(self, u, population=None):
        """Set the units associated with a population timestamps (ms, seconds)"""
        raise NotImplementedError()

    def sort_order(self, population=None):
        return SortOrder.unknown

    def node_ids(self, population=None):
        """ Returns a list of (node-ids, population_name).

        :param population: Name of population, if not set uses the default_population
        :return: A list of node-ids (integers).
        """
        raise NotImplementedError()

    def n_spikes(self, population=None):
        """Get the number of spikes for the given population.

        :param population: population name. If none None will use the default population (when possible).
        :return: unsigned integer, number of spikes.
        """
        raise NotImplementedError()

    def get_times(self, node_id, population=None, time_window=None, **kwargs):
        """Returns a list of spike-times for a given node.

        :param node_id: The id of the node
        :param population: Name of the node-population which the node belongs to. By default will try to use the
            default population (if possible).
        :param time_window: A tuple (min-time, max-time) to limit the returned spikes. By default returns all spikes.
        :param kwargs: optional arguments.
        :return: list of spike times [float]
        """
        raise NotImplementedError()

    def to_dataframe(self, populations=None, sort_order=SortOrder.none, with_population_col=True, **kwargs):
        """Returns a pandas dataframe of the node_ids, populations, and timestamps of the given spikes

        :param populations: string or list of strings, used to only return the dataframes associated with a given
            node population. By default (populations=None) all populations are included
        :param sort_order: 'by_time', 'by_id', 'none' or None. Returns the dataframe sorted within their population.
            By default will not sort and return spikes as they are saved
        :param with_population_col: bool, set to False to not return the 'population' column (useful for really large
            dataframs with only one population). True by default
        :param kwargs:
        :return: A pandas dataframe, unindex, with columns 'node_ids', 'timestamps', and 'population' (optional)
        """
        raise NotImplementedError()

    def spikes(self, populations=None, time_window=None, sort_order=SortOrder.none, **kwargs):
        """Iterate over all the saved spikes, returning a single spike at a time. Will typically be slower than calling
        to_dataframe(), but not require as much memory. To use the generator::

            for node_id, population, timestamp in spike_trains.spikes():
                ...

        :param populations: string or list of strings, used to select specific node_populations. By default all
            populations with spikes data is iterated over
        :param time_window:
        :param sort_order:
        :param kwargs:
        :return:
        """
        raise NotImplementedError()

    def to_sonata(self, path, mode='w', sort_order=SortOrder.none, **kwargs):
        """Write current spike-trains to a sonata hdf5 file

        :param path:
        :param mode:
        :param sort_order:
        :param kwargs:
        :return:
        """
        write_sonata(path=path, spiketrain_reader=self, mode=mode, sort_order=sort_order, **kwargs)

    def to_csv(self, path, mode='w', sort_order=SortOrder.none, **kwargs):
        """Write spikes to csv file

        :param path:
        :param mode:
        :param sort_order:
        :param kwargs:
        :return:
        """
        write_csv(path=path, spiketrain_reader=self, mode=mode, sort_orders=sort_order, **kwargs)

    def to_nwb(self, path, mode='w', **kwargs):
        raise NotImplemented()

    def merge(self, other):
        """Import Another SpikesTrain object into current file, always in-place.

        :param other: Another SpikeTrainsAPI object
        """
        raise NotImplementedError()

    def is_equal(self, other, populations=None, err=0.00001, time_window=None):
        """Compares two SpikeTrains instances to see if they have the same spikes (exlcuding order or their method of
        storage). Use this method instead of == when one of the spike-train instances has extra populations or
        timestamps are stored at a different precision.

        :param other: spike-trains instance being compared
        :param populations: string or list of strings, populations to compare between the two. By default
            (populations=None) will return True only if the two files have the same populations.
        :param err: precision on which two timestamps are compared.
        :param time_window:
        :return: True if the two spike-trains have the same node-ids/timestamps (given the conditions).
        """
        if populations is None:
            # Both must contain the same populations
            populations = self.populations
            if set(other.populations) != set(populations):
                return False
        else:
            # Comparing only a subset of the node populations, make sure both files contains them (or both files don't
            # contain the populations
            populations = [populations] if np.isscalar(populations) else populations
            for p in populations:
                if (p in self.populations) != (p in other.populations):
                    return False

        for p in populations:
            if time_window is None:
                # check that each SpikeTrains contain the same number and ids of nodes so we don't have to iterate
                # through each spike. This won't always work if the user limits the time-window.
                self_nodes = sorted([n for n in self.node_ids(population=p)])
                other_nodes = sorted([n for n in other.node_ids(population=p)])
                if not np.all(self_nodes == other_nodes):
                    return False
            else:
                # If the time-window being checked is restricted
                self_nodes = set([n for n in self.node_ids(p)]) & set([n for n in other.node_ids(p)])

            for node_id in self_nodes:
                # Make sure it's sorted as get_times doesn't guarentee order
                self_ts = np.sort(self.get_times(node_id=node_id, population=p, time_window=time_window))
                other_ts = np.sort(other.get_times(node_id=node_id, population=p, time_window=time_window))
                if len(self_ts) != len(other_ts):
                    return False

                if not np.allclose(self_ts, other_ts, equal_nan=True, atol=err):
                    return False

        return True

    def is_subset(self, other, err=0.00001, strict=False):
        """Checks to see if this given set of spike-trains is a subset of another, which means that every
        (population, node_id, timestamp) that exists in self also exists in other.

        WARNING: It may be possible, possible due to precision, that a node has two spikes at the same time. Right
        now this isn't accounted for, and if self's node 0 has two spikes at 100.00 ms it except other.node[0] has
        two spikes at 100.00 ms as well.
        # TODO: Account for non-uniqueness in on the timestamps

        :param other:
        :param err: precision for comparing two timestamps
        :param strict: bool, if True makes sure that self is a strict subset of other. default False
        :return:
        """
        is_equals = set(self.populations) == set(other.populations)
        for pop in self.populations:
            self_n_spikes = self.n_spikes(population=pop)
            other_n_spikes = other.n_spikes(population=pop)
            is_equals &= other_n_spikes == self_n_spikes

            if self_n_spikes == 0:
                # Sometimes a spikes-file will have a nodes population with no actual spikes, in which case we want
                # to ignore
                continue

            if pop not in other.populations:
                # check that population exists in other
                return False

            if self_n_spikes > other_n_spikes:
                return False

            s_node_ids = set(self.node_ids(population=pop))
            if s_node_ids > set(other.node_ids(population=pop)):
                return False

            for node_id in s_node_ids:
                other_ts_sorted = np.sort(other.get_times(node_id=node_id, population=pop))
                self_ts = self.get_times(node_id=node_id, population=pop)
                indxs = np.searchsorted(other_ts_sorted, self_ts, side='left')

                if np.any(indxs < 0):
                    return False

                if not np.allclose(self_ts, other_ts_sorted[indxs], atol=err):
                    return False

        if strict and is_equals:
            return False

        return True

    def __len__(self):
        total_spikes = 0
        for p in self.populations:
            total_spikes += self.n_spikes(population=p)
        return total_spikes

    def __eq__(self, other):
        return self.is_equal(other=other)

    def __lt__(self, other):
        return self.is_subset(other, strict=True)

    def __le__(self, other):
        return self.is_subset(other)

    def __gt__(self, other):
        return other < self

    def __ge__(self, other):
        return other <= self

    def __ne__(self, other):
        return not self == other  # Consider implementing directly to take advantange of short-circuit evaluation


class SpikeTrainsReadOnlyAPI(SpikeTrainsAPI):
    warning_msg = 'read-only SpikeTrains, trying to add or import spikes will be ignored.'

    def add_spike(self, node_id, timestamp, population=None, **kwargs):
        warnings.warn(SpikeTrainsReadOnlyAPI.warning_msg)
        pass

    def add_spikes(self, node_ids, timestamps, population=None, **kwargs):
        warnings.warn(SpikeTrainsReadOnlyAPI.warning_msg)
        pass

    def import_spikes(self, obj, **kwargs):
        warnings.warn(SpikeTrainsReadOnlyAPI.warning_msg)
        pass

    def flush(self):
        pass

    def close(self):
        pass
