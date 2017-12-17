import pandas as pd
import numpy as np
import csv
import h5py

class Rates(object):
    def __iter__(self):
        return self

    def next(self):
        raise StopIteration


class NormalRates(Rates):
    def __init__(self, t_start, t_end, rate_mu, rate_sigma=5.0):
        self.t_start = t_start
        self.t_end = t_end
        self.period_mu = 1.0/float(rate_mu)
        self.period_sigma = 1.0/float(rate_mu + rate_sigma)

        self._current_t = t_start

    def next(self):
        self._current_t += abs(np.random.normal(self.period_mu, self.period_sigma))
        if self._current_t > self.t_end:
            self._current_t = self.t_start
            raise StopIteration
        else:
            return self._current_t


class SpikesGenerator(object):
    def __init__(self, nodes, t_min=0, t_max=1.0):
        self._t_min = t_min
        self._t_max = t_max

        if isinstance(nodes, basestring):
            nodes_h5 = h5py.File(nodes, 'r')
            nodes = list(nodes_h5['nodes']['node_gid'])

        self._nodes = {n: Rates() for n in nodes}

    def set_rate(self, firing_rate, gids=None, t_start=None, t_end=None):
        t_start = t_start or self._t_min
        assert(t_start >= self._t_min)

        t_end = t_end or self._t_max
        assert(t_end <= self._t_max)

        gids = gids or self._nodes.keys()
        for gid in gids:
            self._nodes[gid] = NormalRates(t_start, t_end, firing_rate)

    def save_csv(self, csv_file_name, in_ms=False):
        conv = 1000.0 if in_ms else 1.0

        with open(csv_file_name, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=' ')
            csv_writer.writerow(['gid', 'spike-times'])
            for gid, rate_gen in self._nodes.items():
                csv_writer.writerow([gid, ','.join(str(r*conv) for r in rate_gen)])

