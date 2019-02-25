# Copyright 2017. Allen Institute. All rights reserved
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
import pandas as pd
import numpy as np
import csv
import h5py
from six import string_types

from bmtk.utils import sonata

class Rates(object):
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration
    
    next = __next__ # For Python 2

class NormalRates(Rates):
    def __init__(self, t_start, t_end, rate_mu, rate_sigma=5.0):
        self.t_start = t_start
        self.t_end = t_end
        self.period_mu = 1.0/float(rate_mu)
        self.period_sigma = 1.0/float(rate_mu + rate_sigma)

        self._current_t = t_start

    def __next__(self):
        self._current_t += abs(np.random.normal(self.period_mu, self.period_sigma))
        if self._current_t > self.t_end:
            self._current_t = self.t_start
            raise StopIteration
        else:
            return self._current_t

    next = __next__ # For Python 2


class SpikesGenerator(object):
    def __init__(self, nodes, populations=None, t_min=0, t_max=1.0):
        self._t_min = t_min
        self._t_max = t_max

        if isinstance(nodes, string_types):
            nodes_h5 = h5py.File(nodes, 'r')
            nodes_grp = nodes_h5['/nodes']
            if populations is None:
                populations = nodes_grp.keys()

            # TODO: Need a way to Use sonata library without having to use node-types
            nodes = []
            for node_pop in populations:
                nodes.extend(nodes_grp[node_pop]['node_id'])

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

