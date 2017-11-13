# Copyright 2017. Allen Institute. All rights reserved.
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
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

from dm_network import DenseNetwork
from mpi4py import MPI
from heapq import heappush, heappop

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


class MPINetwork(DenseNetwork):
    def __init__(self, name, **network_props):
        super(MPINetwork, self).__init__(name, **network_props or {})
        self._edge_assignment = None

    def _add_edges(self, connection_map, i):
        if self._assign_to_rank(0):
            super(MPINetwork, self)._add_edges(connection_map, i)

    def _assign_to_rank(self, i):
        if self._edge_assignment is None:
            self._build_rank_assignments()

        return rank == self._edge_assignment[i]

    def _build_rank_assignments(self):
        """Builds the _edge_assignment array.

        Division of connections is decided by the maximum possible edges (i.e. number of source and target nodes). In
        the end assignment should balance the connection matrix sizes need by each rank.
        """
        rank_heap = []  # A heap of tuples (weight, rank #)
        for a in range(nprocs):
            heappush(rank_heap, (0, a))

        # find the rank with the lowest weight, assign that rank to build the i'th connection matrix, update the rank's
        # weight and re-add to the heap.
        # TODO: sort connection_maps in descending order to get better balance
        self._edge_assignment = []
        for cm in self.get_connections():
            r = heappop(rank_heap)
            self._edge_assignment.append(r[1])
            heappush(rank_heap, (r[0] + cm.max_connections(), r[1]))

