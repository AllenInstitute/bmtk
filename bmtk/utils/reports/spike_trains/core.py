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
from enum import Enum

from bmtk.utils.io import bmtk_world_comm


MPI_rank = bmtk_world_comm.MPI_rank
MPI_size = bmtk_world_comm.MPI_size
comm_barrier = bmtk_world_comm.barrier
comm = bmtk_world_comm.comm


class SortOrder(Enum):
    none = 'none'
    by_id = 'by_id'
    by_time = 'by_time'
    unknown = 'unknown'


# convient method for converting different string values
sort_order_lu = {
    'by_time': SortOrder.by_time,
    'time': SortOrder.by_time,
    'by_id': SortOrder.by_id,
    'id': SortOrder.by_id,
    'node_id': SortOrder.by_id,
    'gid': SortOrder.by_id,
    'none': SortOrder.none,
    'na': SortOrder.none
}


# column names for dataframes, csv headers, h5 datasets
col_timestamps = 'timestamps'
col_node_ids = 'node_ids'
col_population = 'population'
csv_headers = [col_timestamps, col_population, col_node_ids]
pop_na = '<sonata:none>'


def find_conversion(units_old, units_new):
    if units_new is None or units_old is None:
        return 1.0

    if units_old == 's' and units_new == 'ms':
        return 1000.

    if units_old == 'ms' and units_new == 's':
        return 0.001

    return 1.0


def find_file_type(path):
    """Tries to find the input type (sonata/h5, NWB, CSV) from the file-name"""
    if path is None:
        return ''

    path = path.lower()
    if path.endswith('.hdf5') or path.endswith('.hdf') or path.endswith('h5') or path.endswith('.sonata'):
        return 'h5'

    elif path.endswith('.nwb'):
        return 'nwb'

    elif path.endswith('.csv'):
        return 'csv'
