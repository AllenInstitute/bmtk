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
import os
import glob
import csv
import pandas as pd

from bmtk.utils.reports.spike_trains import SpikeTrains, sort_order, sort_order_lu
from bmtk.simulator.pointnet.io_tools import io
from bmtk.simulator.pointnet.nest_utils import nest_version

import nest


try:
    MPI_RANK = nest.Rank()
    N_HOSTS = nest.NumProcesses()

except Exception as e:
    MPI_RANK = 0
    N_HOSTS = 1


def create_spike_detector_nest2(label):
    return nest.Create(
        "spike_detector", 1,
        {
            'label': label,
            'withtime': True,
            'withgid': True,
            'to_file': True
        }
    )


def create_spike_detector_nest3(label):
    return nest.Create(
        "spike_recorder", 1,
        {
            'label': label,
            'record_to': 'ascii'
        }
    )


def read_spikes_file_nest2(spike_trains_writer, gid_map, label):
    for gdf_path in glob.glob(label + '*.gdf'):
        with open(gdf_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            for r in csv_reader:
                p = gid_map.get_pool_id(int(r[0]))
                spike_trains_writer.add_spike(node_id=p.node_id, timestamp=float(r[1]), population=p.population)


def read_spikes_file_nest3(spike_trains_writer, gid_map, label):
    for dat_path in glob.glob(label + '*.dat'):
        report_df = pd.read_csv(dat_path, index_col=False, sep='\t', comment='#')
        for nest_id, spikes_grp in report_df.groupby('sender'):
            gid = gid_map.get_pool_id(nest_id)
            timestamps = spikes_grp['time_ms'].values
            spike_trains_writer.add_spikes(node_ids=gid.node_id, timestamps=timestamps, population=gid.population)


if nest_version[0] >= 3:
    create_spike_detector = create_spike_detector_nest3
    read_spikes_file = read_spikes_file_nest3
    NEST_spikes_file_format = 'dat'
else:
    create_spike_detector = create_spike_detector_nest2
    read_spikes_file = read_spikes_file_nest2
    NEST_spikes_file_format = 'gdf'


class SpikesMod(object):
    """Module use for saving spikes

    """

    def __init__(self, tmp_dir, spikes_file_csv=None, spikes_file=None, spikes_file_nwb=None, spikes_sort_order=None,
                 cache_to_disk=True):
        def _get_path(file_name):
            # Unless file-name is an absolute path then it should be placed in the $OUTPUT_DIR
            if file_name is None:
                return None

            if os.path.isabs(file_name):
                return file_name
            else:
                abs_tmp = os.path.abspath(tmp_dir)
                abs_fname = os.path.abspath(file_name)
                if not abs_fname.startswith(abs_tmp):
                    return os.path.join(tmp_dir, file_name)
                else:
                    return file_name

        self._csv_fname = _get_path(spikes_file_csv)
        self._h5_fname = _get_path(spikes_file)
        self._nwb_fname = _get_path(spikes_file_nwb)

        self._tmp_dir = tmp_dir
        self._tmp_file_base = 'tmp_spike_times'
        self._spike_labels = os.path.join(self._tmp_dir, self._tmp_file_base)

        self._spike_writer = SpikeTrains(cache_dir=tmp_dir, cache_to_disk=cache_to_disk)
        self._spike_writer.delimiter = '\t'
        self._spike_writer.gid_col = 0
        self._spike_writer.time_col = 1
        self._sort_order = sort_order.none if not spikes_sort_order else sort_order_lu[spikes_sort_order]

        self._spike_detector = None

    def initialize(self, sim):
        self._spike_detector = create_spike_detector(self._spike_labels)
        nest.Connect(sim.net.gid_map.gids, self._spike_detector)

    def finalize(self, sim):
        # convert NEST gdf files into SONATA spikes/ format
        # TODO: Create a gdf_adaptor in bmtk/utils/reports/spike_trains to improve conversion speed.
        if MPI_RANK == 0:
            gid_map = sim.net.gid_map
            read_spikes_file(spike_trains_writer=self._spike_writer, gid_map=gid_map, label=self._spike_labels)
        io.barrier()

        if self._csv_fname is not None:
            self._spike_writer.to_csv(self._csv_fname, sort_order=self._sort_order)
            # io.barrier()

        if self._h5_fname is not None:
            # TODO: reimplement with pandas
            self._spike_writer.to_sonata(self._h5_fname, sort_order=self._sort_order)
            # io.barrier()

        if self._nwb_fname is not None:
            self._spike_writer.to_nwb(self._nwb_fname, sort_order=self._sort_order)
            # io.barrier()

        self._spike_writer.close()
        self._clean_files()

    def _clean_files(self):
        if MPI_RANK == 0:
            for nest_file in glob.glob(self._spike_labels + '*.' + NEST_spikes_file_format):
                os.remove(nest_file)
