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
import csv
import h5py
import numpy as np
from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.io import print2log0

from neuron import h
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    pass


pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class SpikesMod(SimulatorMod):
    """Module use for saving spikes

    """

    def __init__(self, tmpdir, csv_filename=None, h5_filename=None):
        self._csv_fname = csv_filename
        self._csv_fhandle = None
        self._csv_writer = None
        self._save_csv = csv_filename is not None

        self._h5_fname = h5_filename
        self._h5_fhandle = None
        self._gids_dataset = None
        self._times_dataset = None
        self._save_h5 = h5_filename is not None

        self._tmpdir = tmpdir
        self._tmp_file = self._get_tmp_filename(MPI_RANK)  # os.path.join(tmpdir, 'tmp_{}_spikes.csv'.format(MPI_RANK))
        self._tmp_spikes_handles = {}
        self._n_spikes_rank = 0
        self._n_spikes_total = 0
        self._row_itr = 0

    def _get_tmp_filename(self, rank):
        return os.path.join(self._tmpdir, 'tmp_{}_spikes.csv'.format(rank))

    def _save_unsorted(self):
        if MPI_RANK == 0:
            for rank in range(N_HOSTS):
                for row in self._tmp_spikes_handles[str(rank)]:
                    self._write_spike(float(row[0]), int(row[1]))

        pc.barrier()

    def _save_sorted(self):
        if N_HOSTS < 2:
            # If there's one rank then the tmp file should already be sorted
            self._save_unsorted()

        else:
            if MPI_RANK == 0:
                # TODO: remove the conditional of running only on rank 0.
                try:
                    # create a list of the first spike time for every rank
                    spikes = []
                    for rank in range(N_HOSTS):
                        spike = self.next_spike(rank)
                        if spike is not None:
                            spikes.append(spike)

                    # Iterate through all the ranks and find the first spike. Write that spike/gid to the output, then
                    # replace that data point with the next spike on the selected rank
                    while spikes:
                        # find which rank has the first spike
                        selected_index = 0
                        selected_time = spikes[0][0]
                        for i, spike in enumerate(spikes[1:]):
                            if spike[0] < selected_time:
                                selected_index = i+1
                                selected_time = spike[0]

                        # write the spike to the file
                        t, gid, rank = spikes.pop(selected_index)
                        # print2log0('{} {}'.format(t, gid))
                        self._write_spike(t, gid)

                        # get the next spike on that rank and replace in spikes table
                        another_spike = self.next_spike(rank)
                        if another_spike is not None:
                            spikes.append(another_spike)

                except Exception as e:
                    print e

            pc.barrier()

    def _write_spike(self, spike_time, gid):
        if MPI_RANK != 0:
            return

        if self._save_csv:
            self._csv_writer.writerow([spike_time, gid])

        if self._save_h5:
            self._gids_dataset[self._row_itr] = gid
            self._times_dataset[self._row_itr] = spike_time

        self._row_itr += 1

    def _close_output(self):
        if MPI_RANK == 0:
            if self._save_csv:
                self._csv_handle.close()

    def _gather_spike_count(self):
        if N_HOSTS == 1:
            return self._n_spikes_rank
        else:
            send_msg = np.array([self._n_spikes_rank], dtype=np.uint)
            recv_buff = np.empty(N_HOSTS, dtype=np.uint)
            # TODO: This is unecessary, just keep a universal list
            comm.Allgather([send_msg, MPI.UNSIGNED_INT], [recv_buff, MPI.UNSIGNED_INT])
            return np.sum(recv_buff)

    def _open_output(self):
        """Creates and opens the files that will be used to display the spiking information"""
        if MPI_RANK == 0 and self._save_csv:
            self._csv_handle = open(self._csv_fname, 'w')
            self._csv_writer = csv.writer(self._csv_handle, delimiter=' ')

        self._n_spikes_total = self._gather_spike_count()
        if MPI_RANK == 0 and self._save_h5:
            self._h5_fhandle = h5py.File(self._h5_fname, 'w')
            self._gids_dataset = self._h5_fhandle.create_dataset("gid", shape=(self._n_spikes_total,), chunks=True,
                                                                 dtype=np.int32)
            self._times_dataset = self._h5_fhandle.create_dataset("time", shape=(self._n_spikes_total,), chunks=True)

    def _delete_tmp_file(self):
        """Removes tempory file used to save spikes during the block"""
        if os.path.exists(self._tmp_file):
            os.remove(self._tmp_file)

    def next_spike(self, rank):
        """Acts as an iterator of the spikes for a given mpi rank.

        :param rank:
        :return: tuple of
        """
        try:
            val = self._tmp_spikes_handles[str(rank)].next()
            return float(val[0]), int(val[1]), rank
        except StopIteration:
            return None

    def initialize(self, sim):
        # TODO: since it's possible that other modules may need to access spikes, set_spikes_recordings() should
        # probably be called in the simulator itself.
        sim.set_spikes_recording()

    def block(self, sim, block_interval):
        # take spikes from Simulator spikes vector and save to the tmp file
        tspikes_handle = open(self._tmp_file, 'a')
        for gid, tVec in sim.spikes_table.items():
            for t in tVec:
                tspikes_handle.write('{:.3f} {}\n'.format(t, gid))
                self._n_spikes_rank += 1

        pc.barrier()  # wait until all ranks have been saved
        sim.set_spikes_recording()  # reset recording vector

    def finalize(self, sim):
        pc.barrier()
        self._tmp_spikes_handles = {str(r): csv.reader(open(self._get_tmp_filename(r), 'r'), delimiter=' ')
                                    for r in range(N_HOSTS)}
        self._open_output()  # create csv and/or h5 files to write to
        self._save_sorted()  # save
        self._close_output()  # flush and close the output files
        pc.barrier()

        self._delete_tmp_file()  # remove the temporary files used to save spike data on each block
