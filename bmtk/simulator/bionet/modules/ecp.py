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
import h5py
import math
import pandas as pd
from neuron import h
import numpy as np

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.utils.sonata.utils import add_hdf5_magic, add_hdf5_version


pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class EcpMod(SimulatorMod):
    def __init__(self, tmp_dir, file_name, electrode_positions, contributions_dir=None, cells=None, variable_name='v',
                 electrode_channels=None, cell_bounds=None):
        self._ecp_output = file_name if os.path.isabs(file_name) else os.path.join(tmp_dir, file_name)
        self._positions_file = electrode_positions
        self._tmp_outputdir = tmp_dir


        if contributions_dir is not None:
            self._save_individ_cells = True
            self._contributions_dir = contributions_dir if os.path.isabs(contributions_dir) else os.path.join(tmp_dir, contributions_dir)
        else:
            self._save_individ_cells = False
            self._contributions_dir = None

        if cell_bounds is None:
            self._cells = cells
        else:
            self._cells = list(np.arange(cell_bounds[0],cell_bounds[1]+1))
        self._rel = None
        self._fih1 = None
        self._rel_nsites = 0
        self._block_size = 0
        self._saved_gids = {}
        self._nsteps = 0

        self._tstep = 0  # accumlative time step
        # self._rel_time = 0  #
        self._block_step = 0  # time step within the given block of time
        self._tstep_start_block = 0
        self._data_block = None
        self._cell_var_files = {}

        self._tmp_ecp_file = self._get_tmp_fname(MPI_RANK)
        self._tmp_ecp_handle = None
        # self._tmp_ecp_dataset = None

        self._local_gids = []

    def _get_tmp_fname(self, rank):
        return os.path.join(self._tmp_outputdir, 'tmp_{}_ecp.h5'.format(MPI_RANK))

    def _create_ecp_file(self, sim):
        dt = sim.dt
        tstop = sim.tstop
        self._nsteps = int(round(tstop/dt))

        # create file to temporary store ecp data on each rank
        self._tmp_ecp_handle = h5py.File(self._tmp_ecp_file, 'a')
        self._tmp_ecp_handle.create_dataset('/ecp/data', (self._nsteps, self._rel_nsites), maxshape=(None, self._rel_nsites),
                                            dtype=np.float64,
                                            chunks=True)

        # only the primary node will need to save the final ecp
        if MPI_RANK == 0:
            with h5py.File(self._ecp_output, 'w') as f5:
                add_hdf5_magic(f5)
                add_hdf5_version(f5)
                f5.create_dataset('/ecp/data', (self._nsteps, self._rel_nsites), maxshape=(None, self._rel_nsites),
                                  dtype=np.float64,
                                  chunks=True)
                f5['/ecp/data'].attrs['units'] = 'mV'
                #f5.attrs['dt'] = dt
                #f5.attrs['tstart'] = 0.0
                #f5.attrs['tstop'] = tstop

                f5.create_dataset('/ecp/time', (3,), data=(0.0, sim.tstop, sim.dt))
                f5['/ecp/time'].attrs['units'] = 'ms'

                # Save channels. Current we record from all channels, may want to be more selective in the future.
                f5.create_dataset('/ecp/channel_id', data=np.arange(self._rel.nsites))

        pc.barrier()

    def _create_cell_file(self, gid):
        if not self._save_individ_cells:
            return

        file_name = os.path.join(self._contributions_dir, '{}.h5'.format(int(gid)))
        file_h5 = h5py.File(file_name, 'a')
        self._cell_var_files[gid] = file_h5
        file_h5.create_dataset('/ecp/data', (self._nsteps, self._rel_nsites), maxshape=(None, self._rel_nsites), chunks=True)
        # self._cell_var_files[gid] = file_h5['ecp']

    def _calculate_ecp(self, sim):
        self._rel = RecXElectrode(self._positions_file)
        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)
            #cell = sim.net.get_local_cell(gid)
            # cell = sim.net.cells[gid]
            self._rel.calc_transfer_resistance(gid, cell.get_seg_coords())

        self._rel_nsites = self._rel.nsites
        sim.h.cvode.use_fast_imem(1)  # make i_membrane_ a range variable

        def set_pointers():
            for gid, cell in sim.net.get_local_cells().items():
            #for gid, cell in sim.net.local_cells.items():
                # for gid, cell in sim.net.cells.items():
                cell.set_im_ptr()
        self._fih1 = sim.h.FInitializeHandler(0, set_pointers)

    def _save_block(self, interval):
        """Add """
        itstart, itend = interval
        self._tmp_ecp_handle['/ecp/data'][itstart:itend, :] += self._data_block[0:(itend - itstart), :]
        self._tmp_ecp_handle.flush()
        self._data_block[:] = 0.0

    def _save_ecp(self, sim):
        """Save ECP from each rank to disk into a single file"""
        block_size = sim.nsteps_block
        nblocks, remain = divmod(self._nsteps, block_size)
        ivals = [i*block_size for i in range(nblocks+1)]
        if remain != 0:
            ivals.append(self._nsteps)

        for rank in range(N_HOSTS):  # iterate over the ranks
            if rank == MPI_RANK:  # wait until finished with a particular rank
                with h5py.File(self._ecp_output, 'a') as ecp_f5:
                    for i in range(len(ivals)-1):
                        ecp_f5['/ecp/data'][ivals[i]:ivals[i+1], :] += self._tmp_ecp_handle['/ecp/data'][ivals[i]:ivals[i+1], :]

            pc.barrier()

    def _save_cell_vars(self, interval):
        itstart, itend = interval

        for gid, data in self._saved_gids.items():
            h5_file = self._cell_var_files[gid]
            h5_file['/ecp/data'][itstart:itend, :] = data[0:(itend-itstart), :]
            h5_file.flush()
            data[:] = 0.0

    def _delete_tmp_files(self):
        if os.path.exists(self._tmp_ecp_file):
            self._tmp_ecp_handle.close()
            os.remove(self._tmp_ecp_file)

    def initialize(self, sim):
        if self._contributions_dir and (not os.path.exists(self._contributions_dir)) and MPI_RANK == 0:
            os.makedirs(self._contributions_dir)
        pc.barrier()

        self._block_size = sim.nsteps_block

        # Get list of gids being recorded
        selected_gids = set(sim.net.get_node_set(self._cells).gids())
        self._local_gids = list(set(sim.biophysical_gids) & selected_gids)

        self._calculate_ecp(sim)
        self._create_ecp_file(sim)

        # ecp data
        self._data_block = np.zeros((self._block_size, self._rel_nsites))

        # create list of all cells whose ecp values will be saved separetly
        self._saved_gids = {gid: np.empty((self._block_size, self._rel_nsites))
                            for gid in self._local_gids}
        for gid in self._saved_gids.keys():
            self._create_cell_file(gid)

        pc.barrier()

    def step(self, sim, tstep):
        for gid in self._local_gids:  # compute ecp only from the biophysical cells
            cell = sim.net.get_cell_gid(gid)
            im = cell.get_im()
            tr = self._rel.get_transfer_resistance(gid)
            ecp = np.dot(tr, im)

            if gid in self._saved_gids.keys():
                # save individual contribution
                self._saved_gids[gid][self._block_step, :] = ecp

            # add to total ecp contribution
            self._data_block[self._block_step, :] += ecp

        self._block_step += 1

    def block(self, sim, block_interval):
        self._save_block(block_interval)
        if self._save_individ_cells:
            self._save_cell_vars(block_interval)

        self._block_step = 0
        self._tstep_start_block = self._tstep

    def finalize(self, sim):
        if self._block_step > 0:
            # just in case the simulation doesn't end on a block step
            self.block(sim, (sim.n_steps - self._block_step, sim.n_steps))

        self._save_ecp(sim)
        self._delete_tmp_files()
        pc.barrier()


class RecXElectrode(object):
    """Extracellular electrode

    """

    def __init__(self, positions):
        """Create an array"""
        # self.conf = conf
        electrode_file = positions  # self.conf["recXelectrode"]["positions"]

        # convert coordinates to ndarray, The first index is xyz and the second is the channel number
        el_df = pd.read_csv(electrode_file, sep=' ')
        self.pos = el_df[['x_pos', 'y_pos', 'z_pos']].T.values
        #self.pos = el_df.as_matrix(columns=['x_pos', 'y_pos', 'z_pos']).T
        self.nsites = self.pos.shape[1]
        # self.conf['run']['nsites'] = self.nsites  # add to the config
        self.transfer_resistances = {}  # V_e = transfer_resistance*Im

    def drift(self):
        # will include function to model electrode drift
        pass

    def get_transfer_resistance(self, gid):
        return self.transfer_resistances[gid]

    def calc_transfer_resistance(self, gid, seg_coords):
        """Precompute mapping from segment to electrode locations"""
        sigma = 0.3  # mS/mm

        r05 = (seg_coords['p0'] + seg_coords['p1']) / 2
        dl = seg_coords['p1'] - seg_coords['p0']

        nseg = r05.shape[1]

        tr = np.zeros((self.nsites, nseg))

        for j in range(self.nsites):  # calculate mapping for each site on the electrode
            rel = np.expand_dims(self.pos[:, j], axis=1)  # coordinates of a j-th site on the electrode
            rel_05 = rel - r05  # distance between electrode and segment centers

            # compute dot product column-wise, the resulting array has as many columns as original
            r2 = np.einsum('ij,ij->j', rel_05, rel_05)

            # compute dot product column-wise, the resulting array has as many columns as original
            rlldl = np.einsum('ij,ij->j', rel_05, dl)
            dlmag = np.linalg.norm(dl, axis=0)  # length of each segment
            rll = abs(rlldl / dlmag)  # component of r parallel to the segment axis it must be always positive
            rT2 = r2 - rll ** 2  # square of perpendicular component
            up = rll + dlmag / 2
            low = rll - dlmag / 2
            num = up + np.sqrt(up ** 2 + rT2)
            den = low + np.sqrt(low ** 2 + rT2)
            tr[j, :] = np.log(num / den) / dlmag  # units of (um) use with im_ (total seg current)

        tr *= 1 / (4 * math.pi * sigma)
        self.transfer_resistances[gid] = tr
