import os
import h5py
from neuron import h
import numpy as np

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.recxelectrode import RecXElectrode


pc = h.ParallelContext()
MPI_RANK = int(pc.id())
N_HOSTS = int(pc.nhost())


class EcpMod(SimulatorMod):
    def __init__(self, ecp_file, positions_file):
        self._ecp_output = ecp_file
        self._positions_file = positions_file
        self._cell_vars_dir = None
        self._rel = None
        self._fih1 = None
        self._rel_nsites = 0
        self._block_size = 0
        self._biophys_gids = []
        self._saved_gids = []
        self._nsteps = 0

        self._tstep = 0  # accumlative time step
        self._rel_time = 0  #
        self._block_step = 0  # time step within the given block of time
        self._tstep_start_block = 0
        self._data_block = None
        self._cell_var_files = {}

    def _create_ecp_file(self, sim):
        dt = sim.dt
        tstop = sim.tstop
        self._nsteps = int(round(tstop/dt))

        if MPI_RANK == 0:
            with h5py.File(self._ecp_output, 'w') as f5:
                f5.create_dataset('ecp', (self._nsteps, self._rel_nsites), maxshape=(None, self._rel_nsites),
                                  chunks=True)
                f5.attrs['dt'] = dt
                f5.attrs['tstart'] = 0.0
                f5.attrs['tstop'] = tstop
        pc.barrier()

    def _create_cell_file(self, gid):
        file_name = os.path.join(self._cell_vars_dir, '{}.h5'.format(int(gid)))
        file_h5 = h5py.File(file_name, 'a')
        self._cell_var_files[gid] = file_h5
        file_h5.create_dataset('ecp', (self._nsteps, self._rel_nsites), maxshape=(None, self._rel_nsites), chunks=True)
        # self._cell_var_files[gid] = file_h5['ecp']

    def _calculate_ecp(self, sim):
        self._rel = RecXElectrode(self._positions_file)
        for gid in self._biophys_gids:
            cell = sim.net.cells[gid]
            self._rel.calc_transfer_resistance(gid, cell.get_seg_coords())

        self._rel_nsites = self._rel.nsites
        sim.h.cvode.use_fast_imem(1)  # make i_membrane_ a range variable

        def set_pointers():
            for gid, cell in sim.net.cells.items():
                cell.set_im_ptr()
        self._fih1 = sim.h.FInitializeHandler(0, set_pointers)

    def _save_ecp(self, interval):
        """Save ECP from each rank to disk into a single file"""
        itstart, itend = interval
        for rank in xrange(N_HOSTS):  # iterate over the ranks
            if rank == MPI_RANK:  # wait until finished with a particular rank
                # TODO: Keep handle open by saving to variable
                with h5py.File(self._ecp_output, 'a') as f5:
                    f5["ecp"][itstart:itend, :] += self._data_block[0:itend - itstart, :]
                    f5.attrs["tsave"] = self._rel_time  # update tsave
                    f5.flush()

                    # TODO: it shouldn't be required to clear data_block every time
                    self._data_block[:] = 0.0

            pc.barrier()  # move on to next rank

    def _save_cell_vars(self, interval):
        itstart, itend = interval
        # for gid in self._cell_var_files[gid] = file_h5

        for gid, data in self._saved_gids.items():
            h5_file = self._cell_var_files[gid]
            h5_file['ecp'][itstart:itend, :] = data[0:(itend-itstart), :]
            h5_file.flush()
            data[:] = 0.0

    def initialize(self, sim):
        self._block_size = sim.nsteps_block
        self._biophys_gids = sim.gids['biophysical']  # gids for biophysical cells on this rank
        self._cell_vars_dir = sim.cell_var_output

        self._calculate_ecp(sim)
        self._create_ecp_file(sim)

        # ecp data
        self._data_block = np.zeros((self._block_size, self._rel_nsites))

        # create list of all cells whose ecp values will be saved separetly
        self._saved_gids = {gid: np.empty((self._block_size, self._rel_nsites))
                            for gid in self._biophys_gids if gid in sim.gids['save_cell_vars']}
        for gid in self._saved_gids:
            self._create_cell_file(gid)

        pc.barrier()

    def step(self, sim, tstep, rel_time=0):
        for gid in self._biophys_gids:  # compute ecp only from the biophysical cells
            cell = sim.net.cells[gid]
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
        self._save_ecp(block_interval)
        self._save_cell_vars(block_interval)

        self._block_step = 0
        self._tstep_start_block = self._tstep

    def finalize(self, sim):
        if self._block_step > 0:
            # just in case the simulation doesn't end on a block step
            self.block(sim, (sim.n_stepsself._block_step-1, sim.n_steps))