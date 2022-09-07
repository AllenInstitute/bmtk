import os
import math
import pandas as pd
import numpy as np
import six
from neuron import h

from bmtk.simulator.bionet.modules.sim_module import SimulatorMod
from bmtk.simulator.bionet.modules.xstim_waveforms import stimx_waveform_factory
from bmtk.simulator.bionet.utils import rotation_matrix
from bmtk.simulator.bionet.io_tools import io


class XStimMod(SimulatorMod):
    def __init__(self, positions_file, waveform, mesh_files_dir=None, cells=None, set_nrn_mechanisms=True,
                 node_set=None):
        self._positions_file = positions_file
        self._mesh_files_dir = mesh_files_dir if mesh_files_dir is not None \
            else os.path.dirname(os.path.realpath(self._positions_file))

        self._waveform = waveform  # TODO: Check if waveform is a file or dict and load it appropiately

        self._set_nrn_mechanisms = set_nrn_mechanisms
        self._electrode = None
        self._cells = cells
        self._local_gids = []
        self._fih = None

    # def __set_extracellular_mechanism(self):
    #     for gid in self._local_gids:

    def initialize(self, sim):
        if self._cells is None:
            # if specific gids not listed just get all biophysically detailed cells on this rank
            self._local_gids = sim.biophysical_gids
        else:
            # get subset of selected gids only on this rank
            self._local_gids = list(set(sim.local_gids) & set(self._all_gids))

        self._electrode = StimXElectrode(self._positions_file, self._waveform, self._mesh_files_dir, sim.dt)
        for gid in self._local_gids:
            # cell = sim.net.get_local_cell(gid)
            cell = sim.net.get_cell_gid(gid)
            cell.setup_xstim(self._set_nrn_mechanisms)
            self._electrode.set_transfer_resistance(gid, cell.seg_coords)

        def set_pointers():
            for gid in self._local_gids:
                cell = sim.net.get_cell_gid(gid)
                # cell = sim.net.get_local_cell(gid)
                cell.set_ptr2e_extracellular()

        self._fih = sim.h.FInitializeHandler(0, set_pointers)

    def step(self, sim, tstep):
        for gid in self._local_gids:
            cell = sim.net.get_cell_gid(gid)
            # Use tstep +1 to match isee-engine existing results. This will make it so that it begins a step earlier
            # than if using just tstep.
            self._electrode.calculate_waveforms(tstep+1)
            vext_vec = self._electrode.get_vext(gid)
            cell.set_e_extracellular(vext_vec)


class StimXElectrode(object):
    """
    Extracellular Stimulating electrode
    """
    def __init__(self, positions_file, waveform, mesh_files_dir, dt):
        self._dt = dt
        self._mesh_files_dir = mesh_files_dir

        stimelectrode_position_df = pd.read_csv(positions_file, sep=' ')

        if 'electrode_mesh_file' in stimelectrode_position_df.columns:
            self.elmesh_files = stimelectrode_position_df['electrode_mesh_file']
        else:
            self.elmesh_files = None
        self.elpos = stimelectrode_position_df[['pos_x', 'pos_y', 'pos_z']].T.values
        self.elrot = stimelectrode_position_df[['rotation_x', 'rotation_y', 'rotation_z']].values
        self.elnsites = self.elpos.shape[1]  # Number of electrodes in electrode file
        self.waveform = stimx_waveform_factory(waveform)

        self.trans_X = {}  # mapping segment coordinates
        self.waveform_amplitude = []
        self.el_mesh = {}
        self.el_mesh_size = []

        self.read_electrode_mesh()
        self.rotate_the_electrodes()
        self.place_the_electrodes()

    def read_electrode_mesh(self):
        if self.elmesh_files is None:
            # if electrode_mesh_file is missing then we still treat the electrode as a mesh on a grid of size 1 with
            # single point at the (0, 0, 0) relative to electrode position
            for el_counter in range(self.elnsites):
                self.el_mesh_size.append(1)
                self.el_mesh[el_counter] = np.zeros((3, 1))

        else:
            # Each electrode has an associate mesh file
            el_counter = 0
            for mesh_file in self.elmesh_files:
                file_path = mesh_file if os.path.isabs(mesh_file) else os.path.join(self._mesh_files_dir, mesh_file)
                mesh = pd.read_csv(file_path, sep=" ")
                mesh_size = mesh.shape[0]
                self.el_mesh_size.append(mesh_size)

                self.el_mesh[el_counter] = np.zeros((3, mesh_size))
                self.el_mesh[el_counter][0] = mesh['x_pos']
                self.el_mesh[el_counter][1] = mesh['y_pos']
                self.el_mesh[el_counter][2] = mesh['z_pos']
                el_counter += 1

    def place_the_electrodes(self):
        transfer_vector = np.zeros((self.elnsites, 3))
        for el in range(self.elnsites):
            mesh_mean = np.mean(self.el_mesh[el], axis=1)
            transfer_vector[el] = self.elpos[:, el] - mesh_mean[:]

        for el in range(self.elnsites):
            new_mesh = self.el_mesh[el].T + transfer_vector[el]
            self.el_mesh[el] = new_mesh.T

    def rotate_the_electrodes(self):
        for el in range(self.elnsites):
            phi_x = self.elrot[el][0]
            phi_y = self.elrot[el][1]
            phi_z = self.elrot[el][2]

            rot_x = rotation_matrix([1, 0, 0], phi_x)
            rot_y = rotation_matrix([0, 1, 0], phi_y)
            rot_z = rotation_matrix([0, 0, 1], phi_z)
            rot_xy = rot_x.dot(rot_y)
            rot_xyz = rot_xy.dot(rot_z)
            new_mesh = np.dot(rot_xyz, self.el_mesh[el])
            self.el_mesh[el] = new_mesh

    def set_transfer_resistance(self, gid, seg_coords):
        rho = 300.0  # ohm cm
        r05 = seg_coords.p05
        nseg = r05.shape[1]
        cell_map = np.zeros((self.elnsites, nseg))
        for el in six.moves.range(self.elnsites):
            mesh_size = self.el_mesh_size[el]
            for k in range(mesh_size):
                rel = np.expand_dims(self.el_mesh[el][:, k], axis=1)
                rel_05 = rel - r05
                r2 = np.einsum('ij,ij->j', rel_05, rel_05)
                r = np.sqrt(r2)
                if not all(i >= 10 for i in r):
                    io.log_exception('External electrode is too close')
                cell_map[el, :] += 1. / r

        cell_map *= (rho / (4 * math.pi)) * 0.01
        self.trans_X[gid] = cell_map

    def calculate_waveforms(self, tstep):
        simulation_time = self._dt * tstep
        # copies waveform elnsites times (homogeneous)
        self.waveform_amplitude = np.zeros(self.elnsites) + self.waveform.calculate(simulation_time)

    def get_vext(self, gid):
        waveform_per_mesh = np.divide(self.waveform_amplitude, self.el_mesh_size)
        v_extracellular = np.dot(waveform_per_mesh, self.trans_X[gid]) * 1E6
        vext_vec = h.Vector(v_extracellular)

        return vext_vec
