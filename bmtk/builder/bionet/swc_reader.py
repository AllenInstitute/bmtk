import numpy as np
from neuron import h

from bmtk.simulator.bionet import nrn
from bmtk.simulator.bionet.morphology import Morphology


class SWCReader(object):
    def __init__(self, swc_file, random_seed=10, fix_axon=True):
        nrn.load_neuron_modules(None, None)
        self._swc_file = swc_file
        self._hobj = h.Biophys1(swc_file)
        if fix_axon:
            self._fix_axon()

        self._morphology = Morphology(self._hobj)
        self._morphology.set_seg_props()
        self._morphology.calc_seg_coords()
        self._prng = np.random.RandomState(random_seed)

        self._secs = []
        self._save_sections()

    def _save_sections(self):
        for sec in self._hobj.all:
            for _ in sec:
                self._secs.append(sec)

    def _fix_axon(self):
        """Removes and refixes axon"""
        axon_diams = [self._hobj.axon[0].diam, self._hobj.axon[0].diam]
        for sec in self._hobj.all:
            section_name = sec.name().split(".")[1][:4]
            if section_name == 'axon':
                axon_diams[1] = sec.diam

        for sec in self._hobj.axon:
            h.delete_section(sec=sec)

        h.execute('create axon[2]', self._hobj)
        for index, sec in enumerate(self._hobj.axon):
            sec.L = 30
            sec.diam = 1

            self._hobj.axonal.append(sec=sec)
            self._hobj.all.append(sec=sec)  # need to remove this comment

        self._hobj.axon[0].connect(self._hobj.soma[0], 1.0, 0)
        self._hobj.axon[1].connect(self._hobj.axon[0], 1.0, 0)

        h.define_shape()

    def find_sections(self, section_names, distance_range):
        return self._morphology.find_sections(section_names, distance_range)

    def choose_sections(self, section_names, distance_range, n_sections=1):
        secs, probs = self.find_sections(section_names, distance_range)
        secs_ix = self._prng.choice(secs, n_sections, p=probs)
        return secs_ix, self._morphology.seg_prop['x'][secs_ix]

    def get_coord(self, sec_ids, sec_xs, soma_center=(0.0, 0.0, 0.0), rotations=None):
        adjusted = self._morphology.get_soma_pos() - np.array(soma_center)
        absolute_coords = []
        for sec_id, sec_x in zip(sec_ids, sec_xs):
            sec = self._secs[sec_id]
            n_coords = int(h.n3d(sec=sec))
            coord_indx = int(sec_x*(n_coords - 1))
            swc_coords = np.array([h.x3d(coord_indx, sec=sec), h.y3d(coord_indx, sec=sec), h.x3d(coord_indx, sec=sec)])
            absolute_coords.append(swc_coords - adjusted)

            if rotations is not None:
                raise NotImplementedError

        return absolute_coords

    def get_dist(self, sec_ids):
        return [self._morphology.seg_prop['dist'][sec_id] for sec_id in sec_ids]

    def get_type(self, sec_ids):
        return [self._morphology.seg_prop['type'][sec_id] for sec_id in sec_ids]

