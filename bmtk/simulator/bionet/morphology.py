# Copyright 2022. Allen Institute. All rights reserved
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
import re
from collections import namedtuple
import warnings
import numpy as np
import pandas as pd
from neuron import h

from bmtk.simulator.bionet.io_tools import io


pc = h.ParallelContext()  # object to access MPI methods


class _LazySegmentProps(object):
    """A Struct for storing coordinate invariant properties of each segment of a cell."""
    def __init__(self, hobj):
        self._hobj = hobj
        self.sec_id = None
        self.type = None
        self.area = None
        self.x = None
        self.x0 = None
        self.x1 = None
        self.dist = None
        self.length = None
        self.dist0 = None
        self.dist1 = None

    def get(self):
        if self.sec_id is None:
            seg_type = []
            seg_area = []
            seg_x = []
            seg_x0 = []
            seg_x1 = []
            seg_dist = []
            seg_length = []
            seg_sec_id = []

            h.distance(sec=self._hobj.soma[0])  # measure distance relative to the soma

            for sec_id, sec in enumerate(self._hobj.all):
                fullsecname = sec.name()
                sec_type = fullsecname.split(".")[1][:4]  # get sec name type without the cell name
                sec_type_swc = Morphology.sec_type_swc[sec_type]  # convert to swc code
                x_range = 1.0 / (sec.nseg * 2)  # used to calculate [x0, x1]

                for seg in sec:
                    seg_area.append(h.area(seg.x))
                    seg_x.append(seg.x)
                    seg_x0.append(seg.x - x_range)
                    seg_x1.append(seg.x + x_range)
                    seg_length.append(sec.L / sec.nseg)
                    seg_type.append(sec_type_swc)  # record section type in a list
                    # seg_dist.append(h.distance(seg.x))  # distance to the center of the segment
                    seg_dist.append(h.distance(seg))
                    seg_sec_id.append(sec_id)

            length_arr = np.array(seg_length)
            dist_arr = np.array(seg_dist)
            dist0_arr = dist_arr - length_arr / 2.0
            dist1_arr = dist_arr + length_arr / 2.0

            self.type = np.array(seg_type)
            self.area = np.array(seg_area)
            self.x = np.array(seg_x)
            self.x0 = np.array(seg_x0)
            self.x1 = np.array(seg_x1)
            self.dist = dist_arr
            self.length = length_arr
            self.dist0 = dist0_arr
            self.dist1 = dist1_arr
            self.sec_id = np.array(seg_sec_id)

        return self


class _LazySegmentCoords(object):
    """A Struct for storing properties of each segment of a cell that vary by soma position and rotation."""
    def __init__(self, hobj):
        self._hobj = hobj
        self.p0 = None
        self.p1 = None
        self.p05 = None
        self.soma_pos = None

    def get(self):
        if self.p0 is None:
            # Iterates through each segment (one section may contain one or more segments). We use NEURON's
            # segment.x to get the mid-point and lenght of each segment we we can use to find and store the beginning
            # (p0) and end (p1) of each segment. We also shift so the middle of the soma is at the origin.
            nseg = np.sum([s.nseg for s in self._hobj.all])

            n3dsoma = 0
            soma_pos = np.zeros(3)
            for sec in self._hobj.soma:
                n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
                n3dsoma += n3d
                for i in range(n3d):
                    soma_pos[0] += h.x3d(i, sec=sec)
                    soma_pos[1] += h.y3d(i, sec=sec)
                    soma_pos[2] += h.z3d(i, sec=sec)

            soma_pos /= n3dsoma

            ix = 0  # segment index
            self.p0 = np.zeros((3, nseg))  # hold the coordinates of segment starting points
            self.p1 = np.zeros((3, nseg))  # hold the coordinates of segment end points
            self.p05 = np.zeros((3, nseg))
            for sec in self._hobj.all:
                n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
                p3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
                l3d = np.zeros(n3d)  # to hold locations of 3D morphology for the current section
                # diam3d = np.zeros(n3d)  # to diameters, at the moment we don't need to keep track of diameter

                for i in range(n3d):
                    p3d[0, i] = h.x3d(i, sec=sec) - soma_pos[0]  # shift coordinates such to place soma at the origin.
                    p3d[1, i] = h.y3d(i, sec=sec) - soma_pos[1]
                    p3d[2, i] = h.z3d(i, sec=sec) - soma_pos[2]
                    # diam3d[i] = h.diam3d(i, sec=sec)
                    l3d[i] = h.arc3d(i, sec=sec)

                l3d /= sec.L  # normalize
                nseg = sec.nseg

                l0 = np.zeros(nseg)  # keep range of segment starting point
                l1 = np.zeros(nseg)  # keep range of segment ending point
                l05 = np.zeros(nseg)

                for iseg, seg in enumerate(sec):
                    l0[iseg] = seg.x - 0.5 * 1 / nseg  # x (normalized distance along the section) for the beginning of the segment
                    l1[iseg] = seg.x + 0.5 * 1 / nseg  # x for the end of the segment
                    l05[iseg] = seg.x

                self.p0[0, ix:ix + nseg] = np.interp(l0, l3d, p3d[0, :])
                self.p0[1, ix:ix + nseg] = np.interp(l0, l3d, p3d[1, :])
                self.p0[2, ix:ix + nseg] = np.interp(l0, l3d, p3d[2, :])

                self.p1[0, ix:ix + nseg] = np.interp(l1, l3d, p3d[0, :])
                self.p1[1, ix:ix + nseg] = np.interp(l1, l3d, p3d[1, :])
                self.p1[2, ix:ix + nseg] = np.interp(l1, l3d, p3d[2, :])

                self.p05[0, ix:ix + nseg] = np.interp(l05, l3d, p3d[0, :])
                self.p05[1, ix:ix + nseg] = np.interp(l05, l3d, p3d[1, :])
                self.p05[2, ix:ix + nseg] = np.interp(l05, l3d, p3d[2, :])

                # x0[ix:ix + nseg] = l0
                # x1[ix:ix + nseg] = l1

                ix += nseg

            # Also calculate the middle of the shifted soma, NOTE: in theory this should be (0, 0, 0), but due to
            # percision the actual soma center might be a little off.
            n3dsoma = 0
            r3dsoma = np.zeros(3)
            for sec in self._hobj.soma:
                n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
                n3dsoma += n3d
                for i in range(n3d):
                    r3dsoma[0] += h.x3d(i, sec=sec) - soma_pos[0]
                    r3dsoma[1] += h.y3d(i, sec=sec) - soma_pos[1]
                    r3dsoma[2] += h.z3d(i, sec=sec) - soma_pos[2]

            r3dsoma /= n3dsoma
            self.soma_pos = r3dsoma

        return self


# For a lot of different cells created from the same morphology and model_processing function, the core segment
# properties will be the same (even if coordinates, synapses, hobjs, etc. are different). In such cases we want
# to calculate and store seg_props only once for all equivelent cells. We can do this by caching the seg_props
# helper function pointer, then all equivelent morphologies (as defined by __hash__ function) use the same
# SegmentProps object
morphology_cache = {}


def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.0)
    b, c, d = -axis*np.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
    ])


class Morphology(object):
    sec_type_swc = {
        'soma': 1, 'somatic': 1,
        'axon': 2, 'axonal': 2,
        'dend': 3, 'basal': 3,
        'apic': 4, 'apical': 4
    }

    def __init__(self, hobj, rng_seed=None, swc_path=None):
        self.hobj = hobj  # h.Biophys1(swc_path)
        self.rng_seed = rng_seed
        self.swc_path = swc_path

        self._prng = np.random.RandomState(self.rng_seed)
        self._sections = None
        self._segments = None
        self._seg_props = _LazySegmentProps(self.hobj)  # None
        self._seg_coords = _LazySegmentCoords(self.hobj)  # None
        self._nseg = None
        self._swc_map = None

        # Used by find_sections() and other methods when building edges/synapses. Should make it faster to look-up
        # cooresponding segments for a large number of syns that target the same area of a cell
        self._trg_segs_cache = {}

    def _copy(self):
        new_morph = Morphology(hobj=self.hobj, swc_path=self.swc_path)
        return new_morph

    def _soma_position_orig(self):
        n3dsoma = 0
        r3dsoma = np.zeros(3)
        for sec in self.hobj.soma:
            n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
            # r3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            n3dsoma += n3d

            for i in range(n3d):
                r3dsoma[0] += h.x3d(i, sec=sec)
                r3dsoma[1] += h.y3d(i, sec=sec)
                r3dsoma[2] += h.z3d(i, sec=sec)

        r3dsoma /= n3dsoma
        return r3dsoma

    @property
    def segments(self):
        if self._segments is None:
            self._segments = []
            for sec in self.hobj.all:
                for seg in sec:
                    self._segments.append(seg)

        return self._segments

    @property
    def sections(self):
        if self._sections is None:
            self._sections = []
            for sec in self.hobj.all:
                self._sections.append(sec)

        return self._sections

    @property
    def n_sections(self):
        return len(self.sections)

    @property
    def seg_props(self):
        return self._seg_props.get()

    @property
    def seg_coords(self):
        """Get the coordinates of all segments of the cells. Each segment is defined by three values, the beginning
        of the segment (seg_coords.p0), the middle of the segment(seg_coords.p05), and the end (seg_coords.p1)."""
        return self._seg_coords.get()

    @property
    def nseg(self):
        if self._nseg is None:
            nseg = 0
            for sec in self.hobj.all:
                nseg += sec.nseg  # get the total # of segments in the cell
            self._nseg = nseg

        return self._nseg

    @property
    def soma_position(self):
        return self.seg_coords.soma_pos

    @property
    def swc_map(self):
        if self._swc_map is None:
            swc_df = pd.read_csv(self.swc_path, sep=' ', names=['id', 'type', 'x', 'y', 'z', 'r', 'pid'], comment='#')
            swc_df = swc_df[['id', 'type']]

            name_indices = []
            for ln in range(len(swc_df)):
                # in read_swc.hoc all non-bifurcating section of rows are assigned a global section-id. But for each
                # type (soma, dend, etc), there is also a nameindex id which is unique within the type, used to assign
                # sections to their name. So lines with sec_id=10 may have nameindex=3 (Biophys1[0].dend[3])
                sec_id = int(self.hobj.nl.point2sec[ln])
                # swc_id = int(self.hobj.nl.pt2id(ln))
                # swc_type = int(self.hobj.nl.type[ln])
                name_index = int(self.hobj.nl.sections.object(sec_id).nameindex)
                name_indices.append(name_index)

            swc_df['nameindex'] = name_indices
            self._swc_map = swc_df

        return self._swc_map

    def get_section(self, section_idx):
        return self.sections[section_idx]

    def set_segment_dl(self, dl):
        """Define number of segments in a cell"""
        self._nseg = 0
        for sec in self.hobj.all:
            sec.nseg = 1 + 2 * int(sec.L/(2*dl))
            self._nseg += sec.nseg  # get the total number of segments in the cell

    def choose_sections(self, section_names, distance_range, n_sections=1, cache=True):
        """Similar to find_sections, but will only N=n_section number of sections_ids/x values randomly selected (may
        return less if there aren't as many sections

        :param section_names: 'soma', 'dend', 'apic', 'axon'
        :param distance_range: [float, float]: distance range of sections from the soma, in um.
        :param n_sections: int: maximum number of sections to select
        :param cache: caches the segments + probs so that next time the same section_names/distance_range values are
            called it will return the same values without recalculating. default True.
        :return: [int], [float]: A list of all section_ids and a list of all segment_x values (as defined by NEURON)
            that meet the given critera.
        """
        secs, probs = self.find_sections(section_names, distance_range, cache=cache)
        secs_ix = self._prng.choice(secs, n_sections, p=probs)
        return secs_ix, self.seg_props.x[secs_ix]

    def find_sections(self, section_names, distance_range, cache=True):
        """Retrieves a list of sections ids and section x's given a section name/type (eg axon, soma, apic, dend) and
        the distance from the soma.

        :param section_names: A list of sections to target, 'soma', 'dend', 'apic', 'axon'
        :param distance_range: [float, float]: distance range of sections from the soma, in um.
        :param cache: caches the segments + probs so that next time the same section_names/distance_range values are
            called it will return the same values without recalculating. default True.
        :return: [float], [float]: A list of all section_ids and a list of all segment_x values (as defined by NEURON)
            that meet the given critera.
        """
        cache_key = (tuple(section_names), tuple(distance_range))
        if cache and cache_key in self._trg_segs_cache:
            return self._trg_segs_cache[cache_key]

        dmin, dmax = distance_range[0], distance_range[1]

        seg_d0 = self.seg_props.dist0
        seg_d1 = self.seg_props.dist1
        seg_length = self.seg_props.length
        seg_type = self.seg_props.type

        # Find the fractional overlap between the segment and the distance range:
        # this is done by finding the overlap between [d0,d1] and [dmin,dmax]
        # np.minimum(seg_d1,dmax) find the smaller of the two end locations
        # np.maximum(seg_d0,dmin) find the larger of the two start locations
        # np.maximum(0,overlap) is used to return zero when segments do not overlap
        # and then dividing by the segment length
        frac_overlap = np.maximum(0, (np.minimum(seg_d1, dmax) - np.maximum(seg_d0, dmin))) / seg_length
        ix_drange = np.where(frac_overlap > 0)  # find indexes with non-zero overlap
        ix_labels = np.array([], dtype=int)

        for tar_sec_label in section_names:  # find indexes within sec_labels
            sec_type = self.sec_type_swc[tar_sec_label]  # get swc code for the section label
            ix_label = np.where(seg_type == sec_type)
            ix_labels = np.append(ix_labels, ix_label)  # target segment indexes

        tar_seg_ix = np.intersect1d(ix_drange, ix_labels)  # find intersection between indexes for range and labels
        tar_seg_length = seg_length[tar_seg_ix] * frac_overlap[tar_seg_ix]  # weighted length of targeted segments
        tar_seg_prob = tar_seg_length / np.sum(tar_seg_length)  # probability of targeting segments

        if cache:
            self._trg_segs_cache[cache_key] = (tar_seg_ix, tar_seg_prob)

        return tar_seg_ix, tar_seg_prob

    def move_and_rotate(self, soma_coords=None, rotation_angles=None, inplace=False):
        old_seg_coords = self.seg_coords
        new_p0 = old_seg_coords.p0.copy()
        new_p1 = old_seg_coords.p1.copy()
        new_p05 = old_seg_coords.p05.copy()
        new_soma_pos = self.soma_position.copy()

        if rotation_angles is not None:
            assert(len(rotation_angles) == 3)
            # try to calc the euler rotation matrix around an arbitary point was causing problems, instead move coords
            # so the the soma center (p05[0]) is at the origin
            soma_pos_mat = new_soma_pos.reshape((3, 1))
            new_p0 = new_p0 - soma_pos_mat
            new_p1 = new_p1 - soma_pos_mat
            new_p05 = new_p05 - soma_pos_mat

            rotx_mat = rotation_matrix([1, 0, 0], rotation_angles[0])
            roty_mat = rotation_matrix([0, 1, 0], rotation_angles[1])
            rotz_mat = rotation_matrix([0, 0, 1], rotation_angles[2])
            rotxyz_mat = np.dot(rotx_mat, roty_mat.dot(rotz_mat))

            new_p0 = np.dot(rotxyz_mat, new_p0) + soma_pos_mat
            new_p1 = np.dot(rotxyz_mat, new_p1) + soma_pos_mat
            new_p05 = np.dot(rotxyz_mat, new_p05) + soma_pos_mat

        if soma_coords is not None:
            assert(len(soma_coords) == 3)
            soma_coords = soma_coords if isinstance(soma_coords, np.ndarray) else np.array(soma_coords)
            displacement = soma_coords - self.soma_position
            displacement = displacement.reshape((3, 1))  # Req to allow adding vector to matrix
            new_p0 += displacement
            new_p1 += displacement
            new_p05 += displacement
            new_soma_pos = soma_coords

        # new_seg_coords = SegmentCoords(p0=new_p0, p1=new_p1, p05=new_p05, soma_pos=new_soma_pos)
        new_seg_coords = _LazySegmentCoords(self.hobj)
        new_seg_coords.p0 = new_p0
        new_seg_coords.p05 = new_p05
        new_seg_coords.p1 = new_p1
        new_seg_coords.soma_pos = new_soma_pos

        if inplace:
            self._seg_coords = new_seg_coords
            return self
        else:
            new_morph = self._copy()
            new_morph._seg_props = self.seg_props
            new_morph._seg_coords = new_seg_coords
            new_morph._soma_pos = new_soma_pos
            return new_morph

    def get_coords(self, sec_id, sec_x):
        segs_indices = np.argwhere(self.seg_props.sec_id == sec_id).flatten()
        return self.seg_coords.p05[:, segs_indices][:, 0]

    def get_swc_id(self, sec_id, sec_x):
        # use sec type and nameindex to find all rows in the swc that correspond to sec_id
        sec = self.sections[sec_id]
        sec_nameindex = self._get_sec_nameindex(sec)
        sec_type = self._get_sec_type(sec)
        filtered_swc = self.swc_map[(self.swc_map['type'] == sec_type) & (self.swc_map['nameindex'] == sec_nameindex)]
        swc_ids = filtered_swc['id'].values

        # use sec_x, a value between [0, 1], to estimate which swc_id/line to choose.
        # NOTE: At the moment it assumes each line in the swc is the same distance apart, making estimating the sec_x
        #  location easy by the number it appears in the squence
        if len(swc_ids) == 0:
            return -1, 0.0
        elif len(swc_ids) == 1:
            return swc_ids[0], sec_x
        else:
            swc_place = np.max((0.0, sec_x*len(swc_ids) - 1.0))
            swc_indx = int(np.ceil(swc_place))
            swc_id = swc_ids[swc_indx]
            swc_dist = swc_indx - swc_place
            return swc_id, swc_dist

    def _get_sec_type(self, sec):
        sec_name = sec.hname()
        sec_type_str = sec_name.split('.')[-1].split('[')[0]
        type_str = self.sec_type_swc[sec_type_str]
        return int(type_str)

    def _get_sec_nameindex(self, sec):
        sec_name = sec.hname()
        sec_type_name = sec_name.split('.')[-1]
        nameindex_str = re.search(r'\[(\d+)\]', sec_type_name).group(1)
        return int(nameindex_str)

    def __eq__(self, other):
        if self.swc_path is None or other.swc_path is None:
            return False
        else:
            return hash(self) == hash(other)

    def __hash__(self):
        """Used for comparing two Morphology objects and for storing as dict keys in seg_props_cache.

        We assume two Morphologies are the same if they have the same morphology file path + same # of segments. This
        is not a good way of determining if two Morphologies are equal and in actuality we would need to compare
        seg_props on an segment-by-segment basis. However that would be comp. expensive for large networks and instead
        do a quick spot test."""
        comp_vals = (self.swc_path if self.swc_path else np.random.randint, self.nseg)
        return hash(comp_vals)

    @classmethod
    def load(cls, hobj=None, morphology_file=None, rng_seed=None, cache_seg_props=True):
        """Factory method, perfered way to create a Morphology object

        :param hobj:
        :param morphology_file:
        :param rng_seed:
        :param cache_seg_props:
        :return:
        """
        if hobj is None and morphology_file is None:
            io.log_exception('Unable to create Morphology, no valid HOC object or morphology file specified')

        elif morphology_file is not None and hobj is None:
            # If the HOC object hasn't already been created try creating it from the morphology
            hobj = h.Biophys1(morphology_file)

        morph = cls(hobj=hobj, swc_path=morphology_file, rng_seed=rng_seed)

        if cache_seg_props and morphology_file is not None:
            if morph in morphology_cache:
                morph._seg_props = morphology_cache[morph][0]
                morph._seg_coords = morphology_cache[morph][1]
            else:
                morphology_cache[morph] = (morph._seg_props, morph._seg_coords)

        return morph
