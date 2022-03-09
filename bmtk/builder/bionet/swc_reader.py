import os
import re
from collections import namedtuple
import numpy as np
import pandas as pd
from neuron import h

from bmtk.simulator.bionet import nrn


SegmentProps = namedtuple('SegmentProps', ['type', 'area', 'x', 'dist', 'length', 'dist0', 'dist1', 'sec_id'])
SegmentCoords = namedtuple('SegmentCoords', ['p0', 'p1', 'p05'])


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


class SWCReader(object):
    sec_type_swc = {
        'soma': 1, 'somatic': 1,
        'axon': 2, 'axonal': 2,
        'dend': 3, 'basal': 3,
        'apic': 4, 'apical': 4
    }

    def __init__(self, swc_path, rng_seed=None):
        nrn.load_neuron_modules(None, None)

        self.swc_path = swc_path
        self.hobj = h.Biophys1(swc_path)
        self.rng_seed = rng_seed

        self._prng = np.random.RandomState(self.rng_seed)
        self._sections = None
        self._n_sections = None
        self._soma_pos = None
        self._seg_props = None
        self._seg_coords = None
        self._nseg = None
        self._swc_map = None
        self._axon_fixed = False
        self._axon_deleted = False


    def _copy(self):
        new_swc = SWCReader(swc_path=self.swc_path, rng_seed=self.rng_seed)
        if self._axon_fixed:
            new_swc.fix_axon()

        if self._axon_deleted:
            new_swc.delete_axon()

        return new_swc

    @property
    def sections(self):
        if self._sections is None:
            self._sections = []
            for sec in self.hobj.all:
                self._sections.append(sec)

        return self._sections

    @property
    def n_sections(self):
        if self._n_sections is None:
            self._n_sections = len(self.sections)

        return self._n_sections

    @property
    def seg_props(self):
        if self._seg_props is None:
            seg_type = []
            seg_area = []
            seg_x = []
            seg_dist = []
            seg_length = []
            seg_sec_id = []

            h.distance(sec=self.hobj.soma[0])  # measure distance relative to the soma

            # for sec in self.hobj.all:
            for sec_id, sec in enumerate(self.sections):
                fullsecname = sec.name()
                sec_type = fullsecname.split(".")[1][:4]  # get sec name type without the cell name
                sec_type_swc = self.sec_type_swc[sec_type]  # convert to swc code

                for seg in sec:
                    seg_area.append(h.area(seg.x))
                    seg_x.append(seg.x)
                    seg_length.append(sec.L / sec.nseg)
                    seg_type.append(sec_type_swc)  # record section type in a list
                    # seg_dist.append(h.distance(seg.x))  # distance to the center of the segment
                    seg_dist.append(h.distance(seg))
                    seg_sec_id.append(sec_id)

            length_arr = np.array(seg_length)
            dist_arr = np.array(seg_dist)
            dist0_arr = dist_arr - length_arr/2.0
            dist1_arr = dist_arr + length_arr/2.0
            self._seg_props = SegmentProps(
                type=np.array(seg_type),
                area=np.array(seg_area),
                x=np.array(seg_x),
                dist=dist_arr,
                length=length_arr,
                dist0=dist0_arr,
                dist1=dist1_arr,
                sec_id=np.array(seg_sec_id)
            )

        return self._seg_props

    @property
    def seg_coords(self):
        if self._seg_coords is None:
            ix = 0  # segment index

            p0 = np.zeros((3, self.nseg))  # hold the coordinates of segment starting points
            p1 = np.zeros((3, self.nseg))  # hold the coordinates of segment end points
            p05 = np.zeros((3, self.nseg))

            # for sec in self.hobj.all:
            for sec in self.sections:
                n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
                p3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
                l3d = np.zeros(n3d)  # to hold locations of 3D morphology for the current section
                diam3d = np.zeros(n3d)  # to diameters

                for i in range(n3d):
                    p3d[0, i] = h.x3d(i, sec=sec)  # - p3dsoma[0]
                    p3d[1, i] = h.y3d(i, sec=sec)  # - p3dsoma[1]  # shift coordinates such to place soma at the origin.
                    p3d[2, i] = h.z3d(i, sec=sec)  # - p3dsoma[2]
                    diam3d[i] = h.diam3d(i, sec=sec)
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

                p0[0, ix:ix + nseg] = np.interp(l0, l3d, p3d[0, :])
                p0[1, ix:ix + nseg] = np.interp(l0, l3d, p3d[1, :])
                p0[2, ix:ix + nseg] = np.interp(l0, l3d, p3d[2, :])

                p1[0, ix:ix + nseg] = np.interp(l1, l3d, p3d[0, :])
                p1[1, ix:ix + nseg] = np.interp(l1, l3d, p3d[1, :])
                p1[2, ix:ix + nseg] = np.interp(l1, l3d, p3d[2, :])

                p05[0, ix:ix + nseg] = np.interp(l05, l3d, p3d[0, :])
                p05[1, ix:ix + nseg] = np.interp(l05, l3d, p3d[1, :])
                p05[2, ix:ix + nseg] = np.interp(l05, l3d, p3d[2, :])

                ix += nseg

            self._seg_coords = SegmentCoords(p0=p0, p1=p1, p05=p05)

        return self._seg_coords

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
        if self._soma_pos is None:
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
            self._soma_pos = r3dsoma

        return self._soma_pos

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

    """
    @soma_position.setter
    def soma_position(self, coords):
        assert(len(coords) == 3)
        self.set_soma_position(coords[0], coords[1], coords[2])

    def set_soma_position(self, x, y, z):
        new_soma_coords = np.array([x, y, z])
        if self._soma_pos is not None and not np.allclose(new_soma_coords, self._soma_pos, equal_nan=True):
            # check if the coordinates are the same as before
            self._seg_coords = None

        self._soma_pos = new_soma_coords
    """

    def get_section(self, section_idx):
        return self.sections[section_idx]

    def fix_axon(self):
        """Removes and refixes axon"""
        axon_diams = [self.hobj.axon[0].diam, self.hobj.axon[0].diam]
        for sec in self.hobj.all:
            section_name = sec.name().split(".")[1][:4]
            if section_name == 'axon':
                axon_diams[1] = sec.diam

        for sec in self.hobj.axon:
            h.delete_section(sec=sec)

        h.execute('create axon[2]', self.hobj)
        for index, sec in enumerate(self.hobj.axon):
            sec.L = 30
            sec.diam = 1

            self.hobj.axonal.append(sec=sec)
            self.hobj.all.append(sec=sec)  # need to remove this comment

        self.hobj.axon[0].connect(self.hobj.soma[0], 1.0, 0)
        self.hobj.axon[1].connect(self.hobj.axon[0], 1.0, 0)

        h.define_shape()

        self._sections = None
        self._n_sections = None
        self._axon_fixed = True

    def delete_axon(self):
        for sec in self.hobj.axon:
            h.delete_section(sec=sec)

        h.define_shape()

        self._sections = None
        self._n_sections = None
        self._axon_deleted = True

    def set_segment_dl(self, dl):
        """Define number of segments in a cell"""
        self._nseg = 0
        for sec in self.hobj.all:
            sec.nseg = 1 + 2 * int(sec.L/(2*dl))
            self._nseg += sec.nseg  # get the total number of segments in the cell

    def choose_sections(self, section_names, distance_range, n_sections=1):
        """Similar to find_sections, but will only N=n_section number of sections_ids/x values randomly selected (may
        return less if there aren't as many sections

        :param section_names: 'soma', 'dend', 'apic', 'axon'
        :param distance_range: [float, float]: distance range of sections from the soma, in um.
        :param n_sections: int: maximum number of sections to select
        :return: [int], [float]: A list of all section_ids and a list of all segment_x values (as defined by NEURON)
            that meet the given critera.
        """
        secs, probs = self.find_sections(section_names, distance_range)
        secs_ix = self._prng.choice(secs, n_sections, p=probs)
        return secs_ix, self.seg_props.x[secs_ix]

    def find_sections(self, section_names, distance_range):
        """Retrieves a list of sections ids and section x's given a section name/type (eg axon, soma, apic, dend) and
        the distance from the soma.

        :param section_names: 'soma', 'dend', 'apic', 'axon'
        :param distance_range: [float, float]: distance range of sections from the soma, in um.
        :return: [float], [float]: A list of all section_ids and a list of all segment_x values (as defined by NEURON)
            that meet the given critera.
        """
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
        ix_labels = np.array([], dtype=np.int)

        for tar_sec_label in section_names:  # find indexes within sec_labels
            sec_type = self.sec_type_swc[tar_sec_label]  # get swc code for the section label
            ix_label = np.where(seg_type == sec_type)
            ix_labels = np.append(ix_labels, ix_label)  # target segment indexes

        tar_seg_ix = np.intersect1d(ix_drange, ix_labels)  # find intersection between indexes for range and labels
        tar_seg_length = seg_length[tar_seg_ix] * frac_overlap[tar_seg_ix]  # weighted length of targeted segments
        tar_seg_prob = tar_seg_length / np.sum(tar_seg_length)  # probability of targeting segments
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

        new_seg_coords = SegmentCoords(p0=new_p0, p1=new_p1, p05=new_p05)

        if inplace:
            self._seg_coords = new_seg_coords
            return self
        else:
            new_swc_reader = self._copy()
            new_swc_reader._seg_coords = new_seg_coords
            new_swc_reader._soma_pos = new_soma_pos

            return new_swc_reader

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
        # Note: At the moment it assumes each line in the swc is the same distance apart, making estimating the sec_x
        #  location easy by the number it appears in the squence
        if len(swc_ids) == 1:
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


swc_cache = {}


def get_swc(cell, morphology_dir=None, use_cache=False):
    cell_pop = cell.get('population', 'default')
    cell_node_id = cell['node_id']
    if use_cache and cell_pop in swc_cache and cell_node_id in swc_cache[cell_pop]:
        return swc_cache[cell_pop][cell_node_id]

    swc_path = cell['morphology']
    if morphology_dir:
        swc_path = os.path.join(morphology_dir, swc_path)

    if not os.path.exists(swc_path) and not swc_path.endswith('.swc'):
        swc_path += '.swc'

    if not os.path.exists(swc_path):
        raise ValueError('File {} does not exists.'.format(swc_path))

    swc = SWCReader(swc_path)

    if cell.get('model_processing', 'NULL') == 'aibs_perisomatic':
        swc.fix_axon()

    if any([cn in cell for cn in ['x', 'y', 'z']]):
        soma_coords = [cell.get('x', 0.0), cell.get('y', 0.0), cell.get('z', 0.0)]
    else:
        soma_coords = None

    if any([cn in cell for cn in ['rotation_angle_xaxis', 'rotation_angle_yaxis', 'rotation_angle_zaxis']]):
        rotation_angles = [
            cell.get('rotation_angle_xaxis', 0.0),
            cell.get('rotation_angle_yaxis', 0.0),
            cell.get('rotation_angle_zaxis', 0.0)
        ]
    else:
        rotation_angles = None

    swc = swc.move_and_rotate(soma_coords=soma_coords, rotation_angles=rotation_angles)

    if use_cache:
        if cell_pop not in swc_cache:
            swc_cache[cell_pop] = {}

        swc_cache[cell_pop][cell_node_id] = swc

    return swc
