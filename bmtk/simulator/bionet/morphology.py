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
import numpy as np
from neuron import h


pc = h.ParallelContext()  # object to access MPI methods


class Morphology(object):
    """Methods for processing morphological data"""
    def __init__(self, hobj):
        """reuse hoc object from one of the cells which share the same morphology/model"""
        self.hobj = hobj
        self.sec_type_swc = {'soma': 1, 'somatic': 1,  # convert section name and section list names
                             'axon': 2, 'axonal': 2,  # into a consistent swc notation
                             'dend': 3, 'basal': 3,
                             'apic': 4, 'apical': 4}
        self.nseg = self.get_nseg()
        self._segments = {}

    def get_nseg(self):
        nseg = 0
        for sec in self.hobj.all:
            nseg += sec.nseg  # get the total # of segments in the cell
        return nseg

    def get_soma_pos(self):
        n3dsoma = 0
        r3dsoma = np.zeros(3)
        for sec in self.hobj.soma:
            n3d = int(h.n3d())  # get number of n3d points in each section
            r3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            n3dsoma += n3d

            for i in range(n3d):
                r3dsoma[0] += h.x3d(i, sec=sec)
                r3dsoma[1] += h.y3d(i, sec=sec)
                r3dsoma[2] += h.z3d(i, sec=sec)

        r3dsoma /= n3dsoma
        return r3dsoma

    def calc_seg_coords(self):
        """Calculate segment coordinates from 3d point coordinates"""
        ix = 0  # segment index

        p3dsoma = self.get_soma_pos()
        self.psoma = p3dsoma
        
        p0 = np.zeros((3, self.nseg))  # hold the coordinates of segment starting points
        p1 = np.zeros((3, self.nseg))  # hold the coordinates of segment end points
        p05 = np.zeros((3, self.nseg))
        d0 = np.zeros(self.nseg)
        d1 = np.zeros(self.nseg)

        for sec in self.hobj.all:
            n3d = int(h.n3d())  # get number of n3d points in each section
            p3d = np.zeros((3, n3d))  # to hold locations of 3D morphology for the current section
            l3d = np.zeros(n3d)  # to hold locations of 3D morphology for the current section
            diam3d = np.zeros(n3d)  # to diameters

            for i in range(n3d):
                p3d[0, i] = h.x3d(i) - p3dsoma[0]
                p3d[1, i] = h.y3d(i) - p3dsoma[1]  # shift coordinates such to place soma at the origin.
                p3d[2, i] = h.z3d(i) - p3dsoma[2]
                diam3d[i] = h.diam3d(i)
                l3d[i] = h.arc3d(i)

            l3d /= sec.L                  # normalize
            nseg = sec.nseg
            
            l0 = np.zeros(nseg)     # keep range of segment starting point 
            l1 = np.zeros(nseg)     # keep range of segment ending point
            l05 = np.zeros(nseg)
            
            for iseg, seg in enumerate(sec):
                l0[iseg] = seg.x - 0.5*1/nseg   # x (normalized distance along the section) for the beginning of the segment
                l1[iseg] = seg.x + 0.5*1/nseg   # x for the end of the segment
                l05[iseg] = seg.x

            p0[0, ix:ix+nseg] = np.interp(l0, l3d, p3d[0, :])
            p0[1, ix:ix+nseg] = np.interp(l0, l3d, p3d[1, :])
            p0[2, ix:ix+nseg] = np.interp(l0, l3d, p3d[2, :])
            d0[ix:ix+nseg] = np.interp(l0, l3d, diam3d[:])

            p1[0, ix:ix+nseg] = np.interp(l1, l3d, p3d[0, :])
            p1[1, ix:ix+nseg] = np.interp(l1, l3d, p3d[1, :])
            p1[2, ix:ix+nseg] = np.interp(l1, l3d, p3d[2, :])
            d1[ix:ix+nseg] = np.interp(l1, l3d, diam3d[:])

            p05[0,ix:ix+nseg] = np.interp(l05, l3d, p3d[0,:])
            p05[1,ix:ix+nseg] = np.interp(l05, l3d, p3d[1,:])
            p05[2,ix:ix+nseg] = np.interp(l05, l3d, p3d[2,:])

            ix += nseg

        self.seg_coords = {}

        self.seg_coords['p0'] = p0
        self.seg_coords['p1'] = p1
        self.seg_coords['p05'] = p05

        self.seg_coords['d0'] = d0
        self.seg_coords['d1'] = d1

        return self.seg_coords

    def set_seg_props(self):
        """Set segment properties which are invariant for all cell using this morphology"""
        seg_type = []
        seg_area = []
        seg_x = []
        seg_dist = []
        seg_length = []

        h.distance(sec=self.hobj.soma[0])   # measure distance relative to the soma
        
        for sec in self.hobj.all:
            fullsecname = sec.name()
            sec_type = fullsecname.split(".")[1][:4] # get sec name type without the cell name
            sec_type_swc = self.sec_type_swc[sec_type]  # convert to swc code

            for seg in sec:

                seg_area.append(h.area(seg.x))
                seg_x.append(seg.x)
                seg_length.append(sec.L/sec.nseg)
                seg_type.append(sec_type_swc)           # record section type in a list
                # seg_dist.append(h.distance(seg.x))  # distance to the center of the segment
                seg_dist.append(h.distance(seg))

        self.seg_prop = {}
        self.seg_prop['type'] = np.array(seg_type)
        self.seg_prop['area'] = np.array(seg_area)
        self.seg_prop['x'] = np.array(seg_x)
        self.seg_prop['dist'] = np.array(seg_dist)
        self.seg_prop['length'] = np.array(seg_length)
        self.seg_prop['dist0'] = self.seg_prop['dist'] - self.seg_prop['length']/2
        self.seg_prop['dist1'] = self.seg_prop['dist'] + self.seg_prop['length']/2

    def get_target_segments(self, edge_type):
        # Determine the target segments and their probabilities of connections for each new edge-type. Save the
        # information for each additional time a given edge-type is used on this morphology
        # TODO: Don't rely on edge-type-table, just use the edge?
        if edge_type in self._segments:
            return self._segments[edge_type]

        else:
            tar_seg_ix, tar_seg_prob = self.find_sections(edge_type.target_sections, edge_type.target_distance)
            self._segments[edge_type] = (tar_seg_ix, tar_seg_prob)
            return tar_seg_ix, tar_seg_prob

        """
        tar_sec_labels = edge_type.target_sections
        drange = edge_type.target_distance
        dmin, dmax = drange[0], drange[1]

        seg_d0 = self.seg_prop['dist0']  # use a more compact variables
        seg_d1 = self.seg_prop['dist1']
        seg_length = self.seg_prop['length']
        seg_area = self.seg_prop['area']
        seg_type = self.seg_prop['type']

        # Find the fractional overlap between the segment and the distance range:
        # this is done by finding the overlap between [d0,d1] and [dmin,dmax]
        # np.minimum(seg_d1,dmax) find the smaller of the two end locations
        # np.maximum(seg_d0,dmin) find the larger of the two start locations
        # np.maximum(0,overlap) is used to return zero when segments do not overlap
        # and then dividing by the segment length
        frac_overlap = np.maximum(0, (np.minimum(seg_d1, dmax) - np.maximum(seg_d0, dmin))) / seg_length
        ix_drange = np.where(frac_overlap > 0)  # find indexes with non-zero overlap
        ix_labels = np.array([], dtype=np.int)

        for tar_sec_label in tar_sec_labels:  # find indexes within sec_labels
            sec_type = self.sec_type_swc[tar_sec_label]  # get swc code for the section label
            ix_label = np.where(seg_type == sec_type)
            ix_labels = np.append(ix_labels, ix_label)  # target segment indexes

        tar_seg_ix = np.intersect1d(ix_drange, ix_labels)  # find intersection between indexes for range and labels
        tar_seg_length = seg_length[tar_seg_ix] * frac_overlap[tar_seg_ix]  # weighted length of targeted segments
        tar_seg_prob = tar_seg_length / np.sum(tar_seg_length)  # probability of targeting segments

        self._segments[edge_type] = (tar_seg_ix, tar_seg_prob)
        return tar_seg_ix, tar_seg_prob
        """

    def find_sections(self, target_sections, distance_range):
        dmin, dmax = distance_range[0], distance_range[1]

        seg_d0 = self.seg_prop['dist0']  # use a more compact variables
        seg_d1 = self.seg_prop['dist1']
        seg_length = self.seg_prop['length']
        seg_area = self.seg_prop['area']
        seg_type = self.seg_prop['type']

        # Find the fractional overlap between the segment and the distance range:
        # this is done by finding the overlap between [d0,d1] and [dmin,dmax]
        # np.minimum(seg_d1,dmax) find the smaller of the two end locations
        # np.maximum(seg_d0,dmin) find the larger of the two start locations
        # np.maximum(0,overlap) is used to return zero when segments do not overlap
        # and then dividing by the segment length
        frac_overlap = np.maximum(0, (np.minimum(seg_d1, dmax) - np.maximum(seg_d0, dmin))) / seg_length
        ix_drange = np.where(frac_overlap > 0)  # find indexes with non-zero overlap
        ix_labels = np.array([], dtype=np.int)

        for tar_sec_label in target_sections:  # find indexes within sec_labels
            sec_type = self.sec_type_swc[tar_sec_label]  # get swc code for the section label
            ix_label = np.where(seg_type == sec_type)
            ix_labels = np.append(ix_labels, ix_label)  # target segment indexes

        tar_seg_ix = np.intersect1d(ix_drange, ix_labels)  # find intersection between indexes for range and labels
        tar_seg_length = seg_length[tar_seg_ix] * frac_overlap[tar_seg_ix]  # weighted length of targeted segments
        tar_seg_prob = tar_seg_length / np.sum(tar_seg_length)  # probability of targeting segments
        return tar_seg_ix, tar_seg_prob