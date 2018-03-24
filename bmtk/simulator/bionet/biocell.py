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
from bmtk.simulator.bionet import utils, nrn
from bmtk.simulator.bionet.cell import Cell

from neuron import h

pc = h.ParallelContext()    # object to access MPI methods


class BioCell(Cell):
    """Implemntation of a morphologically and biophysically detailed type cell.

    """
    def __init__(self, node, bionetwork):
        super(BioCell, self).__init__(node)

        # Set up netcon object that can be used to detect and communicate cell spikes.
        self.set_spike_detector(bionetwork.spike_threshold)

        self._morph = None
        self._seg_coords = {}

        # Determine number of segments and store a list of all sections.
        self._nseg = 0
        self.set_nseg(bionetwork.dL)
        self._secs = []
        self.set_sec_array()

        self._save_conn = False # bionetwork.save_connection
        self._synapses = []
        self._syn_src_net = []
        self._syn_src_gid = []
        self._syn_seg_ix = []
        self._syn_sec_x = []
        self._edge_type_id = []

        # potentially used by ecp module
        self.im_ptr = None
        self.im_Vec = None

        self._segments = None


        '''
        if bionetwork.calc_ecp:
            self.im_ptr = h.PtrVector(self._nseg)  # pointer vector
            self.im_ptr.ptr_update_callback(self.set_im_ptr)   # used for gathering an array of  i_membrane values from the pointer vector
            self.imVec = h.Vector(self._nseg)
        '''

    def set_spike_detector(self, spike_threshold):
        nc = h.NetCon(self.hobj.soma[0](0.5)._ref_v, None, sec=self.hobj.soma[0])  # attach spike detector to cell
        nc.threshold = spike_threshold     
        pc.cell(self.gid, nc)  # associate gid with spike detector

    def set_nseg(self, dL):
        """Define number of segments in a cell"""
        self._nseg = 0
        for sec in self.hobj.all:
            sec.nseg = 1 + 2 * int(sec.L/(2*dL))
            self._nseg += sec.nseg # get the total number of segments in the cell

    def calc_seg_coords(self, morph_seg_coords):
        """Calculate segment coordinates for individual cells"""
        #print self.gid
        #print self._prop_map
        #print self._prop_map.rotation_angle_yaxis(self._props)
        #exit()
        #phi_y = self._props['rotation_angle_yaxis']
        #phi_y = self._props.rotation_angle_yaxis  # self._props['rotation_angle_yaxis']

        phi_y = self._node.rotation_angle_yaxis  # self._props['rotation_angle_zaxis']
        phi_z = self._node.rotation_angle_zaxis  # self._props['rotation_angle_zaxis']

        RotY = utils.rotation_matrix([0, 1, 0], phi_y)  # rotate segments around yaxis normal to pia
        RotZ = utils.rotation_matrix([0, 0, 1], -phi_z) # rotate segments around zaxis to get a proper orientation
        RotYZ = RotY.dot(RotZ)

        # rotated coordinates around z axis first then shift relative to the soma
        self._seg_coords['p0'] = self._pos_soma + np.dot(RotYZ, morph_seg_coords['p0'])
        self._seg_coords['p1'] = self._pos_soma + np.dot(RotYZ, morph_seg_coords['p1'])

    def get_seg_coords(self):
        return self._seg_coords

    @property
    def morphology_file(self):
        # TODO: Get from self._node.morphology_file
        return self._node['morphology_file']

    @property
    def morphology(self):
        return self._morph

    @morphology.setter
    def morphology(self, morphology_obj):
        self.set_morphology(morphology_obj)

    def set_morphology(self, morphology_obj):
        self._morph = morphology_obj

    def get_sections(self):
        return self._secs

    def get_section(self, sec_id):
        return self._secs[sec_id]

    def store_segments(self):
        self._segments = []
        for sec in self._secs:
            for seg in sec:
                self._segments.append(seg)

    def get_segments(self):
        return self._segments

    def set_sec_array(self):
        """Arrange sections in an array to be access by index"""
        secs = []  # build ref to sections
        # TODO: We should calculate and save sections in the morphology object since they should be the same.
        for sec in self.hobj.all:
            for _ in sec:
                secs.append(sec)  # section to which segments belongs

        self._secs = np.array(secs)

    def set_syn_connection(self, edge_prop, src_node, stim=None):
        syn_weight = edge_prop.weight(src_node, self._node)

        if edge_prop.preselected_targets:
            return self._set_connection_preselected(edge_prop, src_node, syn_weight, stim)
        else:
            return self._set_connections(edge_prop, src_node, syn_weight, stim)

    def _set_connection_preselected(self, edge_prop, src_node, syn_weight, stim=None):
        sec_x = edge_prop['sec_x']
        sec_id = edge_prop['sec_id']
        section = self._secs[sec_id]
        delay = edge_prop['delay']
        synapse_fnc = nrn.py_modules.synapse_model(edge_prop['template'])
        syn = synapse_fnc(edge_prop['dynamics_params'], sec_x, section)

        if stim is not None:
            nc = h.NetCon(stim.hobj, syn)  # stim.hobj - source, syn - target
        else:
            nc = pc.gid_connect(src_node.node_id, syn)

        nc.weight[0] = syn_weight
        nc.delay = delay
        self._netcons.append(nc)
        self._synapses.append(syn)
        if self._save_conn:
            self._save_connection(src_gid=src_node.node_id, src_net=src_node.network, sec_x=sec_x, seg_ix=sec_id,
                                  edge_type_id=edge_prop.edge_type_id)

        return 1

    def _set_connections(self, edge_prop, src_node, syn_weight, stim=None):
        tar_seg_ix, tar_seg_prob = self._morph.get_target_segments(edge_prop)
        src_gid = src_node.node_id
        nsyns = edge_prop.nsyns

        # choose nsyn elements from seg_ix with probability proportional to segment area
        segs_ix = self.prng.choice(tar_seg_ix, nsyns, p=tar_seg_prob)
        secs = self._secs[segs_ix]  # sections where synapases connect
        xs = self._morph.seg_prop['x'][segs_ix]  # distance along the section where synapse connects, i.e., seg_x

        # TODO: this should be done just once
        synapses = edge_prop.load_synapses(xs, secs)
        delay = edge_prop['delay']
        self._synapses.extend(synapses)
        if self._save_conn:
            for i in range(nsyns):
                self._save_connection(src_gid, src_node.network, sec_x=xs[i], seg_ix=segs_ix[i],
                                      edge_type_id=edge_prop.edge_type_id)
                # self._save_connection(src_gid, src_node.network, xs[i], segs_ix[i])

        # self._syn_seg_ix.extend(segs_ix)  # use only when need to output synaptic locations
        # self._syn_src_gid.extend([src_gid] * nsyns)

        for syn in synapses:
            # connect synapses
            if stim:
                nc = h.NetCon(stim.hobj, syn)  # stim.hobj - source, syn - target
            else:
                nc = pc.gid_connect(src_gid, syn)

            nc.weight[0] = syn_weight
            nc.delay = delay
            self.netcons.append(nc)
        return nsyns

    def _save_connection(self, src_gid, src_net, sec_x, seg_ix, edge_type_id):
        self._syn_src_gid.append(src_gid)
        self._syn_src_net.append(src_net)
        self._syn_sec_x.append(sec_x)
        self._syn_seg_ix.append(seg_ix)
        self._edge_type_id.append(edge_type_id)

    def get_connection_info(self):
        # TODO: There should be a more effecient and robust way to return synapse information.
        return [[self.gid, self._syn_src_gid[i], self.network_name, self._syn_src_net[i], self._syn_seg_ix[i],
                 self._syn_sec_x[i], self.netcons[i].weight[0], self.netcons[i].delay, self._edge_type_id[i], 0]
                for i in range(len(self._synapses))]

    '''
    def set_syn_connections(self, nsyn, syn_weight, edge_type, src_gid, stim=None):
        """Set synaptic connections"""
        tar_seg_ix, tar_seg_prob = self._morph.get_target_segments(edge_type)

        # choose nsyn elements from seg_ix with probability proportional to segment area
        segs_ix = self.prng.choice(tar_seg_ix, nsyn, p=tar_seg_prob)
        secs = self.secs[segs_ix]  # sections where synapases connect
        xs = self._morph.seg_prop['x'][segs_ix]  # distance along the section where synapse connects, i.e., seg_x

        syn_params = edge_type['params'] #edge_type.syn_params
        set_syns_func = nrn.py_modules.synapse_model(edge_type['set_params_function'])
        syns = set_syns_func(syn_params, xs, secs)
            
        weight = syn_weight
        delay = edge_type['delay']
        self._synapses.extend(syns)
        self._syn_seg_ix.extend(segs_ix)  # use only when need to output synaptic locations
        self._syn_src_gid.extend([src_gid]*len(syns))

        for syn in syns:
            # connect synapses
            if stim:
                nc = h.NetCon(stim.hobj, syn)   # stim.hobj - source, syn - target
            else:
                nc = pc.gid_connect(src_gid, syn)
 
            nc.weight[0] = weight
            nc.delay = delay      
            self.netcons.append(nc)
    '''

    def init_connections(self):
        Cell.init_connections(self)
        self._synapses = []
        self._syn_src_gid = []
        self._syn_seg_ix = []
        self._syn_sec_x = []

    def setup_ecp(self):
        self.im_ptr = h.PtrVector(self._nseg)  # pointer vector
        # used for gathering an array of  i_membrane values from the pointer vector
        self.im_ptr.ptr_update_callback(self.set_im_ptr)
        self.imVec = h.Vector(self._nseg)

    def set_im_ptr(self): 
        """Set PtrVector to point to the i_membrane_"""
        jseg = 0
        for sec in self.hobj.all:  
            for seg in sec:    
                self.im_ptr.pset(jseg,seg._ref_i_membrane_)  # notice the underscore at the end
                jseg += 1

    def get_im(self):
        """Gather membrane currents from PtrVector into imVec (does not need a loop!)"""
        self.im_ptr.gather(self.imVec)
        return self.imVec.as_numpy()  # (nA)

    def print_synapses(self):
        rstr = ''
        for i in xrange(len(self._syn_src_gid)):
            rstr += '{}> <-- {} ({}, {}, {}, {})\n'.format(i, self._syn_src_gid[i], self.netcons[i].weight[0],
                                                           self.netcons[i].delay, self._syn_seg_ix[i],
                                                           self._syn_sec_x[i])
        return rstr
