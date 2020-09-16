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
import six

from neuron import h

pc = h.ParallelContext()    # object to access MPI methods

class ConnectionStruct(object):
    def __init__(self, edge_prop, src_node, syn, connector, is_virtual=False, is_gap_junc=False):
        self._src_node = src_node
        self._edge_prop = edge_prop
        self._syn = syn
        self._connector = connector
        self._is_gj = is_gap_junc
        self._is_virtual = is_virtual

    @property
    def is_virtual(self):
        return self._is_virtual

    @property
    def is_gap_junc(self):
        return self._is_gj

    @property
    def source_node(self):
        return self._src_node

    @property
    def syn_weight(self):
        if self.is_gap_junc:
            return self._connector.g
        else:
            return self._connector.weight[0]

    @syn_weight.setter
    def syn_weight(self, val):
        if self.is_gap_junc:
            self._connector.g = val
        else:
            self._connector.weight[0] = val


class BioCell(Cell):
    """Implemntation of a morphologically and biophysically detailed type cell.

    """
    def __init__(self, node, population_name, bionetwork):
        super(BioCell, self).__init__(node=node, population_name=population_name, network=bionetwork)

        # Set up netcon object that can be used to detect and communicate cell spikes.
        self.set_spike_detector(bionetwork.spike_threshold)

        self._morph = None
        self._seg_coords = {}

        # Determine number of segments and store a list of all sections.
        self._nseg = 0
        self.set_nseg(bionetwork.dL)
        self._secs = []
        self._secs_by_id = []
        self.set_sec_array()

        self._save_conn = False  # bionetwork.save_connection
        self._synapses = []
        self._gap_juncs = []
        self._syn_src_net = []
        self._src_gids = []
        self._syn_seg_ix = []
        self._syn_sec_x = []
        self._edge_type_ids = []
        self._segments = None
        self._connections = []

        # potentially used by ecp module
        self.im_ptr = None
        self.imVec = None

        # used by xstim module
        self.ptr2e_extracellular = None

        self.__extracellular_mech = False

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
        """Update the segment coordinates (after rotations) for individual cells"""
        phi_y = self._node.rotation_angle_yaxis
        phi_z = self._node.rotation_angle_zaxis
        phi_x = self._node.rotation_angle_xaxis

        # Rotate cell
        # TODO: Rotations should follow as described in sonata (https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md).
        #  Need someone with graphics experience to check they are being done correctly (I'm not sure atm).
        RotX = utils.rotation_matrix([1, 0, 0], phi_x)
        RotY = utils.rotation_matrix([0, 1, 0], phi_y)  # rotate segments around yaxis normal to pia
        RotZ = utils.rotation_matrix([0, 0, 1], -phi_z)  # rotate segments around zaxis to get a proper orientation
        RotXYZ = np.dot(RotX, RotY.dot(RotZ))

        # rotated coordinates around z axis first then shift relative to the soma
        self._seg_coords['p0'] = self._pos_soma + np.dot(RotXYZ, morph_seg_coords['p0'])
        self._seg_coords['p1'] = self._pos_soma + np.dot(RotXYZ, morph_seg_coords['p1'])
        self._seg_coords['p05'] = self._pos_soma + np.dot(RotXYZ, morph_seg_coords['p05'])

    def get_seg_coords(self):
        return self._seg_coords

    @property
    def morphology_file(self):
        # TODO: Get from self._node.morphology_file
        return self._node.morphology_file

    @property
    def morphology(self):
        return self._morph

    @morphology.setter
    def morphology(self, morphology_obj):
        self.set_morphology(morphology_obj)

    def set_morphology(self, morphology_obj):
        self._morph = morphology_obj

    def get_sections(self):
        return self._secs_by_id
        #return self._secs

    def get_sections_id(self):
        return self._secs_by_id

    def get_section(self, sec_id):
        return self._secs[sec_id]

    def store_segments(self):
        self._segments = []
        for sec in self._secs_by_id:
            for seg in sec:
                self._segments.append(seg)

    def get_segments(self):
        return self._segments

    def set_sec_array(self):
        """Arrange sections in an array to be access by index"""
        secs = []  # build ref to sections
        self._secs_by_id = []
        for sec in self.hobj.all:
            self._secs_by_id.append(sec)
            for _ in sec:
                secs.append(sec)  # section to which segments belongs

        self._secs = np.array(secs)

    def set_syn_connection(self, edge_prop, src_node, stim=None, gj_ids=None):
        syn_weight = edge_prop.syn_weight(src_node=src_node, trg_node=self._node)

        if edge_prop.is_gap_junction:
            if gj_ids == None:
                raise Exception("Gap junctions must have a gap junction id passed to set_syn_connection.")

            self._edge_props.append(edge_prop)
            self._src_gids.append(src_node.node_id)

            if edge_prop.preselected_targets:
                return self._set_gap_junc_preselected(edge_prop, src_node, syn_weight, gj_ids)
            else:
                return self._set_gap_junc(edge_prop, src_node, syn_weight, gj_ids)
        else:
            if edge_prop.preselected_targets:
                self._edge_props.append(edge_prop)
                self._src_gids.append(src_node.node_id)
                return self._set_connection_preselected(edge_prop, src_node, syn_weight, stim)
            else:
                self._edge_props += [edge_prop]*edge_prop.nsyns
                self._src_gids += [src_node.node_id]*edge_prop.nsyns
                return self._set_connections(edge_prop, src_node, syn_weight, stim)

    def _set_gap_junc_preselected(self, edge_prop, src_node, syn_weight, gj_ids):
        if edge_prop.nsyns < 1:
            return 1

        sec_x = edge_prop['sec_x']
        sec_id = edge_prop['sec_id']
        section = self._secs_by_id[sec_id]

        #Sets up the section to be connected to the gap junction.
        pc.source_var(section(0.5)._ref_v, gj_ids[1], sec=section)

        #Creates gap junction.
        try:
            gap_junc = h.Gap(0.5, sec=section)
        except:
            raise Exception("You need the gap.mod file to create gap junctions.")

        #Attaches the source section to the gap junction.
        pc.target_var(gap_junc, gap_junc._ref_vgap, gj_ids[0])

        gap_junc.g = syn_weight

        self._connections.append(ConnectionStruct(edge_prop, src_node, gap_junc, gap_junc, False, True))

        self._gap_juncs.append(gap_junc)
        self._edge_type_ids.append(edge_prop.edge_type_id)

        if self._save_conn:
            self._save_connection(src_gid=src_node.gid, src_net=src_node.network, sec_x=sec_x, seg_ix=sec_id,
                                  edge_type_id=edge_prop.edge_type_id)

        return 1

    def _set_gap_junc(self, edge_prop, src_node, syn_weight, gj_ids):
        if edge_prop.nsyns < 1:
            return 1

        tar_seg_ix, tar_seg_prob = self._morph.get_target_segments(edge_prop)
        nsyns = 1

        # choose nsyn elements from seg_ix with probability proportional to segment area
        seg_ix = self.prng.choice(tar_seg_ix, nsyns, p=tar_seg_prob)[0]
        sec = self._secs[seg_ix]  # section where synapases connect
        x = self._morph.seg_prop['x'][seg_ix]  # distance along the section where synapse connects, i.e., seg_x

        if edge_prop.nsyns > 1:
            print("Warning: The number of synapses passed in was greater than 1, but only one gap junction will be made.")

        #Sets up the section to be connected to the gap junction.
        pc.source_var(sec(0.5)._ref_v, gj_ids[1], sec=sec)

        #Creates gap junction.
        try:
            gap_junc = h.Gap(0.5, sec=sec)
        except:
            raise Exception("You need the gap.mod file to create gap junctions.")

        #Attaches the source section to the gap junction.
        pc.target_var(gap_junc, gap_junc._ref_vgap, gj_ids[0])

        gap_junc.g = syn_weight

        self._connections.append(ConnectionStruct(edge_prop, src_node, gap_junc, gap_junc, False, True))

        self._gap_juncs.append(gap_junc)
        self._edge_type_ids.append(edge_prop.edge_type_id)

        if self._save_conn:
            self._save_connection(src_gid=src_node.gid, src_net=src_node.network, sec_x=x, seg_ix=sec_ix,
                                  edge_type_id=edge_prop.edge_type_id)

    def _set_connection_preselected(self, edge_prop, src_node, syn_weight, stim=None):
        # TODO: synapses should be loaded by edge_prop.load_synapse
        sec_x = edge_prop['sec_x']
        sec_id = edge_prop['sec_id']
        section = self._secs_by_id[sec_id]
        # section = self._secs[sec_id]
        delay = edge_prop['delay']
        synapse_fnc = nrn.py_modules.synapse_model(edge_prop['model_template'])
        syn = synapse_fnc(edge_prop['dynamics_params'], sec_x, section)

        if stim is not None:
            nc = h.NetCon(stim.hobj, syn)  # stim.hobj - source, syn - target
        else:
            nc = pc.gid_connect(src_node.node_id, syn)

        nc.weight[0] = syn_weight
        nc.delay = delay
        self._connections.append(ConnectionStruct(edge_prop, src_node, syn, nc, stim is not None))

        self._netcons.append(nc)
        self._synapses.append(syn)
        self._edge_type_ids.append(edge_prop.edge_type_id)
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
        synapses = [edge_prop.load_synapses(x, sec) for x, sec in zip(xs, secs)]

        delay = edge_prop['delay']
        self._synapses.extend(synapses)

        # TODO: Don't save this if not needed
        self._edge_type_ids.extend([edge_prop.edge_type_id]*len(synapses))

        for syn in synapses:
            # connect synapses
            if stim:
                nc = h.NetCon(stim.hobj, syn)
            else:
                nc = pc.gid_connect(src_gid, syn)

            nc.weight[0] = syn_weight
            nc.delay = delay
            self.netcons.append(nc)

            self._connections.append(ConnectionStruct(edge_prop, src_node, syn, nc, stim is not None))

        return nsyns

    def connections(self):
        return self._connections

    def _save_connection(self, src_gid, src_net, sec_x, seg_ix, edge_type_id):
        self._src_gids.append(src_gid)
        self._syn_src_net.append(src_net)
        self._syn_sec_x.append(sec_x)
        self._syn_seg_ix.append(seg_ix)
        self._edge_type_id.append(edge_type_id)

    def get_connection_info(self):
        # TODO: There should be a more effecient and robust way to return synapse information.
        return [[self.gid, self._syn_src_gid[i], self.network_name, self._syn_src_net[i], self._syn_seg_ix[i],
                 self._syn_sec_x[i], self.netcons[i].weight[0], self.netcons[i].delay, self._edge_type_id[i], 0]
                for i in range(len(self._synapses))]

    def init_connections(self):
        super(BioCell, self).init_connections()
        self._synapses = []
        self._syn_src_gid = []
        self._syn_seg_ix = []
        self._syn_sec_x = []

    def __set_extracell_mechanism(self):
        if not self.__extracellular_mech:
            for sec in self.hobj.all:
                sec.insert('extracellular')
            self.__extracellular_mech = True

    def setup_ecp(self):
        self.im_ptr = h.PtrVector(self._nseg)  # pointer vector
        # used for gathering an array of  i_membrane values from the pointer vector
        self.im_ptr.ptr_update_callback(self.set_im_ptr)
        self.imVec = h.Vector(self._nseg)

        self.__set_extracell_mechanism()
        #for sec in self.hobj.all:
        #    sec.insert('extracellular')

    def setup_xstim(self, set_nrn_mechanism=True):
        self.ptr2e_extracellular = h.PtrVector(self._nseg)
        self.ptr2e_extracellular.ptr_update_callback(self.set_ptr2e_extracellular)

        # Set the e_extracellular mechanism for all sections on this hoc object
        if set_nrn_mechanism:
            self.__set_extracell_mechanism()
            #for sec in self.hobj.all:
            #    sec.insert('extracellular')

    def set_im_ptr(self):
        """Set PtrVector to point to the _ref_i_membrane_ parameter"""
        jseg = 0
        for sec in self.hobj.all:  
            for seg in sec:
                self.im_ptr.pset(jseg, seg._ref_i_membrane_)  # notice the underscore at the end
                jseg += 1

    def get_im(self):
        """Gather membrane currents from PtrVector into imVec (does not need a loop!)"""
        self.im_ptr.gather(self.imVec)
        # Warning: as_numpy() seems to fail with in neuron 7.4 for python 3
        # return self.imVec.as_numpy()  # (nA)
        return np.array(self.imVec)

    def set_ptr2e_extracellular(self):
        jseg = 0
        for sec in self.hobj.all:
            for seg in sec:
                self.ptr2e_extracellular.pset(jseg, seg._ref_e_extracellular)
                jseg += 1

    def set_e_extracellular(self, vext):
        self.ptr2e_extracellular.scatter(vext)

    def print_synapses(self):
        rstr = ''
        for i in six.moves.range(len(self._syn_src_gid)):
            rstr += '{}> <-- {} ({}, {}, {}, {})\n'.format(i, self._syn_src_gid[i], self.netcons[i].weight[0],
                                                           self.netcons[i].delay, self._syn_seg_ix[i],
                                                           self._syn_sec_x[i])
        return rstr
