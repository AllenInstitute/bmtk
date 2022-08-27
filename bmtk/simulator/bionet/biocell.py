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
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet.morphology import Morphology
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

        # Determine number of segments and store a list of all sections.
        self._secs = []
        self._secs_by_id = []

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

        self._morphology = None
        self._seg_coords = None
        self.build_morphology()

    def build_morphology(self):
        morph_base = Morphology.load(hobj=self.hobj, morphology_file=self.morphology_file, cache_seg_props=True)

        if self._network.dL is not None:
            morph_base.set_segment_dl(self._network.dL)

        # NOTE: most simulations will not require the cells to be shifted and rotated (only modules like ecp, xstim,
        #  etc require it). Only performe a move_and_rotate() function on cell if or when cell.seg_coords is called.
        self._morphology = morph_base
        self.set_sec_array()

    @property
    def morphology_file(self):
        """Value that's stored in SONATA morphology column"""
        return self._node.morphology_file

    @property
    def morphology(self):
        """The actual Morphology object instanstiation"""
        return self._morphology

    @property
    def seg_coords(self):
        """Coordinates for segments/sections of the morphology, need to make public for ecp, xstim, and other
        functionality that needs to compute the soma/dendritic coordinates of each cell"""
        if self._seg_coords is None:
            # Before returning the coordinates make sure to translate and rotate the cell
            phi_x = self._node.rotation_angle_xaxis
            phi_y = self._node.rotation_angle_yaxis
            phi_z = self._node.rotation_angle_zaxis
            self._morphology.move_and_rotate(
                soma_coords=self.soma_position.flatten(),
                rotation_angles=[phi_x, phi_y, phi_z],
                inplace=True
            )

        return self.morphology.seg_coords

    def set_spike_detector(self, spike_threshold):
        nc = h.NetCon(self.hobj.soma[0](0.5)._ref_v, None, sec=self.hobj.soma[0])  # attach spike detector to cell
        nc.threshold = spike_threshold
        pc.cell(self.gid, nc)  # associate gid with spike detector

    def get_sections(self):
        return self._secs_by_id

    def get_sections_id(self):
        return self._secs_by_id

    def get_section(self, sec_id):
        return self._secs[sec_id]

    def set_sec_array(self):
        """Arrange sections in an array to be access by index"""
        # TODO: This should be accessabile in Morphology object
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

        sec_x = edge_prop.afferent_section_pos
        sec_id = edge_prop.afferent_section_id
        section = self._secs_by_id[sec_id]

        # Sets up the section to be connected to the gap junction.
        pc.source_var(section(0.5)._ref_v, gj_ids[1], sec=section)

        # Creates gap junction.
        try:
            gap_junc = h.Gap(0.5, sec=section)
        except:
            raise Exception("You need the gap.mod file to create gap junctions.")

        # Attaches the source section to the gap junction.
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

        tar_seg_ix, tar_seg_prob = self.morphology.find_sections(
            section_names=edge_prop.target_sections,
            distance_range=edge_prop.target_distance,
            cache=True
        )

        nsyns = 1

        # choose nsyn elements from seg_ix with probability proportional to segment area
        seg_ix = self.prng.choice(tar_seg_ix, nsyns, p=tar_seg_prob)[0]
        sec = self._secs[seg_ix]  # section where synapases connect
        x = self.morphology.seg_props.x[seg_ix]  # distance along the section where synapse connects, i.e., seg_x

        if edge_prop.nsyns > 1:
            print("Warning: The number of synapses passed in was greater than 1, but only one gap junction will be made.")

        # Sets up the section to be connected to the gap junction.
        pc.source_var(sec(0.5)._ref_v, gj_ids[1], sec=sec)

        # Creates gap junction.
        try:
            gap_junc = h.Gap(0.5, sec=sec)
        except:
            raise Exception("You need the gap.mod file to create gap junctions.")

        # Attaches the source section to the gap junction.
        pc.target_var(gap_junc, gap_junc._ref_vgap, gj_ids[0])

        gap_junc.g = syn_weight

        self._connections.append(ConnectionStruct(edge_prop, src_node, gap_junc, gap_junc, False, True))

        self._gap_juncs.append(gap_junc)
        self._edge_type_ids.append(edge_prop.edge_type_id)

        # if self._save_conn:
        #     self._save_connection(src_gid=src_node.gid, src_net=src_node.network, sec_x=x, seg_ix=sec_ix,
        #                           edge_type_id=edge_prop.edge_type_id)

    def _set_connection_preselected(self, edge_prop, src_node, syn_weight, stim=None):
        # TODO: synapses should be loaded by edge_prop.load_synapse
        sec_x = edge_prop.afferent_section_pos
        sec_id = edge_prop.afferent_section_id

        section = self._secs_by_id[sec_id]
        # section = self._secs[sec_id]
        delay = edge_prop['delay']
        synapse_fnc = nrn.py_modules.synapse_model(edge_prop['model_template'])
        syn = synapse_fnc(edge_prop['dynamics_params'], sec_x, section)

        if stim is not None:
            nc = h.NetCon(stim.hobj, syn)  # stim.hobj - source, syn - target
        else:
            src_gid = self._network.gid_pool.get_gid(name=src_node.population_name, node_id=src_node.node_id)
            nc = pc.gid_connect(src_gid, syn)

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
        tar_seg_ix, tar_seg_prob = self.morphology.find_sections(
            section_names=edge_prop.target_sections,
            distance_range=edge_prop.target_distance,
            cache=True
        )
        nsyns = edge_prop.nsyns

        if len(tar_seg_ix) == 0:
            msg = 'Could not find target synaptic location for edge-type {}, Please check target_section and/or distance_range properties'.format(edge_prop.edge_type_id)
            io.log_warning(msg, all_ranks=True, display_once=True)
            return 0

        segs_ix = self.prng.choice(tar_seg_ix, nsyns, p=tar_seg_prob)
        secs = self._secs[segs_ix]  # sections where synapases connect
        xs = self.morphology.seg_props.x[segs_ix]  # distance along the section where synapse connects, i.e., seg_x

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
                src_gid = self._network.gid_pool.get_gid(name=src_node.population_name, node_id=src_node.node_id)
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
        self.im_ptr = h.PtrVector(self.morphology.nseg)  # pointer vector
        # used for gathering an array of  i_membrane values from the pointer vector
        self.im_ptr.ptr_update_callback(self.set_im_ptr)
        self.imVec = h.Vector(self.morphology.nseg)

        self.__set_extracell_mechanism()
        # for sec in self.hobj.all:
        #     sec.insert('extracellular')

    def setup_xstim(self, set_nrn_mechanism=True):
        self.ptr2e_extracellular = h.PtrVector(self.morphology.nseg)
        self.ptr2e_extracellular.ptr_update_callback(self.set_ptr2e_extracellular)

        # Set the e_extracellular mechanism for all sections on this hoc object
        if set_nrn_mechanism:
            self.__set_extracell_mechanism()
            # for sec in self.hobj.all:
            #     sec.insert('extracellular')

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


class BioCellSpontSyn(BioCell):
    """Special class that allows certain synapses to spontaneously fire (without spiking) at a specific time.
    """
    def __init__(self, node, population_name, bionetwork):
        super(BioCellSpontSyn, self).__init__(node, population_name=population_name, bionetwork=bionetwork)

        # Get the timestamp at which synapses
        self._syn_timestamps = bionetwork.spont_syns_times
        self._syn_timestamps = [self._syn_timestamps] if np.isscalar(self._syn_timestamps) else self._syn_timestamps
        self._spike_trains = h.Vector(self._syn_timestamps)
        self._vecstim = h.VecStim()
        self._vecstim.play(self._spike_trains)

        self._precell_filter = bionetwork.spont_syns_filter
        assert(isinstance(self._precell_filter, dict))

    def _matches_filter(self, src_node):
        """Check to see if the presynaptic cell matches the criteria specified"""
        for k, v in self._precell_filter.items():
            if isinstance(v, (list, tuple)):
                if src_node[k] not in v:
                    return False
            else:
                if src_node[k] != v:
                    return False
        return True

    def _set_connections(self, edge_prop, src_node, syn_weight, stim=None):
        tar_seg_ix, tar_seg_prob = self.morphology.find_sections(
            section_names=edge_prop.target_sections,
            distance_range=edge_prop.target_distance,
            cache=True
        )
        # tar_seg_ix, tar_seg_prob = self.morphology.get_target_segments(edge_prop)
        src_gid = src_node.node_id
        nsyns = edge_prop.nsyns

        # choose nsyn elements from seg_ix with probability proportional to segment area
        segs_ix = self.prng.choice(tar_seg_ix, nsyns, p=tar_seg_prob)
        secs = self._secs[segs_ix]  # sections where synapases connect
        xs = self.morphology.seg_props.x[segs_ix]  # distance along the section where synapse connects, i.e., seg_x

        synapses = [edge_prop.load_synapses(x, sec) for x, sec in zip(xs, secs)]

        delay = edge_prop['delay']
        self._synapses.extend(synapses)

        for syn in synapses:
            # connect synapses
            if stim:
                nc = h.NetCon(stim.hobj, syn)
            elif self._matches_filter(src_node):
                nc = h.NetCon(self._vecstim, syn)
            else:
                nc = pc.gid_connect(src_gid, syn)
                syn_weight = 0.0

            nc.weight[0] = syn_weight
            nc.delay = delay
            self.netcons.append(nc)

            self._connections.append(ConnectionStruct(edge_prop, src_node, syn, nc, stim is not None))

        return nsyns

    def _set_connection_preselected(self, edge_prop, src_node, syn_weight, stim=None):
        # TODO: synapses should be loaded by edge_prop.load_synapse
        sec_x = edge_prop.afferent_section_pos
        sec_id = edge_prop.afferent_section_id
        section = self._secs_by_id[sec_id]
        # section = self._secs[sec_id]
        delay = edge_prop['delay']
        synapse_fnc = nrn.py_modules.synapse_model(edge_prop['model_template'])
        syn = synapse_fnc(edge_prop['dynamics_params'], sec_x, section)

        if stim is not None:
            nc = h.NetCon(stim.hobj, syn)

        elif self._matches_filter(src_node):
            nc = h.NetCon(self._vecstim, syn)

        else:
            nc = pc.gid_connect(src_node.node_id, syn)
            syn_weight = 0.0

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
