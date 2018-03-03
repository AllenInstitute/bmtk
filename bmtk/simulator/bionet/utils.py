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
import numpy as np
import math
import json
import pandas as pd
import h5py

from neuron import h


def rotation_matrix(axis, theta):
    """Return the rotation matrix associated with counterclockwise rotation about the given axis by theta radians.
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def edge_converter_csv(output_dir, csv_file):
    """urrently being used by BioNetwork.write_connections(), need to refactor

    :param output_dir:
    :param csv_file:
    :return:
    """
    syns_df = pd.read_csv(csv_file, sep=' ')
    for name, group in syns_df.groupby(['trg_network', 'src_network']):
        trg_net, src_net = name
        group_len = len(group.index)
        with h5py.File(os.path.join(output_dir, '{}_{}_edges.h5'.format(trg_net, src_net)), 'w') as conns_h5:
            conns_h5.create_dataset('edges/target_gid', data=group['trg_gid'])
            conns_h5.create_dataset('edges/source_gid', data=group['src_gid'])
            conns_h5.create_dataset('edges/edge_type_id', data=group['edge_type_id'])
            conns_h5.create_dataset('edges/edge_group', data=group['connection_group'])

            group_counters = {group_id: 0 for group_id in group.connection_group.unique()}
            edge_group_indicies = np.zeros(group_len, dtype=np.uint)
            for i, group_id in enumerate(group['connection_group']):
                edge_group_indicies[i] = group_counters[group_id]
                group_counters[group_id] += 1
            conns_h5.create_dataset('edges/edge_group_indicies', data=edge_group_indicies)

            for group_class, sub_group in group.groupby('connection_group'):
                grp = conns_h5.create_group('edges/{}'.format(group_class))
                if group_class == 0:
                    grp.create_dataset('sec_id', data=sub_group['segment'], dtype='int')
                    grp.create_dataset('sec_x', data=sub_group['section'])
                    grp.create_dataset('syn_weight', data=sub_group['weight'])
                    grp.create_dataset('delay', data=sub_group['delay'])
                elif group_class == 1:
                    grp.create_dataset('syn_weight', data=sub_group['weight'])
                    grp.create_dataset('delay', data=sub_group['delay'])
                else:
                    print('Unknown cell group {}'.format(group_class))



##################################################
# TODO: Move these functions to default_setters
##################################################
def set_morphology(hobj, morph_file):
    """Set morphology for the cell from a swc

    :param hobj: NEURON's cell object
    :param morph_file: name of swc file containing 3d coordinates of morphology
    """
    swc = h.Import3d_SWC_read()
    swc.quiet = True
    swc.input(str(morph_file))
    imprt = h.Import3d_GUI(swc, 0)
    imprt.quiet = True
    imprt.instantiate(hobj)


def set_segs(hobj):
    """Define number of segments in a cell

    :param hobj: NEURON's cell object
    :return:
    """
    for sec in hobj.all:
        sec.nseg = 1 + 2 * int(sec.L / 40)


def fix_axon(hobj):
    """Replace reconstructed axon with a stub

    :param hobj: NEURON's cell object
    """
    for sec in hobj.axon:
        h.delete_section(sec=sec)

    h.execute('create axon[2]', hobj)
    for sec in hobj.axon:
        sec.L = 30
        sec.diam = 1
        hobj.axonal.append(sec=sec)
        hobj.all.append(sec=sec)    # need to remove this comment

    hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
    hobj.axon[1].connect(hobj.axon[0], 1, 0)
    h.define_shape()


def fix_axon_all_active(hobj):
    """need temporary because axon is treated differently"""
    pass


def set_params_perisomatic(hobj, params_file_name):
    """Set biophysical parameters for the cell

    :param hobj: NEURON's cell object
    :param params_file_name: name of json file containing biophysical parameters for cell's model which determine
    spiking behavior
    """
    with open(params_file_name) as biophys_params_file:
        biophys_params = json.load(biophys_params_file)

    passive = biophys_params['passive'][0]
    conditions = biophys_params['conditions'][0]
    genome = biophys_params['genome']

    # Set passive properties
    cm_dict = dict([(c['section'], c['cm']) for c in passive['cm']])
    for sec in hobj.all:
        sec.Ra = passive['ra']
        sec.cm = cm_dict[sec.name().split(".")[1][:4]]
        sec.insert('pas')
        for seg in sec:
            seg.pas.e = passive["e_pas"]

    # Insert channels and set parameters
    for p in genome:
        sections = [s for s in hobj.all if s.name().split(".")[1][:4] == p["section"]]

        for sec in sections:
            if p["mechanism"] != "":
                sec.insert(p["mechanism"])
            setattr(sec, p["name"], p["value"])

    # Set reversal potentials
    for erev in conditions['erev']:
        sections = [s for s in hobj.all if s.name().split(".")[1][:4] == erev["section"]]
        for sec in sections:
            sec.ena = erev["ena"]
            sec.ek = erev["ek"]


def set_params_all_active(hobj, params_file_name):
    """Configure a neuron after the cell morphology has been loaded"""
    with open(params_file_name) as biophys_params_file:
        biophys_params = json.load(biophys_params_file)

    passive = biophys_params['passive'][0]
    genome = biophys_params['genome']
    conditions = biophys_params['conditions'][0]

    # Set fixed passive properties
    for sec in hobj.all:
        sec.Ra = passive['ra']
        sec.insert('pas')

    # Insert channels and set parameters
    for p in genome:
        section_array = p["section"]
        mechanism = p["mechanism"]
        param_name = p["name"]
        param_value = float(p["value"])
        if section_array == "glob":
            h(p["name"] + " = %g " % p["value"])
        else:
            if hasattr(hobj, section_array):
                if mechanism != "":
                    print 'Adding mechanism %s to %s' % (mechanism, section_array)
                    for section in getattr(hobj, section_array):
                        if h.ismembrane(str(mechanism), sec=section) != 1:
                            section.insert(mechanism)

                print 'Setting %s to %.6g in %s' % (param_name, param_value, section_array)
                for section in getattr(hobj, section_array):
                    setattr(section, param_name, param_value)

    # Set reversal potentials
    for erev in conditions['erev']:
        erev_section_array = erev["section"]
        ek = float(erev["ek"])
        ena = float(erev["ena"])

        print 'Setting ek to %.6g and ena to %.6g in %s' % (ek, ena, erev_section_array)

        if hasattr(hobj, erev_section_array):
            for section in getattr(hobj, erev_section_array):
                if h.ismembrane("k_ion", sec=section) == 1:
                    setattr(section, 'ek', ek)
                if h.ismembrane("na_ion", sec=section) == 1:
                    setattr(section, 'ena', ena)
        else:
            print "Warning: can't set erev for %s, section array doesn't exist" % erev_section_array
