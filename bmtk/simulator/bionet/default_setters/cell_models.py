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
from neuron import h
try:
    from sklearn.decomposition import PCA
except Exception as e:
    pass

from bmtk.simulator.bionet.pyfunction_cache import add_cell_model, add_cell_processor
from bmtk.simulator.bionet.io_tools import io
from bmtk.simulator.bionet.nml_reader import NMLTree


"""
Functions for loading NEURON cell objects.

Functions will be loaded by bionetwork and called when a new cell object is created. These are for standard models
loaded with Cell-Types json files or their NeuroML equivelent, but may be overridden by the users.
"""

def loadHOC(cell, template_name, dynamics_params):
    # Get template to instantiate
    template_call = getattr(h, template_name)
    if dynamics_params is not None and 'params' in dynamics_params:
        template_params = dynamics_params['params']
        if isinstance(template_params, list):
            # pass in a list of parameters
            hobj = template_call(*template_params)
        else:
            # only a single parameter
            hobj = template_call(template_params)
    else:
        # instantiate template with no parameters
        hobj = template_call()

    # TODO: All "all" section if it doesn't exist
    # hobj.all = h.SectionList()
    # hobj.all.wholetree(sec=hobj.soma[0])
    return hobj


def IntFire1(cell, template_name, dynamics_params):
    """Loads a point integrate and fire neuron"""
    hobj = h.IntFire1()
    hobj.tau = dynamics_params['tau']*1000.0  # Convert from seconds to ms.
    hobj.refrac = dynamics_params['refrac']*1000.0  # Convert from seconds to ms.
    return hobj


def Biophys1(cell, template_name, dynamic_params):
    """Loads a biophysical NEURON hoc object using Cell-Types database objects."""
    morphology_file = cell.morphology_file
    hobj = h.Biophys1(str(morphology_file))
    #fix_axon(hobj)
    #set_params_peri(hobj, dynamic_params)
    return hobj


def Biophys1_nml(json_file):
    # TODO: look at examples to see how to convert .nml files
    raise NotImplementedError()


def Biophys1_dict(cell):
    """ Set parameters for cells from the Allen Cell Types database Prior to setting parameters will replace the
    axon with the stub
    """
    morphology_file = cell['morphology']
    hobj = h.Biophys1(str(morphology_file))
    return hobj


def aibs_perisomatic(hobj, cell, dynamics_params):
    if dynamics_params is not None:
        fix_axon_peri(hobj)
        set_params_peri(hobj, dynamics_params)

    return hobj


def fix_axon_peri(hobj):
    """Replace reconstructed axon with a stub

    :param hobj: hoc object
    """
    for sec in hobj.axon:
        h.delete_section(sec=sec)

    h.execute('create axon[2]', hobj)

    for sec in hobj.axon:
        sec.L = 30
        sec.diam = 1
        hobj.axonal.append(sec=sec)
        hobj.all.append(sec=sec)  # need to remove this comment

    hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
    hobj.axon[1].connect(hobj.axon[0], 1, 0)

    h.define_shape()


def set_params_peri(hobj, biophys_params):
    """Set biophysical parameters for the cell

    :param hobj: NEURON's cell object
    :param biophys_params: name of json file with biophys params for cell's model which determine spiking behavior
    :return:
    """
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


def aibs_allactive(hobj, cell, dynamics_params):
    fix_axon_allactive(hobj)
    set_params_allactive(hobj, dynamics_params)
    return hobj


def fix_axon_allactive(hobj):
    """Replace reconstructed axon with a stub

    Parameters
    ----------
    hobj: instance of a Biophysical template
        NEURON's cell object
    """
    # find the start and end diameter of the original axon, this is different from the perisomatic cell model
    # where diameter == 1.
    axon_diams = [hobj.axon[0].diam, hobj.axon[0].diam]
    for sec in hobj.all:
        section_name = sec.name().split(".")[1][:4]
        if section_name == 'axon':
            axon_diams[1] = sec.diam

    for sec in hobj.axon:
        h.delete_section(sec=sec)

    h.execute('create axon[2]', hobj)
    for index, sec in enumerate(hobj.axon):
        sec.L = 30
        sec.diam = axon_diams[index]  # 1

        hobj.axonal.append(sec=sec)
        hobj.all.append(sec=sec)  # need to remove this comment

    hobj.axon[0].connect(hobj.soma[0], 1.0, 0)
    hobj.axon[1].connect(hobj.axon[0], 1.0, 0)

    h.define_shape()


def set_params_allactive(hobj, params_dict):
    # params_dict = json.load(open(params_file_name, 'r'))
    passive = params_dict['passive'][0]
    genome = params_dict['genome']
    conditions = params_dict['conditions'][0]

    section_map = {}
    for sec in hobj.all:
        section_name = sec.name().split(".")[1][:4]
        if section_name in section_map:
            section_map[section_name].append(sec)
        else:
            section_map[section_name] = [sec]

    for sec in hobj.all:
        sec.insert('pas')
        # sec.insert('extracellular')

    if 'e_pas' in passive:
        e_pas_val = passive['e_pas']
        for sec in hobj.all:
            for seg in sec:
                seg.pas.e = e_pas_val

    if 'ra' in passive:
        ra_val = passive['ra']
        for sec in hobj.all:
            sec.Ra = ra_val

    if 'cm' in passive:
        # print('Setting cm')
        for cm_dict in passive['cm']:
            cm = cm_dict['cm']
            for sec in section_map.get(cm_dict['section'], []):
                sec.cm = cm

    for genome_dict in genome:
        g_section = genome_dict['section']
        if genome_dict['section'] == 'glob':
            io.log_warning("There is a section called glob, probably old json file")
            continue

        g_value = float(genome_dict['value'])
        g_name = genome_dict['name']
        g_mechanism = genome_dict.get("mechanism", "")
        for sec in section_map.get(g_section, []):
            if g_mechanism != "":
                sec.insert(g_mechanism)
            setattr(sec, g_name, g_value)

    for erev in conditions['erev']:
        erev_section = erev['section']
        erev_ena = erev['ena']
        erev_ek = erev['ek']

        if erev_section in section_map:
            for sec in section_map.get(erev_section, []):
                if h.ismembrane('k_ion', sec=sec) == 1:
                    setattr(sec, 'ek', erev_ek)
                if h.ismembrane('na_ion', sec=sec) == 1:
                    setattr(sec, 'ena', erev_ena)
        else:
            io.log_warning("Can't set erev for {}, section array doesn't exist".format(erev_section))


def aibs_perisomatic_directed(hobj, cell, dynamics_params):
    fix_axon_perisomatic_directed(hobj)
    set_params_peri(hobj, dynamics_params)
    return hobj


def aibs_allactive_directed(hobj, cell, dynamics_params):
    fix_axon_allactive_directed(hobj)
    set_params_allactive(hobj, dynamics_params)
    return hobj


def fix_axon_perisomatic_directed(hobj):
    # io.log_info('Fixing Axon like perisomatic')
    all_sec_names = []
    for sec in hobj.all:
        all_sec_names.append(sec.name().split(".")[1][:4])

    if 'axon' not in all_sec_names:
        io.log_exception('There is no axonal recostruction in swc file.')
    else:
        beg1, end1, beg2, end2 = get_axon_direction(hobj)

    for sec in hobj.axon:
        h.delete_section(sec=sec)
    h.execute('create axon[2]', hobj)

    h.pt3dadd(beg1[0], beg1[1], beg1[2], 1, sec=hobj.axon[0])
    h.pt3dadd(end1[0], end1[1], end1[2], 1, sec=hobj.axon[0])
    hobj.all.append(sec=hobj.axon[0])
    h.pt3dadd(beg2[0], beg2[1], beg2[2], 1, sec=hobj.axon[1])
    h.pt3dadd(end2[0], end2[1], end2[2], 1, sec=hobj.axon[1])
    hobj.all.append(sec=hobj.axon[1])

    hobj.axon[0].connect(hobj.soma[0], 0.5, 0)
    hobj.axon[1].connect(hobj.axon[0], 1.0, 0)

    hobj.axon[0].L = 30.0
    hobj.axon[1].L = 30.0

    h.define_shape()

    for sec in hobj.axon:
        # print "sec.L:", sec.L
        if np.abs(30-sec.L) > 0.0001:
            io.log_exception('Axon stub L is less than 30')


def fix_axon_allactive_directed(hobj):
    all_sec_names = []
    for sec in hobj.all:
        all_sec_names.append(sec.name().split(".")[1][:4])

    if 'axon' not in all_sec_names:
        io.log_exception('There is no axonal recostruction in swc file.')
    else:
        beg1, end1, beg2, end2 = get_axon_direction(hobj)

    axon_diams = [hobj.axon[0].diam, hobj.axon[0].diam]
    for sec in hobj.all:
        section_name = sec.name().split(".")[1][:4]
        if section_name == 'axon':
            axon_diams[1] = sec.diam

    for sec in hobj.axon:
        h.delete_section(sec=sec)
    h.execute('create axon[2]', hobj)
    hobj.axon[0].connect(hobj.soma[0], 1.0, 0)
    hobj.axon[1].connect(hobj.axon[0], 1.0, 0)

    h.pt3dadd(beg1[0], beg1[1], beg1[2], axon_diams[0], sec=hobj.axon[0])
    h.pt3dadd(end1[0], end1[1], end1[2], axon_diams[0], sec=hobj.axon[0])
    hobj.all.append(sec=hobj.axon[0])
    h.pt3dadd(beg2[0], beg2[1], beg2[2], axon_diams[1], sec=hobj.axon[1])
    h.pt3dadd(end2[0], end2[1], end2[2], axon_diams[1], sec=hobj.axon[1])
    hobj.all.append(sec=hobj.axon[1])

    hobj.axon[0].L = 30.0
    hobj.axon[1].L = 30.0

    h.define_shape()

    for sec in hobj.axon:
        # io.log_info('sec.L: {}'.format(sec.L))
        if np.abs(30 - sec.L) > 0.0001:
            io.log_exception('Axon stub L is less than 30')


def get_axon_direction(hobj):
    for sec in hobj.somatic:
        n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
        soma_end = np.asarray([h.x3d(n3d - 1, sec=sec), h.y3d(n3d - 1, sec=sec), h.z3d(n3d - 1, sec=sec)])
        mid_point = int(n3d / 2)
        soma_mid = np.asarray([h.x3d(mid_point, sec=sec), h.y3d(mid_point, sec=sec), h.z3d(mid_point, sec=sec)])

    for sec in hobj.all:
        section_name = sec.name().split(".")[1][:4]
        if section_name == 'axon':
            n3d = int(h.n3d(sec=sec))  # get number of n3d points in each section
            axon_p3d = np.zeros((n3d, 3))  # to hold locations of 3D morphology for the current section
            for i in range(n3d):
                axon_p3d[i, 0] = h.x3d(i, sec=sec)
                axon_p3d[i, 1] = h.y3d(i, sec=sec)  # shift coordinates such to place soma at the origin.
                axon_p3d[i, 2] = h.z3d(i, sec=sec)

    # Add soma coordinates to the list
    p3d = np.concatenate(([soma_mid], axon_p3d), axis=0)

    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(p3d)
    unit_v = pca.components_[0]

    mag_v = np.sqrt(pow(unit_v[0], 2) + pow(unit_v[1], 2) + pow(unit_v[2], 2))
    unit_v[0] = unit_v[0] / mag_v
    unit_v[1] = unit_v[1] / mag_v
    unit_v[2] = unit_v[2] / mag_v

    # Find the direction
    axon_end = axon_p3d[-1] - soma_mid
    if np.dot(unit_v, axon_end) < 0:
        unit_v *= -1

    axon_seg_coor = np.zeros((4, 3))
    # unit_v = np.asarray([0,1,0])
    axon_seg_coor[0] = soma_end
    axon_seg_coor[1] = soma_end + (unit_v * 30.)
    axon_seg_coor[2] = soma_end + (unit_v * 30.)
    axon_seg_coor[3] = soma_end + (unit_v * 60.)

    return axon_seg_coor


nml_files = {}  # For caching neuroml file trees
def NMLLoad(cell, template_name, dynamic_params):
    """Convert a NEUROML file to a NEURON hoc cell object.

    Current limitations:
    * Ignores nml morphology section. You must pass in a swc file
    * Only for biophysically detailed cell biophysical components. All properties must be assigned to a segment group.

    :param cell:
    :param template_name:
    :param dynamic_params:
    :return:
    """
    # Last I checked there is no built in way to load a NML file directly into NEURON through the API, instead we have
    # to manually parse the nml file and build the NEUROM cell object section-by-section.
    morphology_file = cell.morphology_file
    hobj = h.Biophys1(str(morphology_file))
    # Depending on if the axon is cut before or after setting cell channels and mechanism can create drastically
    # different results. Currently NML files doesn't produce the same results if you use model_processing directives.
    # TODO: Find a way to specify model_processing directive with NML file
    fix_axon_peri(hobj)

    # Load the hoc template containing a swc initialized NEURON cell
    if template_name in nml_files:
        nml_params = nml_files[template_name]
    else:
        # Parse the NML parameters file xml tree and cache.
        biophys_dirs = cell.network.get_component('biophysical_neuron_models_dir')
        nml_path = os.path.join(biophys_dirs, template_name)
        nml_params = NMLTree(nml_path)
        nml_files[template_name] = nml_params

    # Iterate through the NML tree by section and use the properties to manually create cell mechanisms
    section_lists = [(sec, sec.name().split(".")[1][:4]) for sec in hobj.all]
    for sec, sec_name in section_lists:
        for prop_name, prop_obj in nml_params[sec_name].items():
            if prop_obj.element_tag() == 'resistivity':
                sec.Ra = prop_obj.value

            elif prop_obj.element_tag() == 'specificCapacitance':
                sec.cm = prop_obj.value

            elif prop_obj.element_tag() == 'channelDensity' and prop_obj.ion_channel == 'pas':
                sec.insert('pas')
                setattr(sec, 'g_pas', prop_obj.cond_density)
                for seg in sec:
                    seg.pas.e = prop_obj.erev

            elif prop_obj.element_tag() == 'channelDensity' or prop_obj.element_tag() == 'channelDensityNernst':
                sec.insert(prop_obj.ion_channel)
                setattr(sec, prop_obj.id, prop_obj.cond_density)
                if prop_obj.ion == 'na' and prop_obj:
                    sec.ena = prop_obj.erev
                elif prop_obj.ion == 'k':
                    sec.ek = prop_obj.erev

            elif prop_obj.element_tag() == 'concentrationModel':
                sec.insert(prop_obj.id)
                setattr(sec, 'gamma_' + prop_obj.type, prop_obj.gamma)
                setattr(sec, 'decay_' + prop_obj.type, prop_obj.decay)

    return hobj

def set_extracellular(hobj, cell, dynamics_params):
    for sec in hobj.all:
        sec.insert('extracellular')

    return hobj


add_cell_model(loadHOC, directive='hoc', model_type='biophysical')
add_cell_model(NMLLoad, directive='nml', model_type='biophysical')
add_cell_model(Biophys1, directive='ctdb:Biophys1', model_type='biophysical', overwrite=False)
add_cell_model(Biophys1, directive='ctdb:Biophys1.hoc', model_type='biophysical', overwrite=False)
add_cell_model(IntFire1, directive='nrn:IntFire1', model_type='point_process', overwrite=False)
add_cell_model(IntFire1, directive='nrn:IntFire1', model_type='point_neuron', overwrite=False)


add_cell_processor(aibs_perisomatic, overwrite=False)
add_cell_processor(aibs_allactive, overwrite=False)
add_cell_processor(aibs_perisomatic_directed, overwrite=False)
add_cell_processor(aibs_allactive_directed, overwrite=False)
add_cell_processor(set_extracellular, overwrite=False)
add_cell_processor(set_extracellular, 'extracellular', overwrite=False)
