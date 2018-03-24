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
import json
from neuron import h

from bmtk.simulator.bionet.pyfunction_cache import add_cell_model

"""
Functions for loading NEURON cell objects.

Functions will be loaded by bionetwork and called when a new cell object is created. These are for standard models
loaded with Cell-Types json files or their NeuroML equivelent, but may be overridden by the users.
"""


def IntFire1(cell, template_name, dynamics_params):
    """Loads a point integrate and fire neuron"""
    #model_params = cell.model_params
    hobj = h.IntFire1()
    hobj.tau = dynamics_params['tau']*1000.0  # Convert from seconds to ms.
    hobj.refrac = dynamics_params['refrac']*1000.0  # Convert from seconds to ms.
    return hobj


def Biophys1(cell, template_name, dynamic_params):
    """Loads a biophysical NEURON hoc object using Cell-Types database objects."""
    morphology_file = cell['morphology_file']
    hobj = h.Biophys1(str(morphology_file))
    fix_axon(hobj)
    set_params_peri(hobj, dynamic_params)
    return hobj
    '''
    print cell['dynamics_params']

    if isinstance(cell.model_params, dict):
        # load directly from the dictionary.
        return Biophys1_dict(cell)

    elif isinstance(cell.model_params, basestring):
        # see if the model_params is a NeuroML or json file and load thoses
        file_ext = cell.model_params[-4:].lower()
        if file_ext == '.xml' or file_ext == '.nml':
            return Biophys1_nml(cell.model_params)
        elif file_ext == 'json':
            return Biophys1_dict(json.load(open(cell.model_params, 'r')))
    else:
        raise Exception('Biophys1: Was unable to determin model params type for {}'.format(cell.model_params))
    '''

def Biophys1_nml(json_file):
    # TODO: look at examples to see how to convert .nml files
    raise NotImplementedError()


def Biophys1_dict(cell):
    """ Set parameters for cells from the Allen Cell Types database Prior to setting parameters will replace the
    axon with the stub
    """
    morphology_file = cell['morphology_file']
    hobj = h.Biophys1(str(morphology_file))
    fix_axon(hobj)
    set_params_peri(hobj, cell.model_params)
    return hobj


def Biophys1_adjusted(cell_prop):
    morphology_file_name = str(cell_prop['morphology_file'])
    hobj = h.Biophys1(morphology_file_name)
    fix_axon_allactive(hobj)
    # set_params_peri(hobj, cell_prop.model_params)
    set_params(hobj, cell_prop.model_params)
    return hobj


def set_params(hobj, params_dict):
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
        print('Setting cm')
        for cm_dict in passive['cm']:
            cm = cm_dict['cm']
            for sec in section_map.get(cm_dict['section'], []):
                sec.cm = cm

    for genome_dict in genome:
        g_section = genome_dict['section']
        if genome_dict['section'] == 'glob':
            print("WARNING: There is a section called glob, probably old json file")
            continue

        g_value = float(genome_dict['value'])
        g_name = genome_dict['name']
        g_mechanism = genome_dict.get("mechanism", "")
        for sec in section_map.get(g_section, []):
            if g_mechanism != "":
                sec.insert(g_mechanism)
            setattr(sec, g_name, g_value)

        print('setting {} to {} in {}'.format(g_name, g_value, g_section))

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
            print("Warning: can't set erev for {}, section array doesn't exist".format(erev_section))


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


def fix_axon(hobj):
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


#add_cell_model(Biophys1, directive='ctdb', model_type='biophysical', overwrite=False)
add_cell_model(Biophys1, directive='ctdb:Biophys1', model_type='biophysical', overwrite=False)
add_cell_model(Biophys1, directive='ctdb:Biophys1.hoc', model_type='biophysical', overwrite=False)
add_cell_model(IntFire1, directive='nrn:IntFire1', model_type='point_process', overwrite=False)
#add_cell_model(Biophys1, overwrite=False)
#add_cell_model(Biophys1_adjusted, 'biophysical_adjusted', overwrite=False)
#add_cell_model(Biophys1_adjusted, overwrite=False)
#add_cell_model(IntFire1, 'point_IntFire1', overwrite=False)
#add_cell_model(IntFire1, 'intfire', overwrite=False)
#add_cell_model(IntFire1, overwrite=False)
