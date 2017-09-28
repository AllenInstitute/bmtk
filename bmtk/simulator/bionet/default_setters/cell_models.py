# Allen Institute Software License - This software license is the 2-clause BSD license plus clause a third
# clause that prohibits redistribution for commercial purposes without further permission.
#
# Copyright 20XX. Allen Institute. All rights reserved.
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
# 3. Redistributions for commercial purposes are not permitted without the Allen Institute's written permission. For
# purposes of this license, commercial purposes is the incorporation of the Allen Institute's software into anything for
# which you will charge fees or other compensation. Contact terms@alleninstitute.org for commercial licensing
# opportunities.
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


def IntFire1(node):
    """Loads a point integrate and fire neuron"""
    model_params = node.model_params
    hobj = h.IntFire1()
    hobj.tau = model_params['tau']*1000.0  # Convert from seconds to ms.
    hobj.refrac = model_params['refrac']*1000.0  # Convert from seconds to ms.
    return hobj


def Biophys1(cell):
    """Loads a biophysical NEURON hoc object using Cell-Types database objects."""
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


def Biophys1_nml(json_file):
    # TODO: look at pgleeson examples to see how to convert .nml files
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


add_cell_model(Biophys1, 'biophysical', overwrite=False)
add_cell_model(IntFire1, 'point_IntFire1', overwrite=False)
add_cell_model(IntFire1, 'intfire', overwrite=False)