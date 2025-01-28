import json
from neuron import h


def IntFire1(cell_prop):
    """Set parameters for the IntFire1 cell models."""
    params_file = cell_prop['params_file']

    with open(params_file) as params_file:
        params = json.load(params_file)

    hobj = h.IntFire1()
    hobj.tau = params['tau'] * 1000.0 # Convert from seconds to ms.
    hobj.refrac = params['refrac'] * 1000.0 # Convert from seconds to ms.

    return hobj


def Biophys1(cell_prop):
    """
    Set parameters for cells from the Allen Cell Types database
    Prior to setting parameters will replace the axon with the stub
    """
    morphology_file_name = str(cell_prop['morphology'])
    params_file_name = str(cell_prop['params_file'])

    hobj = h.Biophys1(morphology_file_name)
    fix_axon(hobj)
    set_params_peri(hobj, params_file_name)

    return hobj


def set_params_peri(hobj, params_file_name):
    """Set biophysical parameters for the cell

    Parameters
    ----------
    hobj: instance of a Biophysical template
        NEURON's cell object
    params_file_name: string
        name of json file containing biophysical parameters for cell's model which determine spiking behavior
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


def fix_axon(hobj):
    '''
    Replace reconstructed axon with a stub

    Parameters
    ----------
    hobj: instance of a Biophysical template
        NEURON's cell object
    '''

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




