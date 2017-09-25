from neuron import h
from bmtk.simulator.bionet.nrn import *

@synapse_model
def exp2syn(syn_params, xs, secs):
    '''
    Create a list of exp2syn synapses

    Parameters
    ----------
    syn_params: dict
        parameters of a synapse
    xs: float
        normalized distance along the section

    secs: hoc object
        target section

    Returns
    -------
    syns: synapse objects

    '''
    syns = []

    for x, sec in zip(xs, secs):
        syn = h.Exp2Syn(x, sec=sec)
        syn.e = syn_params['erev']
        syn.tau1 = syn_params['tau1']
        syn.tau2 = syn_params['tau2']
        syns.append(syn)
    return syns

@synapse_model
def Exp2Syn(syn_params, sec_x, sec_id):
    syn = h.Exp2Syn(sec_x, sec=sec_id)
    syn.e = syn_params['erev']
    syn.tau1 = syn_params['tau1']
    syn.tau2 = syn_params['tau2']
    return syn