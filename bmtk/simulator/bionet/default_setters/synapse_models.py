from neuron import h

from bmtk.simulator.bionet.pyfunction_cache import add_synapse_model


def exp2syn(syn_params, xs, secs):
    """Create a list of exp2syn synapses

    :param syn_params: parameters of a synapse
    :param xs: list of normalized distances along the section
    :param secs: target sections
    :return: list of NEURON synpase objects
    """
    syns = []

    for x, sec in zip(xs, secs):
        syn = h.Exp2Syn(x, sec=sec)
        syn.e = syn_params['erev']
        syn.tau1 = syn_params['tau1']
        syn.tau2 = syn_params['tau2']
        syns.append(syn)
    return syns


def Exp2Syn(syn_params, sec_x, sec_id):
    """Create a list of exp2syn synapses

    :param syn_params: parameters of a synapse
    :param sec_x: normalized distance along the section
    :param sec_id: target section
    :return: NEURON synapse object
    """
    syn = h.Exp2Syn(sec_x, sec=sec_id)
    syn.e = syn_params['erev']
    syn.tau1 = syn_params['tau1']
    syn.tau2 = syn_params['tau2']
    return syn


add_synapse_model(exp2syn, overwrite=False)
add_synapse_model(Exp2Syn, overwrite=False)