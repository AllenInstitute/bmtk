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
from neuron import h

from bmtk.simulator.bionet.pyfunction_cache import add_synapse_model
from bmtk.simulator.bionet.nrn import *


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



@synapse_model
def stp1syn(syn_params, xs, secs):
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.stp1syn(x, sec=sec)

        syn.e = syn_params["erev"]
        syn.p0 = 0.5
        syn.tau_r = 200
        syn.tau_1 = 5
        syns.append(syn)

    return syns


@synapse_model
def stp2syn(syn_params, x, sec):
    syn = h.stp2syn(x, sec=sec)
    syn.e = syn_params["erev"]
    syn.p0 = syn_params["p0"]
    syn.tau_r0 = syn_params["tau_r0"]
    syn.tau_FDR = syn_params["tau_FDR"]
    syn.tau_1 = syn_params["tau_1"]
    return syn


@synapse_model
def stp3syn(syn_params, xs, secs):
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.stp3syn(x, sec=sec) # temporary
        syn.e = syn_params["erev"]
        syn.p0 = 0.6
        syn.tau_r0 = 200
        syn.tau_FDR = 2000
        syn.tau_D = 500
        syn.tau_1 = 5
        syns.append(syn)

    return syns


@synapse_model
def stp4syn(syn_params, xs, secs):
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.stp4syn(x, sec=sec)
        syn.e = syn_params["erev"]
        syn.p0 = 0.6
        syn.tau_r = 200
        syn.tau_1 = 5
        syns.append(syn)

    return syns


@synapse_model
def stp5syn(syn_params, x, sec):  # temporary
    syn = h.stp5syn(x, sec=sec)
    syn.e = syn_params["erev"]
    syn.tau_1 = syn_params["tau_1"]
    syn.tau_r0 = syn_params["tau_r0"]
    syn.tau_FDR = syn_params["tau_FDR"]
    syn.a_FDR = syn_params["a_FDR"]
    syn.a_D = syn_params["a_D"]
    syn.a_i = syn_params["a_i"]
    syn.a_f = syn_params["a_f"]
    syn.pbtilde = syn_params["pbtilde"]
    return syn


def stp5isyn(syn_params, xs, secs): # temporary
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.stp5isyn(x, sec=sec)
        syn.e = syn_params["erev"]
        syn.tau_1 = syn_params["tau_1"]
        syn.tau_r0 = syn_params["tau_r0"]
        syn.tau_FDR = syn_params["tau_FDR"]
        syn.a_FDR = syn_params["a_FDR"]
        syn.a_D = syn_params["a_D"]
        syn.a_i = syn_params["a_i"]
        syn.a_f = syn_params["a_f"]
        syn.pbtilde = syn_params["pbtilde"]
        syns.append(syn)

    return syns


@synapse_model
def tmgsyn(syn_params, xs, secs):
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.tmgsyn(x, sec=sec)
        syn.e = syn_params["erev"]
        syn.tau_1 = syn_params["tau_1"]
        syn.tau_rec = syn_params["tau_rec"]
        syn.tau_facil = syn_params["tau_facil"]
        syn.U = syn_params["U"]
        syn.u0 = syn_params["u0"]
        syns.append(syn)

    return syns


@synapse_model
def expsyn(syn_params, x, sec):
    """Create a list of expsyn synapses

    :param syn_params: parameters of a synapse (dict)
    :param x: normalized distance along the section (float)
    :param sec: target section (hoc object)
    :return: synapse objects
    """
    syn = h.ExpSyn(x, sec=sec)
    syn.e = syn_params['erev']
    syn.tau = syn_params["tau1"]
    return syn


@synapse_model
def exp1syn(syn_params, xs, secs):
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.exp1syn(x, sec=sec)
        syn.e = syn_params['erev']
        syn.tau = syn_params["tau_1"]
        syns.append(syn)
    return syns


@synapse_model
def exp1isyn(syn_params, xs, secs):
    syns = []
    for x, sec in zip(xs, secs):
        syn = h.exp1isyn(x, sec=sec)
        syn.e = syn_params['erev']
        syn.tau = syn_params["tau_1"]
        syns.append(syn)
    return syns


add_synapse_model(Exp2Syn, 'exp2syn', overwrite=False)
add_synapse_model(Exp2Syn, overwrite=False)
