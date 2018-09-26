COMMENT
A model (# 3 using terminology of Jung Hoon Lee) of a short-term synaptic plasticity.
The magnitude of the peak conductance is found by solving Eqs (3,5,10) in Hennig, 2013. "Theoretical models of synaptic short term plasticity. Frontiers in computational neuroscience, 7, p.45."

State variables:

	n = fraction of available vesicles 
	D = fraction of non-desensitized receptors
	tau_r = time constant for vesicle replenishment

After each spike the magnitude of the peak conductance changes by the factor w*Pmax, where
w is the static synaptic weight and Pmax is the dynamically changing factor that could be interpreted as a probability of a transmitter release by the presynaptic terminal.

The post_synaptic dynamics of individual synaptic events is modeled by a single exponential synapse with the time constant tau_1

Implemented by Sergey L. Gratiy

ENDCOMMENT


NEURON {
	POINT_PROCESS stp3syn
	RANGE e, i, p0, tau_1, tau_r0, a_FDR, tau_FDR, a_D, tau_D
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	: e = synaptic reversal potential
	e = 0 (mV)

	: p0  = baseline level of release probability
	p0 = 0.3

	: tau_f = facilitation time constant
	tau_f = 1000 (ms) < 0, 1e9 >

	: tau_1 = baseline level of release probability
	tau_1 = 10 (ms)

	: tau_r0 = baseline level of tau_r0
	tau_r0 = 1000 (ms)

	: tau_FDR = time constant for tau_r relaxation
	tau_FDR = 1000 (ms)

	: a_FDR = magnitude of tau_r reduction after each spike
	a_FDR = 0.5

	: tau_D = relaxation time of D 
	tau_D = 100 (ms) 

	: a_D 
	a_D = 0.5 

}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE {
	n
	tau_r
	D
	g
}

INITIAL {
	n=1
	tau_r=tau_r0
	g=0
	D=1
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}

DERIVATIVE state {
	n' = (1-n)/tau_r
	g' = -g/tau_1
	tau_r' = (tau_r0-tau_r)/tau_FDR
	D' = (1-D)/tau_D
}

NET_RECEIVE(weight (umho)) {
	g = g + weight*n*p0*D
	n = n - n*p0
	tau_r = tau_r - a_FDR*tau_r
	D = D - a_D*p0*n*D

}


