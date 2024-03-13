COMMENT
A model (# 2 using terminology of Jung Hoon Lee) of a short-term synaptic plasticity.
The magnitude of the peak conductance is found by solving Eqs (3,5) in Hennig, 2013. "Theoretical models of synaptic short term plasticity. Frontiers in computational neuroscience, 7, p.45."

State variables:

	n = fraction of available vesicles 
	tau_r = time constant for vesicle replenishment

After each spike the magnitude of the peak conductance changes by the factor w*Pmax, where
w is the static synaptic weight and Pmax is the dynamically changing factor that could be interpreted as a probability of a transmitter release by the presynaptic terminal.

The post_synaptic dynamics of individual synaptic events is modeled by a single exponential synapse with the time constant tau_1

Implemented by Sergey L. Gratiy

ENDCOMMENT




NEURON {
	POINT_PROCESS stp2syn
	RANGE e, i, p0, tau_1, tau_r0, a_FDR, tau_FDR, Pmax
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
	(uS) = (microsiemens)

}

PARAMETER {
	: e = synaptic reversal potential
	e = 0 (mV)

	: p0  = baseline level of release probability
	p0 = 0.3

	: tau_1 = baseline level of release probability
	tau_1 = 5 (ms)

	: tau_r0 = baseline level of tau_r0
	tau_r0 = 1000 (ms)

	: tau_FDR = time constant for tau_r relaxation
	tau_FDR = 1000 (ms)

	: a_FDR = magnitude of tau_r reduction after each spike
	a_FDR = 0.5


}

ASSIGNED {
	v (mV)
	i (nA)
	Pmax 
}

STATE {
	n
	tau_r
	g
}

INITIAL {
	n=1
	tau_r=tau_r0
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
	Pmax = n*p0
}

DERIVATIVE state {
	g' = -g/tau_1
	n' = (1-n)/tau_r
	tau_r' = (tau_r0-tau_r)/tau_FDR
}

NET_RECEIVE(weight (umho)) {
	g = g + weight*Pmax
	n = n - n*p0
	tau_r = tau_r - a_FDR*tau_r

}


