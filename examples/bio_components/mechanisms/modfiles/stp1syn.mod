COMMENT
A model (# 1 using terminology of Jung Hoon Lee) of a short-term synaptic plasticity.
The magnitude of the peak conductance is found by solving Eqs (3) in Hennig, 2013. "Theoretical models of synaptic short term plasticity. Frontiers in computational neuroscience, 7, p.45."

State variables:

	n = fraction of available vesicles 

After each spike the magnitude of the peak conductance changes by the factor w*Pmax, where
w is the static synaptic weight and Pmax is the dynamically changing factor that could be interpreted as a probability of a transmitter release by the presynaptic terminal.

The post_synaptic dynamics of individual synaptic events is modeled by a single exponential synapse with the time constant tau_1

Implemented by Sergey L. Gratiy

ENDCOMMENT


NEURON {
	POINT_PROCESS stp1syn
	RANGE e, i, tau_r, p0, tau_1
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
	: tau_r = recovery time constant
	tau_r = 100 (ms)
	: p0  = baseline level of release probability
	p0 = 0.3
	: tau_1 = baseline level of release probability
	tau_1 = 10 (ms)
}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE {
	n
	g
}

INITIAL {
	n=1
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}

DERIVATIVE state {
	n' = (1-n)/tau_r
	g' = -g/tau_1

}

NET_RECEIVE(weight (umho)) {
	g = g + weight*n*p0
	n = n - n*p0
		

}


