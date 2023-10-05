COMMENT
A model (# 5 using terminology of Jung Hoon Lee) of a short-term synaptic plasticity.
The magnitude of the peak conductance is found by solving Eqs (3,4,5,8,10) in Hennig, 2013. "Theoretical models of synaptic short term plasticity. Frontiers in computational neuroscience, 7, p.45."
This file could be used to simulate simpler models by setting to zero the parameters in the unused equations(e.g, a_D,a_i,a_f).

State variables:

	n = fraction of available vesicles 
	D = fraction of non-desensitized receptors
	tau_r = time constant for vesicle replenishment
	pb = baseline vesicle release probability
	p = vesicle release probability

After each spike the magnitude of the peak conductance changes by the factor w*Pmax, where w is the static synaptic weight and Pmax is the activity-dependent factor that could be interpreted as a probability of a transmitter release by the presynaptic terminal.

The post_synaptic dynamics of individual synaptic events is modeled by a single exponential synapse with the time constant tau_1.

Implemented by Sergey L. Gratiy,

ENDCOMMENT

: Declaring parameters as RANGE allows them to change as a function of a position alogn the cell
NEURON {
	POINT_PROCESS stp5syn
	RANGE e, i, tau_1, tau_r0, a_FDR, tau_FDR, a_D, tau_D, a_i, tau_i, a_f, tau_f, pbtilde, Pmax
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

: Declaration of the default values of parameters
PARAMETER {
	: e = synaptic reversal potential
	e = 0 (mV)

	: tau_1 = baseline level of release probability
	tau_1 = 10 (ms)

	: tau_r0 = baseline level of tau_r0
	tau_r0 = 1000 (ms)

	: tau_FDR = time constant for tau_r relaxation
	tau_FDR = 1000 (ms)

	: a_FDR = amount of tau_r reduction after each spike
	a_FDR = 0.5

	: tau_D = relaxation time of D 
	tau_D = 100 (ms) 

	: a_D = amount of desentization
	a_D = 0.5 

	: tau_i = relaxation time for p0 
	tau_i = 100 (ms) 

	: a_i = amount of decrease of baseline probability after each spike 
	a_i = 0.5 

	: tau_f = facilitation time constant (relaxation time constant for p) 
	tau_f = 10 (ms) 

	: a_f = amount of facilitation  (increase of p after each spike)
	a_f = 0.5 

	: pbtilde = baseline level of p0
	pbtilde = 0.5 

}
: Declaration of dependent and external variables that collectively are called ASSIGNED
ASSIGNED {
	v (mV)
	i (nA)
	Pmax
}

: Declaration of the state variables
STATE {
	n
	p
	tau_r
	D
	pb
	g
}

: Initial conditions for the state variables
INITIAL {
	n=1
	p=1
	tau_r=tau_r0
	D=1
	pb=pbtilde
	g=0

}

: Integration method + assignment statements
BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
	Pmax = n*p*D
}

: Definition of teh dynamics between the presynaptic activations
DERIVATIVE state {
	n' = (1-n)/tau_r
	p' = (pb-p)/tau_f
	tau_r' = (tau_r0-tau_r)/tau_FDR
	D' = (1-D)/tau_D
	pb' = (pbtilde-pb)/tau_i
	g' = -g/tau_1

}

: This block defines what happens to the state variables at the moment presynaptic activation
NET_RECEIVE(weight (umho)) {
	g = g + weight*Pmax
	n = n - n*p
	p = p + a_f*(1-p)
	tau_r = tau_r - a_FDR*tau_r
	D = D - a_D*p*n*D
	pb = pb - a_i*pb

}


