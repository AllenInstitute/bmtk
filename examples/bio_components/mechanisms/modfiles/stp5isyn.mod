COMMENT
Current-based version of the conductance-based model stp5syn (refer to the stp5syn.mod)
In this model the synatic current is defined as i = g*e, whereas in the conductance based model i = g*(v - e).
Since the current is proportional to conductance, this model may be used to report the values of conductance by recording the current.

ENDCOMMENT


NEURON {
	POINT_PROCESS stp5isyn
	RANGE e, i, tau_1, tau_r0, a_FDR, tau_FDR, a_D, tau_D, a_i, tau_i, a_f, tau_f, pbtilde, Pmax
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

	: pbtilde = baseline level of pb
	pbtilde = 0.5 

}

ASSIGNED {
	v (mV)
	i (nA)
	Pmax
}

STATE {
	n
	p
	tau_r
	D
	pb
	g
}

INITIAL {
	n=1
	pb=pbtilde
	p=pb
	tau_r=tau_r0
	D=1
	g=0

}

BREAKPOINT {
	SOLVE state METHOD cnexp
:	i = g*(v - e)
	i = g*e
	Pmax = n*p*D

}

DERIVATIVE state {
	g' = -g/tau_1
	n' = (1-n)/tau_r
	p' = (pb-p)/tau_f
	tau_r' = (tau_r0-tau_r)/tau_FDR
	D' = (1-D)/tau_D
	pb' = (pbtilde-pb)/tau_i

}

NET_RECEIVE(weight (umho)) {
	g = g + weight*Pmax
	n = n - n*p
	p = p + a_f*(1-p)
	tau_r = tau_r - a_FDR*tau_r
	D = D - a_D*p*n*D
	pb = pb - a_i*pb

}


