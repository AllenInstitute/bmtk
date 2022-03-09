COMMENT
Current-based version of the conductance-based model exp1syn (refer to the exp1syn.mod)
In this model the synatic current is defined as i = g*e, whereas in the conductance based model i = g*(v - e).
Since the current is proportional to conductance, this model may be used to report the values of conductance by recording the current.


Implemented by Sergey L. Gratiy

ENDCOMMENT

NEURON {
	POINT_PROCESS exp1isyn
	RANGE tau, e, i
	NONSPECIFIC_CURRENT i
}

UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
}

PARAMETER {
	tau = 0.1 (ms) <1e-9,1e9>
	e = 0	(mV)
}

ASSIGNED {
	v (mV)
	i (nA)
}

STATE {
	g (uS)
}

INITIAL {
	g=0
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*e
}

DERIVATIVE state {
	g' = -g/tau
}

NET_RECEIVE(weight (uS)) {
	g = g + weight
}
