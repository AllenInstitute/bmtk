TITLE Voltage-gated potassium channel from Kv3 subunits

COMMENT
Voltage-gated potassium channel with high threshold and fast activation/deactivation kinetics

KINETIC SCHEME: Hodgkin-Huxley (n^4)
n'= alpha * (1-n) - betha * n
g(v) = gbar * n^4 * ( v-ek )

The rate constants of activation (alpha) and deactivation (beta) were approximated by:

alpha(v) = ca * exp(-(v+cva)/cka)
beta(v) = cb * exp(-(v+cvb)/ckb)

Parameters can, cvan, ckan, cbn, cvbn, ckbn are given in the CONSTANT block.
Values derive from least-square fits to experimental data of G/Gmax(v) and taun(v) in Martina et al. J Neurophys. 97 (563-671, 2007.
Model includes a calculation of Kv gating current

Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

Laboratory for Neuronal Circuit Dynamics
RIKEN Brain Science Institute, Wako City, Japan
http://www.neurodynamics.brain.riken.jp

Date of Implementation: April 2007
Contact: akemann@brain.riken.jp

Suffix from Kv3 to Kv3_3

ENDCOMMENT


NEURON {
    THREADSAFE
	SUFFIX Kv3_3
	USEION k READ ek WRITE ik
	NONSPECIFIC_CURRENT i
	RANGE gbar, g, ik, i, igate, nc
	RANGE ninf, taun
	RANGE gateCurrent, gunit
}

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(nA) = (nanoamp)
	(pA) = (picoamp)
	(S)  = (siemens)
	(mS) = (millisiemens)
	(nS) = (nanosiemens)
	(pS) = (picosiemens)
	(um) = (micron)
	(molar) = (1/liter)
	(mM) = (millimolar)		
}

CONSTANT {
	e0 = 1.60217646e-19 (coulombs)
	q10 = 2.7

	ca = 0.22 (1/ms)
	cva = 16 (mV)
	cka = -26.5 (mV)
	cb = 0.22 (1/ms)
	cvb = 16 (mV)
	ckb = 26.5 (mV)
	
	zn = 1.9196 (1)		: valence of n-gate
}

PARAMETER {
	gateCurrent = 0 (1)	: gating currents ON = 1 OFF = 0
	
	gbar = 0.005 (S/cm2)   <0,1e9>
	gunit = 16 (pS)		: unitary conductance 
}

ASSIGNED {
	celsius (degC)
	v (mV)
	
	ik (mA/cm2)
	igate (mA/cm2)
	i (mA/cm2)
 
	ek (mV)
	g (S/cm2)
	nc (1/cm2)
	qt (1)

	ninf (1)
	taun (ms)
	alpha (1/ms)
	beta (1/ms)
}

STATE { n }

INITIAL {
	nc = (1e12) * gbar / gunit
	qt = q10^((celsius-22 (degC))/10 (degC))
	rateConst(v)
	n = ninf
}

BREAKPOINT {
	SOLVE state METHOD cnexp
      g = gbar * n^4 
	ik = g * (v - ek)
	igate = nc * (1e6) * e0 * 4 * zn * ngateFlip()

	if (gateCurrent != 0) { 
		i = igate
	}
}

DERIVATIVE state {
	rateConst(v)
	n' = alpha * (1-n) - beta * n
}

PROCEDURE rateConst(v (mV)) {
	alpha = qt * alphaFkt(v)
	beta = qt * betaFkt(v)
	ninf = alpha / (alpha + beta) 
	taun = 1 / (alpha + beta)
}

FUNCTION alphaFkt(v (mV)) (1/ms) {
	alphaFkt = ca * exp(-(v+cva)/cka) 
}

FUNCTION betaFkt(v (mV)) (1/ms) {
	betaFkt = cb * exp(-(v+cvb)/ckb)
}

FUNCTION ngateFlip() (1/ms) {
	ngateFlip = (ninf-n)/taun 
}


