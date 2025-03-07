TITLE Voltage-gated low threshold potassium current from Kv1 subunits

COMMENT

NEURON implementation of a potassium channel from Kv1.1 subunits
Kinetical scheme: Hodgkin-Huxley m^4, no inactivation

Experimental data taken from:
Human Kv1.1 expressed in xenopus oocytes: Zerr et al., J Neurosci 18, 2842, 2848, 1998
Vhalf = -28.8 +- 2.3 mV; k = 8.1+- 0.9 mV

The voltage dependency of the rate constants was approximated by:

alpha = ca * exp(-(v+cva)/cka)
beta = cb * exp(-(v+cvb)/ckb)

Parameters ca, cva, cka, cb, cvb, ckb
were determined from least square-fits to experimental data of G/Gmax(v) and tau(v).
Values are defined in the CONSTANT block.
Model includes calculation of Kv gating current

Reference: Akemann et al., Biophys. J. (2009) 96: 3959-3976

Laboratory for Neuronal Circuit Dynamics
RIKEN Brain Science Institute, Wako City, Japan
http://www.neurodynamics.brain.riken.jp

Date of Implementation: April 2007
Contact: akemann@brain.riken.jp

Suffix from Kv1 to Kv1_1

ENDCOMMENT


NEURON {
    THREADSAFE
	SUFFIX Kv1_1
	USEION k READ ek WRITE ik
	NONSPECIFIC_CURRENT i
	RANGE g, gbar, ik, i , igate, nc
	RANGE ninf, taun
	RANGE gateCurrent, gunit
}

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(nA) = (nanoamp)
	(pA) = (picoamp)
	(S)  = (siemens)
	(nS) = (nanosiemens)
	(pS) = (picosiemens)
	(um) = (micron)
	(molar) = (1/liter)
	(mM) = (millimolar)		
}

CONSTANT {
	e0 = 1.60217646e-19 (coulombs)
	q10 = 2.7

	ca = 0.12889 (1/ms)
	cva = 45 (mV)
	cka = -33.90877 (mV)

	cb = 0.12889 (1/ms)
      cvb = 45 (mV)
	ckb = 12.42101 (mV) 

	zn = 2.7978 (1)		: valence of n-gate         
}

PARAMETER {
	gateCurrent = 0 (1)	: gating currents ON = 1 OFF = 0
	
	gbar = 0.004 (S/cm2)   <0,1e9>
	gunit = 16 (pS)		: unitary conductance 
}


ASSIGNED {
 	celsius (degC)
	v (mV)	

	ik (mA/cm2)
	i (mA/cm2)
	igate (mA/cm2) 
	ek (mV)
	g  (S/cm2)
	nc (1/cm2)			: membrane density of channel
	
	ninf (1)
	taun (ms)
	alphan (1/ms)
	betan (1/ms)
	qt (1)
}

STATE { n }

INITIAL {
	nc = (1e12) * gbar / gunit
	qt = q10^((celsius-22 (degC))/10 (degC))
	rates(v)
	n = ninf
}

BREAKPOINT {
	SOLVE states METHOD cnexp
      g = gbar * n^4 
	ik = g * (v - ek)
	igate = nc * (1e6) * e0 * 4 * zn * ngateFlip()

	if (gateCurrent != 0) { 
		i = igate
	}
}

DERIVATIVE states {
	rates(v)
	n' = (ninf-n)/taun 
}

PROCEDURE rates(v (mV)) {
	alphan = alphanfkt(v)
	betan = betanfkt(v)
	ninf = alphan/(alphan+betan) 
	taun = 1/(qt*(alphan + betan))       
}

FUNCTION alphanfkt(v (mV)) (1/ms) {
	alphanfkt = ca * exp(-(v+cva)/cka) 
}

FUNCTION betanfkt(v (mV)) (1/ms) {
	betanfkt = cb * exp(-(v+cvb)/ckb)
}

FUNCTION ngateFlip() (1/ms) {
	ngateFlip = (ninf-n)/taun 
}



