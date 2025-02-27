TITLE Calcium dependent potassium channel
: Implemented in Rubin and Cleland (2006) J Neurophysiology
: Parameters from Bhalla and Bower (1993) J Neurophysiology
: Adapted from /usr/local/neuron/demo/release/nachan.mod - squid
:   by Andrew Davison, The Babraham Institute  [Brain Res Bulletin, 2000]

:Suffix from Kca3 to Kca3_1

NEURON {
    THREADSAFE
	SUFFIX Kca3_1
	USEION k READ ek WRITE ik
	USEION ca READ cai
	RANGE gkbar, ik, Yconcdep, Yvdep
	RANGE Yalpha, Ybeta, tauY, Y_inf
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

CONSTANT {
	q10 = 3
}

PARAMETER {
	v (mV)
	dt (ms)
	gkbar= 0.120 (mho/cm2) <0,1e9>
	Ybeta = 0.05 (/ms)
	cai (mM) := 1e-5 (mM)
}


STATE {
	Y
}

ASSIGNED {
	ik (mA/cm2)
	Yalpha   (/ms)
	Yvdep    
	Yconcdep (/ms)
	tauY (ms)
	Y_inf
	ek (mV)

	qt
}

INITIAL {
	rate(v,cai)
	Y = Yalpha/(Yalpha + Ybeta)
	qt = q10^((celsius-37 (degC))/10 (degC))
}

BREAKPOINT {
	SOLVE state METHOD cnexp
	ik = gkbar*Y*(v - ek)
}

DERIVATIVE state {
	rate(v,cai)
	Y' = Yalpha*(1-Y) - Ybeta*Y
}

PROCEDURE rate(v(mV),cai(mM)) {
	vdep(v)
	concdep(cai)
	Yalpha = Yvdep*Yconcdep
	tauY = 1/(Yalpha + Ybeta)
	Y_inf = Yalpha/(Yalpha + Ybeta) /qt
}

PROCEDURE vdep(v(mV)) {
	TABLE Yvdep FROM -100 TO 100 WITH 100
	Yvdep = exp((v*1(/mV)+70)/27)
}

PROCEDURE concdep(cai(mM)) {
	TABLE Yconcdep FROM 0 TO 0.01 WITH 1000
	if (cai < 0.01) {
		Yconcdep = 500(/ms)*( 0.015-cai*1(/mM) )/( exp((0.015-cai*1(/mM))/0.0013) -1 )
	} else {
		Yconcdep = 500(/ms)*0.005/( exp(0.005/0.0013) -1 )
	}
}
