TITLE Medium duration Ca-dependent potassium current
:
:   Ca++ dependent K+ current IC responsible for medium duration AHP
:
:   Original file written by Alain Destexhe, Salk Institute, Nov 3, 1992
:   Modified by Geir Halnes, Norwegian University of Life Sciences, Mar 13, 2011


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX iahp
	USEION k READ ek WRITE ik VALENCE 1
	USEION Ca READ Cai VALENCE 2
      RANGE gkbar, g, minf, taum
	GLOBAL beta, cac, m_inf, tau_m, x
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
}


PARAMETER {
	v (mV)
	ek = -90 (mV)
	celsius = 36 (degC)
	Cai 	= 5e-5 (mM)			: Initial [Ca]i = 50 nM (Cai is simulated by separate mod-file)
	gkbar	= 1.3e-4	(mho/cm2)	: Conductance (modified from hoc-file)
	beta	= 0.02	(1/ms)	: Backward rate constant
	cac	= 4.3478e-4(mM)		: Middle point of m_inf fcn
	taumin	= 1	(ms)		: Minimal value of the time cst
      x       = 2				: Binding cites
}




STATE {
	m
}


ASSIGNED {
	ik 	(mA/cm2)
	g       (mho/cm2)
	m_inf
	tau_m	(ms)
	minf
      taum
	tadj
}


BREAKPOINT { 
	SOLVE states METHOD cnexp
        minf = m_inf
        taum = tau_m
	  g = gkbar*m*m
	  ik = g * (v - ek)
}

DERIVATIVE states { 
	evaluate_fct(v,Cai)
	m' = (m_inf - m) / tau_m
}


UNITSOFF
INITIAL {
:  activation kinetics are assumed to be at 22 deg. C
:  Q10 is assumed to be 3

	VERBATIM
	Cai = _ion_Cai;
	ENDVERBATIM

	tadj = 3 ^ ((celsius-22.0)/10)
	evaluate_fct(v,Cai)
	m = m_inf
      minf = m_inf
      taum = tau_m
}

PROCEDURE evaluate_fct(v(mV),Cai(mM)) {  LOCAL car, tcar
	car = (Cai/cac)^x
	m_inf = car / ( 1 + car )
	tau_m = 1 / beta / (1 + car) / tadj
      if(tau_m < taumin) { tau_m = taumin } 	: min value of time cst
}

UNITSON
