TITLE Low threshold calcium current
:
:   Ca++ current responsible for low threshold spikes (LTS)
:   RETICULAR THALAMUS
:   Differential equations
:
:   Model of Huguenard & McCormick, J Neurophysiol 68: 1373-1383, 1992.
:   
:   Written by Alain Destexhe, Salk Institute, Sept 18, 1992
:   
:    - Biophysical properties of the T current were from recordings of
:    - human recombinant Cav3.2 T-channel in HEK-293 cells
:    - see Vitko et al., J. Neurosci 25(19) :4844-4855, 2005
:    - Q10 and shift parameters are fixed 
:   
:
:   Suffix from CaT3_2 to Cav3_2

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX Cav3_2
	USEION ca READ cai, cao WRITE ica
	RANGE gcabar, m_inf, tau_m, h_inf, tau_h, shift, i,ica
}

UNITS {
	(molar) = (1/liter)
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)

	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	v		(mV)
	celsius	= 36	(degC)
:	eca	= 120	(mV)
	gcabar	= .0008	(mho/cm2)
	shift	= 0 	(mV)
	cai	= 2.4e-4 (mM)		: adjusted for eca=120 mV
	cao	= 2	(mM)
}

STATE {
	m h
}

ASSIGNED {
	ica	(mA/cm2)
	carev	(mV)
	m_inf
	tau_m	(ms)
	h_inf
	tau_h	(ms)
	phi_m
	phi_h
	i	(mA/cm2)
}

BREAKPOINT {
	SOLVE castate METHOD cnexp
	carev = (1e3) * (R*(celsius+273.15))/(2*FARADAY) * log (cao/cai)
	ica = gcabar * m*m*h * (v-carev)
	i = ica		: diagnostic i added to display the current
}

DERIVATIVE castate {
	evaluate_fct(v)

	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
}

UNITSOFF
INITIAL {
:
:   Activation functions and kinetics were obtained from
:   Vitko et al., 2005 at 23-25 deg.
:   Transformation to 36 deg assuming Q10 of 5 and 3 for m and h
:   (as in Coulter et al., J Physiol 414: 587, 1989)
:
	phi_m = 5 ^ (12/10)
	phi_h = 3 ^ (12/10)

	evaluate_fct(v)

	m = m_inf
	h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) { 
:

	m_inf = 1.0 / ( 1 + exp(-(v+shift+54.8)/7.4) )
	h_inf = 1.0 / ( 1 + exp((v+shift+85.5)/7.18) )

	tau_m = ( 1.9 + 1.0 / ( exp((v+shift+37.0)/11.9) + exp(-(v+shift+131.6)/21) ) ) / phi_m
	tau_h = 13.7 + (1942 + exp((v+shift+164)/9.2)) / (1 + exp((v+shift+89.3)/3.7) ) / phi_h
}
UNITSON
