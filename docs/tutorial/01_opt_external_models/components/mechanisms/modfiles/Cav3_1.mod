TITLE Low threshold calcium current Cerebellum Purkinje Cell Model

COMMENT

Kinetics adapted to fit the Cav3.1 Iftinca et al 2006, Temperature dependence of T-type Calcium channel gating, NEUROSCIENCE

Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*

*Article available as Open Access

PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513

Written by Haroon Anwar, Computational Neuroscience Unit, Okinawa Institute of Science and Technology, 2010.
Contact: Haroon Anwar (anwar@oist.jp)

Suffix from CaT3_1 to CaV3_1

ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
        SUFFIX Cav3_1
        USEION ca READ cai, cao WRITE ica VALENCE 2
        RANGE g, pcabar, minf, taum, hinf, tauh
	RANGE ica, m ,h

    }

UNITS {
        (molar) = (1/liter)
        (mV) =  (millivolt)
        (mA) =  (milliamp)
        (mM) =  (millimolar)

}

CONSTANT {
	F = 9.6485e4 (coulombs)
	R = 8.3145 (joule/kelvin)
	q10 = 3
}

PARAMETER {
        v               (mV)
        celsius (degC)
        eca (mV)
	pcabar  = 2.5e-4 (cm/s)
        cai  (mM)           : adjusted for eca=120 mV
	cao  (mM)
	
	v0_m_inf = -52 (mV)
	v0_h_inf = -72 (mV)
	k_m_inf = -5 (mV)
	k_h_inf = 7  (mV)
	
	C_tau_m = 1
	A_tau_m = 1.0
	v0_tau_m1 = -40 (mV)
	v0_tau_m2 = -102 (mV)
	k_tau_m1 = 9 (mV)
	k_tau_m2 = -18 (mV)
	
	C_tau_h = 15
	A_tau_h = 1.0
	v0_tau_h1 = -32 (mV)
	k_tau_h1 = 7 (mV)
	
    }
    

STATE {
        m h
}

ASSIGNED {
        ica     (mA/cm2)
	g        (coulombs/cm3) 
        minf
        taum   (ms)
        hinf
        tauh   (ms)
	T (kelvin)
	E (volt)
	zeta
	qt
}

BREAKPOINT {
	SOLVE castate METHOD cnexp 

        ica = (1e3) *pcabar*m*m *h * g
}

DERIVATIVE castate {
        evaluate_fct(v)

        m' = (minf - m) / taum
        h' = (hinf - h) / tauh
}

FUNCTION ghk( v (mV), ci (mM), co (mM), z )  (coulombs/cm3) {
    E = (1e-3) * v
      zeta = (z*F*E)/(R*T)


    if ( fabs(1-exp(-zeta)) < 1e-6 ) {
        ghk = (1e-6) * (z*F) * (ci - co*exp(-zeta)) * (1 + zeta/2)
    } else {
        ghk = (1e-6) * (z*zeta*F) * (ci - co*exp(-zeta)) / (1-exp(-zeta))
    }
}


UNITSOFF
INITIAL {
	
	T = kelvinfkt (celsius)

        evaluate_fct(v)
        m = minf
        h = hinf
	qt = q10^((celsius-37 (degC))/10 (degC))
}

PROCEDURE evaluate_fct(v(mV)) { 

        minf = 1.0 / ( 1 + exp((v  - v0_m_inf)/k_m_inf) )
        hinf = 1.0 / ( 1 + exp((v - v0_h_inf)/k_h_inf) )
        if (v<=-90) {
	taum = 1
	} else {
	taum = ( C_tau_m + A_tau_m / (exp((v - v0_tau_m1)/ k_tau_m1) + exp((v - v0_tau_m2)/k_tau_m2))) / qt
	}
	tauh = ( C_tau_h + A_tau_h / exp((v - v0_tau_h1)/k_tau_h1) ) / qt
	g = ghk(v, cai, cao, 2)
}

FUNCTION kelvinfkt( t (degC) )  (kelvin) {
    kelvinfkt = 273.19 + t
}

UNITSON
