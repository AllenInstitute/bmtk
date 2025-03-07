TITLE Large conductance Ca2+ activated K+ channel mslo

COMMENT

Parameters from Cox et al. (1987) J Gen Physiol 110:257-81 (patch 1).

Current Model Reference: Anwar H, Hong S, De Schutter E (2010) Controlling Ca2+-activated K+ channels with models of Ca2+ buffering in Purkinje cell. Cerebellum*

*Article available as Open Access

PubMed link: http://www.ncbi.nlm.nih.gov/pubmed/20981513


Written by Sungho Hong, Okinawa Institute of Science and Technology, March 2009.
Contact: Sungho Hong (shhong@oist.jp)

Suffix from mslo to Kca1_1

ENDCOMMENT

NEURON {
  SUFFIX Kca1_1
  USEION k READ ek WRITE ik
  USEION ca READ cai
  RANGE g, gbar, ik

}

UNITS { 
    (mV) = (millivolt)
    (S) = (siemens)
    (molar) = (1/liter)
    (mM) = (millimolar)
    FARADAY = (faraday) (kilocoulombs)
    R = (k-mole) (joule/degC)
}

CONSTANT {
    q10 = 3
}

PARAMETER {
    gbar = 0.01 (S/cm2)
    
    Qo = 0.73
    Qc = -0.67
    
    k1 = 1.0e3 (/mM)
    onoffrate = 1 (/ms)
    
    L0 = 1806
    Kc = 11.0e-3 (mM)
    Ko = 1.1e-3 (mM)
    
    pf0 = 2.39e-3  (/ms)
    pf1 = 7.0e-3  (/ms)
    pf2 = 40e-3   (/ms)
    pf3 = 295e-3  (/ms)
    pf4 = 557e-3  (/ms)
    
    pb0 = 3936e-3 (/ms)
    pb1 = 1152e-3 (/ms)
    pb2 = 659e-3  (/ms)
    pb3 = 486e-3  (/ms)
    pb4 = 92e-3  (/ms)
}

ASSIGNED {
    : rates
    c01    (/ms)
    c12    (/ms)
    c23    (/ms)
    c34    (/ms)
    o01    (/ms)
    o12    (/ms)
    o23    (/ms)
    o34    (/ms)
    f0     (/ms)
    f1     (/ms)
    f2     (/ms)
    f3     (/ms)
    f4     (/ms)

    c10    (/ms)
    c21    (/ms)
    c32    (/ms)
    c43    (/ms)
    o10    (/ms)
    o21    (/ms)
    o32    (/ms)
    o43    (/ms)
    b0     (/ms)
    b1     (/ms)
    b2     (/ms)
    b3     (/ms)
    b4     (/ms)
    
    v            (mV)
    cai          (mM)
    ek           (mV)
    ik           (milliamp/cm2)
    g            (S/cm2)
    celsius      (degC)
}

STATE {
    C0 FROM 0 TO 1
    C1 FROM 0 TO 1
    C2 FROM 0 TO 1
    C3 FROM 0 TO 1
    C4 FROM 0 TO 1
    O0 FROM 0 TO 1
    O1 FROM 0 TO 1
    O2 FROM 0 TO 1
    O3 FROM 0 TO 1
    O4 FROM 0 TO 1
}

BREAKPOINT {
    SOLVE activation METHOD sparse
    g = gbar * (O0 + O1 + O2 + O3 + O4)
    ik = g * (v - ek)
}

INITIAL {
:    rates(v, cai)
:    SOLVE seqinitial
    SOLVE activation STEADYSTATE sparse
}

KINETIC activation {
    rates(v, cai)
    ~ C0 <-> C1      (c01,c10)
    ~ C1 <-> C2      (c12,c21)
    ~ C2 <-> C3      (c23,c32)
    ~ C3 <-> C4      (c34,c43)
    ~ O0 <-> O1      (o01,o10)
    ~ O1 <-> O2      (o12,o21)
    ~ O2 <-> O3      (o23,o32)
    ~ O3 <-> O4      (o34,o43)
    ~ C0 <-> O0      (f0 , b0)
    ~ C1 <-> O1      (f1 , b1)
    ~ C2 <-> O2      (f2 , b2)
    ~ C3 <-> O3      (f3 , b3)
    ~ C4 <-> O4      (f4 , b4)

CONSERVE C0 + C1 + C2 + C3 + C4 + O0 + O1 + O2 + O3 + O4 = 1
}

PROCEDURE rates(v(mV), ca (mM)) { 
    LOCAL qt, alpha, beta
    
    qt = q10^((celsius-23 (degC))/10 (degC))
    
    c01 = 4*ca*k1*onoffrate*qt
    c12 = 3*ca*k1*onoffrate*qt
    c23 = 2*ca*k1*onoffrate*qt
    c34 = 1*ca*k1*onoffrate*qt
    o01 = 4*ca*k1*onoffrate*qt
    o12 = 3*ca*k1*onoffrate*qt
    o23 = 2*ca*k1*onoffrate*qt
    o34 = 1*ca*k1*onoffrate*qt
    
    c10 = 1*Kc*k1*onoffrate*qt
    c21 = 2*Kc*k1*onoffrate*qt
    c32 = 3*Kc*k1*onoffrate*qt
    c43 = 4*Kc*k1*onoffrate*qt
    o10 = 1*Ko*k1*onoffrate*qt
    o21 = 2*Ko*k1*onoffrate*qt
    o32 = 3*Ko*k1*onoffrate*qt
    o43 = 4*Ko*k1*onoffrate*qt
    
    alpha = exp(Qo*FARADAY*v/R/(273.15 + celsius))
    beta  = exp(Qc*FARADAY*v/R/(273.15 + celsius))
    
    f0  = pf0*alpha*qt
    f1  = pf1*alpha*qt
    f2  = pf2*alpha*qt
    f3  = pf3*alpha*qt
    f4  = pf4*alpha*qt
    
    b0  = pb0*beta*qt
    b1  = pb1*beta*qt
    b2  = pb2*beta*qt
    b3  = pb3*beta*qt
    b4  = pb4*beta*qt
}
