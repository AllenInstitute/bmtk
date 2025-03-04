TITLE I-h HCN1 channel from Kamilla Angelo, Michael London,Soren R. Christensen, and Michael Hausser 2007 J. of Neurosci.
COMMENT

We call it HCN1 as PC express only HCN1 Santoro et al. 2000
:aggiunta di correzione per Q10 by ERICA GRANDI

ENDCOMMENT

NEURON {
	SUFFIX HCN1
	USEION h READ eh WRITE ih VALENCE 1 
	RANGE gbar, hinf,tauh,ratetau,ih
	RANGE hinf,tauh,eh
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}


CONSTANT {
	q10=3
}

PARAMETER {
    v 		(mV)
    : eh  =-34.4	(mV)        
    gbar=.0001 	(mho/cm2)
    ratetau = 1 (ms)
    rec_temp = 23 (deg) : we set it here at room temperature as in Angelo et al. they forogot tp mention the recording temperature
    ljp = 9.3 (mV) : liquid_junction_potential
    v_inf_half_noljp = -90.3 (mV)
    v_inf_k = 9.67 (mV)
    v_tau_const = 0.0018 (1)
    v_tau_half1_noljp = -68 (mV)
    v_tau_half2_noljp = -68 (mV)
    v_tau_k1 = -22 (mv)
    v_tau_k2 = 7.14 (mv)
 }

STATE {
    h
}

ASSIGNED {
    eh (mV)
    ih (mA/cm2)
    hinf      
    tauh
    celsius (deg)
    v_inf_half (mV)
    v_tau_half1 (mV)
    v_tau_half2 (mV)
    qt
}

INITIAL {

    : ADD Q10 correction!!!!! FATTO!!!
    qt = q10^((celsius-37 (degC))/10 (degC))
    v_inf_half = (v_inf_half_noljp - ljp)
    v_tau_half1 = (v_tau_half1_noljp - ljp)
    v_tau_half2 = (v_tau_half2_noljp - ljp)
    
    rate(v)
    h=hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ih = h*gbar*(v-eh)
}

DERIVATIVE states {  
    rate(v)
    h' =  (hinf - h)/tauh
}

PROCEDURE rate(v (mV)) {
    : hinf=1/( 1+exp((90+v)/9.67) )
    : tauh=ratetau*1/(0.0018*( exp((v+68)/-22) + exp((v+68)/7.14) ))
    hinf = 1 / (1+exp( (v-v_inf_half) / v_inf_k) )
    tauh = (ratetau / (v_tau_const * ( exp( (v-v_tau_half1) / v_tau_k1) + exp( (v-v_tau_half2) / v_tau_k2) )))/qt
}
















