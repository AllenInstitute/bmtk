TITLE CaV 3.3 CA3 hippocampal neuron

COMMENT
    Cell model: CA3 hippocampal neuron
    
    Created by jun xu @ Clancy Lab of Cornell University Medical College on 3/27/05
    
    Geometry: single-compartment model modified on 04/19/07 
    Xu J, Clancy CE (2008) Ionic mechanisms of endogenous bursting in CA3 hippocampal pyramidal neurons: 
        a model study. PLoS ONE 3:e2056- [PubMed]

ENDCOMMENT 
 
 
 NEURON	{
        : CaT--alpha 1I CaV3.3
	SUFFIX Cav3_3
	USEION ca READ cai, cao WRITE ica
	RANGE gCav3_3bar, pcabar, ica, tau_l, tau_n, n_inf, l_inf
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
    gCav3_3bar = 0.00001 (S/cm2)
    vhalfn = -41.5  :mv
    vhalfl = -69.8
    kn = 6.2
    kl = -6.1
    q10 = 2.3
    pcabar = 0.0001 : cm/s to check!!!
    z= 2
    F = 96520 : Farady constant (coulomb/mol)
    R = 8.3134 : gas constant (J/K.mol)
    PI = 3.14    
}

ASSIGNED	{
	v	(mV)
	ica	(mA/cm2)
	gCav3_3	(S/cm2)
	n_inf
	tau_n
	l_inf
	tau_l
	cai     (mM)
	cao     (mM)
	qt 
	T  : absolute temperature (K)
	ghk
	w
}

STATE	{ 
	n
	l
}

BREAKPOINT	{
    SOLVE states METHOD cnexp
    ica = gCav3_3bar*pcabar*n*n*l*ghk
}

DERIVATIVE states	{
	rates()
	n' = (n_inf-n)/tau_n
	l' = (l_inf-l)/tau_l
    }
    
INITIAL{
	T = celsius+273.14
	qt = pow(q10,(celsius-28)/10)
	rates()
	n = n_inf
	l = l_inf
}

PROCEDURE rates(){
	n_inf = 1/(1+exp(-(v-vhalfn)/kn))
	l_inf = 1/(1+exp(-(v-vhalfl)/kl))
	
        if (v > -60) {
            tau_n = (7.2+0.02*exp(-v/14.7))/qt
	    tau_l = (79.5+2.0*exp(-v/9.3))/qt
        }else{
            tau_n = (0.875*exp((v+120)/41))/qt
	    tau_l = 260/qt
        }
	
      w = v*0.001*z*F/(R*T)
      ghk = -0.001*z*F*(cao-cai*exp(w))*w/(exp(w)-1)	
}
