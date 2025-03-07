TITLE Cerebellum Granule Cell Model

COMMENT
        KA channel
   
	Author: E.D'Angelo, T.Nieus, A. Fontana
	Last revised: Egidio 3.12.2003

:Suffix from GRC_KA to Kv4_3
ENDCOMMENT

NEURON { 
	SUFFIX Kv4_3
	USEION k READ ek WRITE ik 
	RANGE gkbar, ik, g, alpha_a, beta_a, alpha_b, beta_b
	RANGE Aalpha_a, Kalpha_a, V0alpha_a
	RANGE Abeta_a, Kbeta_a, V0beta_a
	RANGE Aalpha_b, Kalpha_b, V0alpha_b
	RANGE Abeta_b, Kbeta_b, V0beta_b
	RANGE V0_ainf, K_ainf, V0_binf, K_binf
	RANGE a_inf, tau_a, b_inf, tau_b 
} 
 
UNITS { 
	(mA) = (milliamp) 
	(mV) = (millivolt) 
} 
 
PARAMETER { 
	Aalpha_a = 0.8147 (/ms) :4.88826
	Kalpha_a = -23.32708 (mV)
	V0alpha_a = -9.17203 (mV)
	Abeta_a = 0.1655 (/ms)   : 0.99285	
	Kbeta_a = 19.47175 (mV)
	V0beta_a = -18.27914 (mV)

	Aalpha_b = 0.0368 (/ms)  : 0.11042 
	Kalpha_b = 12.8433 (mV)
	V0alpha_b = -111.33209 (mV)   
	Abeta_b = 0.0345(/ms)   : 0.10353 
	Kbeta_b = -8.90123 (mV)
	V0beta_b = -49.9537 (mV)

	V0_ainf = -38(mV)
	K_ainf = -17(mV)

	V0_binf = -78.8 (mV)
	K_binf = 8.4 (mV)
	v (mV) 
	gkbar= 0.0032 (mho/cm2) :0.003 
	celsius = 30 (degC) 
} 

STATE { 
	a
	b 
} 

ASSIGNED { 
	ik (mA/cm2) 
	a_inf 
	b_inf 
	tau_a (ms) 
	tau_b (ms) 
	g (mho/cm2) 
	alpha_a (/ms)
	beta_a (/ms)
	alpha_b (/ms)
	beta_b (/ms)
	ek (mV)
} 
 
INITIAL { 
	rate(v) 
	a = a_inf 
	b = b_inf 
} 
 
BREAKPOINT { 
	SOLVE states METHOD derivimplicit 
	g = gkbar*a*a*a*b 
	ik = g*(v - ek)
	alpha_a = alp_a(v)
	beta_a = bet_a(v) 
	alpha_b = alp_b(v)
	beta_b = bet_b(v) 
} 
 
DERIVATIVE states { 
	rate(v) 
	a' =(a_inf - a)/tau_a 
	b' =(b_inf - b)/tau_b 
} 
 
FUNCTION alp_a(v(mV))(/ms) { LOCAL Q10
	Q10 = 3^((celsius-25.5(degC))/10(degC))
:	alp_a = Q10*Aalpha_a*exp(Kalpha_a*(v-V0alpha_a)) 
:	alp_a = -0.04148(/mV-ms)*linoid(v+67.697(mV),-3.857(mV))
	alp_a = Q10*Aalpha_a*sigm(v-V0alpha_a,Kalpha_a)
} 
 
FUNCTION bet_a(v(mV))(/ms) { LOCAL Q10
	Q10 = 3^((celsius-25.5(degC))/10(degC))
:	bet_a = Q10*Abeta_a*exp(Kbeta_a*(v-V0beta_a)) 
:	bet_a = 0.0359(/mV-ms)*linoid(v+45.878(mV),23.654(mV))
	bet_a = Q10*Abeta_a/(exp((v-V0beta_a)/Kbeta_a))
} 
 
FUNCTION alp_b(v(mV))(/ms) { LOCAL Q10
	Q10 = 3^((celsius-25.5(degC))/10(degC))
:	alp_b = Q10*Aalpha_b*exp(Kalpha_b*(v-V0alpha_b)) 
:	alp_b = 0.356(/mV-ms)*linoid(v+231.03(mV),17.8(mV))
	alp_b = Q10*Aalpha_b*sigm(v-V0alpha_b,Kalpha_b)
} 
 
FUNCTION bet_b(v(mV))(/ms) { LOCAL Q10
	Q10 = 3^((celsius-25.5(degC))/10(degC))
:	bet_b = Q10*Abeta_b*exp(Kbeta_b*(v-V0beta_b)) 
:	bet_b = -0.00825(/mV-ms)*linoid(v+43.284(mV),-8.927(mV))
	bet_b = Q10*Abeta_b*sigm(v-V0beta_b,Kbeta_b)
} 
 
PROCEDURE rate(v (mV)) {LOCAL a_a, b_a, a_b, b_b 
	TABLE a_inf, tau_a, b_inf, tau_b 
	DEPEND Aalpha_a, Kalpha_a, V0alpha_a, 
	       Abeta_a, Kbeta_a, V0beta_a,
               Aalpha_b, Kalpha_b, V0alpha_b,
               Abeta_b, Kbeta_b, V0beta_b, celsius FROM -100 TO 30 WITH 13000 
	a_a = alp_a(v)  
	b_a = bet_a(v) 
	a_b = alp_b(v)  
	b_b = bet_b(v) 
	a_inf = 1/(1+exp((v-V0_ainf)/K_ainf)) 
	tau_a = 1/(a_a + b_a) 
	b_inf = 1/(1+exp((v-V0_binf)/K_binf))
	tau_b = 1/(a_b + b_b) 
: Bardoni Belluzzi data
:	a_inf = 1/(1+exp(-(v+46.7)/19.8))
:	tau_a = 0.41*exp(-(v+43.5)/42.8)+0.167
:	b_inf = 1/(1+exp((v+78.8)/8.4))
:	tau_b = 10.8 + 0.03*v + 1/(57.9*exp(0.127*v)+0.000134*exp(-0.059*v))
}

FUNCTION linoid(x (mV),y (mV)) (mV) {
        if (fabs(x/y) < 1e-6) {
                linoid = y*(1 - x/y/2)
        }else{
                linoid = x/(exp(x/y) - 1)
        }
} 

FUNCTION sigm(x (mV),y (mV)) {
                sigm = 1/(exp(x/y) + 1)
}
