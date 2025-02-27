TITLE SK2 multi-state model Cerebellum Golgi Cell Model

COMMENT

Author:Sergio Solinas, Lia Forti, Egidio DAngelo
Based on data from: Hirschberg, Maylie, Adelman, Marrion J Gen Physiol 1998
Last revised: May 2007

Published in:
             Sergio M. Solinas, Lia Forti, Elisabetta Cesana, 
             Jonathan Mapelli, Erik De Schutter and Egidio D`Angelo (2008)
             Computational reconstruction of pacemaking and intrinsic 
             electroresponsiveness in cerebellar golgi cells
             Frontiers in Cellular Neuroscience 2:2

Suffix from SK2 to Kca2_2

ENDCOMMENT

NEURON{
	SUFFIX Kca2_2
	USEION ca READ cai
	USEION k READ ek WRITE ik 
	RANGE gkbar, g, ik, tcorr
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(molar) = (1/liter)
	(mM) = (millimolar)
}

PARAMETER {
	celsius  (degC)
	cai (mM)
	gkbar = 0.038 (mho/cm2)
	Q10 = 3 (1)
	diff = 3 (1) : diffusion factor

: rates ca-indipendent
	invc1 = 80e-3  ( /ms)
	invc2 = 80e-3  ( /ms)
	invc3 = 200e-3 ( /ms)

	invo1 = 1      ( /ms)
	invo2 = 100e-3 ( /ms)
	diro1 = 160e-3 ( /ms)
	diro2 = 1.2    ( /ms)

: rates ca-dipendent
	dirc2 = 200 ( /ms-mM )
	dirc3 = 160 ( /ms-mM )
	dirc4 = 80  ( /ms-mM )

}

ASSIGNED{ 
	v	(mV) 
	ek	(mV) 
	g	(mho/cm2) 
	ik	(mA/cm2) 
	invc1_t  ( /ms)
	invc2_t  ( /ms)
	invc3_t  ( /ms)
	invo1_t  ( /ms)
	invo2_t  ( /ms)
	diro1_t  ( /ms)
	diro2_t  ( /ms)
	dirc2_t  ( /ms-mM)
	dirc3_t  ( /ms-mM)
	dirc4_t  ( /ms-mM)
	tcorr	 (1)

	dirc2_t_ca  ( /ms)
	dirc3_t_ca  ( /ms)
	dirc4_t_ca  ( /ms)
} 

STATE {
	c1
	c2
	c3
	c4
	o1
	o2
}

BREAKPOINT{ 
	SOLVE kin METHOD sparse 
	g = gkbar*(o1+o2)	:(mho/cm2)
	ik = g*(v-ek)		:(mA/cm2)
} 

INITIAL{
	rate(celsius)
	SOLVE kin STEADYSTATE sparse
} 

KINETIC kin{ 
	rates(cai/diff) 
	~c1<->c2 (dirc2_t_ca, invc1_t) 
	~c2<->c3 (dirc3_t_ca, invc2_t) 
	~c3<->c4 (dirc4_t_ca, invc3_t) 
	~c3<->o1 (diro1_t, invo1_t) 
	~c4<->o2 (diro2_t, invo2_t) 
	CONSERVE c1+c2+c3+c4+o2+o1=1 
} 

FUNCTION temper (Q10, celsius (degC)) {
	temper = Q10^((celsius -23(degC)) / 10(degC)) 
}

PROCEDURE rates(cai(mM)){
	dirc2_t_ca = dirc2_t*cai
	dirc3_t_ca = dirc3_t*cai
	dirc4_t_ca = dirc4_t*cai 
} 

PROCEDURE rate (celsius(degC)) {
	tcorr = temper (Q10,celsius)
	invc1_t = invc1*tcorr  
	invc2_t = invc2*tcorr
	invc3_t = invc3*tcorr 
	invo1_t = invo1*tcorr 
	invo2_t = invo2*tcorr 
	diro1_t = diro1*tcorr 
	diro2_t = diro2*tcorr 
	dirc2_t = dirc2*tcorr
	dirc3_t = dirc3*tcorr
	dirc4_t = dirc4*tcorr
}
