: this model is built-in to neuron with suffix epsp
: Schaefer et al. 2003

COMMENT
modified from syn2.mod
injected current with exponential rise and decay current defined by
         i = 0 for t < onset and
         i=amp*((1-exp(-(t-onset)/tau0))-(1-exp(-(t-onset)/tau1)))
          for t > onset

	compare to experimental current injection:
 	i = - amp*(1-exp(-t/t1))*(exp(-t/t2))

	-> tau1==t2   tau0 ^-1 = t1^-1 + t2^-1
ENDCOMMENT
					       
INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	POINT_PROCESS epsp
	RANGE onset, tau0, tau1, imax, i, myv
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(umho) = (micromho)
}

PARAMETER {
	onset=0  (ms)
	tau0=0.2 (ms)
	tau1=3.0 (ms)
	imax=0 	 (nA)
	v	 (mV)
}

ASSIGNED { i (nA)  myv (mV)}

LOCAL   a[2]
LOCAL   tpeak
LOCAL   adjust
LOCAL   amp

BREAKPOINT {
	myv = v
        i = curr(t)
}

FUNCTION myexp(x) {
	if (x < -100) {
	myexp = 0
	}else{
	myexp = exp(x)
	}
}

FUNCTION curr(x) {				
	tpeak=tau0*tau1*log(tau0/tau1)/(tau0-tau1)
	adjust=1/((1-myexp(-tpeak/tau0))-(1-myexp(-tpeak/tau1)))
	amp=adjust*imax
	if (x < onset) {
		curr = 0
	}else{
		a[0]=1-myexp(-(x-onset)/tau0)
		a[1]=1-myexp(-(x-onset)/tau1)
		curr = -amp*(a[0]-a[1])
	}
}
