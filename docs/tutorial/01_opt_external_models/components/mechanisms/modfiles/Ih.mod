:Comment :
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih
	NONSPECIFIC_CURRENT ihcn
	RANGE gbar, g, ihcn, shift1, shift2, shift3, shift4, shift5, shift6
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gbar = 0.00001 (S/cm2)
	ehcn = -45.0 (mV)
	shift1 = 154.9
	shift2 = 11.9
	shift3 = 0
	shift4 = 33.1
	shift5 = 6.43
	shift6 = 193
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	g	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	g = gbar*m
	ihcn = g*(v-ehcn)
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
}

PROCEDURE rates(){
	UNITSOFF
				if(v == -shift1){
						v = v + 0.0001
				}
				if(shift4 == 0){
						shift4 = shift4 + 0.0001
				}
				if(shift2 == 0){
						shift2 = shift2 + 0.0001
				}
		mAlpha =  0.001*(shift5)*(v+shift1)/(exp((v+shift1)/(shift2))-1)
		mBeta  =  0.001*(shift6)*exp((v+shift3)/(shift4))
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
