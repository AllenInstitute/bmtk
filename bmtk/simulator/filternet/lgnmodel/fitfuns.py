import os
from math import *
import numpy as np
import numpy.fft as npft


def makeFitStruct_GLM(dtsim, kbasprs, nkt, flag_exp):
    gg = {}
    gg['k'] = []
    gg['dc'] = 0
    gg['kt'] = np.zeros((nkt,1))
    gg['ktbas'] = []
    gg['kbasprs'] = kbasprs
    gg['dt'] = dtsim
    
    nkt = nkt
    if flag_exp==0:    
        ktbas = makeBasis_StimKernel(kbasprs,nkt)
    else:
        ktbas = makeBasis_StimKernel_exp(kbasprs,nkt)
    
    gg['ktbas'] = ktbas
    gg['k'] = gg['ktbas']*gg['kt']
    
    return gg


def makeBasis_StimKernel(kbasprs, nkt):
    neye = kbasprs['neye']
    ncos = kbasprs['ncos']
    kpeaks = kbasprs['kpeaks']
    kdt = 1
    b = kbasprs['b']
    delays_raw = kbasprs['delays']
    delays = delays_raw[0].astype(int)
    
    ylim = np.array([100.,200.])  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!HARD-CODED FOR NOW 

    yrnge = nlin(ylim + b*np.ones(np.shape(kpeaks)))
    db = (yrnge[-1]-yrnge[0])/(ncos-1)

    ctrs = nlin(np.array(kpeaks))  # yrnge
    mxt = invnl(yrnge[ncos-1]+2*db)-b
    kt0 = np.arange(0, mxt, kdt)  # -delay
    nt = len(kt0)
    e1 = np.tile(nlin(kt0 + b*np.ones(np.shape(kt0))), (ncos, 1))
    e2 = np.transpose(e1)    
    e3 = np.tile(ctrs, (nt, 1))

    kbasis0 = []
    for kk in range(ncos):
        kbasis0.append(ff(e2[:,kk],e3[:,kk],db))

    #Concatenate identity vectors
    nkt0 = np.size(kt0, 0)
    a1 = np.concatenate((np.eye(neye), np.zeros((nkt0,neye))),axis=0)
    a2 = np.concatenate((np.zeros((neye,ncos)),np.array(kbasis0).T),axis=0)
    kbasis = np.concatenate((a1, a2),axis=1)
    kbasis = np.flipud(kbasis)
    nkt0 = np.size(kbasis,0)

    if nkt0 < nkt:
        kbasis = np.concatenate((np.zeros((nkt - nkt0, ncos + neye)), kbasis), axis=0)
    elif nkt0 > nkt:
        kbasis = kbasis[-1-nkt:-1, :]
    
    kbasis = normalizecols(kbasis)

    # Add delays for both functions. tack on delays (array of 0s) to the end of the function, then readjusts the second
    # function so both are the same size.
    kbasis2_0 = np.concatenate((kbasis[:, 0], np.zeros((delays[0], ))), axis=0)
    kbasis2_1 = np.concatenate((kbasis[:, 1], np.zeros((delays[1], ))), axis=0)
    len_diff = delays[1] - delays[0]
    kbasis2_1 = kbasis2_1[len_diff:]

    # combine and renormalize
    kbasis2 = np.zeros((len(kbasis2_0), 2))
    kbasis2[:, 0] = kbasis2_0
    kbasis2[:, 1] = kbasis2_1
    kbasis2 = normalizecols(kbasis2)
    return kbasis2


def makeBasis_StimKernel_exp(kbasprs,nkt):
    ks = kbasprs['ks']
    b = kbasprs['b']
    x0 = np.arange(0,nkt)
    kbasis = np.zeros((nkt,len(ks)))
    for ii in range(len(ks)):
        kbasis[:,ii] = invnl(-ks[ii]*x0)  # (1.0/ks[ii])*
    
    kbasis = np.flipud(kbasis)     
    # kbasis = normalizecols(kbasis)
    return kbasis  


def nlin(x):
    eps = 1e-20
    # x.clip(0.)
    return np.log(x+eps)


def invnl(x):
    eps = 1e-20
    return np.exp(x)-eps


def ff(x, c, dc):
    rowsize = np.size(x,0)
    m = []
    for i in range(rowsize): 
        xi = x[i]
        ci = c[i]
        val=(np.cos(np.max([-pi, np.min([pi, (xi-ci)*pi/dc/2])])) + 1)/2
        m.append(val)
        
    return np.array(m)


def normalizecols(A):
    B = A/np.tile(np.sqrt(sum(A**2,0)),(np.size(A,0),1))
    return B
    
def sameconv(A,B):
    
    am = np.size(A)
    bm = np.size(B)
    nn = am+bm-1
    
    q = npft.fft(A,nn)*npft.fft(np.flipud(B),nn)
    p = q
    G = npft.ifft(p)
    G = G[range(am)]
    
    return G
