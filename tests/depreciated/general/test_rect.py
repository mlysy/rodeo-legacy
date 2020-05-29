from scipy import integrate
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

#Local file
from probDE.Bayesian.cov_rect import cov_vv_re, cov_xv_re, cov_xx_re


def R_re(t, s):
    if s < t+1 and s > t-1:
        return 1
    else:
        return 0
    
def RR_re(z, t, s):
    return R_re(t,z)*R_re(s,z)

def Q_re(t, s):
    return max(0, min(s+1,t) - max(0, s-1))

def QQ_re(z, t, s):
    return Q_re(t,z)*Q_re(s,z)

def QR_re(z, t, s):
    return Q_re(t,z)*R_re(s,z)

def cov_vv_re2(t,s):
    return integrate.quad(RR_re, -1, max(t+1,s+1), args=(t,s))

def cov_xx_re2(t,s):
    return integrate.quad(QQ_re, -1, max(t+1,s+1), args=(t,s))

def cov_xv_re2(t,s):
    return integrate.quad(QR_re, -1, max(t+1,s+1), args=(t,s))

def Q_re2(t, s):
    return integrate.quad(R_re, 0, t, args=(s,))

def test_cov_re(tseq):
    N = len(tseq)
    Sigma_an_vv = np.zeros((N,N))
    Sigma_an_xx = np.zeros((N,N))
    Sigma_an_xv = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            Sigma_an_vv[i,j] = cov_vv_re2(tseq[i], tseq[j])[0]
            Sigma_an_xx[i,j] = cov_xx_re2(tseq[i], tseq[j])[0]
            Sigma_an_xv[i,j] = cov_xv_re2(tseq[i], tseq[j])[0]
    
    Sigma_nu_vv = cov_vv_re(tseq, tseq, 1, 1)
    Sigma_nu_xx = cov_xx_re(tseq, tseq, 1, 1)
    Sigma_nu_xv = cov_xv_re(tseq, tseq, 1, 1)

    return Sigma_an_vv, Sigma_an_xx, Sigma_an_xv, Sigma_nu_vv, Sigma_nu_xx, Sigma_nu_xv
