from scipy import integrate
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

#Local file
sys.path.insert(0, os.path.abspath('../'))
from BayesODE.cov_fun import cov_vv_ex, cov_xv_ex, cov_xx_ex

def cov_vv_ex2(t, s, gamma, alpha):
    return exp(-abs(t-s)/gamma)/alpha

def cov_xv_ex2(t, s):
    return integrate.quad(cov_vv_ex2, 0, t, args=(s,1,1))

def cov_xx_ex2(t, s):
    return integrate.dblquad(cov_vv_ex2, 0, t, lambda s: 0, s, args=(1,1))


def test_cov_ex(tseq):
    N = len(tseq)
    Sigma_an_xx = np.zeros((N,N))
    Sigma_an_xv = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            Sigma_an_xx[i,j] = cov_xx_ex2(tseq[i], tseq[j])[0]
            Sigma_an_xv[i,j] = cov_xv_ex2(tseq[i], tseq[j])[0]
    
    Sigma_nu_xx = cov_xx_ex(tseq, tseq, 1, 1)
    Sigma_nu_xv = cov_xv_ex(tseq, tseq, 1, 1)

    return Sigma_an_xx, Sigma_an_xv, Sigma_nu_xx, Sigma_nu_xv
