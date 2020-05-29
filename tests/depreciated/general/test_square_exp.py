from scipy import integrate
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

#Local file
from probDE.Bayesian.cov_square_exp import cov_vv_se, cov_xv_se, cov_xx_se

def R_se(t, s):
    return exp(-(s - t)**2/2)

def RR_se(z, t, s):
    return R_se(t,z)*R_se(s,z)

def Q_se(t,s):
    return sqrt(pi/2)*(erf(s/sqrt(2)) - erf((s-t)/sqrt(2)))

def QQ_se(z, t, s):
    return Q_se(t,z)*Q_se(s,z)

def QR_se(z, t, s):
    return Q_se(t,z)*R_se(s,z)

def cov_vv_se2(t,s):
    return integrate.quad(RR_se, -np.inf, np.inf, args=(t,s))

def cov_xx_se2(t,s):
    return integrate.quad(QQ_se, -np.inf, np.inf, args=(t,s))

def cov_xv_se2(t,s):
    return integrate.quad(QR_se, -np.inf, np.inf, args=(t,s))

def test_cov_se(tseq):
    N = len(tseq)
    Sigma_an_vv = np.zeros((N,N))
    Sigma_an_xx = np.zeros((N,N))
    Sigma_an_xv = np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            Sigma_an_vv[i,j] = cov_vv_se2(tseq[i], tseq[j])[0]
            Sigma_an_xx[i,j] = cov_xx_se2(tseq[i], tseq[j])[0]
            Sigma_an_xv[i,j] = cov_xv_se2(tseq[i], tseq[j])[0]
    
    Sigma_nu_vv = cov_vv_se(tseq, tseq, 1, 1)
    Sigma_nu_xx = cov_xx_se(tseq, tseq, 1, 1)
    Sigma_nu_xv = cov_xv_se(tseq, tseq, 1, 1)

    return Sigma_an_vv, Sigma_an_xx, Sigma_an_xv, Sigma_nu_vv, Sigma_nu_xx, Sigma_nu_xv
