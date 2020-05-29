from scipy import integrate
from scipy import stats
import scipy as sc
import numpy as np
from math import exp, erf, sqrt, pi

from probDE.Bayesian import cov_vv_ex, cov_xv_ex, cov_xx_ex
from probDE.utils.utils import mvncond

def cov_yy_ex(t1, t2, gamma, alpha):
    n1 = len(t1)
    n2 = len(t2)
    # T/F array of length N giving indices of elements
    iv1 = np.array([False, True] * n1)
    ix1 = ~iv1
    iv2 = np.array([False, True] * n2)
    ix2 = ~iv2
    Sigma = np.zeros((2*n1,2*n2))
    # var of v
    Sigma[np.ix_(iv1, iv2)] = cov_vv_ex(t1, t2, gamma, alpha)
    # cross covariances
    Sigma[np.ix_(ix1, iv2)] = cov_xv_ex(t1, t2, gamma, alpha)
    Sigma[np.ix_(iv1, ix2)] = cov_xv_ex(t2, t1, gamma, alpha).T
    # var of x
    Sigma[np.ix_(ix1, ix2)] = cov_xx_ex(t1, t2, gamma, alpha)
    return Sigma
