from scipy import integrate
from scipy import stats
import scipy as sc
import numpy as np
import sys
import os
from math import exp, erf, sqrt, pi

sys.path.insert(0, os.path.abspath('../'))
from BayesODE.cov_fun import cov_vv_ex, cov_xv_ex, cov_xx_ex

# def cov_vv_ex(t, s, gamma, alpha):
#     return exp(-abs(t-s)/gamma)/alpha

def mvCond(mu, Sigma, icond):
    """
    For y ~ N(mu, Sigma), returns A, b, and V 
    such that y[~icond] | y[icond] ~ N(A * y[icond] + b, V).
    """
    # if y1 = y[~icond] and y2 = y[icond],
    # should have A = Sigma12 * Sigma22^{-1}
    A = np.dot(Sigma[np.ix_(~icond, icond)],sc.linalg.cho_solve(sc.linalg.cho_factor(Sigma[np.ix_(icond,icond)]), np.identity(sum(icond))))
    b = mu[~icond] - np.dot(A, mu[icond]) # mu1 - A * mu2
    # Sigma11 - A * Sigma21
    V = Sigma[np.ix_(~icond,~icond)] - np.dot(A, Sigma[np.ix_(icond,~icond)]) 
    return A, b, V


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

gamma = 0.35
alpha = .6
n1 = 3
n2 = 2
t1 = np.array(sorted(abs(stats.norm.rvs(size = n1))))
t2 = max(t1) + np.array(sorted(abs(stats.norm.rvs(size = n2))))
T = np.concatenate((t1, t2))
Sigma = cov_yy_ex(T, T, gamma, alpha)
print(np.linalg.eigvalsh(Sigma))

# condition on t1
icond = np.array([True] * (2*n1) + [False] * (2*n2))
icond
mu = np.zeros(2*(n1+n2))
A, b, V = mvCond(mu, Sigma, icond)
print(np.linalg.eigvalsh(V))
A


# -----------

# def cov_yy_ex(t, s, gamma, alpha):
#     """A 4 x 4 matrix containing the cov(y_t, y_s), where y_t = (x_t, v_t).
#     The order of variables is (x_t, v_t, x_s, v_s).
#     """
#     V = np.zeros((4,4))
#     Err = np.zeros((4,4))
#     # all v's
#     V[1,1] = cov_vv_ex(t, t, gamma, alpha)
#     V[1,3] = cov_vv_ex(t, s, gamma, alpha)
#     V[3,1] = V[1,3]
#     V[3,3] = cov_vv_ex(s, s, gamma, alpha)
#     # cov(x_t, v_t): 0 and 1
#     V[0,1], Err[0,1] = integrate.quad(cov_vv_ex, 0, t, args=(t,gamma,alpha))
#     V[1,0], Err[1,0] = V[0,1], Err[0,1]
#     # cov(x_t, v_s): 0 and 3
#     V[0,3], Err[0,3] = integrate.quad(cov_vv_ex, 0, t, args=(s,gamma,alpha))
#     V[3,0], Err[3,0] = V[0,3], Err[0,3]
#     # cov(x_s, v_t): 2 and 1
#     V[2,1], Err[2,1] = integrate.quad(cov_vv_ex, 0, s, args=(t,gamma,alpha))
#     V[1,2], Err[1,2] = V[2,1], Err[2,1]
#     # cov(x_s, v_s): 2 and 3
#     V[2,3], Err[2,3] = integrate.quad(cov_vv_ex, 0, s, args=(s,gamma,alpha))
#     V[3,2], Err[3,2] = V[2,3], Err[2,3]
#     # cov(x_t, x_t): 0 and 0
#     V[0,0], Err[0,0] = integrate.dblquad(cov_vv_ex, 0, t, lambda u: 0, t,
#                                          args=(gamma,alpha))
#     # cov(x_t, x_s): 0 and 2
#     V[0,2], Err[0,2] = integrate.dblquad(cov_vv_ex, 0, t, lambda u: 0, s,
#                                          args=(gamma,alpha))
#     V[2,0], Err[2,0] = V[0,2], Err[0,2]
#     # cov(x_s, x_s): 2 and 2
#     V[2,2], Err[2,2] = integrate.dblquad(cov_vv_ex, 0, s, lambda u: 0, s,
#                                          args=(gamma,alpha))
#     return V, Err

# gamma = 0.35
# alpha = .6
# t = abs(stats.norm.rvs(size = 1))
# s = t + abs(stats.norm.rvs(size = 1))
# Sigma, Err = cov_yy_ex(t, s, gamma, alpha)

# # make sure variance is positive
# print(np.linalg.eigvalsh(Sigma))

# # y_s | y_t
# icond = np.array([True, True, False, False])
# mu = np.zeros(4)
# A, b, V = mvCond(mu, Sigma, icond)
# print(np.linalg.eigvalsh(V))
# A
# # shows that v_s is independent of x_t, t < s
# # however, x_s depends on all of y_t.
