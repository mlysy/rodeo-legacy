"""
.. module:: ode_bayes
    :synopsis: Bayesian solver for univariate ODEs.
"""
import numpy as np
from math import sqrt

def ode_bayes(fun, tseq, x0, Sigma_vv, Sigma_xx, Sigma_xv, vstar=None):
    """Bayesian solver of ODE problem :math:`dx_t/dt = f(x_t, t)`.

    :param fun: ODE function, taking two `float` parameters and returning a `float`.
    :type fun: function
    :param tseq: Vector of `N` timepoints at which :math:`x_t` will be calculated.
    :type tseq: float
    :param x0: Initial condition :math:`x(t_0) = x_0`.
    :type x0: float
    :param Sigma_vv: `N x N` prior covariance matrix **cov(v(tseq), v(tseq))**.
    :type Sigma_vv: float
    :param Sigma_xx: `N x N` prior covariance matrix **cov(x(tseq), x(tseq))**.    
    :type Sigma_xx: float
    :param Sigma_xv: `N x N` prior cross-covariance matrix **cov(x(tseq), v(tseq))**, where :math:`v_t = dx_t/dt`.
    :type Sigma_xv: float
    :param vstar: default None, predetermined V_t after N interrogations
    :type vstar: None or float
    :param mu_star: default None, predetermined mean of x(tseq)
    :type mu_star: None or float
    :param Sigma_star: default None, predetermined covariance matrix **cov(x(tseq), x(tseq))**
    :type Sigma_star: None or float
    :returns: A vector of length `N` corresponding to the probabilistic solution **x(tseq)** and mu_x, Sigma_xx
    :rtype: float, float, float
    """

    N = len(tseq)
    mu_v = np.array([0]*N) # prior mean of v(tseq)
    mu_x = x0 + mu_v # prior mean of x(tseq)
    for i in range(N):
        Sigma_vstar = Sigma_vv[i,i] if i == 0 else 2 * Sigma_vv[i,i]
        if  vstar is None:
            xt = np.random.normal(mu_x[i], sqrt(Sigma_xx[i,i])) # interrogation of x_t
            vt = fun(xt, tseq[i]) # interrogation of v_t
            # mean and variance updates
            mu_x = mu_x + Sigma_xv[:,i]*1/Sigma_vstar *(vt - mu_v[i])
            mu_v = mu_v + Sigma_vv[:,i]*(vt - mu_v[i])*1/Sigma_vstar
        else:
            mu_x = mu_x + Sigma_xv[:,i]*1/Sigma_vstar *(vstar[i] - mu_v[i])
            mu_v = mu_v + Sigma_vv[:,i]*(vstar[i] - mu_v[i])*1/Sigma_vstar
        Sigma_xx = Sigma_xx - 1/Sigma_vstar*np.outer(Sigma_xv[:,i], Sigma_xv[:,i])
        Sigma_xv = Sigma_xv - 1/Sigma_vstar*np.outer(Sigma_xv[:,i], Sigma_vv[i,:])
        Sigma_vv = Sigma_vv - 1/Sigma_vstar*np.outer(Sigma_vv[:,i], Sigma_vv[i,:])

    return np.random.multivariate_normal(mu_x, Sigma_xx), mu_x, Sigma_xx
