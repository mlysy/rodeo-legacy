"""
.. module:: bayes_ode
    :synopsis: Bayesian solver for univariate ODEs.
"""
import numpy as np
from math import sqrt

def bayes_ode(fun, tseq, x0, Sigma_vv, Sigma_xx, Sigma_xv, vstar=None):
    """Bayesian solver of ODE problem :math:`dx_t/dt = f(x_t, t)`.

    Parameters
    ---------- 

    fun: function
        ODE function, taking two `float` parameters and returning a `float`.
    tseq: ndarray(dim_x)
        Vector of `N` timepoints at which :math:`x_t` will be calculated.
    x0: float
        Initial condition :math:`x(t_0) = x_0`.
    Sigma_vv: ndarray(dim_x, dim_x)
        `N x N` prior covariance matrix :math:`cov(v(tseq), v(tseq))`.
    Sigma_xx: ndarray(dim_x, dim_x)
        `N x N` prior covariance matrix :math:`cov(x(tseq), x(tseq))`.    
    Sigma_xv: ndarray(dim_x, dim_x)
        `N x N` prior cross-covariance matrix :math:`cov(x(tseq), v(tseq))`, where :math:`v_t = dx_t/dt`.
    vstar: None, ndarray(dim_x) 
        default None, predetermined :math:`V_t` after N interrogations
    
    Returns
    -------

    solution: ndarray(dim_x)
        vector of length `N` corresponding to the probabilistic solution :math:`x(tseq)`
    mu_x: ndarray(dim_x)
        vector of length `N` corresponding to the mean of x(tseq)
    Sigma_xx: ndarray(dim_x, dim_x)
        `N x N` prior covariance matrix :math:`cov(x(tseq), x(tseq))`.  

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
    
    solution = np.random.multivariate_normal(mu_x, Sigma_xx)

    return solution, mu_x, Sigma_xx
