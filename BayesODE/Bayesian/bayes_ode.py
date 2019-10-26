"""
.. module:: bayes_ode
    :synopsis: Bayesian solver for univariate ODEs.

"""
import numpy as np
from math import sqrt

def bayes_ode(fun, tseq, x0, vvSigma, xxSigma, xvSigma, vstar=None):
    """Bayesian solver of ODE problem :math:`dx_t/dt = f(x_t, t)`.

    Parameters
    ---------- 
    fun: function
        ODE function, taking two `float` parameters and returning a `float`.
    tseq: ndarray(dim_x)
        Vector of :math:`N` timepoints at which :math:`x_t` will be calculated.
    x0: float
        Initial condition :math:`x(t_0) = x0`.
    vvSigma: ndarray(dim_x, dim_x)
        :math:`N x N` prior covariance matrix :math:`cov(v(tseq), v(tseq))`.
    xxSigma: ndarray(dim_x, dim_x)
        :math:`N x N` prior covariance matrix :math:`cov(x(tseq), x(tseq))`.    
    xvSigma: ndarray(dim_x, dim_x)
        :math:`N x N` prior cross-covariance matrix :math:`cov(x(tseq), v(tseq))`, where :math:`v_t = dx_t/dt`.
    vstar: None, ndarray(dim_x) 
        default None, predetermined :math:`V_t` after N interrogations
    
    Returns
    -------
    X: ndarray(dim_x)
        vector of length :math:`N` corresponding to the probabilistic solution :math:`x(tseq)`
    mu_x: ndarray(dim_x)
        vector of length :math:`N` corresponding to the mean of :math:`x(tseq)`
    xxSigma: ndarray(dim_x, dim_x)
        :math:`N x N` prior covariance matrix :math:`cov(x(tseq), x(tseq))`.  

    """
    N = len(tseq)
    mu_v = np.array([0]*N) # prior mean of v(tseq)
    mu_x = x0 + mu_v # prior mean of x(tseq)
    for i in range(N):
        Sigma_vstar = vvSigma[i, i] if i == 0 else 2 * vvSigma[i, i]
        if  vstar is None:
            xt = np.random.normal(mu_x[i], sqrt(xxSigma[i, i])) # interrogation of x_t
            vt = fun(xt, tseq[i]) # interrogation of v_t
            # mean and variance updates
            mu_x = mu_x + xvSigma[:, i] / Sigma_vstar * (vt - mu_v[i])
            mu_v = mu_v + vvSigma[:, i] * (vt - mu_v[i]) / Sigma_vstar
        else:
            mu_x = mu_x + xvSigma[:, i] / Sigma_vstar * (vstar[i] - mu_v[i])
            mu_v = mu_v + vvSigma[:, i] * (vstar[i] - mu_v[i]) * Sigma_vstar
        xxSigma = xxSigma - 1/Sigma_vstar * np.outer(xvSigma[:, i], xvSigma[:, i])
        xvSigma = xvSigma - 1/Sigma_vstar * np.outer(xvSigma[:, i], vvSigma[i, :])
        vvSigma = vvSigma - 1/Sigma_vstar * np.outer(vvSigma[:, i], vvSigma[i, :])
    
    X = np.random.multivariate_normal(mu_x, xxSigma)
    return X, mu_x, xxSigma
