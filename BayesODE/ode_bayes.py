"""
.. module:: ode_bayes
    :synopsis: Bayesian solver for univariate ODEs.
"""

def ode_bayes(fun, tseq, x0, Sigma_vv, Sigma_xx, Sigma_xv):
    """Bayesian solver of ODE problem `dx_t/dt = f(x_t, t)`.

    .. note::
    - Please find out how to properly document `type` for non-scalars...

    :param fun: ODE function, taking two `float` parameters and returning a `float.
    :type fun: function
    :param tseq: Vector of `N` timepoints at which `x_t` will be calculated.
    :type tseq: float
    :param x0: Initial condition `x(t0) = x0`.
    :type x0: float
    :param Sigma_vv: `N x N` prior covariance matrix `cov(v(tseq), v(tseq))`.
    :type Sigma_vv: float
    :param Sigma_xx: `N x N` prior covariance matrix `cov(x(tseq), x(tseq))`.    :type Sigma_xx: float
    :param Sigma_xv: `N x N` prior cross-covariance matrix `cov(x(tseq), v(tseq))`, where `v_t = dx_t/dt`.
    :type Sigma_xv: float
    :returns: A vector of length `N` corresponding to the probabilistic solution `x(tseq)`.
    :rtype: float
    """

    N = len(tseq)
    mu_v = np.array([0]*N) # prior mean of v(tseq)
    mu_x = u + mu_v # prior mean of x(tseq)
    for i in range(N):
        xt = np.random.normal(mu_x[i], Sigma_xx[i,i]) # interrogation of x_t
        vt = f(xt, t[i]) # interrogation of v_t
        # mean and variance updates
        mu_x = mu_x + Sigma_xv[:,i]*1/Sigma_vv[i,i] *(vt - mu_v[i])
        mu_v = mu_v + Sigma_vv[:,i]*(vt - mu_v[i])*1/Sigma_vv[i,i]
        Sigma_xx = Sigma_xx - 1/Sigma_vv[i,i]*np.outer(Sigma_xv[:,i], Sigma_xv[:,i])
        Sigma_xv = Sigma_xv - 1/Sigma_vv[i,i]*np.outer(Sigma_xv[:,i], Sigma_vv[i,:])
        Sigma_vv = Sigma_vv - 1/Sigma_vv[i,i]*np.outer(Sigma_vv[:,i], Sigma_vv[i,:])
    
    return np.random.multivariate_normal(mu_x, Sigma_xx)
