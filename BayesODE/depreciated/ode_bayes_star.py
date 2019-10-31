"""
.. module:: ode_bayes_star
    :synopsis: Compute the mean and variance of p(X∣V⋆)p(X∣V⋆)
"""
import numpy as np

def ode_bayes_star(mu_x, mu_v, Sigma_vv, Sigma_xx, Sigma_xv, vstar):
    """Computes the mean and variance of :math:`p(\XX \mid \VV_\star)`.
    :param mu_x: `N` prior mean vector of mean of :math:`x_t`
    :type mu_x: float
    :param mu_v: `N` prior mean vector of mean of :math:`v_t`
    :type mu_v: float
    :param Sigma_vv: `N x N` prior covariance matrix **cov(v(tseq), v(tseq))**.
    :type Sigma_vv: float
    :param Sigma_xx: `N x N` prior covariance matrix **cov(x(tseq), x(tseq))**.    
    :type Sigma_xx: float
    :param Sigma_xv: `N x N` prior cross-covariance matrix **cov(x(tseq), v(tseq))**, where :math:`v_t = dx_t/dt`.
    :type Sigma_xv: float
    :param vstar: predetermined V_t after N interrogations
    :type vstar: float
    :returns: The mean and variance of :math:`p(\XX \mid \VV_\star)
    :rtype: (float, float)
    """

    sigma_xx = np.matrix(Sigma_xx)
    sigma_xv = np.matrix(Sigma_xv)
    sigma_vv = np.matrix(Sigma_vv)
    mu_x = mu_x.reshape(len(mu_x),1)
    mu_v = mu_v.reshape(len(mu_v),1)
    vstar = vstar.reshape(len(vstar),1)

    mu_bar = mu_x + sigma_xv*np.linalg.pinv(sigma_vv).dot(vstar - mu_v)
    sigma_bar = sigma_xx - sigma_xv*np.linalg.pinv(sigma_vv)*sigma_xv.T
    
    return np.matrix.flatten(mu_bar), sigma_bar