"""
.. module:: cov_rect

Covariance and cross-covariance functions for the solution process x_t and its derivative 

.. math:: v_t = dx_t/dt 

under the rectangular-kernel correlation model.
"""
#from numba import jit
import numpy as np

#@jit
def cov_vv_re(t, s, gamma, alpha):
    """
    Computes the covariance function for the derivative :math:`v_t`. 

    Parameters
    ----------
    t : ndarray(dim_t)
        Time vector t
    s : ndarray(dim_s)
        Time vector s
    gamma : float
        Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    alpha : float 
        Covariance scale parameter.
    
    Returns
    -------
    vvSigma : ndarray(dim_t, dim_s) 
        Evaluates :math:`cov(v_t, v_s)`.

    """
    t_len = len(t)
    s_len = len(s)
    vvSigma = np.zeros((t_len, s_len))
    
    for i in range(t_len):
        for j in range(s_len):
            vvSigma[i, j] = (min(t[i], s[j])- max(t[i], s[j]) + 2*gamma) * (min(t[i], s[j]) - max(t[i], s[j]) > -2*gamma) / alpha
    return vvSigma

#@jit
def cov_xv_re(t,s,gamma,alpha):
    """
    Computes the cross-covariance function for the solution process :math:`x_t` and its derivative :math:`v_t`. 
 
    Parameters
    ----------
    
    t : ndarray(dim_t)
        Time vector t
    s : ndarray(dim_s)
        Time vector s
    gamma : float
        Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    alpha : float 
        Covariance scale parameter.
    
    Returns
    -------
    xvSigma : ndarray(dim_t, dim_s)
        Evaluates :math:`cov(x_t, v_s)`.

    """
    t_len = len(t)
    s_len = len(s)
    xvSigma = np.zeros((t_len, s_len))

    for i in range(t_len):
        for j in range(s_len):
            xvSigma[i, j] = (2*gamma * (min(t[i] - gamma, s[j] + gamma) - max(gamma, s[j] - gamma)) * (min(t[i] - gamma, s[j] + gamma) > max(gamma,s[j]-gamma)) \
            + (0.5*min(gamma, min(t[i] - gamma, s[j] + gamma))**2 + gamma*min(gamma, min(t[i] - gamma, s[j] + gamma)) - 0.5*(s[j] - gamma)**2 -gamma * (s[j] - gamma)) \
                * (min(gamma,min(t[i]-gamma,s[j]+gamma)) > (s[j]-gamma)) \
            + ((t[i] + gamma) * min(t[i] + gamma, s[j] + gamma) - 0.5*min(t[i] + gamma, s[j] + gamma)**2 - (t[i] + gamma) * max(gamma, max(t[i] - gamma, s[j] - gamma)) + 0.5*max(gamma, max(t[i] - gamma, s[j] - gamma))**2) \
                * (min(t[i] + gamma, s[j] + gamma) > max(gamma, max(t[i] - gamma, s[j] - gamma))) \
            + t[i] * (-max(t[i], s[j]) + 2*gamma) * (-max(t[i], s[j]) > -2*gamma))

    xvSigma = xvSigma/alpha
    return xvSigma

#@jit
def cov_xx_re(t,s,gamma,alpha):
    """
    Computes the covariance function for the solution process :math:`x_t`. 
 
    Parameters
    ----------
    t : ndarray(dim_t)
        Time vector t
    s : ndarray(dim_s)
        Time vector s
    gamma : float
        Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    alpha : float 
        Covariance scale parameter.
    
    Returns
    -------
    xxSigma : ndarray(dim_t, dim_s) 
        Evaluates :math:`cov(x_t, x_s)`.

    """
    t_len = len(t)
    s_len = len(s)
    xxSigma = np.zeros((t_len, s_len))

    for i in range(t_len):
        for j in range(s_len):
            xxSigma[i, j] = ((4*gamma**2) * (min(t[i], s[j]) - 2*gamma) * (min(t[i], s[j]) > (2*gamma)) \
            + (2*gamma) * ((s[j] + gamma) * min(t[i] - gamma, s[j] + gamma) - 0.5*min(t[i] - gamma, s[j] + gamma)**2 \
            - (s[j] + gamma) * max(gamma, s[j] - gamma) + 0.5*max(gamma, s[j] - gamma)**2) * (min(t[i] - gamma, s[j] + gamma) > max(gamma, s[j] - gamma)) \
            + ((1/3) * min(gamma, min(t[i] - gamma, s[j] - gamma))**3 + gamma*min(gamma, min(t[i] - gamma,s[j] - gamma))**2 \
            + gamma**2*min(gamma, min(t[i] - gamma, s[j] - gamma))+(1/3)*(gamma)**3)*(min(gamma, min(t[i] - gamma, s[j] - gamma)) > (-gamma)) \
            + s[j]*(0.5*min(gamma, t[i] - gamma)**2 + gamma * min(gamma, t[i]-gamma)- 0.5*(s[j] - gamma)**2 - gamma * (s[j] - gamma)) * (min(gamma, t[i] - gamma) > (s[j] - gamma)) \
            + 2*gamma*((t[i] + gamma) * min(t[i] + gamma, s[j] - gamma) - 0.5*min(t[i] + gamma, s[j] - gamma)**2 - (t[i] + gamma) * max(gamma, t[i] - gamma) \
            + 0.5*max(gamma, t[i] - gamma)**2) * (min(t[i] + gamma, s[j] - gamma) > max(gamma, t[i] - gamma)) \
            + ((t[i] + gamma) * (s[j] + gamma) * min(t[i] + gamma, s[j] + gamma) - 0.5*(t[i] + s[j] + 2*gamma) * min(t[i] + gamma, s[j] + gamma)**2 \
            + (1/3)*min(t[i] + gamma, s[j] + gamma)**3 - (t[i] + gamma) * (s[j] + gamma) * max(gamma, max(t[i] - gamma, s[j] - gamma)) \
            + 0.5*(t[i] + s[j] + 2*gamma) * max(gamma, max(t[i] - gamma, s[j] - gamma))**2 \
            - (1/3)*max(gamma, max(t[i] - gamma, s[j] - gamma))**3) * (min(t[i], s[j]) > max(0, max(t[i] - 2*gamma, s[j] - 2*gamma))) \
            + t[i]*(0.5*min(gamma, s[j] - gamma)**2 + gamma*min(gamma, s[j] - gamma)- 0.5*(t[i] - gamma)**2 - gamma*(t[i] - gamma)) * (min(gamma, s[j] - gamma) > (t[i] - gamma)) \
            + t[i]*s[j]*(2*gamma - max(t[i], s[j]))*(2*gamma > max(t[i], s[j])))

    xxSigma = xxSigma/alpha
    return xxSigma
    
