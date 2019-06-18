"""
.. module:: cov_exp

Covariance and cross-covariance functions for the solution process x_t and its derivative 

.. math:: v_t = dx_t/dt 

under the exponential correlation model.
"""
from numba import jit
from math import exp
import numpy as np

@jit
def cov_vv_ex(t,s,gamma,alpha):
    """Computes the covariance function for the derivative :math:`v_t`. 

    Parameters
    ----------
    
    t: [N] :obj:`numpy.ndarray` of float
        Time vector t
    s: [N] :obj:`numpy.ndarray` of float
        Time vector s
    gamma: float
        Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    alpha: float 
        Covariance scale parameter.
    
    Returns
    -------
    
    float
        Evaluates :math:`cov(v_t, v_s)`.
    """
    t_len = len(t)
    s_len = len(s)
    Sigma_vv = np.zeros((t_len, s_len))

    for i in range(t_len):
        for j in range(s_len):
            Sigma_vv[i,j] = exp(-abs(t[i]-s[j])/gamma)/alpha

    return Sigma_vv

@jit
def cov_xv_ex(t,s,gamma,alpha):
    """Computes the cross-covariance function for the solution process :math:`x_t` and its derivative :math:`v_t`. 
 
    Parameters
    ----------
    
    t: [N] :obj:`numpy.ndarray` of float
        Time vector t
    s: [N] :obj:`numpy.ndarray` of float
        Time vector s
    gamma: float
        Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    alpha: float 
        Covariance scale parameter.
    
    Returns
    -------
    
    float
        Evaluates :math:`cov(x_t, v_s)`.
    """
    t_len = len(t)
    s_len = len(s)
    Sigma_xv = np.zeros((t_len, s_len))

    for i in range(t_len):
        for j in range(s_len):
            if t[i] >= s[j]:
                Sigma_xv[i,j] = (2*gamma - gamma*exp(-s[j]/gamma) - gamma*exp((s[j]-t[i])/gamma))/alpha
            else:
                Sigma_xv[i,j] = (gamma*exp(-s[j]/gamma)*(exp(t[i]/gamma)-1))/alpha
        
    return Sigma_xv

@jit
def cov_xx_ex(t,s,gamma,alpha):
    """Computes the covariance function for the solution process :math:`x_t`. 
 
    Parameters
    ----------
    
    t: [N] :obj:`numpy.ndarray` of float
        Time vector t
    s: [N] :obj:`numpy.ndarray` of float
        Time vector s
    gamma: float
        Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    alpha: float 
        Covariance scale parameter.
    
    Returns
    -------
    
    float
        Evaluates :math:`cov(x_t, x_s)`.
    """
    t_len = len(t)
    s_len = len(s)
    Sigma_xx = np.zeros((t_len, s_len))
    gamma2 = gamma*gamma

    for i in range(t_len):
        for j in range(s_len):
            Sigma_xx[i,j] = (-gamma2 + gamma2*exp(-t[i]/gamma) - gamma2*exp(-abs(t[i]-s[j])/gamma) + gamma2*exp(-s[j]/gamma) + 2*gamma*min(t[i],s[j]))/alpha
    
    return Sigma_xx
