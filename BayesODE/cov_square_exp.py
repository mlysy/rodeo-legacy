"""
.. module:: cov_square_exp
    :synopsis: Covariance and cross-covariance functions for the solution process x_t and its derivative v_t = dx_t/dt under the squared-exponential correlation model, cov(v_t, v_s) = exp(-|t-s|^2/gamma^2).
"""

from numba import jit
from math import exp, sqrt, pi, erf
import numpy as np

@jit
def cov_vv_se(t, s, gamma, alpha):
    """Computes the covariance function for the derivative :math:`v_t`. 

    :param t: Time point at t
    :type t: float
    :param s: Time point at s
    :type s: float
    :param gamma: Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    :type gamma: float
    :param alpha: Covariance scale parameter.
    :type alpha: float
    :returns: Evaluates :math:`cov(v_t, v_s)`.
    :rtype: float
    """
    # no point in dragging around sqrt(pi), etc.
    # instead, the output should be:
    # z = (t-s)/gamma
    # return exp(-z*z) # better to use z*z instead of z**2, google it :)
    # I've left your original code for now...
    t_len = len(t)
    s_len = len(s)
    Sigma_vv = np.zeros((t_len, s_len))
    gamma2 = gamma*gamma

    for i in range(t_len):
        for j in range(s_len):
            Sigma_vv[i,j] = exp(-((s[j] - t[i])**2) / (4 * gamma2)) * sqrt(pi) * gamma / alpha

    return Sigma_vv

@jit
def cov_xv_se(t, s, gamma, alpha):
    """Computes the cross-covariance function for the solution process :math:`x_t` and its derivative :math:`v_t`. 

    :param t: time point at t
    :type t: float
    :param s: time point at s
    :type s: float
    :param gamma: Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    :type gamma: float
    :param alpha: Covariance scale parameter.
    :type alpha: float
    :returns: Evaluates :math:`cov(x_t, v_s)`.
    :rtype: float
    """
    t_len = len(t)
    s_len = len(s)
    Sigma_xv = np.zeros((t_len, s_len))
    gamma2 = gamma*gamma

    for i in range(t_len):
        for j in range(s_len):
            Sigma_xv[i,j] = pi * gamma2 * erf((t[i] - s[j]) / (2 * gamma)) + pi * gamma2 * erf(s[j] / (2 * gamma))
            
    Sigma_xv = Sigma_xv/alpha
    
    return Sigma_xv

@jit
def cov_xx_se(t, s, gamma, alpha):
    """Computes the covariance function for the solution process :math:`x_t`. 

    :param t: time point at t
    :type t: float
    :param s: time point at s
    :type s: float
    :param gamma: Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    :type gamma: float
    :param alpha: Covariance scale parameter.
    :type alpha: float
    :returns: Evaluates :math:`cov(x_t, x_s)`.
    :rtype: float
    """
    t_len = len(t)
    s_len = len(s)
    Sigma_xx = np.zeros((t_len, s_len))
    gamma2 = gamma*gamma
    gamma3 = gamma*gamma2

    for i in range(t_len):
        for j in range(s_len):
            Sigma_xx[i,j] = pi * gamma2 * (s[j]) * erf(s[j] / (2 * gamma)) \
                + 2 * sqrt(pi) * gamma3 * exp(-(s[j]**2) / (4 * gamma2)) \
                - pi * gamma2 * (t[i] - s[j]) * erf((t[i] - s[j]) / (2 * gamma)) \
                - 2 * sqrt(pi) * gamma3 * exp(-(t[i] - s[j])**2 / (4 * gamma2)) \
                + pi * gamma2 * t[i] * erf(t[i] / (2 * gamma)) \
                + 2 * sqrt(pi) * gamma3 * exp(-(t[i]**2) / (4 * gamma2)) \
                - 2 * sqrt(pi) * gamma3
            
    Sigma_xx = Sigma_xx/alpha

    return Sigma_xx


