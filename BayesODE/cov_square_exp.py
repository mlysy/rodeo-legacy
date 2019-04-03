"""
.. module:: cov_square_exp
    :synopsis: Covariance and cross-covariance functions for the solution process x_t and its derivative v_t = dx_t/dt under the squared-exponential correlation model, cov(v_t, v_s) = exp(-|t-s|^2/gamma^2).
"""

from numba import jit
from math import exp, sqrt, pi, erf

@jit
def cov_vv_se(t, s, gamma):
    """Computes the covariance function for the derivative :math:`v_t`. 

    :param t: Time point at t
    :type t: float
    :param s: Time point at s
    :type s: float
    :param gamma: Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    :type gamma: float
    :returns: Evaluates :math:`cov(v_t, v_s)`.
    :rtype: float
    """
    # no point in dragging around sqrt(pi), etc.
    # instead, the output should be:
    # z = (t-s)/gamma
    # return exp(-z*z) # better to use z*z instead of z**2, google it :)
    # I've left your original code for now...
    gamma2 = gamma*gamma

    return exp(-((s - t)**2) / (4 * gamma2)) * sqrt(pi) * gamma

@jit
def cov_xv_se(t, s, gamma):
    """Computes the cross-covariance function for the solution process :math:`x_t` and its derivative :math:`v_t`. 

    :param t: time point at t
    :type t: float
    :param s: time point at s
    :type s: float
    :param gamma: Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    :type gamma: float
    :returns: Evaluates :math:`cov(x_t, v_s)`.
    :rtype: float
    """
    gamma2 = gamma*gamma

    ans = pi * gamma2 * erf((t - s) / (2 * gamma)) + pi * gamma2 * erf(s / (2 * gamma))
    return ans

@jit
def cov_xx_se(t, s, gamma):
    """Computes the covariance function for the solution process :math:`x_t`. 

    :param t: time point at t
    :type t: float
    :param s: time point at s
    :type s: float
    :param gamma: Decorrelation time, such that :math:`cov(v_t, v_{t+\gamma}) = 1/e`.
    :type gamma: float
    :returns: Evaluates :math:`cov(x_t, x_s)`.
    :rtype: float
    """
    gamma2 = gamma*gamma
    gamma3 = gamma*gamma2

    ans = pi * gamma2 * (s) * erf(s / (2 * gamma)) \
        + 2 * sqrt(pi) * gamma3 * exp(-(s**2) / (4 * gamma2)) \
        - pi * gamma2 * (t - s) * erf((t - s) / (2 * gamma)) \
        - 2 * sqrt(pi) * gamma3 * exp(-(t - s)**2 / (4 * gamma2)) \
        + pi * gamma2 * s * erf(t / (2 * gamma)) \
        + 2 * sqrt(pi) * gamma3 * exp(-(t**2) / (4 * gamma2)) \
        - 2 * sqrt(pi) * gamma3
    return ans


