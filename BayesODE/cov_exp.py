"""
.. module:: cov_exp
    :synopsis: Covariance and cross-covariance functions for the solution process x_t and its derivative v_t = dx_t/dt under the exponential correlation model.
"""
from numba import jit
from math import exp

@jit
def cov_vv_ex(t,s,gamma):
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

    return exp(-abs(t-s)/gamma)

@jit
def cov_xv_ex(t,s,gamma):
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

    if t >= s:
        ans = 2*gamma - gamma*exp(-s/gamma) - gamma*exp((s-t)/gamma)
    else:
        ans = gamma*exp(-s/gamma)*(exp(t/gamma)-1)
        
    return ans

@jit
def cov_xx_ex(t,s,gamma):
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
    ans = -gamma2 + gamma2*exp(-t/gamma) - gamma2*exp(t-s/gamma) + gamma2*exp(-s/gamma) + 2*gamma*min(t,s)
    return ans
