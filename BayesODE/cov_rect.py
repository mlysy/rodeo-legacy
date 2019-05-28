"""
.. module:: cov_rect
    :synopsis: Covariance and cross-covariance functions for the solution process x_t and its derivative v_t = dx_t/dt under the rectangular-kernel correlation model.
"""
from numba import jit
import numpy as np

@jit
def cov_vv_re(t,s,gamma,alpha):
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
    t_len = len(t)
    s_len = len(s)
    Sigma_vv = np.zeros((t_len, s_len))
    
    for i in range(t_len):
        for j in range(s_len):
            Sigma_vv[i,j] = (min(t[i],s[j])- max(t[i],s[j]) + 2*gamma)*(min(t[i],s[j]) - max(t[i],s[j]) > -2*gamma)/alpha
    
    return Sigma_vv

@jit
def cov_xv_re(t,s,gamma,alpha):
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
    Sigma_xv = np.zeros((t_len, s_len))

    for i in range(t_len):
        for j in range(s_len):
            Sigma_xv[i,j] = ((2*gamma)*(min(t[i]-gamma,s[j]+gamma) - max(gamma,s[j]-gamma))*(min(t[i]-gamma,s[j]+gamma) > max(gamma,s[j]-gamma)) \
            +(0.5*min(gamma,min(t[i]-gamma,s[j]+gamma))**2 + (gamma)*min(gamma,min(t[i]-gamma,s[j]+gamma)) - 0.5*(s[j]-gamma)**2 -(gamma)*(s[j]-gamma))*(min(gamma,min(t[i]-gamma,s[j]+gamma)) > (s[j]-gamma)) \
            +((t[i]+gamma)*min(t[i]+gamma,s[j]+gamma) - 0.5*min(t[i]+gamma,s[j]+gamma)**2 - (t[i]+gamma)*max(gamma,max(t[i]-gamma,s[j]-gamma)) + 0.5*max(gamma,max(t[i]-gamma,s[j]-gamma))**2)*(min(t[i]+gamma,s[j]+gamma) > max(gamma,max(t[i]-gamma,s[j]-gamma))) \
            + (t[i])*(-max(t[i],s[j]) + 2*gamma)*(-max(t[i],s[j]) > -2*gamma))

            Sigma_xv = Sigma_xv/alpha
            
    return Sigma_xv

@jit
def cov_xx_re(t,s,gamma,alpha):
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
    Sigma_xx = np.zeros((t_len, s_len))

    for i in range(t_len):
        for j in range(s_len):
            Sigma_xx[i,j] =((4*gamma**2)*(min(t[i],s[j])-2*gamma)*(min(t[i],s[j])>(2*gamma)) \
            +(2*gamma)*((s[j]+gamma)*min(t[i]-gamma,s[j]+gamma) - 0.5*min(t[i]-gamma,s[j]+gamma)**2 - (s[j]+gamma)*max(gamma,s[j]-gamma) + 0.5*max(gamma,s[j]-gamma)**2)*(min(t[i]-gamma,s[j]+gamma)>max(gamma,s[j]-gamma)) \
            +((1/3)*min(gamma,min(t[i]-gamma,s[j]-gamma))**3 + gamma*min(gamma,min(t[i]-gamma,s[j]-gamma))**2 + (gamma)**2*min(gamma,min(t[i]-gamma,s[j]-gamma))+(1/3)*(gamma)**3)*(min(gamma,min(t[i]-gamma,s[j]-gamma))>(-gamma)) \
            +(s[j])*(0.5*min(gamma,t[i]-gamma)**2 + (gamma)*min(gamma,t[i]-gamma)- 0.5*(s[j]-gamma)**2 - (gamma)*(s[j]-gamma))*(min(gamma,t[i]-gamma)>(s[j]-gamma)) \
            +(2*gamma)*((t[i]+gamma)*min(t[i]+gamma,s[j]-gamma)-0.5*min(t[i]+gamma,s[j]-gamma)**2-(t[i]+gamma)*max(gamma,t[i]-gamma) + 0.5*max(gamma,t[i]-gamma)**2)*(min(t[i]+gamma,s[j]-gamma)>max(gamma,t[i]-gamma)) \
            +((t[i]+gamma)*(s[j]+gamma)*min(t[i]+gamma,s[j]+gamma) - 0.5*(t[i]+s[j]+2*gamma)*min(t[i]+gamma,s[j]+gamma)**2 + (1/3)*min(t[i]+gamma,s[j]+gamma)**3 - (t[i]+gamma)*(s[j]+gamma)*max(gamma,max(t[i]-gamma,s[j]-gamma)) + 0.5*(t[i]+s[j]+2*gamma)*max(gamma,max(t[i]-gamma,s[j]-gamma))**2 - (1/3)*max(gamma,max(t[i]-gamma,s[j]-gamma))**3)*(min(t[i],s[j])>max(0,max(t[i]-2*gamma,s[j]-2*gamma))) \
            +(t[i])*(0.5*min(gamma,s[j]-gamma)**2 + (gamma)*min(gamma,s[j]-gamma)- 0.5*(t[i]-gamma)**2 - (gamma)*(t[i]-gamma))*(min(gamma,s[j]-gamma)>(t[i]-gamma)) \
            +(t[i])*(s[j])*(2*gamma-max(t[i],s[j]))*(2*gamma>max(t[i],s[j])))

            Sigma_xx = Sigma_xx/alpha
    
    return Sigma_xx
    
