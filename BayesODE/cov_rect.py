"""
.. module:: cov_rect
    :synopsis: Covariance and cross-covariance functions for the solution process `x_t` and its derivative `v_t = dx_t/dt` under the rectangular-kernel correlation model.
"""

@jit
def cov_vv_re(t,s,gamma):
    """Calculate covariance function of `v_t`.

    Evaluates `cov(v_t, v_s)`.
    """
    return (min(t,s)-max(t,s) + 2*gamma)*(min(t,s) - max(t,s) > -2*gamma)

@jit
def cov_xv_re(t,s,gamma):
    """Calculate cross-covariance between `x_t` and `v_t`.

    Evaluates `cov(x_t, v_s)`.
    """
    ans =((4*gamma**2)*(min(t,s)-2*gamma)*(min(t,s)>(2*gamma)) \
    +(2*gamma)*((s+gamma)*min(t-gamma,s+gamma) - 0.5*min(t-gamma,s+gamma)**2 - (s+gamma)*max(gamma,s-gamma) + 0.5*max(gamma,s-gamma)**2)*(min(t-gamma,s+gamma)>max(gamma,s-gamma)) \
    +((1/3)*min(gamma,min(t-gamma,s-gamma))**3 + gamma*min(gamma,min(t-gamma,s-gamma))**2 + (gamma)**2*min(gamma,min(t-gamma,s-gamma))+(1/3)*(gamma)**3)*(min(gamma,min(t-gamma,s-gamma))>(-gamma)) \
    +(s)*(0.5*min(gamma,t-gamma)**2 + (gamma)*min(gamma,t-gamma)- 0.5*(s-gamma)**2 - (gamma)*(s-gamma))*(min(gamma,t-gamma)>(s-gamma)) \
    +(2*gamma)*((t+gamma)*min(t+gamma,s-gamma)-0.5*min(t+gamma,s-gamma)**2-(t+gamma)*max(gamma,t-gamma) + 0.5*max(gamma,t-gamma)**2)*(min(t+gamma,s-gamma)>max(gamma,t-gamma)) \
    +((t+gamma)*(s+gamma)*min(t+gamma,s+gamma) - 0.5*(t+s+2*gamma)*min(t+gamma,s+gamma)**2 + (1/3)*min(t+gamma,s+gamma)**3 - (t+gamma)*(s+gamma)*max(gamma,max(t-gamma,s-gamma)) + 0.5*(t+s+2*gamma)*max(gamma,max(t-gamma,s-gamma))**2 - (1/3)*max(gamma,max(t-gamma,s-gamma))**3)*(min(t,s)>max(0,max(t-2*gamma,s-2*gamma))) \
    +(t)*(0.5*min(gamma,s-gamma)**2 + (gamma)*min(gamma,s-gamma)- 0.5*(t-gamma)**2 - (gamma)*(t-gamma))*(min(gamma,s-gamma)>(t-gamma)) \
    +(t)*(s)*(2*gamma-max(t,s))*(2*gamma>max(t,s)))

    return ans
    
@jit
def cov_xx_re(t,s,gamma):
    """Calculate covariance function of `x_t`.

    Evaluates `cov(x_t, x_s)`.
    """
    ans= ((2*gamma)*(min(t-gamma,s+gamma) - max(gamma,s-gamma))*(min(t-gamma,s+gamma) > max(gamma,s-gamma)) \
    +(0.5*min(gamma,min(t-gamma,s+gamma))**2 + (gamma)*min(gamma,min(t-gamma,s+gamma)) - 0.5*(s-gamma)**2 -(gamma)*(s-gamma))*(min(gamma,min(t-gamma,s+gamma)) > (s-gamma)) \
    +((t+gamma)*min(t+gamma,s+gamma) - 0.5*min(t+gamma,s+gamma)**2 - (t+gamma)*max(gamma,max(t-gamma,s-gamma)) + 0.5*max(gamma,max(t-gamma,s-gamma))**2)*(min(t+gamma,s+gamma) > max(gamma,max(t-gamma,s-gamma))) \
    + (t)*(-max(t,s) + 2*gamma)*(-max(t,s) > -2*gamma))
    
    return ans
