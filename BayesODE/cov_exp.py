"""
.. module:: cov_exp
    :synopsis: Covariance and cross-covariance functions for the solution process `x_t` and its derivative `v_t = dx_t/dt` under the exponential correlation model.
"""

@jit
def cov_vv_ex(t,s,gamma):
    return exp(-abs(t-s)/gamma)

@jit
def cov_xv_ex(t,s,gamma):
    if t >= s:
        ans = 2*gamma - gamma*exp(-s/gamma) - gamma*exp((s-t)/gamma)
    else:
        ans = gamma*exp(-s/gamma)*(exp(t/gamma)-1)
        
    return ans

@jit
def cov_xx_ex(t,s,gamma):
    ans = -gamma**2 + gamma**2*exp(-t/gamma) - gamma**2*exp(abs(t-s)/gamma) + gamma**2*exp(-s/gamma) + 2*gamma*min(t,s)
    return ans
