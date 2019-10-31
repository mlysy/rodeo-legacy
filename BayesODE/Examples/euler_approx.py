"""
.. module:: euler_approx

Uses the Euler approximation for univariate ordinary differential equations (ODEs).

"""
import numpy as np

def euler_approx(fun, tseq, x_init):
    """
    Pad x0 with p-len(x0) 0s at the end of x0.

    Parameters
    ----------
    fun : function 
        ODE function :math:`w' x_t = F(x_t, t)` taking arguments :math:`x` and :math:`t`.
    x_init : ndarray(n_dim_ode)
        The initial value of the IVP problem. 
    tseq : ndarray(n_dim_time)
        Time points to evaluate the Euler approximation.

    Returns
    euler_x : ndarray(n_dim_time, n_dim_ode)
        Approximation of the solution at each time point in tseq.

    """
    h = tseq[1] - tseq[0]
    euler_x = np.zeros((len(tseq), len(x_init)-1))
    euler_x[0] = x_init[0:-1]
    for t in range(len(tseq)-1):
        euler_x[t+1] = euler_x[t] + h*fun(euler_x[t], tseq[t])
    return euler_x
    