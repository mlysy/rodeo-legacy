"""
.. module:: higher_mvCond

Calculates parameters used in kalman_ode_higher.

"""
import numpy as np

from probDE.Kalman.var_car import var_car
from probDE.Kalman._mou_car import _mou_car

def higher_mvncond(delta_t, roots, sigma):    
    """
    Calculate wgtState(A), and varState(V) in Y_{n+1} ~ p(Y_{n+1} | Y_n) using 
    .. :math: `A = e^{-\Gamma \Delta t}` and 
    .. :math: `V = V_{\Delta t}`.
        
    Parameters
    ----------
    delta_t : ndarray(1)
        A vector containing the step size between simulation points
    roots : ndarray(n_dim_roots)
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma : float
        Parameter in mOU volatility matrix

    Returns
    -------
    wgtState : ndarray(n_dim_roots, n_dim_roots)
        :math: `A = e^{-\Gamma \Delta t}`
    varState : ndarray(n_dim_roots, n_dim_roots)
        :math: `V = V_{\Delta t}`
    """
    _, Q = _mou_car(roots, sigma)
    Q_inv = np.linalg.pinv(Q)
    varState = var_car(delta_t, roots, sigma)[0]
    wgtState = np.matmul(Q * np.exp(-roots*delta_t[0]), Q_inv)
    return wgtState, varState
    
