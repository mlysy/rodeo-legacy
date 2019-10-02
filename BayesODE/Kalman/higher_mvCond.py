"""
.. module:: higher_mvCond

Calculates parameters used in kalman_ode_higher.

"""
import numpy as np
from BayesODE.var_car import var_car
from BayesODE._mou_car import _mou_car

def higher_mvCond(delta_t, roots, sigma):
    
    """Calculate A, and V in Y_{n+1} ~ p(Y_{n+1} | Y_n) using 
    .. :math: `A = e^{-\Gamma \Delta t}` and 
    .. :math: `V = V_{\Delta t}`.
        
    Parameters
    ----------
    
    delta_t : [1] :obj:`numpy.ndarray` of float
        Step size between simulation points
    roots : [p] :obj:`numpy.ndarray` of float
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma : float
        Parameter in mOU volatility matrix

    Returns
    -------
    
    A : [p,p]  :obj:`numpy.ndarray` of float
        :math: `A = e^{-\Gamma \Delta t}`
    V : [p,p] :obj:`numpy.ndarray` of float
        :math: `V = V_{\Delta t}`
    """

    delta = np.array(-roots)
    _, Q = _mou_car(roots, sigma)
    Q_inv = np.linalg.pinv(Q)
    V = var_car(delta_t, roots, sigma)[0]
    A = np.matmul(Q * np.exp(-delta * delta_t[0]), Q_inv)

    return A, V
    