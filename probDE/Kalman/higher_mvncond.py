"""
.. module:: higher_mvncond

Calculate the state transition matrix, and state variance matrix used in the model in Kalman solver.

"""
import numpy as np

from probDE.Kalman.var_car import var_car
from probDE.Kalman._mou_car import _mou_car

def higher_mvncond(delta_t, roots, sigma):    
    """
    Calculate the state transition matrix, and state variance matrix used in the model in Kalman solver.
        
    Args:
        delta_t (ndarray(1)): A vector containing the step size between simulation points.
        roots (ndarray(n_dim_roots)): Roots to the p-th order polynomial of the car(p) 
            process (roots must be negative).
        sigma (float): Parameter in mOU volatility matrix

    Returns:
        (tuple):
        - **wgtState** (ndarray(n_dim_roots, n_dim_roots)): The state transition matrix defined in 
          Kalman solver.
        - **varState** (ndarray(n_dim_roots, n_dim_roots)): The state variance matrix defined in
          Kalman solver.
    """
    _, Q = _mou_car(roots, sigma)
    Q_inv = np.linalg.pinv(Q)
    varState = var_car(delta_t, roots, sigma)[:, :, 0]
    wgtState = np.matmul(Q * np.exp(-roots*delta_t[0]), Q_inv, order='F')
    return wgtState, varState
    