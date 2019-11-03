"""
.. module:: _mou_car

Calculates parameters for the mOU CAR(p) process.

"""
import numpy as np
from math import exp
        
def _mou_car(roots, sigma=1., test=False):
    """
    Calculates parameters for the mOU CAR(p) process.
    
    Parameters
    ----------
    roots: ndarray(n_dim_roots)
        Roots to the p-th order polynomial of the car(p) process (roots must be negative)
    sigma: float
        Parameter in mOU volatility matrix
    test: bool
        If True, return Sigma, and Gamma.

    Returns
    -------
    Gamma : ndarray(n_dim_roots, n_dim_roots)
        :math:`\Gamma` in CAR process.

    Sigma : ndarray(n_dim_roots, n_dim_roots)
        :math:`\Sigma` in CAR process.
    
    Sigma_tilde: ndarray(n_dim_roots, n_dim_roots)
        :math:`\widetilde{\Sigma}` in CAR process.

    Q : ndarray(n_dim_roots, n_dim_roots)
        :math:`Q` in CAR process.

    """
    D = np.diag(roots)
    p = len(roots)
    Q = np.zeros((p, p))
    row = np.ones(p)
        
    for i in range(p):
        Q[i] = row
        row = -row * roots
    Q_inv = np.linalg.pinv(Q)

    if test:
        Sigma = np.zeros((p, p))
        Sigma[p-1, p-1] = sigma * sigma
        Gamma = np.zeros((p,p)) # Q*D*Q^-1
        Gamma[range(p - 1),range(1, p)] = -1.
        Gamma[p-1] = np.linalg.multi_dot([Q[p-1], D, Q_inv])
        
        return Gamma, Sigma


    Sigma = np.zeros(p)
    Sigma[p-1] = sigma * sigma
    Sigma_tilde = np.matmul(Q_inv * Sigma, Q_inv.T)
    
    return Sigma_tilde, Q
